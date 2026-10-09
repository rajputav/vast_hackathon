"""Minimal app on the team's VSS stack: checks every service, searches, plays clips, asks Cosmos,
and runs courtroom trials (court/) streamed to the browser exhibit by exhibit."""
import asyncio
import base64
import json
import os
import queue
import threading
import uuid

import httpx
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel
from starlette.background import BackgroundTask

from court.trial import iter_trial

VSS_URL = os.environ["VSS_URL"].rstrip("/")
VSS_USERNAME = os.environ["VSS_USERNAME"]
VSS_PASSWORD = os.environ["VSS_PASSWORD"]
COSMOS_URL = os.environ["COSMOS_URL"].rstrip("/")
COSMOS_TOKEN = os.environ["COSMOS_TOKEN"]
WANDB_URL = "https://api.inference.wandb.ai/v1"
WANDB_API_KEY = os.environ.get("WANDB_API_KEY", "")
WANDB_PROJECT = os.environ.get("WANDB_PROJECT", "")  # "<team>/<project>"
INDEX = os.path.join(os.path.dirname(__file__), "index.html")

app = FastAPI()
client = httpx.AsyncClient(timeout=httpx.Timeout(300.0, connect=15.0))
_token: str | None = None
_token_lock = asyncio.Lock()


async def vss_token(refresh: bool = False) -> str:
    global _token
    async with _token_lock:
        if _token and not refresh:
            return _token
        r = await client.post(
            f"{VSS_URL}/api/v1/auth/login",
            json={"username": VSS_USERNAME, "password": VSS_PASSWORD},
        )
        r.raise_for_status()
        _token = r.json()["access_token"]
        return _token


async def vss_stream(source: str, headers: dict | None = None, stream: bool = False) -> httpx.Response:
    """GET a segment from VSS (token goes in the query string); re-login once on 401."""
    for attempt in range(2):
        req = client.build_request(
            "GET",
            f"{VSS_URL}/api/v1/videos/stream",
            params={"source": source, "token": await vss_token(refresh=attempt > 0)},
            headers=headers or {},
        )
        resp = await client.send(req, stream=stream)
        if resp.status_code != 401:
            return resp
        await resp.aclose()
    return resp


@app.get("/health")
async def health():
    return {"ok": True}


@app.get("/")
async def index():
    return FileResponse(INDEX, media_type="text/html")


@app.get("/api/check")
async def check():
    """Reachability of each service the app depends on."""
    async def probe(name, coro):
        try:
            r = await coro
            return name, {"ok": r.status_code == 200, "status": r.status_code}
        except Exception as exc:
            return name, {"ok": False, "error": str(exc)[:200]}

    async def vss():
        await vss_token(refresh=True)
        return await client.get(f"{VSS_URL}/health")

    results = await asyncio.gather(
        probe("vss", vss()),
        probe("cosmos", client.get(f"{COSMOS_URL}/v1/models", headers={"Authorization": f"Bearer {COSMOS_TOKEN}"})),
        probe("wandb", client.get(
            f"{WANDB_URL}/models",
            headers={"Authorization": f"Bearer {WANDB_API_KEY}", "OpenAI-Project": WANDB_PROJECT},
        )),
    )
    return dict(results)


class SearchReq(BaseModel):
    query: str
    top_k: int = 8


@app.post("/api/search")
async def search(req: SearchReq):
    for attempt in range(2):
        r = await client.post(
            f"{VSS_URL}/api/v1/search",
            headers={"Authorization": f"Bearer {await vss_token(refresh=attempt > 0)}"},
            json={"query": req.query, "top_k": req.top_k, "llm_top_n": 1, "min_similarity": 0.2},
        )
        if r.status_code != 401:
            break
    if r.status_code != 200:
        raise HTTPException(r.status_code, r.text[:500])
    return {
        "results": [
            {
                "source": x["source"],
                "filename": x["source"].rsplit("/", 1)[-1],
                "score": x.get("similarity_score"),
                "caption": x.get("reasoning_content") or "",
            }
            for x in r.json().get("results", [])
        ]
    }


@app.get("/api/clip")
async def clip(source: str, request: Request):
    """Relay a segment so the VSS token never reaches the browser. Range-capable for seeking."""
    fwd = {"Range": request.headers["range"]} if "range" in request.headers else {}
    resp = await vss_stream(source, headers=fwd, stream=True)
    if resp.status_code >= 400:
        await resp.aclose()
        raise HTTPException(resp.status_code, "could not stream clip")
    headers = {k: resp.headers[k] for k in ("content-length", "content-range", "accept-ranges") if k in resp.headers}
    return StreamingResponse(
        resp.aiter_raw(),
        status_code=resp.status_code,
        headers=headers,
        media_type="video/mp4",
        background=BackgroundTask(resp.aclose),
    )


class AskReq(BaseModel):
    source: str
    question: str = "Explain what is happening in this video."


@app.post("/api/ask")
async def ask(req: AskReq):
    """Send the whole segment to Cosmos and return its answer."""
    resp = await vss_stream(req.source)
    if resp.status_code != 200:
        raise HTTPException(resp.status_code, "could not fetch clip")
    b64 = base64.b64encode(resp.content).decode()
    headers = {"Authorization": f"Bearer {COSMOS_TOKEN}"}
    model = (await client.get(f"{COSMOS_URL}/v1/models", headers=headers)).json()["data"][0]["id"]
    r = await client.post(
        f"{COSMOS_URL}/v1/chat/completions",
        headers=headers,
        json={
            "model": model,
            "max_tokens": 800,
            "temperature": 0.2,
            "messages": [{"role": "user", "content": [
                {"type": "text", "text": req.question},
                {"type": "video_url", "video_url": {"url": f"data:video/mp4;base64,{b64}"}},
            ]}],
        },
    )
    if r.status_code != 200:
        raise HTTPException(502, f"Cosmos error {r.status_code}: {r.text[:300]}")
    return {"answer": r.json()["choices"][0]["message"]["content"].strip()}


# ---------------------------------------------------------------------------
# Courtroom: POST /api/trial starts one; GET /api/trial/{id}/events streams it (SSE);
# GET /api/trial/{id} is the current snapshot. Event payloads are court.trial.iter_trial events.
# ---------------------------------------------------------------------------

_DONE = object()


class Trial:
    def __init__(self, req: "TrialReq"):
        self.id = uuid.uuid4().hex[:8]
        self.request = req.model_dump()
        self.events: list[dict] = []          # everything emitted so far, for late joiners / snapshot
        self.subscribers: list[queue.Queue] = []
        self.done = False
        self.error: str | None = None
        self.lock = threading.Lock()

    def publish(self, event: dict) -> None:
        with self.lock:
            self.events.append(event)
            subs = list(self.subscribers)
        for q in subs:
            q.put(event)

    def finish(self) -> None:
        with self.lock:
            self.done = True
            subs = list(self.subscribers)
        for q in subs:
            q.put(_DONE)

    def subscribe(self) -> tuple[list[dict], queue.Queue | None]:
        """Return (history, live queue). Queue is None if the trial already finished."""
        with self.lock:
            history = list(self.events)
            if self.done:
                return history, None
            q: queue.Queue = queue.Queue()
            self.subscribers.append(q)
            return history, q

    def unsubscribe(self, q: queue.Queue) -> None:
        with self.lock:
            if q in self.subscribers:
                self.subscribers.remove(q)

    def snapshot(self) -> dict:
        with self.lock:
            exhibits = [e["record"] for e in self.events if e["type"] == "exhibit"]
            verdict = next((e["trial"] for e in self.events if e["type"] == "verdict"), None)
            docket = next((e["exhibits"] for e in self.events if e["type"] == "docket"), None)
        return {
            "id": self.id,
            "request": self.request,
            "docket": docket,
            "exhibits": exhibits,
            "trial": verdict,
            "done": self.done,
            "error": self.error,
        }


TRIALS: dict[str, Trial] = {}


class TrialReq(BaseModel):
    claim: str
    camera: str | None = None
    k: int = 10
    subpoenas: int | None = None  # cap on tapes viewed; None = the court views every exhibit
    whole_videos: bool = False    # try every segment of each video the search turns up


def _run_trial(trial: Trial) -> None:
    req = trial.request
    try:
        for event in iter_trial(
            req["claim"], req["camera"], req["k"], req["subpoenas"], whole_videos=req["whole_videos"]
        ):
            trial.publish(event)
    except Exception as exc:  # the UI needs to hear about it, not a dead stream
        trial.error = f"{type(exc).__name__}: {exc}"[:300]
        trial.publish({"type": "failed", "error": trial.error})  # not "error": that is EventSource's own event
    finally:
        trial.finish()


@app.post("/api/trial")
async def start_trial(req: TrialReq):
    trial = Trial(req)
    TRIALS[trial.id] = trial
    threading.Thread(target=_run_trial, args=(trial,), daemon=True, name=f"trial-{trial.id}").start()
    return {"id": trial.id}


@app.get("/api/trial/{trial_id}")
async def get_trial(trial_id: str):
    trial = TRIALS.get(trial_id)
    if trial is None:
        raise HTTPException(404, "no such trial")
    return trial.snapshot()


@app.get("/api/trial/{trial_id}/events")
async def trial_events(trial_id: str):
    trial = TRIALS.get(trial_id)
    if trial is None:
        raise HTTPException(404, "no such trial")

    def sse(event: dict) -> str:
        return f"event: {event['type']}\ndata: {json.dumps(event)}\n\n"

    async def stream():
        history, q = trial.subscribe()
        for event in history:
            yield sse(event)
        if q is None:
            yield "event: done\ndata: {}\n\n"
            return
        loop = asyncio.get_running_loop()

        def next_item():
            try:
                return q.get(timeout=15)
            except queue.Empty:
                return None

        try:
            while True:
                item = await loop.run_in_executor(None, next_item)
                if item is None:
                    yield ": keep-alive\n\n"      # keep nginx from closing an idle stream
                    continue
                if item is _DONE:
                    yield "event: done\ndata: {}\n\n"
                    return
                yield sse(item)
        finally:
            trial.unsubscribe(q)

    return StreamingResponse(
        stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.get("/api/cameras")
async def cameras():
    """camera_id values the archive knows, for the claim form."""
    for attempt in range(2):
        r = await client.get(
            f"{VSS_URL}/api/v1/metadata/values",
            params={"field": "camera_id", "limit": 100},
            headers={"Authorization": f"Bearer {await vss_token(refresh=attempt > 0)}"},
        )
        if r.status_code != 401:
            break
    if r.status_code != 200:
        raise HTTPException(r.status_code, r.text[:300])
    return {"cameras": r.json().get("values", [])}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", "8080")))
