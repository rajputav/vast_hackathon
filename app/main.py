"""Minimal app on the team's VSS stack: checks every service, searches, plays clips, asks Cosmos."""
import asyncio
import base64
import os

import httpx
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel
from starlette.background import BackgroundTask

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


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", "8080")))
