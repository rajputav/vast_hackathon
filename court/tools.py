"""Court retrieval tools — ``search``/``neighbors`` on the VSS retrieval API, ``look`` on Cosmos3-Reason.

Set ``COURT_DEMO=1`` to use the hand-written fixtures instead of the live stack.
"""

from __future__ import annotations

import base64
import glob
import json
import os
import re
import socket
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Any

# When set, forces the fake-data path even after real VSS implementations exist.
DEMO_MODE: bool = os.environ.get("COURT_DEMO") == "1"

# ---------------------------------------------------------------------------
# VSS backend: credentials, cached JWT, HTTP helper
# ---------------------------------------------------------------------------

# Skill-documented names first, then the aliases the deployed app uses.
_ENV_ALIASES: dict[str, tuple[str, ...]] = {
    "INGRESS_URL": ("INGRESS_URL", "VSS_URL", "BACKEND"),
    "USERNAME": ("USERNAME", "VSS_USERNAME"),
    "PASSWORD": ("PASSWORD", "VSS_PASSWORD"),
    "GPU_BEARER_TOKEN": ("GPU_BEARER_TOKEN", "COSMOS_TOKEN"),
    "COSMOS3_REASON_URL": ("COSMOS3_REASON_URL", "COSMOS_URL"),
    "COSMOS3_REASON_MODEL": ("COSMOS3_REASON_MODEL",),
}
# Shared GPU host documented in .cursor/skills/gpu (the URL is not in the team config).
_DEFAULTS: dict[str, str] = {"COSMOS3_REASON_URL": "http://166.19.38.112:8001"}
_CONFIG_GLOB = "/config/*.config"
_HTTP_TIMEOUT = float(os.environ.get("COURT_VSS_TIMEOUT", "120"))
_LOOK_TIMEOUT = float(os.environ.get("COURT_LOOK_TIMEOUT", "60"))

_config_cache: dict[str, str] | None = None
_token: str | None = None
_token_lock = threading.Lock()


def _read_team_config() -> dict[str, str]:
    """Parse ``/config/<team>.config`` (KEY=VALUE lines) once. Values are never logged."""
    global _config_cache
    if _config_cache is not None:
        return _config_cache
    values: dict[str, str] = {}
    for path in sorted(glob.glob(_CONFIG_GLOB)):
        try:
            with open(path, encoding="utf-8") as fh:
                for raw in fh:
                    line = raw.strip()
                    if not line or line.startswith("#") or "=" not in line:
                        continue
                    if line.startswith("export "):
                        line = line[len("export "):]
                    key, _, val = line.partition("=")
                    val = val.strip()
                    if len(val) >= 2 and val[0] == val[-1] and val[0] in "\"'":
                        val = val[1:-1]
                    values.setdefault(key.strip(), val)
        except OSError:
            continue
    _config_cache = values
    return values


def _setting(name: str) -> str:
    """Resolve a setting from the environment (any alias) or the team config file."""
    for alias in _ENV_ALIASES.get(name, (name,)):
        val = os.environ.get(alias)
        if val:
            return val
    cfg = _read_team_config()
    for alias in _ENV_ALIASES.get(name, (name,)):
        if cfg.get(alias):
            return cfg[alias]
    if name in _DEFAULTS:
        return _DEFAULTS[name]
    raise RuntimeError(
        f"{name} is not set. Export it or make sure {_CONFIG_GLOB} exists (see config.example)."
    )


def _base_url() -> str:
    return _setting("INGRESS_URL").rstrip("/")


def _login() -> str:
    body = json.dumps({"username": _setting("USERNAME"), "password": _setting("PASSWORD")}).encode()
    req = urllib.request.Request(
        f"{_base_url()}/api/v1/auth/login",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=_HTTP_TIMEOUT) as resp:
            payload = json.load(resp)
    except urllib.error.HTTPError as exc:
        if exc.code == 401:
            raise RuntimeError("VSS login rejected (401): check USERNAME/PASSWORD for this tenant.") from None
        raise RuntimeError(f"VSS login failed with HTTP {exc.code}") from None
    token = payload.get("access_token")
    if not token:
        raise RuntimeError("VSS login response had no access_token")
    return token


def _get_token(refresh: bool = False) -> str:
    """Return the cached JWT, logging in (or re-logging in when ``refresh``) as needed."""
    global _token
    with _token_lock:
        if _token and not refresh:
            return _token
        _token = _login()
        return _token


def _api(
    method: str,
    path: str,
    *,
    params: dict[str, Any] | None = None,
    body: dict[str, Any] | None = None,
    token_in_query: bool = False,
) -> Any:
    """Call ``/api/v1/<path>`` with the cached JWT; on 401 re-login once and retry, on 502/503/504
    back off and retry up to three times.

    Raises :class:`urllib.error.HTTPError` for non-401 failures so callers can
    treat e.g. 404 as "not available".
    """
    refreshed = False
    for attempt in range(4):
        token = _get_token(refresh=refreshed and attempt > 0)
        query = dict(params or {})
        headers = {"Accept": "application/json"}
        if token_in_query:
            query["token"] = token
        else:
            headers["Authorization"] = f"Bearer {token}"
        url = f"{_base_url()}/api/v1/{path.lstrip('/')}"
        if query:
            url += "?" + urllib.parse.urlencode(query)
        data = None
        if body is not None:
            data = json.dumps(body).encode()
            headers["Content-Type"] = "application/json"
        req = urllib.request.Request(url, data=data, headers=headers, method=method)
        try:
            with urllib.request.urlopen(req, timeout=_HTTP_TIMEOUT) as resp:
                raw = resp.read()
                return json.loads(raw) if raw else None
        except urllib.error.HTTPError as exc:
            if exc.code == 401 and not refreshed:
                refreshed = True
                continue  # token expired or revoked: refresh and retry once
            if exc.code in (502, 503, 504) and attempt < 3:
                # The shared backend drops requests under load; one blip shouldn't sink a 60-exhibit trial.
                time.sleep(1.5 * (attempt + 1))
                continue
            raise
    raise RuntimeError("unreachable")  # pragma: no cover


def _playback_url(source: str) -> str:
    """Seekable proxy stream URL; this endpoint takes the JWT as ``?token=``."""
    query = urllib.parse.urlencode({"source": source, "token": _get_token()})
    return f"{_base_url()}/api/v1/videos/stream?{query}"


def _fetch_segment_bytes(source: str, timeout: float) -> bytes:
    """Download a segment clip through ``/videos/stream``; re-login once on 401."""
    for attempt in range(2):
        if attempt:
            _get_token(refresh=True)
        req = urllib.request.Request(_playback_url(source), headers={"Accept": "video/mp4"})
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return resp.read()
        except urllib.error.HTTPError as exc:
            if exc.code == 401 and attempt == 0:
                continue
            raise
    raise RuntimeError("unreachable")  # pragma: no cover


# ---------------------------------------------------------------------------
# Cosmos3-Reason GPU endpoint (OpenAI-compatible chat; see .cursor/skills/gpu)
# ---------------------------------------------------------------------------

_cosmos_model: str | None = None
_LOOK_INSTRUCTION = (
    "Answer with exactly one of yes / no / unclear on the first line, "
    "then one sentence of reasoning."
)
_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)
_ANSWER_RE = re.compile(r"\b(yes|no|unclear)\b", re.IGNORECASE)


def _cosmos_headers() -> dict[str, str]:
    return {
        "Authorization": f"Bearer {_setting('GPU_BEARER_TOKEN')}",
        "Content-Type": "application/json",
        "Accept": "application/json",
    }


def witness_model() -> str | None:
    """Model id that answers :func:`look` (or None if not resolvable); never raises."""
    try:
        return _cosmos_model_id(timeout=10.0)
    except Exception:  # noqa: BLE001
        return None


def _cosmos_model_id(timeout: float) -> str:
    """Model id from env/config, else discovered once via ``GET /v1/models``."""
    global _cosmos_model
    if _cosmos_model:
        return _cosmos_model
    try:
        _cosmos_model = _setting("COSMOS3_REASON_MODEL")
        return _cosmos_model
    except RuntimeError:
        pass
    req = urllib.request.Request(f"{_setting('COSMOS3_REASON_URL').rstrip('/')}/v1/models", headers=_cosmos_headers())
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        data = json.load(resp)
    _cosmos_model = data["data"][0]["id"]
    return _cosmos_model


def _cosmos_chat(content: list[dict[str, Any]], *, timeout: float, max_tokens: int = 200) -> str:
    body = {
        "model": _cosmos_model_id(timeout=min(timeout, 10.0)),
        "messages": [{"role": "user", "content": content}],
        "max_tokens": max_tokens,
        "temperature": 0,
    }
    req = urllib.request.Request(
        f"{_setting('COSMOS3_REASON_URL').rstrip('/')}/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers=_cosmos_headers(),
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        data = json.load(resp)
    return (data["choices"][0]["message"].get("content") or "").strip()


def _parse_look(text: str) -> dict[str, str]:
    """First line → ``answer`` (yes/no/unclear), remainder → ``reason``."""
    text = _THINK_RE.sub("", text).strip()
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if not lines:
        return {"answer": "unclear", "reason": "empty model response"}
    first, rest = lines[0], lines[1:]
    match = _ANSWER_RE.search(first)
    answer = match.group(1).lower() if match else "unclear"
    # Tolerate "Yes. The truck is ..." on a single line: keep the trailing text as reason.
    tail = first[match.end():].lstrip(" .:,;-—") if match else first
    reason = " ".join(([tail] if tail else []) + rest).strip()
    return {"answer": answer, "reason": reason or first}


def _vss_look(segment_id: str, question: str) -> dict:
    """Send the segment clip + question to Cosmos3-Reason; never raises."""
    deadline = time.monotonic() + _LOOK_TIMEOUT

    def remaining() -> float:
        left = deadline - time.monotonic()
        if left <= 0:
            raise TimeoutError
        return left

    try:
        if not segment_id.startswith("s3://"):
            return {"answer": "unclear", "reason": f"unknown segment id: {segment_id}"}
        clip = _fetch_segment_bytes(segment_id, timeout=remaining())
        b64 = base64.b64encode(clip).decode()
        content = [
            {"type": "text", "text": f"{question.strip()}\n{_LOOK_INSTRUCTION}"},
            {"type": "video_url", "video_url": {"url": f"data:video/mp4;base64,{b64}"}},
        ]
        return _parse_look(_cosmos_chat(content, timeout=remaining()))
    except (TimeoutError, socket.timeout):
        return {"answer": "unclear", "reason": "timeout"}
    except urllib.error.URLError as exc:
        if isinstance(exc.reason, (TimeoutError, socket.timeout)):
            return {"answer": "unclear", "reason": "timeout"}
        detail = f"HTTP {exc.code}" if isinstance(exc, urllib.error.HTTPError) else str(exc.reason)
        return {"answer": "unclear", "reason": f"error: {detail}"}
    except Exception as exc:  # noqa: BLE001 - look() must never break the caller
        return {"answer": "unclear", "reason": f"error: {type(exc).__name__}: {exc}"[:200]}


# ---------------------------------------------------------------------------
# VSS response → contract dict
# ---------------------------------------------------------------------------


def _as_dict(value: Any) -> dict[str, Any]:
    """VastDB returns JSON columns as strings; accept either form."""
    if isinstance(value, dict):
        return value
    if isinstance(value, str) and value.strip():
        try:
            parsed = json.loads(value)
        except ValueError:
            return {}
        return parsed if isinstance(parsed, dict) else {}
    return {}


def _yolo_counts(hit: dict[str, Any]) -> dict[str, int]:
    """``{class: count}`` from ``object_counts`` (falls back to ``perception_json``)."""
    counts = _as_dict(hit.get("object_counts")) or _as_dict(hit.get("perception_json")).get("object_counts") or {}
    out: dict[str, int] = {}
    for cls, n in counts.items():
        try:
            out[str(cls)] = int(n)
        except (TypeError, ValueError):
            continue
    return out


_NO_DETECTOR: dict[str, Any] = {"model": None, "frames": 0, "frames_seen": {}}


def _detections(hit: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """``(bboxes, detector)`` from the YOLO sidecar.

    ``bboxes`` are the boxes of the segment's densest frame (object_counts are "max per
    frame", so that frame matches them). ``detector`` summarises coverage: the detector
    model, how many frames it ran on, and in how many frames each class was seen — so a
    court can tell "seen in 3 of 150 frames as 'boat'" from "seen throughout". Both are
    best-effort: no sidecar (404) or any failure yields ``([], _NO_DETECTOR)``.
    """
    source = hit.get("source")
    if not source:
        return [], dict(_NO_DETECTOR)
    try:
        sidecar = _api("GET", "videos/detections", params={"source": source})
    except (urllib.error.URLError, RuntimeError, ValueError):
        return [], dict(_NO_DETECTOR)
    frames = (sidecar or {}).get("frames") or []
    if not frames:
        return [], dict(_NO_DETECTOR)

    frames_seen: dict[str, int] = {}
    for f in frames:
        for cls in {d.get("label") or d.get("class") for d in (f.get("detections") or [])}:
            if cls:
                frames_seen[cls] = frames_seen.get(cls, 0) + 1
    detector = {
        "model": (sidecar or {}).get("source"),  # e.g. "yolo11_coco"
        "frames": len(frames),
        "frames_seen": dict(sorted(frames_seen.items(), key=lambda kv: -kv[1])),
    }

    motion = _person_motion(frames, float((sidecar or {}).get("fps") or 30.0))
    if motion is not None:
        # Pixel speed from a camera that is itself moving measures the camera, not the person.
        moving = bool(_MOVING_CAMERA.search(str(hit.get("camera_id") or "")))
        motion["admissible"] = not moving
        if moving:
            motion["inadmissible_because"] = "moving camera"
            motion["abrupt"] = False
    detector["motion"] = motion

    best = max(frames, key=lambda f: len(f.get("detections") or []))
    boxes: list[dict[str, Any]] = []
    for det in best.get("detections") or []:
        bbox = det.get("bbox") or det.get("xyxy")
        if not bbox or len(bbox) != 4:
            continue
        boxes.append(
            {
                "class": det.get("label") or det.get("class"),
                "confidence": det.get("confidence"),
                "xyxy": [float(v) for v in bbox],
            }
        )
    return boxes, detector


_MOVING_CAMERA = re.compile(r"gopro|dash|body|helmet|bike|handheld|mobile|drone", re.IGNORECASE)
_MOTION_EDGE_S = 0.3        # ignore the first/last 0.3 s: tracks start and end with jitter
_MOTION_MIN_TRACK = 0.5     # the person must be tracked in at least half the frames
_MOTION_MIN_MEDIAN = 40.0   # px/s; below this the person is standing and any ratio is noise
_MOTION_SPIKE = 2.5         # peak / median at which we call it an abrupt change


def _person_motion(frames: list[dict[str, Any]], fps: float) -> dict[str, Any] | None:
    """Kinematics of the longest-tracked person, from per-frame YOLO boxes.

    A person struck by something changes speed abruptly; one walking or pushing does not.
    Centroids are linked greedily frame to frame, speeds smoothed over five samples, and the
    peak compared with the median. Pixel speeds only mean anything from a fixed camera; the
    court is told the ratio and the moment, and weighs it. None when nobody is tracked.
    """
    tracks: list[list[tuple[float, float, float]]] = []
    for i, f in enumerate(frames):
        t = float(f.get("time_sec") if f.get("time_sec") is not None else i / fps)
        cents = [
            ((b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0)
            for d in (f.get("detections") or [])
            if (d.get("label") or d.get("class")) == "person"
            for b in [d.get("bbox") or d.get("xyxy") or []]
            if len(b) == 4
        ]
        used: set[int] = set()
        for tr in tracks:
            lt, lx, ly = tr[-1]
            if t - lt > 0.5:
                continue
            cands = [(j, ((cx - lx) ** 2 + (cy - ly) ** 2) ** 0.5) for j, (cx, cy) in enumerate(cents) if j not in used]
            if not cands:
                break
            j, dist = min(cands, key=lambda c: c[1])
            if dist < 120:
                tr.append((t, *cents[j]))
                used.add(j)
        tracks.extend([(t, *c)] for j, c in enumerate(cents) if j not in used)
    if not tracks:
        return None
    tr = max(tracks, key=len)
    if len(tr) < _MOTION_MIN_TRACK * len(frames) or len(tr) < 10:
        return None
    speeds = [
        ((((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5) / (t2 - t1), t2)
        for (t1, x1, y1), (t2, x2, y2) in zip(tr, tr[1:])
        if t2 > t1
    ]
    v = [s for s, _ in speeds]
    smooth = [sum(v[max(0, i - 2): i + 3]) / len(v[max(0, i - 2): i + 3]) for i in range(len(v))]
    median = sorted(smooth)[len(smooth) // 2]
    end = tr[-1][0]
    mid = [i for i, (_, t) in enumerate(speeds) if _MOTION_EDGE_S <= t <= end - _MOTION_EDGE_S]
    if not mid or median < _MOTION_MIN_MEDIAN:
        return {"tracked_frames": len(tr), "median_px_s": round(median), "peak_px_s": None, "peak_t": None, "spike_ratio": None}
    pk = max(mid, key=lambda i: smooth[i])
    return {
        "tracked_frames": len(tr),
        "median_px_s": round(median),
        "peak_px_s": round(smooth[pk]),
        "peak_t": round(speeds[pk][1], 2),
        "spike_ratio": round(smooth[pk] / median, 2),
        "abrupt": smooth[pk] / median >= _MOTION_SPIKE,
    }


def _bboxes(hit: dict[str, Any]) -> list[dict[str, Any]]:
    """Per-object boxes for the segment's densest frame (see :func:`_detections`)."""
    return _detections(hit)[0]


def _to_segment(hit: dict[str, Any], *, with_bboxes: bool) -> dict[str, Any]:
    source = hit.get("source") or ""
    bboxes, detector = _detections(hit) if with_bboxes else ([], dict(_NO_DETECTOR))
    return {
        "id": source or hit.get("filename") or "",
        "video": hit.get("original_video") or source,
        "ts": float(hit.get("segment_start_sec") or 0.0),
        "caption": (hit.get("reasoning_content") or "").strip(),
        # The model that wrote the caption. look() calls the same endpoint, so a court should
        # know when its tape witness is also its stenographer.
        "captioner": hit.get("cosmos_model"),
        "camera": hit.get("camera_id"),
        "yolo": _yolo_counts(hit),
        "bboxes": bboxes,
        "detector": detector,
        "playback_url": _playback_url(source) if source else "",
    }


def _vss_search(
    query: str,
    camera: str | None = None,
    k: int = 10,
    *,
    min_similarity: float = 0.1,
    with_bboxes: bool = True,
) -> list[dict]:
    body: dict[str, Any] = {
        "query": query,
        "top_k": max(1, min(int(k), 100)),
        "llm_top_n": 1,  # backend requires >= 1; we ignore the synthesis
        "min_similarity": min_similarity,
        "include_public": True,
    }
    if camera:
        body["metadata_filters"] = {"camera_id": camera}
    resp = _api("POST", "search", body=body) or {}
    hits = resp.get("results") or []
    return [_to_segment(h, with_bboxes=with_bboxes) for h in hits[: body["top_k"]]]


def _vss_neighbors(segment_id: str, *, with_bboxes: bool = True) -> tuple[dict | None, dict | None]:
    """(prev, next) by ``segment_start_sec`` within the segment's parent video.

    Resolves the parent via ``GET /videos/metadata?source=`` and lists its
    segments via ``GET /tools/segments?original_video=``. Unknown ids → (None, None).
    """
    if not segment_id.startswith("s3://"):
        return None, None
    try:
        meta = _api("GET", "videos/metadata", params={"source": segment_id}) or {}
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            return None, None
        raise
    parent = meta.get("original_video")
    if not parent:
        return None, None
    listing = _api("GET", "tools/segments", params={"original_video": parent}) or {}
    rows = listing.get("segments") if isinstance(listing, dict) else listing
    rows = [r for r in (rows or []) if r.get("source")]
    rows.sort(key=lambda r: (float(r.get("segment_start_sec") or 0.0), int(r.get("segment_number") or 0)))
    idx = next((i for i, r in enumerate(rows) if r["source"] == segment_id), None)
    if idx is None:
        return None, None
    prev = _to_segment(rows[idx - 1], with_bboxes=with_bboxes) if idx > 0 else None
    nxt = _to_segment(rows[idx + 1], with_bboxes=with_bboxes) if idx < len(rows) - 1 else None
    return prev, nxt


# ---------------------------------------------------------------------------
# Demo fixtures
# ---------------------------------------------------------------------------

# Shared bbox payload used by the three demo segments (caption/yolo disagreement fixture).
_SHARED_BBOXES: list[dict[str, Any]] = [
    {
        "class": "bus",
        "confidence": 0.91,
        "xyxy": [120.0, 80.0, 420.0, 300.0],
    },
    {
        "class": "bicycle",
        "confidence": 0.84,
        "xyxy": [450.0, 220.0, 520.0, 310.0],
    },
]

_DEMO_VIDEO = "demo://nyc_streets/bike_lane_chunk_0001.mp4"


def _segment(
    *,
    segment_id: str,
    ts: float,
    caption: str,
    yolo: dict[str, int],
) -> dict[str, Any]:
    return {
        "id": segment_id,
        "video": _DEMO_VIDEO,
        "ts": ts,
        "caption": caption,
        "yolo": dict(yolo),
        "bboxes": [dict(b) for b in _SHARED_BBOXES],
        "playback_url": f"demo://playback/{segment_id}",
    }


# Three consecutive demo segments; bboxes are identical across all three.
_DEMO_PREV = _segment(
    segment_id="demo-seg-001",
    ts=10.0,
    caption="traffic approaches the intersection",
    yolo={"bus": 1, "bicycle": 1},
)
_DEMO_HIT = _segment(
    segment_id="demo-seg-002",
    ts=15.0,
    caption="a truck is stopped in the bike lane",
    yolo={"bus": 1, "bicycle": 1},
)
_DEMO_NEXT = _segment(
    segment_id="demo-seg-003",
    ts=20.0,
    caption="vehicles clear the crosswalk",
    yolo={"bus": 1, "bicycle": 1},
)

_DEMO_BY_ID: dict[str, dict[str, Any]] = {
    _DEMO_PREV["id"]: _DEMO_PREV,
    _DEMO_HIT["id"]: _DEMO_HIT,
    _DEMO_NEXT["id"]: _DEMO_NEXT,
}

_DEMO_ORDER = [_DEMO_PREV["id"], _DEMO_HIT["id"], _DEMO_NEXT["id"]]


def _demo_search(query: str, camera: str | None = None, k: int = 10) -> list[dict]:
    """Return exactly one hand-written segment (caption/yolo mismatch fixture)."""
    del query, camera, k  # unused in demo path
    return [dict(_DEMO_HIT)]


def _demo_neighbors(segment_id: str) -> tuple[dict | None, dict | None]:
    """Return (prev, next) in the demo video; None at edges."""
    if segment_id not in _DEMO_BY_ID:
        return None, None
    idx = _DEMO_ORDER.index(segment_id)
    prev = dict(_DEMO_BY_ID[_DEMO_ORDER[idx - 1]]) if idx > 0 else None
    nxt = dict(_DEMO_BY_ID[_DEMO_ORDER[idx + 1]]) if idx < len(_DEMO_ORDER) - 1 else None
    return prev, nxt


def _demo_look(segment_id: str, question: str) -> dict:
    """Fixed visual answer that contradicts the truck-in-bike-lane caption."""
    del segment_id, question  # unused in demo path
    return {
        "answer": "no",
        "reason": "A bus is parked at a stop; the bike lane is clear.",
    }


def search(query: str, camera: str | None = None, k: int = 10) -> list[dict]:
    """Search the archive for segments matching ``query``.

    Returns a list of segment dicts, each with keys:
    ``id``, ``video``, ``ts``, ``caption``, ``yolo``, ``bboxes``, ``playback_url``.

    ``id``/``playback_url`` refer to the 5-second segment clip, ``video`` to its
    parent upload, ``ts`` to the segment start (seconds into the parent).
    ``camera`` filters on the ``camera_id`` metadata column when given.
    """
    if DEMO_MODE:
        return _demo_search(query, camera=camera, k=k)
    return _vss_search(query, camera=camera, k=k)


def neighbors(segment_id: str) -> tuple[dict | None, dict | None]:
    """Return ``(prev, next)`` segments in the same video, or ``None`` at edges.

    Each segment uses the same dict shape as :func:`search`. Ordering is by
    ``segment_start_sec`` within the parent ``video``.
    """
    if DEMO_MODE:
        return _demo_neighbors(segment_id)
    return _vss_neighbors(segment_id)


def segments(video: str) -> list[dict]:
    """Every segment of one parent ``video``, in order, in the :func:`search` dict shape."""
    if DEMO_MODE:
        return [dict(_DEMO_BY_ID[sid]) for sid in _DEMO_ORDER if _DEMO_BY_ID[sid]["video"] == video]
    listing = _api("GET", "tools/segments", params={"original_video": video}) or {}
    rows = listing.get("segments") if isinstance(listing, dict) else listing
    rows = [r for r in (rows or []) if r.get("source")]
    rows.sort(key=lambda r: (float(r.get("segment_start_sec") or 0.0), int(r.get("segment_number") or 0)))
    return [_to_segment(r, with_bboxes=True) for r in rows]


def look(segment_id: str, question: str) -> dict:
    """Answer a yes/no visual question about a segment.

    Returns ``{"answer": "yes"|"no"|"unclear", "reason": str}``.

    ``segment_id`` is the segment clip URI returned by :func:`search` (``id``).
    The clip is sent to Cosmos3-Reason as base64 video with the question; a
    60s overall budget applies and failures yield ``unclear`` instead of raising.
    """
    if DEMO_MODE:
        return _demo_look(segment_id, question)
    return _vss_look(segment_id, question)
