"""Court retrieval tools — committed signatures; demo stubs for now."""

from __future__ import annotations

import os
from typing import Any

# When set, forces the fake-data path even after real VSS implementations exist.
DEMO_MODE: bool = os.environ.get("COURT_DEMO") == "1"

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
    """
    if DEMO_MODE:
        return _demo_search(query, camera=camera, k=k)
    # Real VSS path not wired yet — stubs still return fake data.
    return _demo_search(query, camera=camera, k=k)


def neighbors(segment_id: str) -> tuple[dict | None, dict | None]:
    """Return ``(prev, next)`` segments in the same video, or ``None`` at edges.

    Each segment uses the same dict shape as :func:`search`.
    """
    if DEMO_MODE:
        return _demo_neighbors(segment_id)
    # Real VSS path not wired yet — stubs still return fake data.
    return _demo_neighbors(segment_id)


def look(segment_id: str, question: str) -> dict:
    """Answer a yes/no visual question about a segment.

    Returns ``{"answer": "yes"|"no"|"unclear", "reason": str}``.
    """
    if DEMO_MODE:
        return _demo_look(segment_id, question)
    # Real VSS path not wired yet — stubs still return fake data.
    return _demo_look(segment_id, question)
