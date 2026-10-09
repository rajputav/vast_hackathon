"""Court trial — put a claim about the archive on trial, one exhibit at a time.

    python -m court.trial --claim "..." --camera nyc_bike_gopro-1 --k 10 --subpoenas 4 --out transcript.txt
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time
from pathlib import Path
from typing import Any

from court.llm import chat
from court.tools import look, neighbors, search

COURT_DIR = Path(__file__).parent
LEDGER = COURT_DIR / "ledger.jsonl"

# Defense and judge make the calls that decide the verdict. The fast default (8B) mislabeled a
# truck-vs-bus mismatch as 'scope' in 5 of 6 runs; 70B got 'object' 6 of 6. Prosecutor and
# bailiff only write prose, so they keep the fast default.
REASONING_MODEL = os.environ.get("COURT_REASONING_MODEL", "meta-llama/Llama-3.3-70B-Instruct")

PROSECUTOR = (
    "You are the prosecutor in a trial where the defendant is a pattern of behavior in a video archive. "
    "Given a claim and one segment's caption, state in one confident sentence what this exhibit shows in "
    "support of the claim. Never mention detections. Never hedge."
)

DEFENSE = (
    "You are defense counsel. Captions are written by a vision-language model and are hearsay; object "
    "detections and adjacent segments are physical evidence. Given the claim, the caption, the YOLO class "
    "counts and bboxes for this segment, and the captions and YOLO counts of the previous and next segments, "
    "decide whether to object. Object with type 'object' if the caption names an object class the detector "
    "did not see or a different class is present (e.g. caption says truck, detector saw bus). Object with "
    "type 'motion' if the caption says stopped/parked/blocking/stationary but bboxes change substantially "
    "across segments, or says moving/passing but bboxes are stable. Object with type 'scope' if the caption "
    "describes something real but not what the claim requires (wrong lane, wrong actor, wrong place). "
    "Otherwise 'none'. Return only JSON: {\"objection\": \"none|object|motion|scope\", \"reason\": \"one "
    "sentence citing the specific evidence\"}."
)

JUDGE = (
    "You are the judge. You rule on one exhibit given the claim, the caption, the detections, and the "
    "defense's objection. ADMIT only if the detections corroborate the caption on the point the claim needs "
    "and no valid objection stands. STRIKE if the objection is supported by the evidence. SUBPOENA if the "
    "caption and detections cannot settle it — write one yes/no question that a model watching the clip "
    "could answer to decide the point. Return only JSON: {\"ruling\": \"ADMIT|STRIKE|SUBPOENA\", "
    "\"question\": \"...\", \"line\": \"one dry, formal, slightly theatrical sentence announcing the ruling\"}."
)

BAILIFF = (
    "You are the bailiff. Given the claim and the counts of admitted and struck exhibits, read the verdict "
    "in one formal, slightly theatrical sentence. State the number of exhibits admitted and struck."
)

OBJECTIONS = ("none", "object", "motion", "scope")
RULINGS = ("ADMIT", "STRIKE", "SUBPOENA")

DEFENSE_SCHEMA = {
    "type": "object",
    "properties": {"objection": {"enum": list(OBJECTIONS)}, "reason": {"type": "string"}},
    "required": ["objection", "reason"],
}
JUDGE_SCHEMA = {
    "type": "object",
    "properties": {
        "ruling": {"enum": list(RULINGS)},
        "question": {"type": "string"},
        "line": {"type": "string"},
    },
    "required": ["ruling", "question", "line"],
}

LOOK_TO_RULING = {"yes": "ADMIT", "no": "STRIKE", "unclear": "STRIKE"}


def _neighbor_view(segment: dict | None) -> dict | None:
    if segment is None:
        return None
    return {"caption": segment.get("caption", ""), "yolo": segment.get("yolo", {})}


def _clean(text: str) -> str:
    """One line, no wrapping quotes — models like to quote their in-character lines."""
    return " ".join(str(text).split()).strip().strip('"').strip()


def _first_sentence(text: str) -> str:
    """Models tack on extras ('Number of exhibits admitted: 2 ...') after the asked-for sentence."""
    text = _clean(text)
    match = re.match(r'(.+?[.!?])["”]?(\s|$)', text)
    return _clean(match.group(1)) if match else text


def _chunk_name(segment: dict) -> str:
    """'s3://…/20261008_073013_GOPR0130_chunk_0015.mp4' -> 'GOPR0130_chunk_0015'."""
    name = Path(str(segment.get("video") or segment.get("id") or "?")).stem
    return re.sub(r"^\d{8}_\d{6}_", "", name)


def prosecute(claim: str, segment: dict) -> str:
    return _clean(chat(PROSECUTOR, f"Claim: {claim}\nCaption: {segment['caption']}"))


def defend(claim: str, segment: dict, prev: dict | None, nxt: dict | None) -> dict:
    evidence = {
        "claim": claim,
        "caption": segment["caption"],
        "yolo": segment.get("yolo", {}),
        "bboxes": segment.get("bboxes", []),
        "previous_segment": _neighbor_view(prev),
        "next_segment": _neighbor_view(nxt),
    }
    result = chat(DEFENSE, json.dumps(evidence, indent=2), json_schema=DEFENSE_SCHEMA, model=REASONING_MODEL)
    objection = str(result.get("objection", "none")).lower()
    return {
        "objection": objection if objection in OBJECTIONS else "none",
        "reason": _clean(result.get("reason", "")),
    }


def rule(claim: str, segment: dict, objection: dict) -> dict:
    evidence = {
        "claim": claim,
        "caption": segment["caption"],
        "yolo": segment.get("yolo", {}),
        "objection": objection,
    }
    result = chat(JUDGE, json.dumps(evidence, indent=2), json_schema=JUDGE_SCHEMA, model=REASONING_MODEL)
    ruling = str(result.get("ruling", "")).upper()
    if ruling not in RULINGS:  # an unparseable ruling can't be settled on the papers
        ruling = "SUBPOENA"
    question = _clean(result.get("question", ""))
    if ruling == "SUBPOENA" and not question:
        question = f"Does this clip show the following: {claim}?"
    return {"ruling": ruling, "question": question, "line": _clean(result.get("line", ""))}


def run_trial(claim: str, camera: str | None, k: int, subpoenas: int) -> tuple[list[str], dict]:
    lines: list[str] = []
    budget = subpoenas
    admitted: list[dict] = []
    struck = 0

    exhibits = search(claim, camera=camera, k=k)
    for n, segment in enumerate(exhibits, start=1):
        prev, nxt = neighbors(segment["id"])
        # ts is relative to the parent chunk, so the chunk name is what tells exhibits apart.
        where = f"{camera} · {_chunk_name(segment)}" if camera else _chunk_name(segment)
        lines.append(f"EXHIBIT {n}  [{where} @ {segment['ts']:g}s]")
        lines.append(f"  PROSECUTION: {prosecute(claim, segment)}")

        objection = defend(claim, segment, prev, nxt)
        if objection["objection"] == "none":
            lines.append("  DEFENSE:     No objection.")
        else:
            lines.append(f"  DEFENSE:     Objection ({objection['objection']}). {objection['reason']}")

        decision = rule(claim, segment, objection)
        lines.append(f"  JUDGE:       {decision['line']}")

        ruling = decision["ruling"]
        subpoenaed = False
        if ruling == "SUBPOENA":
            if budget > 0:
                budget -= 1
                subpoenaed = True
                viewed = look(segment["id"], decision["question"])
                answer = str(viewed.get("answer", "unclear")).lower()
                ruling = LOOK_TO_RULING.get(answer, "STRIKE")
                lines.append(f"  THE COURT VIEWS THE TAPE: {_clean(viewed.get('reason', ''))}")
            else:
                ruling = "STRIKE"
                lines.append("  THE COURT DECLINES TO VIEW THE TAPE: subpoena budget exhausted; exhibit struck.")

        if ruling == "ADMIT":
            admitted.append(segment)
        else:
            struck += 1

        with LEDGER.open("a") as f:
            f.write(json.dumps({
                "claim": claim,
                "segment_id": segment["id"],
                "ruling": ruling,
                "objection": objection["objection"],
                "reason": objection["reason"],
                "subpoenaed": subpoenaed,
                "ts": segment["ts"],
            }) + "\n")
        lines.append("")

    total = len(admitted) + struck
    confidence = len(admitted) / total if total else 0.0
    guilty = bool(total) and confidence >= 0.5
    # The outcome is decided here, not by the bailiff; without it the model can announce
    # "guilty" on a record of zero admitted exhibits.
    outcome = "guilty" if guilty else "not guilty — case dismissed"
    # 70B: the 8B bailiff ignored the one-sentence format and appended raw counts.
    bailiff = _first_sentence(chat(
        BAILIFF,
        f"Claim: {claim}\nExhibits admitted: {len(admitted)}\nExhibits struck: {struck}\n"
        f"Outcome (already decided): {outcome}",
        model=REASONING_MODEL,
    ))
    lines.append(f"VERDICT  [admitted {len(admitted)} · struck {struck} · confidence {confidence:.2f}]")
    lines.append(f"  BAILIFF:     {bailiff}")

    if guilty:
        report = write_report(claim, camera, admitted, struck, confidence)
        lines.append(f"SENTENCE: guilty — hazard report drafted to {report}")
    else:
        lines.append("SENTENCE: not guilty — case dismissed")

    summary = {"admitted": len(admitted), "struck": struck, "confidence": confidence}
    return lines, summary


def write_report(claim: str, camera: str | None, admitted: list[dict], struck: int, confidence: float) -> str:
    stamp = time.strftime("%Y%m%d_%H%M%S")
    path = COURT_DIR / f"report_{stamp}.md"
    rows = "\n".join(
        f"| {seg['id']} | {seg['ts']} | {seg['caption']} | {seg.get('playback_url', '')} |"
        for seg in admitted
    )
    path.write_text(
        f"# Hazard report\n\n"
        f"**Claim:** {claim}  \n"
        f"**Camera:** {camera or 'all'}  \n"
        f"**Verdict:** guilty — {len(admitted)} admitted, {struck} struck, confidence {confidence:.2f}  \n"
        f"**Filed:** {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        f"## Admitted exhibits\n\n"
        f"| Segment | ts | Caption | Playback |\n|---|---|---|---|\n{rows}\n"
    )
    return str(path.relative_to(COURT_DIR.parent))


def main() -> None:
    parser = argparse.ArgumentParser(description="Put a claim about the video archive on trial.")
    parser.add_argument("--claim", required=True, help="the behavior on trial, in plain language")
    parser.add_argument("--camera", default=None, help="limit the search to one camera")
    parser.add_argument("--k", type=int, default=10, help="number of exhibits to search for")
    parser.add_argument("--subpoenas", type=int, default=4, help="max clips the court may view")
    parser.add_argument("--out", default=None, help="also write the transcript here")
    args = parser.parse_args()

    lines, _ = run_trial(args.claim, args.camera, args.k, args.subpoenas)
    transcript = "\n".join(lines)
    print(transcript)
    if args.out:
        Path(args.out).write_text(transcript + "\n")


if __name__ == "__main__":
    main()
