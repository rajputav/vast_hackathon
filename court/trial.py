"""Court trial — put a claim about the archive on trial, one exhibit at a time.

    python -m court.trial --claim "..." --camera nyc_bike_gopro-1 --k 10 --subpoenas 4 \
        --out transcript.txt --json trial.json

Every exhibit becomes one record (see ``try_exhibit``) that the transcript, the ledger, the
hazard report and ``--json`` all render; UIs should consume the record, not the prose.
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
# Where the ledger and hazard reports go. In the pod the code is a read-only ConfigMap
# mount, so deploy.sh points this at /tmp/court.
OUT_DIR = Path(os.environ.get("COURT_OUT_DIR", str(COURT_DIR)))
LEDGER = OUT_DIR / "ledger.jsonl"

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
    "You are defense counsel. Captions are written by a vision-language model and are hearsay; YOLO "
    "detections and adjacent segments are physical evidence. Your duty is to object whenever this exhibit "
    "does not prove the claim. 'none' is reserved for exhibits where the caption asserts every element of "
    "the claim and the detections corroborate the classes it names. Agreeing that a truck is present is not "
    "a defense.\n\n"
    "Step 1 — break the claim into its required elements: the actor (an object class), the state or action "
    "(stopped, parked, moving, crossing...), and the place (in the bike lane, in the crosswalk, at the dock...). "
    "Frequency words in the claim (routinely, often, repeatedly, always) are NOT elements of a single exhibit; "
    "the pattern is decided at verdict by counting admitted exhibits. Never object because one clip does not "
    "show a pattern. Step 2 — check the caption against each element. Step 3 — check the caption's actor "
    "against the detections.\n\n"
    "How to read the evidence: 'yolo' is the maximum count per class seen in any frame of this 5-second "
    "segment and is the authoritative detection record. 'bboxes' are boxes from one representative frame, so a "
    "class present in 'yolo' but missing from 'bboxes' is NOT a contradiction. 'previous_segment' and "
    "'next_segment' are the adjacent 5-second segments of the same video.\n\n"
    "Objection types:\n"
    "- 'object': the caption's actor class is absent from 'yolo', or 'yolo' shows a different class where the "
    "actor should be (caption says truck, yolo saw only bus). Never object 'object' because bboxes omit a class "
    "or because counts differ.\n"
    "- 'motion': the caption says stopped/parked/stationary/blocking but the adjacent segments show the actor "
    "moving or gone; or the caption says moving/passing while the adjacent segments describe it stopped in the "
    "same place.\n"
    "- 'scope': the caption describes something real but does not assert every element of the claim — wrong "
    "actor class, wrong state (passing by or driving in traffic is not blocking), or wrong place (a truck in a "
    "traffic lane, at the curb, or 'on the street' is not in the bike lane unless the caption says so). A caption "
    "that never mentions the place or state the claim requires is 'scope', not 'none'. In the reason, say "
    "which it is: 'state never asserted' (the caption describes no blocking/stopping at all) or 'place unstated' "
    "(the caption asserts the actor and state but names only 'the lane', 'the road' or no place). The judge "
    "strikes the first outright and may view the tape for the second.\n"
    "- 'none': every element is asserted by the caption and the actor is corroborated by 'yolo'.\n\n"
    "Return only JSON: {\"objection\": \"none|object|motion|scope\", \"reason\": \"one sentence naming the "
    "missing or contradicted element and the specific evidence\"}."
)

JUDGE = (
    "You are the judge. You rule on one exhibit given the claim, the caption, the detections, and the "
    "defense's objection. ADMIT only if the caption asserts what the claim requires, the detections corroborate "
    "it, and no valid objection stands. Overrule and ADMIT when the objection is contradicted by the record "
    "(an 'object' objection where 'yolo' contains the class; a 'scope' objection where the caption plainly "
    "asserts the element; any objection that one clip does not prove a pattern — frequency words like "
    "'routinely' are decided at verdict, not per exhibit). STRIKE when the objection is supported by the record: "
    "the caption never asserts the required state or action at all, or the detections contradict the actor. "
    "Subpoenas are scarce: SUBPOENA only when the caption asserts the actor AND the state and the sole open "
    "question is the place — for example the caption says 'parked, partially blocking the lane' and the question "
    "is whether that lane is the bike lane.\n\n"
    "When you SUBPOENA, write two yes/no questions for a model that sees only the pixels of a 5-second clip "
    "(it has not read the caption or the claim). 'question' decides the open element; 'foundation' checks the "
    "premise that question takes for granted — that the place, marking or object it refers to is actually "
    "visible (e.g. 'Is a painted or signed bike lane visible on this street?'). Both must be neutral: ask about "
    "visible markings, signs and positions; do not presuppose the answer; do not accept the camera's own "
    "vehicle or equipment as evidence of anything. The exhibit is admitted only if both answers are yes.\n\n"
    "Return only JSON: {\"ruling\": \"ADMIT|STRIKE|SUBPOENA\", \"question\": \"...\", \"foundation\": \"...\", "
    "\"line\": \"one dry, formal, slightly theatrical sentence announcing the ruling\"}."
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
        "foundation": {"type": "string"},
        "line": {"type": "string"},
    },
    "required": ["ruling", "question", "foundation", "line"],
}

LOOK_TO_RULING = {"yes": "ADMIT", "no": "STRIKE", "unclear": "STRIKE"}
ANSWERS = ("yes", "no", "unclear")


def _neighbor_view(segment: dict | None) -> dict | None:
    if segment is None:
        return None
    return {
        "caption": segment.get("caption", ""),
        "yolo": segment.get("yolo", {}),
        "bboxes": segment.get("bboxes", []),
    }


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
    return {
        "ruling": ruling,
        "question": question,
        "foundation": _clean(result.get("foundation", "")),
        "line": _clean(result.get("line", "")),
    }


def view_tape(segment_id: str, question: str, foundation: str) -> dict:
    """Put the judge's questions to the clip; both must be answered yes to admit.

    Two independent framings have to agree before one 5-second clip decides an exhibit —
    a single leading question ("is the truck in the bike lane?") was being answered yes on
    the strength of the rider's own handlebars. Returns the subpoena record.
    """
    t0 = time.monotonic()
    answers: list[dict] = []
    for q in (question, foundation):
        if not q:
            continue
        viewed = look(segment_id, q)
        answer = str(viewed.get("answer", "unclear")).lower()
        answers.append({
            "question": q,
            "answer": answer if answer in ANSWERS else "unclear",
            "reason": _clean(viewed.get("reason", "")),
        })
    got = [a["answer"] for a in answers]
    if got and all(a == "yes" for a in got):
        verdict = "yes"
    elif "no" in got:
        verdict = "no"
    else:  # unclear, or no questions at all
        verdict = "unclear"
    main = answers[0] if answers else {"question": question, "answer": "unclear", "reason": "no question"}
    return {
        "question": main["question"],
        "answer": verdict,
        "reason": main["reason"],
        "answers": answers,
        "seconds": round(time.monotonic() - t0, 2),
    }


# ---------------------------------------------------------------------------
# Per-exhibit record — the one object the transcript, ledger, report and UI all render.
# ---------------------------------------------------------------------------

# Final disposition of an exhibit. NOT_REACHED = the judge wanted the tape but the subpoena
# budget was spent; that is a court-capacity limit, not an evidence failure, so it counts
# neither for nor against the claim.
DISPOSITIONS = ("ADMIT", "STRIKE", "NOT_REACHED")
# Why a STRIKE happened.
STRUCK_FOR = ("objection", "tape", "unclear")


def _public_segment(segment: dict) -> dict:
    """The search() contract dict minus playback_url, which embeds the VSS JWT.

    UIs should play ``id`` through their own proxy (the app has ``/api/clip?source=``).
    """
    return {k: v for k, v in segment.items() if k != "playback_url"}


def _neighbor_record(segment: dict | None) -> dict | None:
    if segment is None:
        return None
    return {
        "id": segment.get("id"),
        "ts": segment.get("ts"),
        "caption": segment.get("caption", ""),
        "yolo": segment.get("yolo", {}),
    }


def try_exhibit(claim: str, n: int, segment: dict, camera: str | None, budget: int) -> tuple[dict, int]:
    """Run one exhibit through prosecution, defense, judge and (maybe) the tape.

    Returns the exhibit record and the remaining subpoena budget.
    """
    prev, nxt = neighbors(segment["id"])
    prosecution = prosecute(claim, segment)
    defense = defend(claim, segment, prev, nxt)
    judge = rule(claim, segment, defense)

    subpoena: dict | None = None
    disposition = judge["ruling"]
    struck_for: str | None = None
    if disposition == "SUBPOENA":
        if budget > 0:
            budget -= 1
            subpoena = view_tape(segment["id"], judge["question"], judge["foundation"])
            disposition = LOOK_TO_RULING[subpoena["answer"]]
            if disposition == "STRIKE":
                struck_for = "unclear" if subpoena["answer"] == "unclear" else "tape"
        else:
            disposition = "NOT_REACHED"
    elif disposition == "STRIKE":
        struck_for = "objection"

    record = {
        "n": n,
        "camera": camera,
        "chunk": _chunk_name(segment),
        "segment": _public_segment(segment),
        "neighbors": {"prev": _neighbor_record(prev), "next": _neighbor_record(nxt)},
        "prosecution": prosecution,
        "defense": defense,
        "judge": judge,
        "subpoena": subpoena,
        "disposition": disposition,
        "struck_for": struck_for,
    }
    return record, budget


def exhibit_lines(record: dict) -> list[str]:
    """Transcript lines for one exhibit record."""
    seg = record["segment"]
    # ts is relative to the parent chunk, so the chunk name is what tells exhibits apart.
    where = f"{record['camera']} · {record['chunk']}" if record["camera"] else record["chunk"]
    lines = [
        f"EXHIBIT {record['n']}  [{where} @ {seg['ts']:g}s]",
        f"  PROSECUTION: {record['prosecution']}",
    ]
    defense = record["defense"]
    if defense["objection"] == "none":
        lines.append("  DEFENSE:     No objection.")
    else:
        lines.append(f"  DEFENSE:     Objection ({defense['objection']}). {defense['reason']}")
    lines.append(f"  JUDGE:       {record['judge']['line']}")
    if record["judge"]["ruling"] == "SUBPOENA":
        if record["subpoena"]:
            sub = record["subpoena"]
            for a in sub["answers"]:
                lines.append(f"  THE COURT ASKS: {a['question']}")
                lines.append(f"  THE TAPE ANSWERS: {a['answer']} — {a['reason']}")
            if len(sub["answers"]) > 1 and sub["answer"] != sub["answers"][0]["answer"]:
                lines.append(f"  THE COURT FINDS: {sub['answer']} — the answers do not agree.")
        else:
            lines.append("  THE COURT DECLINES TO VIEW THE TAPE: subpoena budget exhausted; exhibit not reached.")
    lines.append("")
    return lines


def _strike_breakdown(records: list[dict]) -> dict[str, int]:
    counts = {reason: 0 for reason in STRUCK_FOR}
    for r in records:
        if r["disposition"] == "STRIKE" and r["struck_for"] in counts:
            counts[r["struck_for"]] += 1
    return counts


def iter_trial(claim: str, camera: str | None, k: int, subpoenas: int):
    """Run a trial, yielding events as the court works so a UI can show it unfolding.

    Events (dicts with a ``type`` key):
      ``docket``  — ``{"exhibits": n}`` once search() has returned
      ``exhibit`` — ``{"record": <exhibit record>, "lines": [transcript lines]}`` per exhibit
      ``verdict`` — ``{"trial": <full trial object>, "lines": [verdict lines]}`` last
    """
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    records: list[dict] = []
    budget = subpoenas

    exhibits = search(claim, camera=camera, k=k)
    yield {"type": "docket", "exhibits": len(exhibits)}

    for n, segment in enumerate(exhibits, start=1):
        record, budget = try_exhibit(claim, n, segment, camera, budget)
        records.append(record)
        with LEDGER.open("a") as f:
            f.write(json.dumps({"claim": claim, **record}) + "\n")
        yield {"type": "exhibit", "record": record, "lines": exhibit_lines(record)}

    lines, trial = _verdict(claim, camera, k, subpoenas, budget, records)
    yield {"type": "verdict", "trial": trial, "lines": lines}


def run_trial(claim: str, camera: str | None, k: int, subpoenas: int) -> tuple[list[str], dict]:
    """Try every exhibit search() returns.

    Returns ``(transcript_lines, trial)`` where ``trial`` carries the per-exhibit records
    under ``exhibits`` plus the verdict, so UIs render the same object the transcript does.
    """
    lines: list[str] = []
    trial: dict = {}
    for event in iter_trial(claim, camera, k, subpoenas):
        if event["type"] in ("exhibit", "verdict"):
            lines.extend(event["lines"])
        if event["type"] == "verdict":
            trial = event["trial"]
    return lines, trial


def _verdict(
    claim: str, camera: str | None, k: int, subpoenas: int, budget: int, records: list[dict]
) -> tuple[list[str], dict]:
    """Tally the records, hear the bailiff, maybe file the report."""
    lines: list[str] = []
    admitted = [r for r in records if r["disposition"] == "ADMIT"]
    struck = sum(r["disposition"] == "STRIKE" for r in records)
    not_reached = sum(r["disposition"] == "NOT_REACHED" for r in records)
    breakdown = _strike_breakdown(records)
    # Only exhibits the court actually decided count. An exhibit the court could not afford to
    # view says nothing about the claim, so it is excluded from the denominator.
    decided = len(admitted) + struck
    confidence = len(admitted) / decided if decided else 0.0
    guilty = bool(decided) and confidence >= 0.5
    # The outcome is decided here, not by the bailiff; without it the model can announce
    # "guilty" on a record of zero admitted exhibits.
    outcome = "guilty" if guilty else "not guilty — case dismissed"
    # 70B: the 8B bailiff ignored the one-sentence format and appended raw counts.
    bailiff = _first_sentence(chat(
        BAILIFF,
        f"Claim: {claim}\nExhibits admitted: {len(admitted)}\nExhibits struck: {struck}\n"
        f"Exhibits not reached (subpoena budget exhausted; not counted): {not_reached}\n"
        f"Outcome (already decided): {outcome}",
        model=REASONING_MODEL,
    ))
    tally = f"admitted {len(admitted)} · struck {struck}"
    if not_reached:
        tally += f" · not reached {not_reached}"
    lines.append(f"VERDICT  [{tally} · confidence {confidence:.2f}]")
    if struck:
        why = " · ".join(f"{v} {k}" for k, v in breakdown.items() if v)
        lines.append(f"  STRUCK FOR:  {why}")
    lines.append(f"  BAILIFF:     {bailiff}")

    trial = {
        "claim": claim,
        "camera": camera,
        "k": k,
        "subpoenas": subpoenas,
        "subpoenas_used": subpoenas - budget,
        "exhibits": records,
        "verdict": {
            "admitted": len(admitted),
            "struck": struck,
            "not_reached": not_reached,
            "struck_for": breakdown,
            "confidence": confidence,
            "guilty": guilty,
            "bailiff": bailiff,
        },
        "report": None,
    }
    if guilty:
        trial["report"] = write_report(trial)
        lines.append(f"SENTENCE: guilty — hazard report drafted to {trial['report']}")
    else:
        lines.append("SENTENCE: not guilty — case dismissed")
    return lines, trial


def write_report(trial: dict) -> str:
    """Hazard report from the trial record: admitted exhibits, then what was struck and why."""
    stamp = time.strftime("%Y%m%d_%H%M%S")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / f"report_{stamp}.md"
    v = trial["verdict"]
    admitted = [r for r in trial["exhibits"] if r["disposition"] == "ADMIT"]
    struck = [r for r in trial["exhibits"] if r["disposition"] != "ADMIT"]

    def row(r: dict) -> str:
        seg = r["segment"]
        return f"| {r['n']} | {r['chunk']} | {seg['ts']:g}s | {seg['id']} | {seg['caption']} |"

    def struck_row(r: dict) -> str:
        seg = r["segment"]
        if r["disposition"] == "NOT_REACHED":
            why = "not reached: subpoena budget exhausted (not counted)"
        elif r["struck_for"] == "objection":
            why = f"objection ({r['defense']['objection']}): {r['defense']['reason']}"
        elif r["subpoena"]:
            why = f"tape ({r['subpoena']['answer']}): {r['subpoena']['reason']}"
        else:
            why = r["struck_for"] or ""
        return f"| {r['n']} | {r['chunk']} | {seg['ts']:g}s | {r['disposition']} | {why} |"

    not_reached = f", {v['not_reached']} not reached" if v["not_reached"] else ""
    path.write_text(
        f"# Hazard report\n\n"
        f"**Claim:** {trial['claim']}  \n"
        f"**Camera:** {trial['camera'] or 'all'}  \n"
        f"**Verdict:** guilty — {v['admitted']} admitted, {v['struck']} struck{not_reached}, "
        f"confidence {v['confidence']:.2f}  \n"
        f"**Filed:** {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        f"## Admitted exhibits\n\n"
        f"| # | Chunk | ts | Segment | Caption |\n|---|---|---|---|---|\n"
        + "\n".join(row(r) for r in admitted)
        + "\n\n## Struck and not-reached exhibits\n\n"
        f"| # | Chunk | ts | Disposition | Why |\n|---|---|---|---|---|\n"
        + ("\n".join(struck_row(r) for r in struck) or "| – | – | – | – | none |")
        + "\n"
    )
    try:
        return str(path.relative_to(COURT_DIR.parent))
    except ValueError:  # OUT_DIR outside the repo (e.g. /tmp/court in the pod)
        return str(path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Put a claim about the video archive on trial.")
    parser.add_argument("--claim", required=True, help="the behavior on trial, in plain language")
    parser.add_argument("--camera", default=None, help="limit the search to one camera")
    parser.add_argument("--k", type=int, default=10, help="number of exhibits to search for")
    parser.add_argument("--subpoenas", type=int, default=4, help="max clips the court may view")
    parser.add_argument("--out", default=None, help="also write the transcript here")
    parser.add_argument("--json", default=None, help="write the full trial record (per-exhibit dicts + verdict) here")
    args = parser.parse_args()

    lines, trial = run_trial(args.claim, args.camera, args.k, args.subpoenas)
    transcript = "\n".join(lines)
    print(transcript)
    if args.out:
        Path(args.out).write_text(transcript + "\n")
    if args.json:
        Path(args.json).write_text(json.dumps(trial, indent=2) + "\n")


if __name__ == "__main__":
    main()
