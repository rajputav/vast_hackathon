# Exhibit A: the video archive on trial

**Repo:** `<REPO LINK>`  
**Artifact:** [`transcript.txt`](transcript.txt), a full courtroom transcript from one trial

## One-line pitch

Put a claim about your video archive on trial: one agent prosecutes with search, another cross-examines every exhibit against physical evidence, a judge rules. Nothing counts until it survives.

## The problem

Video search runs on captions, and captions are written by a model under someone else's prompt. They hallucinate (a "truck in the bike lane" that is really a bus at a stop) and they leave out whatever the prompt never asked about. Every search tool today asks you to believe them.

## How it works

1. **File a claim:** *"Trucks routinely block the bike lane on this route."*
2. **Prosecution** gathers exhibits from the VSS archive using hybrid search (captions plus visual embeddings).
3. **Defense** cross-examines each exhibit. It treats the caption as hearsay and checks it against YOLO detections and the adjacent segments. It objects on **object** (the caption names a class the detector didn't see), **motion** (the caption says "stopped" but the boxes move) or **scope** (real, but not what the claim needs).
4. **The judge** rules on each exhibit:
   - **ADMIT** it,
   - **STRIKE** it, or
   - **SUBPOENA** the tape: a direct Cosmos3-Reason call on the clip with a yes/no question written for this claim.
5. **Verdict:** the counts of admitted and struck exhibits, with a confidence score, and every ruling recorded in a ledger.
6. **Sentence = action:** a guilty verdict drafts a hazard report listing the admitted exhibits with playback links. Otherwise the case is dismissed.

## Evidence sources

| Source | What it is | Depends on the ingestion prompt? |
|---|---|---|
| Captions | Cosmos-written description of each segment | **Yes**: it says only what the prompt asked about |
| Detections | YOLO11 classes, counts and bounding boxes | **No** |
| Visual embeddings | Cosmos-Embed1 vectors of the pixels | **No** |

Only the captions are hearsay. The defense deliberately builds its case on the other two.

## Why subpoenas instead of re-ingest

Re-ingest re-captions whole videos and waits on the pipeline. A subpoena asks one precise question about one clip and answers in seconds, with no pipeline dependency. Today's re-ingest outage didn't touch the design. Re-ingest is reserved for claims the index has never observed.

## What's next

- **Claim compiler:** turns a plain-language claim into a jurisdiction (cameras, time window) and stipulations (facts both sides accept), drawn from a briefing on the corpus.
- **Relevance feedback:** verdicts move the search in embedding space, toward admitted exhibits and away from struck ones.
- **Evidence ledger in VastDB:** rulings become queryable, surfacing unreliable captions across trials.
- **Audience docket:** viewers vote before the trial, then watch the court overrule them.
- **Web UI:** a live courtroom with clips beside the transcript.

*Built on VAST S3, DataEngine and VastDB; NVIDIA Cosmos3-Reason, Cosmos-Embed1 and YOLO11 on CoreWeave; Weights & Biases serverless inference for the courtroom agents.*
