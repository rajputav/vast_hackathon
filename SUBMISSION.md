# Exhibit A: the video archive on trial

**Live app:** https://team-47-app.thecosmoslabs.com/app/  
**Repo:** https://github.com/rajputav/vast_hackathon  
**Artifact:** [`transcript.txt`](transcript.txt), a full courtroom transcript from one trial  
**Team 47:** Avinash Rajput, Pradhyumn Thakur

## One-line pitch

Put a claim about your video archive on trial: one agent prosecutes with search, another cross-examines every exhibit against physical evidence, a judge rules, and the tape itself takes the stand. Nothing counts until it survives.

## The problem

Video search runs on captions, and captions are written by a model under someone else's prompt. They hallucinate (a "truck in the bike lane" that is really a bus at a stop) and they leave out whatever the prompt never asked about. Every search tool today asks you to believe them. Exhibit A refuses.

## How a trial runs

1. **File a claim:** *"Someone was struck by a forklift."* or *"Trucks routinely block the bike lane on this route."* Pick a camera or try the whole archive.
2. **Docket.** Hybrid search over the VSS archive (captions plus Cosmos-Embed1 visual embeddings) returns *k* 5-second segments.
3. **Prosecution** (Llama 3.1 8B) states what each exhibit shows in support of the claim.
4. **Defense** (Llama 3.3 70B) treats the caption as hearsay and tests it against evidence that did not come from the caption: the YOLO11 detection record, per-class frame coverage, the person's tracked motion, and the ±5 s neighbouring segments. It objects on **object** (the caption names an actor the detector didn't see), **motion** (the caption says "stopped" but the boxes move) or **scope** (real, but not what the claim needs). Frequency words like "routinely" are never objectionable per exhibit; they are decided at verdict.
5. **The court vets the objection.** A deterministic guard withdraws any *object* objection that is not grounded in a COCO class. YOLO has no word for "forklift", so "yolo saw a boat, not a forklift" is the detector guessing, not evidence. The withdrawn objection stays on the record, struck through.
6. **The record.** From a fixed camera, the detector's own per-frame boxes give the person's speed. An abrupt mid-clip spike (×2.5 or more over the median) is entered as physical evidence consistent with contact; a steady speed is consistent with walking or pushing. This is the one witness in the room that is neither the captioner nor the tape. It is ruled inadmissible from moving cameras (the GoPro), where pixel speed measures the camera.
7. **The judge** (Llama 3.3 70B) rules on the papers, strike-only, and writes two neutral yes/no questions for a model that sees only pixels: one testing the claim's *action* ("Does the forklift make physical contact with a person?") and a *foundation* question testing the premise ("Is a forklift visible?"). Presence alone can never admit.
8. **The tape.** The court puts three questions to Cosmos3-Reason on the GPU cluster with the raw clip: the claim itself, in the court's own neutral wording ("Does this clip itself show an instance of the following: … Judge only what is visible, not what is typical."), plus the judge's question and foundation. Every answer must be **yes** to admit; the tape outranks the caption and the detector once viewed. The court notes on the record that the tape witness is the same model that wrote the caption, so their agreement is one witness heard twice, not corroboration.
9. **Verdict.** Admitted, struck (and why: objection, tape, unclear), and not-reached (tape budget spent; not counted). An incident claim needs one admitted exhibit; a pattern claim needs most of them. Confidence is admitted over decided.
10. **Sentence = action.** A guilty verdict drafts a hazard report listing the admitted exhibits and what was struck. Otherwise the case is dismissed.

Every step produces one per-exhibit record. The CLI transcript, the JSON ledger, the hazard report and the courtroom UI all render that same object.

## The courtroom

A single-page web app, streamed live over SSE. File a claim, watch exhibits land on the docket and get tried one by one: the clip beside the transcript; YOLO chips with frame coverage ("boat × 1 · 3/150f", dimmed when it is a flicker); the motion chip; the caption folded away under "hearsay"; PROSECUTION, DEFENSE, THE RECORD, JUDGE, THE COURT ASKS, THE TAPE ANSWERS; a stamp. The verdict panel shows the confidence bar, the strike breakdown and the bailiff's sentence. Trials reattach by URL and the transcript downloads.

## Evidence sources

| Source | What it is | Depends on the ingestion prompt? | Independent of Cosmos? |
|---|---|---|---|
| Captions | Cosmos-written description of each segment | **Yes**: it says only what the prompt asked about | No |
| The tape | Cosmos3-Reason answering a yes/no question on the raw clip | No | No: same model as the captioner, and the record says so |
| Detections | YOLO11 classes, counts, bounding boxes, frame coverage | No | **Yes** |
| Person kinematics | Speed profile tracked from the detector's boxes | No | **Yes** |
| Visual embeddings | Cosmos-Embed1 vectors of the pixels | No | No |

Only the captions are hearsay. The defense builds its case on the rest, and the court is told which witnesses are really the same witness.

## Why subpoenas instead of re-ingest

Re-ingest re-captions whole videos and waits on the pipeline. A subpoena asks one precise question about one clip and answers in seconds, with no pipeline dependency. An ingest outage on build day did not touch the design.

## Tools and stack

| Layer | What we used | How |
|---|---|---|
| Video index | **VAST VSS retrieval API** (`/api/v1/search`, `/videos/detections`, `/videos/metadata`, `/tools/segments`, `/videos/stream`) | JWT login with a cached token that re-logs-in on 401; hybrid search with `metadata_filters`; segment neighbours by timestamp; per-frame YOLO sidecar for boxes, coverage and kinematics; clips proxied for playback |
| Captions and embeddings | **NVIDIA Cosmos3-Reason** (`cosmos3-nano-reasoner`), **Cosmos-Embed1** | Pre-ingested by the VSS pipeline; captions treated as hearsay |
| Detections | **YOLO11 (COCO-80)** sidecar | Max-per-frame counts, densest-frame boxes, frames-seen per class, greedy person tracking → smoothed speed profile and spike ratio |
| Tape witness | **Cosmos3-Reason on the GPU cluster** (OpenAI-compatible `/v1/chat/completions`, base64 `video_url`) | One direct call per subpoena question; 60 s shared deadline; returns `unclear / timeout` on failure, never raises |
| Courtroom agents | **Weights & Biases serverless inference**: Llama 3.1 8B (prosecutor, bailiff), Llama 3.3 70B (defense, judge) | JSON-schema constrained outputs; deterministic guards in code where the rules matter |
| App | **FastAPI** + vanilla JS, SSE | `POST /api/trial` → `GET /api/trial/{id}/events` replays history then streams; `/api/clip` proxies playback |
| Deploy | **Kubernetes** on the team namespace: `python:3.12-slim` pod, `app/` and the `court/` package mounted from ConfigMaps (no registry), creds from a Secret, Ingress at `/app` | `./deploy.sh` rebuilds the ConfigMaps and rolls the deployment |
| Storage | **VAST S3** (chunks, segments, detection sidecars), **VastDB** behind the retrieval API | Read through the retrieval API |

## What we learned

- **The detector's vocabulary is a courtroom fact.** Half of our early false strikes were "no forklift in yolo", a class YOLO cannot emit. Rules about what a piece of evidence is *capable* of showing had to be code, not prompt.
- **A judge will admit on "is a forklift visible?" if you let it.** Left to write the deciding question, the judge asked whatever settled the objection and convicted 53 of 60 forklift clips on presence. Now the court itself asks the tape the claim; the judge's questions test the action and the foundation. Result: 3 of 60 admitted, each a collision on tape.
- **When the only vision model is the one that wrote the captions, say so on the record.** The person kinematics from the detector boxes were the one genuinely independent signal we had. On the forklift footage they line up with the tape: ×3 speed spikes on the angles where contact was found, a stationary person on the angles where it was not.

## What's next

- **Claim compiler:** turn a plain-language claim into a jurisdiction (cameras, time window) and stipulations (facts both sides accept).
- **Relevance feedback:** verdicts move the search in embedding space, toward admitted exhibits and away from struck ones.
- **Evidence ledger in VastDB:** rulings become queryable, surfacing unreliable captions across trials.
- **A second vision witness:** an independent VLM so the tape and the captioner are not the same model.
- **Audience docket:** viewers vote before the trial, then watch the court overrule them.

*Built on VAST S3, DataEngine and VastDB; NVIDIA Cosmos3-Reason, Cosmos-Embed1 and YOLO11 on CoreWeave; Weights & Biases serverless inference for the courtroom agents.*

**License:** MIT ([LICENSE](LICENSE))
