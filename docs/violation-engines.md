# Violation engines

The detection engines this project generates training images for. Each engine is one violation class of the
target system; its `id` is the key, its `name` the class name the images and sidecars use. Records as supplied by
the user on 2026-10-06 (from the engine configuration):

| Field | `forklift-charging-multiple-skids-horizontally` | `forklift-pushing-multiple-lsps` |
|---|---|---|
| `name` | Forklift Charging Multiple Skids Horizontally | Forklift Pushing Multiple Lsps |
| `title` | null | null |
| `color` | `#974935` | `#960399` |
| `sms` / `email` | true / true | true / true |
| `show_on_home` | true | true |
| `seq_no_format` | `fcmsh` | `f_p_m_l` |
| `created_at` | 2025-10-13T05:09:22.000Z | 2025-10-09T09:47:01.000Z |
| `updated_at` | 2025-11-07T05:20:32.000Z | 2025-11-07T05:20:38.000Z |

```json
[
  {
    "id": "forklift-charging-multiple-skids-horizontally",
    "name": "Forklift Charging Multiple Skids Horizontally",
    "title": null,
    "color": "#974935",
    "sms": true,
    "email": true,
    "show_on_home": true,
    "seq_no_format": "fcmsh",
    "created_at": "2025-10-13T05:09:22.000Z",
    "updated_at": "2025-11-07T05:20:32.000Z"
  },
  {
    "id": "forklift-pushing-multiple-lsps",
    "name": "Forklift Pushing Multiple Lsps",
    "title": null,
    "color": "#960399",
    "sms": true,
    "email": true,
    "show_on_home": true,
    "seq_no_format": "f_p_m_l",
    "created_at": "2025-10-09T09:47:01.000Z",
    "updated_at": "2025-11-07T05:20:38.000Z"
  }
]
```

## Input kinds

Every input frame is a valid frame of one input kind, the thing it shows, and each kind feeds one engine: a case of
that engine turns frames of the kinds it lists into its violation. Inputs are sorted into folders named by kind
(user decision, 2026-10-06):

| Input kind (folder name) | The frame shows | Feeds engine | Cases |
|---|---|---|---|
| `forklift-with-lsp-cargo` | a forklift pushing one LSP loaded with cargo | `forklift-pushing-multiple-lsps` | V1, V2, V6 |
| `forklift-with-lsp-empty` | a forklift pushing one empty LSP | `forklift-pushing-multiple-lsps` | V1, V2, V6 |
| `forklift-with-cargo-no-lsp` | a forklift carrying a cargo load (on a SKID) on its forks, no LSP | `forklift-charging-multiple-skids-horizontally` | Carry 2 cargo side by side, Carry 2 cargo stacked |
| `forklift-empty` | a forklift with empty forks | `forklift-charging-multiple-skids-horizontally` | Carry 2 cargo side by side, Carry 2 cargo stacked |
| `floor-closed` | a floor hatch with its cover closed, flush with the floor | `floor-opening` | Open with person, Open no person, Open main door closed, Open main door open |

A folder of inputs holds these kind folders (any of them), each with its .jpg / .png frames (sub-folders inside a
kind folder are searched too); a zip holds the same folders, at its top or inside one folder:

```
cam01/
├─ forklift-with-lsp-cargo/      frame_0145.jpg, frame_0517.jpg, ...
├─ forklift-with-lsp-empty/
├─ forklift-with-cargo-no-lsp/
└─ forklift-empty/
```

Other folders and loose images are left out, and a job reports them. A job is for one engine and uses only the
frames of that engine's kinds; it reports how many frames of the other kinds it left out.

Prompt rules for the pushing cases (2026-10-06): a new LSP copies the LSP the forklift pushes in the frame (colour,
wear, edges, thickness, CCTV look); only when the frame shows no LSP does image 2 stand in: the fixed grid of real CCTV crops of single empty LSPs `work/ref-lsp.jpg` (17 tiles from ch10, many pushed by a forklift; its grid and captions must not be copied). No catalogue3d LSP render goes in any more; the SKID is still the frame's catalogue3d render. A new LSP
lies against the sheet behind it, edges touching with no gap and no overlap, never on top of the original, and takes
the original's size, never the cargo's. In V2 the three sheets are packed tight in one straight row, a thin seam
between plates. The prompt also says what an LSP, a SKID and a cargo are, with their real sizes (LSP about
1.9 x 1.85 m and 9 cm thick, SKID 1.2 x 1.0 m and 13 cm high, cargo about the SKID's footprint), and that they stack
floor, LSP, SKID, cargo.

On a `forklift-with-lsp-empty` frame the original LSP stays empty: only the new LSPs get cargo (user decision,
2026-10-06). So V1 and V2 add one or two LSPs with cargo in front of the empty one, and V6 adds one empty LSP, which
leaves two empty LSPs.

## `forklift-pushing-multiple-lsps`

A forklift pushes cargo on more than one LSP at once; exactly one LSP is valid
([request decisions](requirements/2026-09-28-gary-request.md): `is_violation = lsp_count >= 2`). This is the
violation the project was started for: `config/scenarios.json` family `push` (`event_class` "Forklift Pushing
Multiple Lsps") lists its scenarios V1-V8 and the look-alike valid scenes N1-N5, and
[spec section 1.1](superpowers/specs/2026-09-28-violation-image-synth-design.md) describes them.

Cases generated so far:

| Case | Catalogue | How | Status |
|---|---|---|---|
| Push 2 LSP with Cargo (`push_2_lsp_cargo`) | V1 (in series, 2 LSPs, cargo on each) | a second LSP of the same size in front of the loaded one, with a cargo copied from the loads in the frame | POC, 49 frames of ch10/ch13/ch14/ch15 (2026-10-06) |
| Push 3 LSP with Cargo (`push_3_lsp_cargo`) | V2 (in series, 3 LSPs) | two more LSPs of the same size in a straight row in front, a cargo on each | first output 2026-10-06: Astra joined the chain to an idle LSP already on the floor |
| Push 2 LSP, extra LSP empty (`push_2_lsp_empty_extra`) | V6 (rule to confirm with Terry) | a second LSP of the same size in front, left empty | first output 2026-10-06 |

Every case is one Astra edit of a real frame (`synth/try_astra_edit_batch.py` `edit_frame`): the ONE change of its
prompt is written per input kind, and the kind also gives the prompt's description of the frame. The input kinds,
cases and prompt text are in `synth/violation_cases.py`; a new case is a new entry there and shows up in the batch
CLI (`--engine`, `--case`, `--input-kind`) and the service at once.

## `forklift-charging-multiple-skids-horizontally`

A forklift carries more than one SKID at once, placed side by side (horizontally) across its forks. Not covered
by `config/scenarios.json` yet (its `carry_on_forks` family is about LSPs on the forks, class "Forklift Lifting
Multiple Lsps").

| Case | Catalogue | How | Status |
|---|---|---|---|
| Carry 2 cargo side by side (`carry_2_skids_side_by_side`) | none (draft) | two cargo loads, each on its own SKID, side by side across the fork carriage; each SKID turned lengthwise along the forks (its long side parallel to the tines). From `forklift-with-cargo-no-lsp` the carried load gets a SKID if none shows, and a second loaded SKID goes beside it. Every load sits all the way back against the forklift's vertical mast. Only the SKID reference goes in | draft after the user's first example prompt; placement rules from the user (2026-10-06) |
| Carry 2 cargo stacked (`carry_2_cargo_stacked`) | none (draft) | two cargo loads on SKIDs stacked one on top of the other, each turned crosswise (its long side across both tines); the upper SKID sits squarely on the lower load, both about the same footprint. From `forklift-with-cargo-no-lsp` the carried load is the lower one, turned crosswise if it lies otherwise. Every load sits all the way back against the forklift's vertical mast. Only the SKID reference goes in | draft (user, 2026-10-06); first outputs 2026-10-06 (before the orientation and mast rules); to confirm that a vertical stack counts for an engine named "horizontally" |

## `floor-opening`

A floor hatch is left open: its cover is lifted and the floor opening below shows (user, 2026-10-08). One input
kind, `floor-closed` (the hatch closed). Target camera: CMW01 channel 54 (`rtsp://<user>:<pass>@192.168.137.23/cam/realmonitor?channel=54&subtype=0`).
The engine colour `#d4570f` is ours until the engine configuration gives one.

| Case | How | Status |
|---|---|---|
| Open, person in scene (`open_with_person`) | the hatch open; the people of the frame stay, one worker is added beside the hatch when the frame shows nobody | draft |
| Open, nobody in scene (`open_no_person`) | the hatch open; every person removed, the hatch left unattended | draft |
| Open, main door closed (`open_main_door_closed`) | the hatch open; the room's main door shut (closed if the frame shows it open); people unchanged | draft |
| Open, main door open (`open_main_door_open`) | the hatch open; the main door wide open, light from beyond it; people unchanged | draft |

The hatch cover's angle is random per output (user, 2026-10-08: not only upright): `fill()` in
`synth/violation_cases.py` draws a pose from the task's seed and its change text (so the four cases of one frame
differ, a rerun keeps its angle, a rebuild gets a new one): slightly open (20-35 degrees, the pit only a wedge under
the cover), half open (40-60), mostly open (65-80), upright (85-95) or flung back past upright (100-120). `edit_frame`
fills it, records `lid_angle` in its result row, and the angle is in each run's `prompt.txt`.

Every case takes one reference, `work/floor-opening/ref-floor-opening.jpg`: two frames of the user's clip
(Drive file `1N7hGHcvkBSuYJmdkv7d71Paeal73WLm0`, CMW01 G02-CAM00, 960x540) side by side, the hatch closed (0.2 s) and
open (6.5 s); the prompt copies only how an open hatch looks, never that room, its person or its text. Rebuilt with:

```
ffmpeg -ss 0.2 -i ref-video.mp4 -frames:v 1 closed.png; ffmpeg -ss 6.5 -i ref-video.mp4 -frames:v 1 open.png
ffmpeg -i closed.png -i open.png -filter_complex "[0][1]hstack=inputs=2" work/floor-opening/ref-floor-opening.jpg
```

The engine has its own prompt context (`prompt` in `synth/violation_cases.py`: setting, what a floor hatch, floor
opening and main door are, what stays unchanged, the camera's timestamp kept), so its prompt names no warehouse, LSP
or laser line, and no catalogue3d render is made for it (`edit_frame` runs `synth.catalogue_reference_map` only for a
SKID reference). Inputs from the camera: `python -m synth.grab_rtsp_frames "<rtsp url>" --kind floor-closed
--count 10 --every 20 --out work/inputs/cmw01-ch54` (10 frames, 20 s apart).

## Generation service

`service/app.py` (FastAPI) with its page `service/static/index.html`:

```
.venv/bin/python -m uvicorn service.app:app --host 127.0.0.1 --port 8000     # open http://127.0.0.1:8000
```

The page takes the input frames in one of three ways, all sorted by input kind (above): a folder path on the
server (free text, relative to the repository or absolute), a zip of such a folder, or images uploaded into one box
per kind. It then takes the engine, its cases and the number of variants per frame. Generate starts a job of one
task per (frame of the engine's kinds, chosen case that takes the frame's kind, variant), each one Astra edit and
one output; all jobs share one pool of 3 threads. Every
state change of a task (running, done with its image, failed with its error) is pushed to the page over a WebSocket
(`/ws/jobs/<job id>`) as it happens; outputs download one by one, or as one zip once the job has finished (one folder per input kind). Packages: `requirements-service.txt`.

Jobs are kept on disk, one folder each in `work/service/jobs/<job id>/`: `job.json` (engine, cases, variants, notes,
the input frames per kind and every task's state, seconds and tokens, rewritten on every change), `inputs/<kind>/`
(a copy of every input frame, whether it came from a folder path, a zip or an upload, so the job keeps them when
the source folder changes), `runs/`, `outputs/<kind>/` and `thumbs/`. The service loads every `job.json` when it
starts, so `#job=<id>` in the page URL and the page's list of recent jobs open any past job with its inputs and
outputs side by side. A task that was waiting or running when the service stopped is marked interrupted and is
never run again on its own: a job only runs when someone starts it.

Tokens: new runs use `codex exec --json` and read `turn.completed.usage`: input, cached input,
output and reasoning output. Total = input + output; cached input and reasoning are subsets, not added again.
Each task stores `token_attempts` and an aggregate `token_usage`; retries preserve earlier usage and logs in
`runs/<kind>/<task>/attempt-NNNN/`. Failed attempts count when usage is available. Missing usage stays unknown;
the UI labels incomplete totals. Imagegen's own usage is not collected by this counter.
On load, legacy scalar `tokens` values migrate to input tokens, with output/cache unknown. Migration is
idempotent and persists to job.json. The compatibility `tokens` scalar now means known input + output.
Estimates use input tokens only (outputs × historical median, with the 10th–90th percentile range), so legacy
input-only readings are not mixed with new total-token readings.

Codex plan usage (`service/codex_status.py`, `GET /api/codex`): the service reads the signed-in plan's usage limit
with the Codex app-server's read-only `account/rateLimits/read` (cached 60 s): the plan (2026-10-06: `prolite`), the
window (weekly), the percent of it used and when it resets, and whether the limit is reached. Codex gives no token
budget, so the room left is measured in outputs: every reading (at the start and the end of each job, and on the page's
refresh every minute) goes into `work/service/usage.json` with the outputs and tokens the service had made by then,
and the percent per output is the rise in percent over the rise in outputs between readings of one window (pairs
across a reset, or with no new output, are left out; Codex use outside the service still lowers the room through the
percent used). It is trusted after 6 outputs. The page shows the plan, the percent used, the reset time and the
outputs the plan still has room for after the queued ones. A new job whose outputs exceed that room, or any job while
Codex reports the limit reached, is refused (HTTP 409) and the page says why; its "Run anyway" button sends the job
with `force=1`. While the room is not known yet (no calibration, or no reading), jobs run and the page says the room
is unknown.

Rebuild (`POST /api/jobs/<job id>/tasks/<task id>/rebuild`, the Rebuild button on an output card): one more output
for the same frame and case, with the user's own request for that image (up to 1000 characters) appended to the
case's ONE change as "Additional request for this image, which takes priority over anything above it conflicts
with". The earlier output stays; the new one is a task of the same job, shown beside the same input frame, named
`<frame>__<case>__v<n>__r<round>.png`, with another seed for the catalogue references. A rebuild of a rebuild counts
the rounds from the first output. It takes the same Codex usage check as a job (one output; "Run anyway" passes
`force=1`).

Rebuild many (`POST /api/jobs/<job id>/rebuild`, form fields `task` repeated once per output, `note`, `force`): the
same rebuild for each chosen output, all with one request, e.g. the reviewer's feedback on every bad output. On the
page, a box on each finished, failed or interrupted card chooses it, and "Select shown" chooses every such output the
filter shows (Bad, Unreviewed, Failed, ...); a bar at the bottom takes the request. The usage check counts all the
outputs, and nothing is added when one of the choices is unknown or still being made. After it, the review menu
switches to Review unreviewed, where the new outputs wait.

Review (the Review button on a job): a full-screen deck of the job's finished outputs, chosen as Review all (unreviewed
first), Review unreviewed (only outputs with no mark yet, such as new rebuilds), Review good or Review bad. The output fills the screen, its input frame sits top left. Right arrow / swipe
right / Good marks it good and moves on, left / Bad marks it bad, up and down move without marking, X / "Exclude input"
takes that input frame (and all its outputs) out of review and downloads, with an Undo. Marks are stored per output
(`review` in `job.json`, `POST /api/jobs/<id>/tasks/<task id>/review`, verdict good|bad|clear); excluded frames as
`excluded_inputs` (`POST /api/jobs/<id>/inputs/<kind>/<name>/exclude`, excluded=1|0). The download menu takes all,
good or bad outputs (never those of excluded frames), in one folder or a folder per case:
`GET /api/jobs/<id>/download.zip?filter=all|good|bad&layout=flat|by_case`, written to a temporary file on disk.

