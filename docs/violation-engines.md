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
| `forklift-with-cargo-no-lsp` | a forklift carrying a cargo load (on a SKID) on its forks, no LSP | `forklift-charging-multiple-skids-horizontally` | Carry 2 SKIDs side by side |
| `forklift-empty` | a forklift with empty forks | `forklift-charging-multiple-skids-horizontally` | Carry 2 SKIDs side by side |

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
| Carry 2 SKIDs side by side (`carry_2_skids_side_by_side`) | none (draft) | from `forklift-empty`: two SKIDs with cargo side by side across the forks; from `forklift-with-cargo-no-lsp`: a second SKID with cargo beside the carried load. Only the SKID reference goes in | draft after the user's first example prompt (2026-10-06); not generated yet |

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
(`/ws/jobs/<job id>`) as it happens; outputs download one by one, or as one zip once the job has finished (one folder per input kind). Jobs live
in memory (a restart forgets them, `#job=<id>` in the page URL reopens one while the service runs); their files are in
`work/service/jobs/<job id>/`. Packages: `requirements-service.txt`.
