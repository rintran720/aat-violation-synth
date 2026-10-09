# Synthetic Data Generation

A tool that makes **variants of real camera images**. The input is one image of a real scene (a CCTV frame); each
output is that same image with **one change** made by an image-editing model (Codex `gpt-6-astra`, imagegen), for
example a forklift pushing two LSPs instead of one, or a floor hatch opened. An output does not have to be a safety
violation: it is any controlled variant of the scene that a dataset needs (positives, negatives, rare cases).

The repository started as "AAT violation synth" (making violation images for one customer); that history is in
`docs/`. Code, prompts and docs from then still say "violation" in places; read it as "variant".

## Concepts

- **Project**: groups jobs; may use only the engines granted to it. `Default` holds the jobs made before projects
  and is hidden.
- **Engine**: one family of variants (e.g. *Forklift Pushing Multiple Lsps*, *Charging Multiple Skids*,
  *Floor Opening*), with its prompt context (setting, scene prefix, objects, keep rules).
- **Input kind**: what an input image shows (`forklift-with-lsp-cargo`, `floor-closed`, ...). Inputs come as a zip
  of folders named by kind; each kind feeds one engine.
- **Output case**: one change of an engine. It has a prompt per input kind it takes (no prompt = it does not take
  that kind) and makes exactly ONE output per input image it takes.
- **Object reference**: one reference sheet per object (`LSP`, `SKID`, `CARGO`, `FLOOR_OPENING`): one image of the
  object from several angles plus the text that describes it. Prompts name them as `{ID}`; each output numbers them
  (`image 2`, `image 3`, ...) in the order of the case's references. Sheets live in `references/<ID>/`; replaced
  ones are kept in `references/<ID>/previous/`. `python -m synth.reference_sheet SKID|CARGO` rebuilds a sheet from
  the 3D catalogue.
- **Rebuild**: redo of an output the reviewer rejected: the rejected image goes to the model as the last image with
  the reviewer's feedback.

## Where things are

- `service/` — FastAPI service and web UI (`service/static/`): Generate, Projects, Engines, Prompts, Object
  references pages. `app.py` runs jobs (3 edits at a time), `store.py` the database (projects, catalogue, prompts),
  `job_records.py` each job's full state, `reference_store.py` the object references, `catalogue_api.py` the
  management API.
- `synth/try_astra_edit_batch.py` — `edit_frame()`: one model edit of one image (prompt template, reference images).
- `synth/violation_cases.py` — the engines, kinds and prompts as first seeded into the database. The database is the
  source of truth now: change prompts on the Prompts page (or its API), not here.
- Jobs: files in `work/service/jobs/<job id>/` (inputs, runs with each prompt and model log, outputs); their state
  in the database, `job.json` is a copy.

## Run and test

- Database: MySQL, configured in `.env` (`MYSQL_*`, see `.env.example`); tables are created and upgraded on start.
- Service on 192.168.2.240: `systemctl --user restart violation-generator.service` (port 8000; public at
  https://ai-data-generation.viact.ai behind Cloudflare, whose WAF can block zip uploads).
- **Never restart while a job has outputs queued or running** (check `GET /api/jobs` counts): they become
  interrupted and must be resumed.
- Tests: `.venv/bin/python -m pytest tests -q` (sqlite, temporary folders; no model calls).
