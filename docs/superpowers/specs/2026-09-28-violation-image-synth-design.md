# AAT Violation Image Synthesis — Design Spec (POC)

- Date: 2026-09-28
- Status: Draft. Direction approved (Approach B; fixed violation "Forklift Pushing Multiple Lsps"; per-camera input: 1 clean background image + ~10 reference images). Updated 2026-09-29: SKID asset required and placed as idle distractors; scenario catalogue (section 1.1); scope narrowed: this ticket generates images, sidecars and scripts for Terry, who labels, trains and tests (sections 1 and 10)
- Project folder: `aat-violation-synth/`
- Requirements and decisions: [`docs/requirements/2026-09-28-gary-request.md`](../../requirements/2026-09-28-gary-request.md) (see its "LSP reference" and "Violation clip reference" sections)
- Companion plan: [`docs/superpowers/plans/2026-09-28-violation-image-synth-poc.md`](../plans/2026-09-28-violation-image-synth-poc.md)

## 1. Goal

The AAT client detects warehouse safety violations from CCTV (RTSP). Key objects: **forklift**, **LSP**, **SKID** (pallet), **cargo**. There are too few real violation images to train a detector, so we **synthesize violation images** that look as if the real camera recorded them. This ticket generates those images (violation and valid) with JSON sidecars and the generation scripts, and hands them to Terry. Terry labels them, converts them to the label format he needs, trains the model and tests it.

**Violation in scope (fixed): "Forklift Pushing Multiple Lsps".** A forklift that pushes cargo with **more than one LSP at the same time** is a violation: at most one LSP is valid, two or more is a violation. An LSP is a large, thin plastic slip sheet laid flat on the floor. Stretch-wrapped cargo sits on it and the forklift pushes the sheet to move the cargo. Idle LSPs with no forklift pushing them are **not** a violation. Reference: `docs/requirements/lsp-reference.png` (5 idle LSPs laid end to end) and `docs/requirements/violation-clip-frames/` (a real violation where 2 LSPs are pushed in series).

**Camera (concept):** one fixed viewpoint — exactly one clean background image plus ~10 reference images showing the objects from that same viewpoint, with its own calibration (scale, floor plane, lens). It is not tied to a physical device: any set of images sharing one viewpoint counts as one camera, and one physical camera that moves (e.g. PTZ presets) gives several cameras. `camera_id` names this viewpoint.

**POC goal:** take **1 camera** with its two separate inputs: exactly **1 clean background image** (no forklift, LSP, SKID or cargo), which is the base of every output and the input to MoGe-2 calibration, and **~10 reference images** of the same view showing the forklift, LSPs, SKIDs and cargo at different positions. The reference images are used only to extract objects and are never used as output backgrounds. SAM3 extracts the objects from the reference images into a crop library (asset references, textures, size cross-check, lighting reference), MoGe-2 calibrates the camera once from the clean background, and Astra 6 rebuilds the forklift, the cargo types and the SKID as 3D assets. Each output inserts **a forklift pushing cargo on a chain of LSPs in series** ahead of its forks, along its heading, onto the clean background: **2 LSPs** in total (the primary case, as in the real clip) or, less often, **3**, mixed by `lsp_count_weights` in `config.json`. As a **proposed addition (not in Gary's request, which asks for violation images only)**, the same pipeline also produces **synthetic valid images**: a forklift pushing exactly 1 LSP (`valid_fraction` in `config.json`) and some idle LSP rows without a forklift (`idle_row_fraction`). The forklift, cargo and LSPs are rendered only in synthetic images, while real valid images have none, so without these images the model could learn the shortcut "rendered objects = violation"; with them it learns to count pushed LSPs instead of spotting rendered objects. Terry's A/B evaluation (section 10) will show whether they are needed; they can be dropped by setting both fractions to 0. Every output also places **0–2 idle SKIDs** (with or without cargo on top) as scene distractors inside `floor_region`, away from the forklift path (`skid_count_weights` in `config.json`); they are never part of the event. In the scenario catalogue (section 1.1) these are V1, V2, N1 and N2. Produce **~10 violation images and a few (3) valid images** that:

1. match the input frame format exactly: 960×540 frame from an H.264 stream at 5 fps, same file format and mode, H.264-like compression artifacts, CCTV noise and blur, and the burned-in OSD (the `SENSTAR` logo top-left and the timestamp top-right) left untouched;
2. look plausible at CCTV scale next to the real reference images. The inserted forklift, cargo and SKIDs need the right size, shape, colour and shading, must stand on the floor and cast matching shadows. An LSP is a thin strip only tens of pixels wide, partly hidden by the forklift and cargo, so the violation is subtle: each LSP needs the correct scale, must lie flat on the floor in series ahead of the forks, must be occluded correctly, and must match the real sheet's colour, wear and gloss;
3. each come with a JSON metadata sidecar containing the inserted objects (class, 3D pose, projected 2D box; idle SKIDs have class `skid`), `lsp_count`, `is_violation` (`lsp_count >= 2`) and, for violation images only, a 2D **event box** of class `Forklift Pushing Multiple Lsps` (the target model/class name) that covers the forklift, the cargo and all LSPs, never the idle SKIDs. The existing detector labels one box per event, so this box is a labeling hint for Terry. Valid images have no event box.

Out of scope for the POC: more cameras (viewpoints), mass generation (1000+ images), video sequences, lens-distortion modelling (it is only checked), other violation types, and the catalogue scenarios beyond V1, V2, N1 and N2 (section 1.1). Outside this ticket (Terry's work): labeling, conversion to the label format, training, the A/B evaluation and the RTSP live test.

**Success measure:** for this ticket, realistic images in the exact input format that pass human review (plan Task 15). Image realism is necessary but not sufficient for the project: the synthetic images pay off only if a model trained with them detects real violations better than a real-only baseline on held-out real footage. Terry runs that check (section 10); if he reports no gain, we tune realism or the scenario mix and regenerate.

**Prerequisite (pending: to be provided by stakeholders):** **raw footage without overlays** from the target camera, as two separate sets: exactly 1 clean background image (no forklift, LSP, SKID or cargo) and ~10 reference images with forklifts, LSPs, SKIDs and cargo at different positions (object extraction only), taken while the camera did not move. The provided violation clip is the annotated output of an existing detector (masks, boxes, arrow, label `Multiple LSPs ID: 5`). It is a reference only and must never be used as a background or training image. Until raw footage arrives, a frame from that clip may be used only to develop and smoke-test the deterministic code.

**Prerequisite (pending: to be provided by stakeholders):** the **Ubuntu GPU workstation** (section 2). It does not exist yet.

### 1.1 Scenario catalogue: Forklift Pushing Multiple Lsps

The catalogue covers this one violation only. Astra 6 generates variations for each approved scenario along these axes: forklift heading and position in the view, LSP gap and offset, cargo type and height, SKID distractors, lighting (plan `prompts/02_propose_variations.md`). The POC covers V1 and V2 (violations) and N1 and N2 (valid).

Violation scenarios (`is_violation = true`, pushed `lsp_count >= 2`):

| # | Scenario | What the image shows | Priority |
|---|---|---|---|
| V1 | In series, 2 LSPs | Forklift pushes cargo on 2 LSPs end to end (as in the real clip) | POC |
| V2 | In series, 3 LSPs | Same with 3 LSPs | POC |
| V3 | Stacked LSPs | 2+ LSPs stacked on top of each other under the cargo | Next |
| V4 | Staggered or overlapping | The extra LSP is offset sideways or partly overlaps the first | Next |
| V5 | Side by side | 2 LSPs next to each other, pushed together as a wide load | Next |
| V6 | Empty extra LSP | Cargo only on the first LSP; the extra LSP is empty | Next |
| V7 | Heavily occluded | The extra LSP is mostly hidden by cargo, the forklift, racks or a SKID | Next |
| V8 | Turning while pushing | The LSP chain is angled as the forklift turns | Later |

Look-alike valid scenes (hard negatives, `is_violation = false`). Synthetic valid images are our proposed addition, not part of Gary's request (which asks for violation images only): they stop the model learning "rendered objects = violation" and make it count pushed LSPs instead (see the POC goal):

| # | Scene | What the image shows | Priority |
|---|---|---|---|
| N1 | One LSP | Forklift pushing cargo on exactly 1 LSP | POC |
| N2 | Idle LSP row | Idle LSPs in a row, no forklift (as in `lsp-reference.png`) | POC |
| N3 | Forklift near idle LSPs | Forklift driving past or parked next to idle LSPs, not pushing them | Next |
| N4 | One LSP with distractors | Forklift pushing 1 LSP with a SKID or an idle LSP nearby | Next |
| N5 | SKID on forks | Forklift carrying a SKID on its forks, no LSP | Later |

Rules to confirm with Terry before generation: is V3 (stacked) a violation? Is V6 (empty extra LSP) a violation? Where is the boundary when a forklift touches an idle LSP (N3/N4)?

Other violation types (aisle blocking, overstacking, pedestrians, …) are out of scope.

## 2. Machine and tools

Workstation (pending: to be provided by stakeholders; it does not exist yet). Required: Ubuntu, an NVIDIA GPU, CUDA and SAM3.

An early Blender 5.2.2 feature check was run on a Windows machine and passed; Ubuntu is not tested yet.

| Tool | Role | Notes |
|---|---|---|
| Blender (>= 4.2, required by the CLI-Anything Blender harness) | Proxy scene, forklift / cargo / SKID / LSP assets, Cycles render with shadow catcher | Headless: `blender --background --python script.py -- args` |
| Codex CLI, model "GPT Astra 6" | Agent: proxy scene, 3D asset rebuild from real crops (forklift, cargo types, SKID) over several passes, LSP asset spec, scenario template, optional variation proposals per catalogue scenario (section 1.1) | Non-interactive: `codex exec`. Codex sign-in with GPT Astra 6 available; usage bounded by the plan's limits (no API key, no budget to set) |
| CLI-Anything Blender harness (`cli-anything-blender`) | The agent's CLI for authoring Blender scenes as JSON projects | Verified capabilities and gaps in section 6 |
| MoGe-2 (`Ruicheng/moge-2-vitl-normal`) | Metric point map and normalized intrinsics from the clean background (once per camera) | Camera calibration, floor plane, metric scale |
| SAM3 | Text-prompted masks: floor and racks on the clean background (plane fit, proxies); forklift, LSP, SKID and cargo on the reference images (crop library, size and position cross-check, textures) | Required on the workstation (CUDA GPU) |
| ffmpeg | Extract frames from raw video; H.264 round-trip of inserted pixels | Required on the workstation |
| Python 3.12 + numpy + Pillow + pytest | Deterministic glue: object crop library, calibration, LSP texture rectification, randomizer, composite and format matching, verification, contact sheet | No frameworks |

## 3. Approach B (chosen): keep the real clean background, render only what we insert

The camera's clean background frame is **the background of every output**. Blender renders **only the inserted objects (forklift, cargo, LSPs, idle SKIDs) and their shadows** on a transparent film. The proxy scene covers only the static geometry of the background (floor, racks, walls, columns) and is never visible. It exists to (a) catch the shadows of inserted objects and (b) occlude inserted objects behind real static ones. Inserted objects occlude each other in the render itself (the forklift and the cargo hide parts of the LSPs). The render is composited over the clean background and then format-matched, with the reference images as the reference for lighting, colour, shadows, noise, blur and compression.

### Why not a full re-render (Approach A)

A full re-render would put the whole image through the domain gap: CG textures, CG lighting and CG noise everywhere. The detector would learn "rendered = violation" and fail on real RTSP frames. With Approach B:

- almost every pixel comes from the real camera (real texture, light, noise, compression, OSD);
- only the inserted objects are synthetic, and we spend our effort matching them: assets rebuilt from real crops with real-crop textures, a texture rectified from real LSPs, lights derived from real shadows, blur, noise, gain, and H.264/JPEG re-encoding;
- the proxy geometry can be crude boxes, because it only needs to be in the right place;
- synthetic valid images (our proposed addition, not in Gary's request) go through the same pipeline, so rendering alone does not separate the classes and the model has to count pushed LSPs instead of spotting rendered objects.

The cost is that we need an accurate camera (intrinsics, pose, metric scale), lights whose shadows agree with the real frames, and forklift, cargo and SKID assets that pass as real at CCTV scale (section 8, top risk). MoGe-2 provides the camera, once per camera, cross-checked with real object sizes. The agent sets up the lights from the real shadows in the reference images, and the agent and a human review them.

## 4. Architecture

```
  data/raw/cam01.mp4 (RAW, no overlays) --ffmpeg fps=5--> data/frames/*.png --human picks, per camera-->
      data/input/cam01/background.png      (clean: no forklift / LSP / SKID / cargo)
      data/input/cam01/objects/obj_*.png   (~10 reference images: forklift, LSP, SKID, cargo at different positions; object extraction only, never a background)
                           |
                           v
  [SAM3] synth/segment.py
    background -> work/masks/background/{floor,rack}.png
    objects    -> work/masks/obj_*/*.png, work/refs/<frame>_<class>_<rank>.png + index.json
                  (object crop library: crop, mask, 2D box, floor-contact pixel)
                           |
          +----------------+-------------------------------+---------------------------------+
          v                                                v                                 |
  [MoGe-2] synth/calibrate.py   (once per camera)    synth/lsp_texture.py                    |
    clean background -> K, floor plane, metric scale (homography rectify LSP instances)      |
    cross-check: real LSP sizes, object positions    -> work/textures/lsp_<k>.png            |
    -> work/camera.json, points_world.npy,                     |                             |
       scene_facts.json, calibration.json                      |                             |
          |                                                    |                             |
          v                                                    |                             |
  [Codex exec + GPT Astra 6 + cli-anything-blender]  (agent, once per camera)                |
    prompts/01_proxy_scene.md -> work/proxy.blend (static proxies: floor, racks + lights)    |
        ^ loop: render_variants.py --debug-proxy + composite overlay (agent + human)         |
    prompts/03_lsp_params.md  -> work/lsp_params.json (size, thickness, corners, gloss)      |
    prompts/05_build_asset.md -> work/assets_raw/{forklift,cargo_<type>,skid}.blend <- crops +
        ^ loop: 3 passes + previews, re-prompted until the details show (agent + human)
                           |                                   |
                           v                                   v
    blender/build_lsp.py      -> work/assets/lsp_<k>.blend  (rounded thin sheet + real LSP texture)
    blender/texture_asset.py  -> work/assets/{forklift,cargo_<type>,skid}.blend  (real crop on tex_* materials)
    prompts/04_scenario.md    -> work/scenario.json  (floor region, headings, forks-to-LSP, in-series step, jitter)
    [optional] prompts/02_propose_variations.md -> work/variations/<scenario>.json  (catalogue scenario, section 1.1)
                           |
                           v   (deterministic, no LLM per image)
    synth/randomize.py         -> work/variants.json      (seeded forklift + cargo + N LSPs; N=1 valid, N>=2 violation; idle LSP rows; 0-2 idle SKIDs)
    blender/render_variants.py -> work/renders/<id>.png   (RGBA: forklift, cargo, LSPs, SKIDs + caught shadows)
                               -> work/out/<id>.json      (sidecar; event box "Forklift Pushing Multiple Lsps" if violation)
    synth/composite.py         -> work/out/<id>.png       (on the clean background: OSD protected, alpha-over, blur, noise, local H.264)
    synth/verify.py            -> format / OSD / sidecar checks (exit code)
    synth/contact_sheet.py     -> work/contact_sheet.jpg  (background + outputs, object and event boxes)
                           |
                           v
                  HUMAN visual review (acceptance)
```

## 5. Data flow and conventions

- **Inputs per camera** (`config.json`: `camera_id`, `background_image`, `object_frames_dir`): two separate inputs of the same fixed view. `background_image` is exactly 1 clean background image (no forklift, LSP, SKID or cargo): the base of every output and the input to MoGe-2 calibration. `object_frames_dir` is the reference images folder: ~10 reference images showing the forklift, LSPs, SKIDs and cargo at different positions, used only for object extraction (SAM3 crops, masks and sizes as asset references, textures, size cross-check and lighting reference) and never as output backgrounds. One `work_dir` per camera; its calibration and assets are reused by every image of that camera.
- **Object library** (`synth/segment.py`): SAM3 masks the floor and racks on the clean background, and the forklift, LSP, SKID and cargo on every reference image. For the top-3 instances per class and reference image it stores a tight crop (pixels outside the mask filled with the object's mean colour, so the crop can serve as a texture), and `work/refs/index.json` lists class, frame, score, 2D box, area and bottom-centre (floor-contact) pixel.
- **World frame:** metres, Z up, floor is `z = 0`, origin is the floor point below the camera, +Y is camera forward projected on the floor, +X is right.
- **Camera** (`work/camera.json`, from MoGe-2 on the clean background, once per camera): `width`, `height`, `lens_mm`, `sensor_width_mm` (36, sensor fit horizontal), `shift_x`, `shift_y`, `matrix_world` (4×4 in the Blender camera convention), `hfov_deg`, `K_norm`. MoGe intrinsics are normalized (`fx/W`, `fy/H`, `cx/W`, `cy/H`):
  - `lens_mm = fx_n · 36`
  - `shift_x = 0.5 − cx_n`
  - `shift_y = (cy_n − 0.5) · H / W` (valid for `W ≥ H`)
- **Pose:** RANSAC plane fit on MoGe points inside the SAM3 floor mask of the clean background. The normal that points towards the camera becomes world +Z, and the camera height is the plane distance.
- **Metric scale:** MoGe-2 is metric, and the calibration cross-checks it with the SAM3 objects. All images of one camera share the same viewpoint, so the LSP mask pixels of an reference image map onto the background's floor points. The footprint of every fully visible LSP (minimum-area rectangle; partly covered sheets are skipped by their aspect ratio) is compared with the known LSP size (`lsp_size_m`, nominal 1.1 × 1.1 m until supplied). `work/calibration.json` records the measured sizes, a `suggested_scale_correction` and the floor positions where each object class was seen. `scale_correction` in `config.json` is the knob.
- **Pixel → world** for the agent: `work/points_world.npy` has shape `(H//4, W//4, 3)`. The world XYZ of pixel `(u, v)` is `P[v//4, u//4]`.
- **LSP asset:** built deterministically by `blender/build_lsp.py` from `work/lsp_params.json` (`size_x_m`, `size_y_m`, `thickness_m` ≈ 0.01–0.02, `corner_radius_m`, `roughness`), as a rounded-corner thin sheet with planar UVs. Its base colour is a **real LSP texture**: SAM3 LSP instance masks are perspective-rectified with a 4-corner homography into 512×512 images. Instances with a low mask fill, i.e. partly covered by cargo, are skipped. There is one `.blend` per texture (`lsp_<k>`), and the randomizer picks a texture per placement. That, plus small offsets, rotation and optional 90° turns, gives the appearance variety ("wear"). Origin is at the footprint centre on the floor, and +Y is the pushing direction. The exact LSP size will be supplied by the user: `lsp_size_m` in `config.json` (`[x, y]` in metres, `null` = use the measured value) then overrides the measured `size_x_m`/`size_y_m` in `work/lsp_params.json`, and the measurement is kept as `measured_size_m`.
- **Forklift, cargo and SKID assets:** Astra 6 rebuilds each object from its crops with the harness (primitives, modifiers, Principled materials) in 3 passes with preview renders, and is re-prompted with the previews until the details show: `forklift`, one `cargo_<type>` per cargo type, and `skid` (required: idle SKIDs appear in every POC image). Output: `work/assets_raw/<name>.blend`. `blender/texture_asset.py` then box-projects the largest real crop onto the materials the agent named `tex_*` and saves `work/assets/<name>.blend`. Same convention as the LSP: metres, origin at the footprint centre on the floor, +Y = front / pushing direction.
- **Proxy** (`work/proxy.blend`), authored by the agent with CLI-Anything. Every mesh becomes a **shadow catcher** at render time: invisible in Combined, but it still blocks camera rays, so it both catches shadows and occludes. The proxy holds only the static geometry of the clean background: `proxy_floor` (required) and `proxy_<what>` boxes for racks, walls and columns. There are no forklift, cargo, LSP or SKID proxies. Lights in the proxy are the scene lights. Cameras from authored files are discarded.
- **Scenario template** (`work/scenario.json`): `assets` (`forklift`, `cargo`, `lsp`, `skid` lists), `lsp_thickness_m`, `floor_region` (world rectangle of open floor where forklifts are really seen), `heading_deg` ([min, max]), `forks_to_lsp_m` (forklift origin to the first LSP centre), `arrangements`, `jitter`, `skid_height_m`, `skid_clearance_m`, `idle_row_lsps`. The LSP chain lies ahead of the forks along the heading: LSP k sits at `forks_to_lsp_m` + k · step in the forklift's local frame, with `in_series` = `[0, size_y + gap, 0]` as primary. The cargo stands on the first LSP. A placement whose forklift origin or LSP centres leave `floor_region` is drawn again. The number of LSPs is not in the template: `valid_fraction` in `config.json` sets the share of valid images (exactly 1 LSP), and `lsp_count_weights` maps the total LSP count of a violation image to a relative weight (default `{"2": 3, "3": 1}`: 2 LSPs primary, 3 less often). `idle_row_fraction` sets the share of valid images that show only a row of idle LSPs (`idle_row_lsps` sheets in series, no forklift, no cargo; scenario N2). Idle SKIDs: `skid_count_weights` in `config.json` maps 0, 1 or 2 SKIDs per image to a relative weight (default `{"0": 1, "1": 2, "2": 1}`). Each SKID lies inside `floor_region`, at least `skid_clearance_m` to the side of the forklift's lane (the line through the forklift and its LSP chain) and from the other SKIDs; about half carry a cargo on top (at `skid_height_m`). They are scene distractors in valid and violation images alike, so they are no class cue, and they never enter the event box.
- **Variants** (`work/variants.json`): seeded placements (forklift + cargo + N LSPs or an idle LSP row, plus 0–2 idle SKIDs, `heading_deg`, `lsp_count`, `is_violation`, `skid_count`). This is what makes later mass generation LLM-free.
- **Render:** Cycles, film transparent, view transform `Standard`, 8-bit RGBA PNG at the input resolution. Shadows appear as black with partial alpha (Blender ≥ 3.0 shadow catcher behaviour).
- **Composite:** render alpha is zeroed inside `osd_boxes`. Then `out = blur(rgb·a) + bg·(1 − blur(a)) + noise·blur(a)`, where noise σ is estimated from the clean background.
  - Video-frame input (PNG): pixels near the inserted object are replaced by a libx264 round-trip (`h264_crf`) to get block and chroma artifacts. The background stays bit-exact.
  - JPEG input: the output is saved with the input's quantization tables, subsampling, EXIF and ICC.
- **Metadata sidecar** (`work/out/<id>.json`): `image`, `camera_id`, `source_image` (the clean background), `violation_id`, `arrangement`, `heading_deg`, `lsp_count` (LSPs pushed by the forklift, 0 for an idle LSP row), `is_violation` (`lsp_count >= 2`), `skid_count`, `objects[]` (every inserted object, idle SKIDs included as class `skid` named `skid_<k>`, cargo on them named `skid_<k>_cargo`: `name`, `class`, `asset`, `location`, `rot_z_deg`, `size_m`, `bbox_2d`), `event` (violation images: `class: "Forklift Pushing Multiple Lsps"`, `bbox_2d` = projection of the forklift, the cargo and all LSPs, never the idle SKIDs or their cargo; valid images: `null`). Boxes ignore occlusion. The sidecar JSON is the POC output and a labeling hint; conversion to the label format Terry needs is Terry's work, outside this ticket.

## 6. CLI-Anything: verified facts

Sources, read from `main` on 2026-09-28: README <https://github.com/HKUDS/CLI-Anything>; harness source <https://github.com/HKUDS/CLI-Anything/tree/main/blender/agent-harness> (`cli_anything/blender/blender_cli.py`, `utils/bpy_gen.py`, `utils/blender_backend.py`, `core/*.py`, `setup.py`, `skills/SKILL.md`); Codex skill <https://github.com/HKUDS/CLI-Anything/tree/main/codex-skill>.

Verified from source:

1. Package `cli-anything-blender` (Python ≥ 3.10; deps `click`, `prompt-toolkit`), entry point `cli-anything-blender`. Source install: `cd CLI-Anything/blender/agent-harness && pip install -e .`. The README also shows `pip install cli-anything-hub`, `cli-hub install blender`.
2. It is **stateful over a JSON project file** (`*.blend-cli.json`), not a live Blender session. Global options are `--json`, `--project PATH` and `--dry-run`. One-shot commands auto-save. Running with no subcommand opens a REPL.
3. Command groups:
   - `scene`: `new`, `open`, `save`, `info`, `profiles`, `json`
   - `object`: `add`, `remove`, `duplicate`, `transform`, `set`, `list`, `get`
   - `material`: `create`, `assign`, `set`, `list`, `get`
   - `modifier`: `list-available`, `info`, `add`, `remove`, `set`, `list`
   - `camera`: `add`, `set`, `set-active`, `list`
   - `light`: `add`, `set`, `list`
   - `animation`
   - `render`: `settings`, `info`, `presets`, `execute`, `script`
   - `preview`: `recipes`, `capture`, `latest`, `live ...`
   - `session`: `status`, `undo`, `redo`, `history`
4. `object add` only takes primitives (`cube, sphere, cylinder, cone, plane, torus, monkey, empty`) with `--location/-l x,y,z`, `--rotation/-r` (degrees), `--scale/-s` and `--param/-p key=value`. `cube` and `plane` default to `size=2.0`.
5. Supported features:
   - Modifiers: `subdivision_surface, mirror, array, bevel, solidify, decimate, boolean, smooth`.
   - Materials: Principled colour, metallic, roughness, specular.
   - Lights: `point, sun, spot, area`.
   - Camera properties: `location, rotation, focal_length, sensor_width, clip_start, clip_end, type, name, dof_*`. There is **no lens shift and no sensor fit**.
6. Rendering generates a complete bpy script from the JSON. The script deletes all objects first and ends with `bpy.ops.render.render(...)`, and is meant to be run with `blender --background --python script.py`.
   - `render script OUTPUT` prints the script to stdout.
   - `render execute OUTPUT` writes `_render_script.py` next to `OUTPUT`. In the current source it does **not** launch Blender.
   - `render settings --transparent` sets film transparent.
7. **Not supported:**
   - arbitrary Python
   - mesh import (OBJ/FBX/glTF)
   - vertex editing
   - image textures
   - shadow catcher / holdout flags
   - camera lens shift
   - compositor setup
   - the `--collection` option of `object add` (not emitted in the generated script)
8. `codex-skill/` (installed by `bash CLI-Anything/codex-skill/scripts/install.sh` into `${CODEX_HOME:-~/.codex}/skills/cli-anything`) is for **building or refining harnesses**, not for using one. The Blender harness ships its own usage skill at `cli_anything/blender/skills/SKILL.md`.

What this means for the design:

- The agent uses `cli-anything-blender` to **author** the proxy scene and the geometry of the 3D assets (forklift, cargo types, SKID), naming the materials that should carry the real look `tex_*`.
- The exported bpy script is run by our `blender/cli_to_blend.py`, with the render call neutralized, to save a `.blend`.
- The things the harness cannot do live in small deterministic bpy scripts that we own: the image-textured LSP (`blender/build_lsp.py`), real-crop textures on the agent's assets (`blender/texture_asset.py`), the calibrated camera with lens shift, the shadow catcher, placement, colour settings, instancing and metadata (`blender/render_variants.py`).
- If the harness proves too limiting, CLI-Anything's "Refine" mode could add commands (shadow catcher, lens shift, image texture). The POC does not need this.

## 7. Codex CLI: verified facts

Sources, read 2026-09-28:

- The reference page <https://developers.openai.com/codex/cli/reference>, which redirects to <https://learn.chatgpt.com/docs/developer-commands?surface=cli>.
- The source <https://github.com/openai/codex/blob/main/codex-rs/exec/src/cli.rs> and `codex-rs/utils/cli/src/shared_options.rs`.

Facts:

- `codex exec` (alias `codex e`) runs non-interactively. Options:
  - `--model/-m`
  - `--sandbox/-s {read-only|workspace-write|danger-full-access}`
  - `--cd/-C DIR`, `--add-dir DIR`
  - `--image/-i FILE[,FILE...]` (`num_args = 1..`, comma-delimited)
  - `--json`, `--output-last-message/-o FILE`, `--output-schema FILE`
  - `--skip-git-repo-check`, `--ephemeral`, `-c key=value`
  - `--dangerously-bypass-approvals-and-sandbox`
- `PROMPT` can be `-` (or omitted) to read from stdin. If stdin is piped and a prompt is also given, stdin is appended as a `<stdin>` block.
- `codex exec resume [SESSION_ID] [--last]` accepts `--image`.
- Because `--image` takes 1+ values, the plan puts the prompt `-` **before** the options and `--image` **last**.
- Skills load from `~/.codex/skills`. `AGENTS.md` in the project root carries the project instructions, and the plan puts the shared conventions there once.

## 8. Risks and mitigations

| Risk | Effect | Mitigation in POC |
|---|---|---|
| Rendered forklift, cargo and SKIDs not realistic enough (top risk: they are now the largest synthetic regions) | Domain gap; the detector learns "rendered forklift = violation" and fails on RTSP | Assets rebuilt from real crops of the same camera, with real-crop textures. Several Astra 6 passes with previews and a human detail gate (plan Task 13). Synthetic valid images from the same pipeline (proposed addition, not in Gary's request), so rendering is no class cue. Human review against the reference images. Terry's A/B (section 10) is the final judge |
| No raw footage (only the annotated clip exists) | Cannot produce valid outputs | Hard prerequisite, pending: to be provided by stakeholders. The deterministic code is developed on a clip frame, and final outputs require a raw clean background image and raw reference images |
| No truly clean frame (the floor is never empty) | No background, or leftover objects in every output | Pick the emptiest frame and keep leftovers out of `floor_region`. A temporal-median background is future work |
| The physical camera moved between the reference images and the background (PTZ, vibration), so they no longer share one viewpoint | Wrong size cross-check and floor positions | Pick all frames from one fixed period; frames from another PTZ preset form a separate camera. Check the `work/calibration.json` positions against the overlay |
| Subtle target: an LSP is a thin strip of tens of px, partly occluded | Small pose errors make it float, vanish or poke through the cargo | Forklift, cargo and LSPs are rendered together, so their mutual occlusion is exact. The in-series step comes from the measured LSP size, the heading and forks-to-LSP offset from the scenario. A human reviews full-resolution crops |
| Idle SKIDs placed on the forklift path or on the LSP chain | Implausible scene; a SKID inside the event could teach a wrong cue | The randomizer keeps every SKID at least `skid_clearance_m` beside the forklift's lane and from the other SKIDs, inside `floor_region`. SKIDs never enter the event box and appear in valid and violation images alike. Human review of the check images (plan Task 14) |
| Proxy inaccuracy (floor extent, racks) | Wrong occlusion, shadows cut | Debug overlay loop, `points_world.npy` for metric lookups, placements only inside `floor_region` (open floor where forklifts are really seen) |
| MoGe scale or floor error; camera far from the objects | Wrong object sizes | RANSAC inlier ratio in `scene_facts.json`, automatic cross-check with the measured size of real LSPs in the reference images (`work/calibration.json`, `suggested_scale_correction`), `scale_correction` knob, `lsp_size_m` once the user supplies the real size |
| Lighting mismatch | Inserted objects too bright or dark, wrong shadows | World colour = mean linear colour of the clean background. The agent derives lights from the real shadows in the reference images. `gain` knob. Textures come from real crops, so albedo and wear are close by construction |
| Texture source differs (other camera, other light) | Colour cast | Crops come from the reference images of the same camera; other sources (`lsp-reference.png`) only as a fallback. `gain` knob. Several textures, so no single bad one dominates |
| Lens distortion (wide-angle CCTV, visible in the clip) | Straight proxies do not match curved real edges near the borders | `floor_region` keeps the inserted objects away from the borders. The overlay shows distortion. An undistort/redistort step is future work |
| Compression mismatch | Inserted region too clean or too blocky | Local libx264 round-trip with a tunable `h264_crf`, compared visually at 100 % crop |
| Every image shares one background | The detector memorizes the background | Valid and violation images share it too, so it is no class cue. Mass generation adds more backgrounds and cameras (viewpoints) |
| OSD overwritten | Obvious artifact, detector shortcut | `osd_boxes` alpha mask in composite, checked by `synth/verify.py` |
| Codex sandbox blocks Blender | Agent cannot render | Spike (plan Task 1). Fallback is `--dangerously-bypass-approvals-and-sandbox` on the dedicated workstation |
| Codex plan usage limit reached mid-way through asset building | Asset and scene prompt rounds pause; the one-time agent phase (plan Tasks 11–14) may take 1–2 extra days | Save every agent output to files (`work/`) so re-runs do not re-call the agent. Batch requests into fewer prompts. Use `codex exec resume` to keep the context instead of opening new sessions. When limited, defer the next prompt round until the limit resets. Image generation is unaffected: it is scripted and calls no LLM |
| Composite in sRGB, black-only shadows | Slight tone error | Accepted for the POC. Upgrade path: Shadow Catcher pass (EXR), `bg × pass`, then alpha-over |

## 9. Future work (not in POC)

- Richer labeling hints in the sidecar, if Terry asks for them:
  - exact per-object masks from a Blender object-index pass;
  - occlusion-aware boxes.
- SAM3 as a realism QA check.
- Mass generation (~1000 images): same chain with `n_images: 1000`, more clean backgrounds and more cameras, i.e. viewpoints (calibration, proxy scene and asset check once per camera; every image of the camera is scripted), plus the optional variation proposals per catalogue scenario (`prompts/02`). Recommended: Terry runs a quick A/B on ~200 images before we generate 1,000 (section 10).
- The catalogue scenarios beyond the POC (V3–V8, N3–N5, section 1.1) once Terry confirms their rules; a temporal-median background when no empty frame exists, lens distortion model, Shadow Catcher pass compositing.
- Temporal consistency for short synthetic clips (the camera runs at 5 fps).

## 10. Recommended evaluation (done by Terry)

Realistic images are not the goal in themselves. The synthetic data has to make the trained model better on real footage. This evaluation is Terry's work, outside this ticket; it is kept here as a recommendation.

- **Held-out real test set.** It is never used for training or as a synthetic background. It holds:
  - raw violation clips (a forklift pushing ≥ 2 LSPs);
  - raw valid frames (a forklift pushing exactly 1 LSP);
  - idle-LSP frames (no forklift pushing).

  To avoid leakage, exclude every frame from the same clip or time window as any frame used as a clean background or reference image. Real violations are rare, so staged violations recorded on the camera and replays of recorded clips are allowed.
- **A/B comparison** with identical training settings (same architecture, hyperparameters, epochs and seed):
  - Model A: real data only (baseline);
  - Model B: real + synthetic data (violation and valid images).
- **Dose-response (optional):** train B with 0 / 250 / 500 / 1000 synthetic images to see whether more synthetic data helps, saturates or hurts.
- **Metrics** on the class `Forklift Pushing Multiple Lsps`: recall, precision and mAP@0.5, plus false alarms on valid 1-LSP frames and idle-LSP frames (the model must not fire when exactly 1 LSP is pushed). Report them per camera. False alarms of B on real valid frames must not rise: the synthetic valid images (our proposed addition) exist for that. Training B with and without them (both fractions set to 0) shows whether they are needed.
- **Early signal (recommended):** Terry runs a quick A/B on ~200 synthetic images before we generate 1,000. If B shows no gain, we go back and tune realism instead of scaling.
- **Pass criterion:** B beats A on recall by an agreed margin, with no drop in precision and no increase in false alarms. The thresholds are Terry's decision.
- **Then** Terry's RTSP live test (final step; Gate 3 in the project plan) confirms the result on the live stream.
- **Our part:** when Terry reports no gain (or more false alarms), we tune realism (assets, lighting, `gain`, `h264_crf`) or the scenario mix (`lsp_count_weights`, `valid_fraction`, `idle_row_fraction`, `skid_count_weights`), regenerate and hand over the new batch.

## 11. Open questions

Pending inputs (to be provided by stakeholders; not open questions):

- Raw no-overlay images per camera (POC: this camera), as two separate sets: exactly 1 clean background image (no forklift, LSP, SKID or cargo) plus ~10 reference images with forklifts, LSPs, SKIDs and cargo at several positions (including a forklift pushing exactly 1 LSP and fully visible idle LSPs), used only for object extraction; taken while the camera did not move. Prerequisite for plan Task 3.
- The Ubuntu GPU workstation (section 2): Ubuntu, an NVIDIA GPU, CUDA and SAM3. Prerequisite for plan Task 1.
- The exact LSP size. Until then it is measured (agent + MoGe); once supplied it goes into `lsp_size_m` in `config.json`.

Terry's decisions (outside this ticket):

- Label format (YOLO or COCO), one box per event (as the existing system does) or per object, and the conversion from our JSON sidecars.
- The held-out real test set, the A/B pass thresholds and the RTSP pass criteria (section 10).

Open:

1. How are multiple LSPs arranged in real violations besides "in series"? Stacked? Side by side? (Scenarios V3–V5 in section 1.1.) (Counts are decided: 2 LSPs in total by default, 3 less often.)
2. Do the real sheets have a lip or folded edge?
3. The Codex model slug for "GPT Astra 6". The plan uses `gpt-astra-6`, and Task 1 confirms or corrects it. Task 1 also records which account Codex is signed in with (personal or team) and the plan's usage limits (Codex runs on a signed-in plan, not an API key, so there is no budget to set).
4. Does the Codex `workspace-write` sandbox allow `blender --background` and `ffmpeg`? Can Codex view local images mid-session? (Task 1 spike. The plan re-attaches images with `--image`, which always works.)
5. Is `cli-anything-blender` on PyPI? (Its SKILL.md says `pip install cli-anything-blender`. The plan installs from source.)
6. Does `cli-anything-blender render script` print anything besides the script? (The spike checks that the output compiles.)
7. Blender 5.x: is `material.use_nodes` / `world.use_nodes` still settable? (The scripts tolerate both.)
8. SAM3 text prompts that work for LSPs (`"slip sheet"` in `config.json`; alternatives are `"blue plastic sheet"` and `"floor mat"`). The spike records the one that segments reliably.
9. OSD box coordinates. They were measured on the annotated clip as `[0, 0, 90, 24]` and `[760, 0, 960, 26]`, and must be re-checked on raw footage.
10. Does every camera have a truly empty frame (no forklift, LSP, SKID or cargo on the floor)? If not, a temporal-median background is needed.
11. Do material names given with `material create` survive in the exported bpy script (needed for `tex_*`)? `blender/texture_asset.py` stops with an error if no `tex_*` material exists.
12. Which forklift models, cargo types and SKID types appear on each camera (one asset each)?
13. Scenario rules to confirm with Terry before generation (section 1.1): is V3 (stacked) a violation? Is V6 (empty extra LSP) a violation? Where is the boundary when a forklift touches an idle LSP (N3/N4)?
