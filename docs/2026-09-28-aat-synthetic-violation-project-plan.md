# AAT Synthetic Violation Images — Project Plan

Sep 28, 2026 · @John

> Source: https://claude.ai/artifact/1GLh9whjEtZMwYkoSzBDCo#310a6be0-82a0.m1bz2j73sn4.1115~approach
> Saved from the user-provided text on September 29, 2026. Embedded diagrams were not included; their placeholders are preserved below. Statuses and estimates describe the source plan, not a fresh verification of the workspace.

This ticket generates realistic "Forklift Pushing Multiple Lsps" violation images and look-alike valid images from real CCTV backgrounds: a small proof of concept first, then ~1,000 images. It ends at handover: Terry labels the images, converts them to the needed format, retrains and re-tests the model. Estimated duration is about 2 weeks (8–11 working days) for one engineer. The clock starts when stakeholders provide the raw footage and the Ubuntu GPU workstation; neither is available yet.

## Background

The AAT detector lacks enough real violation images to train on, so we will synthesize them. The project calls for an agent pipeline built on Blender and GPT Astra 6; Qwen 3.8 did not give good enough results.

- **Target model / class:** "Forklift Pushing Multiple Lsps".
- **Rule:** a forklift pushing cargo on exactly 1 LSP is valid. Pushing 2 or more LSPs at once is a violation. Idle LSPs with no forklift are not a violation.
- **LSP:** a large, thin blue plastic slip sheet (roughly square, ~1–2 cm thick). Cargo sits on it and the forklift pushes the sheet. A SKID is a pallet; it appears in scenes as an idle object and matters for later violation types.
- **Camera (concept, not a device):** one fixed viewpoint, meaning a clean background image plus the frames that show objects from that same viewpoint, with its own calibration (scale, floor, lens). Any image set sharing one viewpoint counts as one camera; a physical camera that moves (e.g. PTZ presets) gives several. Throughout this plan, "per camera" means per viewpoint.
- **Camera format to match:** 960×540, H.264, 5 fps, with a burned-in logo and timestamp.
- **Why it is hard:** in the real clip the extra LSP is a thin strip only tens of pixels wide, partly hidden by the forklift and cargo.

## Approach

We keep a real clean background from each camera and render only the inserted objects (forklift, cargo, LSPs, SKIDs) with their shadows. A full 3D re-render of the whole scene would look synthetic; the model would learn "rendered = violation" and fail on live RTSP.

[embedded content: generation pipeline · 9 stages, input and output of each]

The input follows the requested workflow, per camera: one clean background image with no objects, plus about 10 reference images from the same viewpoint showing forklifts, LSPs, SKIDs and cargo at different positions. The background is the base of every output; the reference images are only used to extract the objects. SAM3 extracts each object. MoGe-2 measures the clean background, and the object sizes cross-check it to calibrate depth and scale once per camera. GPT Astra 6, driving Blender through CLI-Anything, rebuilds the extracted objects as 3D assets and composes the violation scenes. Once assets and calibration exist, 1,000 images are a scripted batch with no LLM call per image.

- **Proposed addition: synthetic valid images.** The original request asks only for violation images. We propose also rendering a few valid scenes (forklift pushing exactly 1 LSP, idle LSP row) with the same pipeline, so the model learns to count pushed LSPs instead of spotting rendered objects. The A/B test shows whether they are needed; they can be switched off in config. Idle SKIDs, with or without cargo, are placed as distractors away from the forklift path.
- Precision work (textures from real crops, calibrated camera, shadow catcher) is done by small scripts we own, because the CLI-Anything Blender harness cannot do image textures, shadow catchers or lens shift.
- Labeling is done by Terry, outside this ticket. Each image already ships with a JSON sidecar holding `lsp_count`, `is_violation` and an event box.

## Scenario catalogue: Forklift Pushing Multiple Lsps

Every scenario is a case of the one target violation, plus look-alike valid scenes that teach the model where the line is. The POC covers V1, V2, N1 and N2; the rest follow once the rules below are confirmed with Terry.

**Violation scenarios** (2 or more LSPs pushed)

| # | Scenario | What the image shows | Priority |
| --- | --- | --- | --- |
| V1 | In series, 2 LSPs | Forklift pushes cargo on 2 LSPs end to end, as in the real clip | POC |
| V2 | In series, 3 LSPs | Same with 3 LSPs | POC |
| V3 | Stacked LSPs | 2+ LSPs stacked on top of each other under the cargo | Next |
| V4 | Staggered or overlapping | The extra LSP is offset sideways or partly overlaps the first | Next |
| V5 | Side by side | 2 LSPs next to each other, pushed together as a wide load | Next |
| V6 | Empty extra LSP | Cargo only on the first LSP; the extra LSP is empty | Next |
| V7 | Heavily occluded | The extra LSP is mostly hidden by cargo, the forklift, racks or a SKID | Next |
| V8 | Turning while pushing | The LSP chain is angled as the forklift turns | Later |

**Look-alike valid scenes** (hard negatives, not a violation; proposed addition to stop the model from learning "rendered = violation")

| # | Scene | What the image shows | Priority |
| --- | --- | --- | --- |
| N1 | One LSP | Forklift pushing cargo on exactly 1 LSP | POC |
| N2 | Idle LSP row | Idle LSPs in a row with no forklift, as in the reference image | POC |
| N3 | Forklift near idle LSPs | Forklift driving past or parked next to idle LSPs, not pushing them | Next |
| N4 | One LSP with distractors | Forklift pushing 1 LSP with a SKID or an idle LSP nearby | Next |
| N5 | SKID on forks | Forklift carrying a SKID on its forks, no LSP | Later |

Astra 6 generates variations of each approved scenario: forklift heading and position in view, LSP gap and offset, cargo type and height, SKID distractors and lighting.

Rules to confirm with Terry before generating: is V3 (stacked) a violation, is V6 (empty extra LSP) a violation, and where is the line when a forklift touches an idle LSP (N3, N4)?

## Preparation

Three inputs block the proof of concept and are not yet provided: raw footage without overlays, the Ubuntu GPU workstation, and working GPT Astra 6 access. Terry supports all model, engine and image information.

| # | Item | Detail | Owner | Status |
| --- | --- | --- | --- | --- |
| 1 | Raw CCTV footage, no overlays | Not yet provided. Per camera: 1 clean background image (no objects) plus ~10 reference images of forklift, LSP, SKID and cargo at several positions | Terry | Pending |
| 2 | Exact LSP dimensions | Length, width, thickness. Until then the pipeline measures them from the image | Terry | Pending |
| 3 | Violation reference clip | 8 s clip of a forklift pushing 2 LSPs (annotated, reference only) | Terry | Pending |
| 4 | LSP reference image | 5 idle LSPs on the floor, used for texture and size cross-check | Terry | Ready |
| 5 | Workstation | Not yet provided. Needs Ubuntu, NVIDIA GPU, CUDA and SAM3 | Eric | Pending |
| 6 | Blender 5.2.2 LTS (pinned) | Must run headless on the workstation. Passed the feature check on Windows; Ubuntu not yet tested | John | Ready |
| 7 | Codex CLI + GPT Astra 6 | Codex sign-in on the workstation; confirm Astra 6 is available in Codex, the sandbox can run Blender, and the plan's usage limits | Harry | To verify |
| 8 | CLI-Anything, MoGe-2, ffmpeg | Install the Blender harness; Hugging Face access for MoGe-2 weights | John | To verify |
| 9 | Realism review | ~30 min per batch on a contact sheet | John, confirmed by Terry | Pending |
| 10 | Handover format | Folder layout, image format and sidecar fields Terry needs for labeling. Needed before Phase 3 | Terry | Pending |

### Blender check (Windows, 2026-09-28)

Blender 5.2.2 LTS covers every Blender step in Approach B. A headless test script rendered one 3-LSP violation image end to end on Windows with an RTX 3080, with no errors.

| Requirement | Result |
| --- | --- |
| Headless run (`blender -b -P`) | Pass |
| GPU render | Pass: Cycles sees the RTX 3080 via OptiX and CUDA; 3.5 s per image at 32 samples |
| 960×540 output | Pass |
| Calibrated camera (fx, cx, cy → focal length + lens shift; extrinsic matrix) | Pass |
| LSP image texture from a real crop | Pass |
| Shadow catcher on the real floor | Pass |
| Holdout proxy occludes the extra LSPs | Pass; proxy shadow turned off (`visible_shadow = False`) because the real frame already has the forklift shadow |
| Alpha-over onto the real CCTV frame | Pass (compositor) |
| JSON sidecar: `lsp_count`, `is_violation`, event box from projected 2D boxes | Pass |
| Reproducible seeds | Pass |
| Mesh import for MoGe-2 output (OBJ, PLY, STL, glTF) | Pass |
| H.264 encode | Pass in Blender; ffmpeg 8.1.1 is also installed |

- **Version choice:** pin 5.2.2 LTS on both Windows and Ubuntu. It is the newest LTS, and the pipeline code is new, so there is no reason to target 4.2.
- **API change to watch:** from 5.x, the compositor uses `scene.compositing_node_group` instead of `scene.node_tree`, and `Material.use_nodes` / `World.use_nodes` will be removed in 6.0. Code written for 4.2 may break.
- **Not yet tested:** Ubuntu, CLI-Anything and Codex (neither is installed yet), SAM3, MoGe-2, and realism with real footage (the test used a stand-in background and texture).

- [ ] Install CLI-Anything and confirm it can drive Blender 5.2.2 on Windows. If it only supports 4.x, revisit the version choice.

## Implementation roadmap

The work runs in three build phases plus a handover, about 2 weeks in total, and each phase must pass its gate before the next starts. Durations are working days for one engineer, counted from when the raw footage and the workstation are provided.

AI agents and models do the heavy work inside this ticket:

- **3D modelling by agent.** GPT Astra 6, driving Blender through CLI-Anything, rebuilds forklift, LSP, SKID and cargo from the extracted crops, so there is no manual 3D modelling.
- **Scenario variations by agent.** Astra 6 generates the variations of each approved scenario (LSP count, arrangement, forklift heading, cargo type).
- **Perception by AI models.** SAM3 segments the objects and MoGe-2 estimates depth and geometry, so camera calibration needs no manual measurement.
- **Generation is scripted.** After the one-time agent run per camera, 1,000 images are a batch job with no LLM call per image.

[embedded content: implementation roadmap · 3 build phases + handover]

1. **Phase 0: Setup and inputs (1–2 days)**
   1. Receive, for the first camera, 1 clean background image plus ~10 reference images with forklift, LSP, SKID and cargo at several positions.
   2. Verify the workstation: GPU, SAM3, Blender headless, MoGe-2.
   3. Install CLI-Anything and Codex CLI; confirm the GPT Astra 6 model ID and that Codex can run Blender, and record the Codex account and its usage limits.
2. **Phase 1: Proof of concept (5–6 days)**
   1. SAM3 extracts forklift, LSP, SKID and cargo from the ~10 reference images into an object crop library.
   2. MoGe-2 measures the clean background; object sizes cross-check it to calibrate depth and scale for the camera.
   3. Astra 6 rebuilds forklift, LSP, SKID and cargo as 3D assets in Blender over several prompt rounds; real crops become their textures.
   4. Astra 6 generates variations of the approved scenarios (LSP count, arrangement, forklift heading); scripts place forklift, cargo and LSPs on the floor, with idle SKIDs as distractors.
   5. Generate ~10 violation images plus a few valid ones, check format and review them on a contact sheet.
3. **Phase 2: Scale up (2–3 days)**
   1. Add more cameras (viewpoints): one clean background and one calibration per camera; the 3D assets are reused.
   2. Hand ~200 images to Terry for a quick A/B (recommended), then generate ~1,000 images by script with random seeds, no LLM per image.
   3. Spot-check a random sample for realism.
4. **Phase 3: Handover (Terry takes over)**
   1. Deliver violation and valid images, JSON sidecars, generation scripts and review notes.
   2. Terry labels the images, converts them to the needed format, retrains and re-tests the model.
   3. If Terry's results show no gain, we tune realism or the scenario mix and regenerate.

## Recommended evaluation (done by Terry)

Terry runs this check after labeling, outside this ticket. It measures model gain on real footage, not image realism. Our part is to tune realism or the scenario mix and regenerate when the gain is missing.

| Model | Training data | Role |
| --- | --- | --- |
| A | Real images only | Baseline |
| B | Real images + synthetic violations | Candidate |

1. **Hold out a real test set.** Raw violation clips (2+ LSPs), valid frames (exactly 1 LSP) and idle LSPs. None of it is used for training or as a background; frames from the same clip as a background are excluded. If real violations are too rare, stage a few on camera.
2. **Train A and B with identical settings** (same architecture, hyperparameters, epochs, seed).
3. **Compare on the test set,** per camera: recall, precision and mAP@0.5 for "Forklift Pushing Multiple Lsps", plus false alarms on 1-LSP and idle-LSP frames.
4. **Early check in Phase 2:** run a small A/B with ~200 synthetic images before generating 1,000. No gain means tuning realism first, not scaling.
5. **Optional:** train B with 250 / 500 / 1,000 synthetic images to see where the gain stops.

**Pass:** B beats A on recall by an agreed margin with no increase in false alarms. Thresholds are agreed with Terry; the RTSP live test then confirms the result. The early check fits inside Phase 2, and the A/B runs within Phase 3.

## Deliverables and acceptance

The first go/no-go point is the end of Phase 1: at least 7 of 10 generated images must pass a person's realism review.

| Phase | Deliverable | Accepted when |
| --- | --- | --- |
| 0 | Working toolchain on the workstation; spike notes | Blender renders headless, Codex + Astra 6 can drive it, SAM3 and MoGe-2 run on GPU |
| 1 | 10 violation images plus a few synthetic valid images from 1 camera, JSON sidecars, contact sheet, reusable scripts | ≥ 7/10 look real at CCTV scale; ≥ 8/10 show the 2+ LSPs clearly; same format as the camera (960×540, OSD untouched) |
| 2 | ~1,000 violation images across the available camera viewpoints | Random sample passes the same review; generation reproducible from seeds |
| 3 | Handover package: violation and valid images, JSON sidecars, generation scripts, review notes | Terry confirms the package is usable for labeling and training |

## Risks

The biggest risk is schedule, not technology: without raw footage, Phase 1 cannot produce usable images.

| Risk | Impact | Mitigation |
| --- | --- | --- |
| Raw footage or workstation arrives late | Phase 1 blocked; only code can be developed | Build and test the scripts on the reference clip meanwhile; final images need raw frames |
| Rendered forklift, cargo, LSP or SKID looks fake | Model learns "rendered = violation" | Textures from real crops, several Astra 6 prompt rounds, synthetic valid images rendered the same way, human review, A/B against a real-only baseline |
| Wrong object size or position | Implausible scenes | Real LSP dimensions from Terry override the measured value; per-camera calibration cross-checked with SAM3 object sizes and signed off by a person |
| Lighting or compression mismatch | Synthetic region stands out | Match blur, noise and H.264 artifacts to the camera; tunable settings reviewed per batch |
| CLI-Anything or Codex limits | Agent cannot finish a step | Precision steps already moved to our own scripts; Phase 0 spike confirms the rest early |
| Codex plan usage limit reached mid-way through asset building | Prompt rounds pause; Phase 1 may take 1–2 extra days | Save every agent output to files so re-runs need no new calls; batch requests into fewer prompts; resume sessions instead of starting new ones; image generation is unaffected |
| Synthetic images alone do not transfer to RTSP | Retrained model underperforms live | Terry's A/B and RTSP tests catch it; we tune realism or the scenario mix and regenerate |

## Decisions needed

Phase 0 starts as soon as these are in place.

- [ ] Provide Codex access with GPT Astra 6 (signed-in plan). Usage is bounded by the plan's limits and there is no budget to set; the agent runs only once per camera and variations are scripted, so usage stays small.
- [ ] Provide the Ubuntu workstation with an NVIDIA GPU, CUDA and SAM3.
- [ ] Terry to provide raw no-overlay footage, exact LSP dimensions and model/engine details, and confirm the scenario rules (V3, V6, N3/N4).
- [ ] Terry to confirm the handover format (folder layout, image format, sidecar fields).
