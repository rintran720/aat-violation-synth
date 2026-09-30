# Real-forklift anchor POC

## Goal

Test the new composition direction on one same-camera frame that already contains a real forklift. Preserve every source forklift pixel and render only LSP, cargo and SKID geometry.

## Input and anchor

- Frame: `data/cam01/background2.jpg` (1920×1080), with one large forklift on the right and clear floor extending in front of its forks.
- Camera: reuse `work/camera.json` and the floor XYZ map `work/points_world.npy` generated for the same fixed viewpoint from the clean plate.
- Anchor configuration: `work/anchor_poc.json`; two manually identified same-side wheel/floor contact pixels: front `[1100,550]`, rear `[1250,575]`.
- Lift both image points through the MoGe floor map. The resulting wheelbase is 1.6462 m versus 1.68 m in the authored forklift model (ratio 0.9799). The fitted base is `[2.1313,10.8363,0]` m and heading is 57.844°; translation is the mean fit from both wheel landmarks.
- The anchor points are manual estimates, not SAM3 keypoints. MoGe was not rerun because camera and floor calibration are shared; SAM3 was not needed for this manual-anchor image-only spike.

## Scene and output

Render two 1.9×1.85×0.09 m LSPs in series, cargo on the first sheet only (V6), and one idle 1.2×1.0×0.151 m SKID alongside as a distractor. The real forklift is not loaded into the beauty scene. Blender renders only the existing textured assets and floor shadow; `synth/composite_anchor_poc.py` places that RGBA render over the photographic frame.

- Preview: `work/out/anchor_poc_v6.jpg`
- Lossless composite: `work/out/anchor_poc_v6.png`
- Sidecar: `work/out/anchor_poc_v6.json`
- Transparent render: `work/renders/anchor_poc_v6.png`
- Reopenable Blender scene: `work/scenes/anchor_poc_v6.blend`

The source image remains unchanged outside the render support in the PNG. The event box and object boxes are in the sidecar. The 1920×1080 output was visually checked at full resolution; cargo rests on the first LSP, the second sheet remains empty, and the idle SKID is clear of the forklift/load path.

## Limits and next POC gate

This establishes a usable initial anchor and shows the reused object assets can be placed against the real forklift. It is not yet a fully automatic anchor or occlusion solution: wheel contact pixels were selected by eye, and there is no SAM3 forklift mask or depth-tested forklift holdout. In this first image the load is layered over the source where it overlaps; a segmented/proxy-depth pass must be added before broad placement variations. The synthetic shadow is approximate, and visibility is visually reviewed rather than validated with per-instance ID masks.

Next gate: refine the wheel/fork landmarks and add a forklift mask or depth proxy so the original forklift correctly occludes the rendered cargo/LSP. Then compare a one-LSP valid scene with this two-LSP violation before generating more layouts.

## Follow-up frame: `s_018.jpg`

At the user's request, reran the V6 layout on `data/cam01/rtsp_samples/s_018.jpg`, where the forklift is more visible and its forks point into the open foreground aisle. The same-viewpoint MoGe floor map was reused. The initial wheel-anchored render was superseded after review because wheel landmarks did not constrain the fork head closely enough. The current render directly anchors to the visible fork-tip and heel centers.

- Preview: `work/out/anchor_poc_s018_forkaxis.jpg`
- Lossless composite: `work/out/anchor_poc_s018_forkaxis.png`
- Sidecar: `work/out/anchor_poc_s018_forkaxis.json`
- Transparent render: `work/renders/anchor_poc_s018_forkaxis.png`
- Reopenable Blender scene: `work/scenes/anchor_poc_s018_forkaxis.blend`

The compositing check confirms the background is unchanged outside the rendered support. Fork landmarks remain manual estimates; the source forklift does not yet occlude rendered objects with a SAM3 mask/depth proxy.

## Additional frame sweep

Reviewed the requested frames `s_026`, `s_039`, `s_056`, `s_057`, `s_067`, `s_071`, `s_073`, and `s_075`. The first four were not rendered: the forklift is small or obstructed around its forks, so the anchor landmarks are too ambiguous for a useful placement comparison. Rendered V6 variants for `s_067`, `s_071`, `s_073`, and `s_075`, with the idle SKID disabled to keep the comparison focused on alignment.

- Contact sheet: `work/out/anchor_poc_candidates_contact.jpg`
- `s_067`: `work/out/anchor_poc_s067_v6.jpg` — clearest forward-axis test among these; the tall cargo covers some of the source truck because this POC has no forklift occlusion mask.
- `s_071`: `work/out/anchor_poc_s071_v6.jpg` — useful side/back angle, but existing floor loads compete with the inserted load.
- `s_073`: `work/out/anchor_poc_s073_v6.jpg` — the source already has cargo/pallets in the fork path, so the synthetic load overlaps them.
- `s_075`: `work/out/anchor_poc_s075_v6.jpg` — the source forklift is already carrying cargo, making this a poor background for adding another synthetic load.

## Fork-axis correction

The wheel-based placement shifted the insert away from the forklift's front. The renderer now accepts two direct floor landmarks—the midpoint of the visible fork tips and the midpoint of the fork heels—and solves the model pose from those points. This aligns the authored forks to the photographed fork axis before placing the LSP and cargo. The same-camera MoGe map is reused; SAM3 is not required for this manual image-only POC.

- Corrected comparison sheet: `work/out/anchor_poc_forkaxis_contact.jpg`
- `s_018`: `work/out/anchor_poc_s018_forkaxis.jpg`
- `s_067`: `work/out/anchor_poc_s067_forkaxis.jpg`
- `s_071`: `work/out/anchor_poc_s071_forkaxis.jpg`
- `s_073`: `work/out/anchor_poc_s073_forkaxis.jpg`

The fork-axis solves have baseline ratios near 1.0 for `s_067`, `s_071`, and `s_073`. These images still need a forklift holdout/depth mask; some source tines and truck parts are covered by the inserted cargo. `s_073` also has real floor loads near the insert, so `s_067` is the clearest corrected test among these frames.

## Reproduce

```bash
blender -b --python-exit-code 1 --python blender/render_anchor_poc.py -- config.json --anchor work/anchor_poc.json --output-stem anchor_poc_v6
PYTHONPATH=. .venv/bin/python -m synth.composite_anchor_poc --config config.json --anchor work/anchor_poc.json --output-stem anchor_poc_v6

# Corrected direct fork-axis solve for s_018
blender -b --python-exit-code 1 --python blender/render_anchor_poc.py -- config.json --anchor work/anchor_poc_s018.json --output-stem anchor_poc_s018_forkaxis
PYTHONPATH=. .venv/bin/python -m synth.composite_anchor_poc --config config.json --anchor work/anchor_poc_s018.json --output-stem anchor_poc_s018_forkaxis
```

## Multi-landmark fit spike (2026-09-30)

Implemented an image-space weighted least-squares fit with four fork tip/heel landmarks, giving fork points higher weight and recording per-point reprojection errors. A 12 px RMSE gate rejects a fit before rendering, so uncertain landmarks cannot silently produce a misleading image. The existing MoGe floor-map fork-axis render remains the current visual reference.

The first manually marked landmark attempt did not pass the gate: reprojection RMSE was 34.6 px, and the exploratory render visibly covered the real forklift. That output is retained only for diagnosis at `work/out/anchor_poc_s018_multilandmark2.jpg`; the attempted points are in `work/anchor_poc_s018_landmarks_attempt.json`. The landmark solver is not yet validated for this camera. Next: get a reliable forklift mask and landmarks from GPU inference, then rerun and check the RMSE and occlusion before accepting the render.

### Local model setup and MoGe run

Installed the repository-pinned SAM3 and MoGe source into ignored `vendor/`, and installed their Python dependencies plus PyTorch 2.10 CPU into the ignored `.venv`. This host exposes no `nvidia-smi` or NVIDIA device, so the CUDA setup from `setup.sh` cannot run here. Both model APIs import successfully. Downloaded the public `Ruicheng/moge-2-vitl-normal` checkpoint (about 1.3 GB) and ran it locally on `s_018.jpg` at inference resolution level 5; the full-size point map is `work/points_s_018_moge.npy`, and the valid-pixel map is `work/points_s_018_moge_valid.npy`. The lower inference resolution is a CPU workaround and needs comparison before it replaces the existing camera calibration.

Logged into the user's selected Hugging Face account and verified access to `facebook/sam3`; the 3.4 GB SAM3 checkpoint is now cached under ignored `work/huggingface/`. CPU inference on `s_018` was attempted, but this pinned SAM3 build assumes CUDA in a few more places and CPU BF16 prompt inference did not finish successfully. Do not copy `work/huggingface/token` or `stored_tokens` to another machine; authenticate there with an account that has SAM3 access.

### GPU handoff

The other workstation should have an NVIDIA GPU with a CUDA 12.8-capable driver. From a copy of this checkout with `data/` available:

```bash
bash setup.sh
HF_HOME=work/huggingface .venv/bin/hf auth login --force
HF_HOME=work/huggingface .venv/bin/python -m synth.segment  # project Stage 2: background + references
HF_HOME=work/huggingface .venv/bin/python -m synth.segment data/cam01/rtsp_samples/s_018.jpg forklift
HF_HOME=work/huggingface .venv/bin/python -m synth.calibrate  # project Stage 3
HF_HOME=work/huggingface .venv/bin/python -m synth.infer_moge_frame data/cam01/rtsp_samples/s_018.jpg --resolution-level 9
```

Bring back `work/masks/s_018/`, `work/refs/`, `work/refs/index.json`, `work/camera.json`, `work/points_world.npy`, `work/scene_facts.json`, `work/calibration.json`, and the full-resolution `work/points_s_018_moge.npy` plus `work/points_s_018_moge_valid.npy`. Do not copy the Hugging Face token/cache. `synth.infer_moge_frame` saves the exact-frame MoGe point map and valid mask; the standard calibrator remains based on the clean camera background. The official SAM3 setup requires checkpoint access; MoGe installation and checkpoint loading follow the [official MoGe repository](https://github.com/microsoft/MoGe).
