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

### Follow-up frame sweep and reusable scene profiles (2026-09-30)

Revisited all eight requested backgrounds. `s_026` has an exploratory V6 render at `work/out/anchor_poc_s026_aligned_v1.jpg` (tip `[955,345]`, heel `[955,333]`, measured baseline 0.999 m, ratio 0.925); the user judged it temporarily acceptable. `s_039` has `work/out/anchor_poc_s039_aligned_v1.jpg` (tip `[886,265]`, heel `[858,254]`, baseline 0.944 m, ratio 0.874); it visibly misses the truck and is rejected. These two configs are `work/anchor_poc_s_026_aligned.json` and `work/anchor_poc_s_039_aligned.json`.

The old `s_067` fork-axis points were wrong: they forced a vertical image-space axis although the visible tines run diagonally. A four-endpoint estimate had 68.62 px RMSE and was rejected. Replacing it with two fork-axis midpoints at the model tine height gave 3.14 px RMSE and produced `work/out/anchor_poc_s067_heightfit_v1.jpg`; this is a better geometric fit, though the synthetic load still overlaps real floor objects and needs visual acceptance. For `s_071`, the height-aware fit gave 0.36 px RMSE (`work/out/anchor_poc_s071_heightfit_v1.jpg`), but the rendered sheet/load collides with the real blue pallet, so this frame is not a clean candidate. For `s_073`, fit RMSE was 1.35 px, but the second LSP projects to y=1090 and clips; reject it for a two-LSP scene. `s_075` already has cargo on its forks and its old wheel-anchor baseline ratio is 0.790, so reject it as an insert background.

The SAM3 scene-mask run for `s_056`/`s_057` did not complete on this CPU-only workstation. The 3.4 GB checkpoint loaded, but float32 CPU inference did not finish in the bounded attempt; no masks were produced. Their fork direction is not validated. The latest comparison sheet is `work/out/aat_s_frame_sweep_contact.jpg`. LSP geometry has its bottom at z=0; cargo starts at z=0 and is placed at the 9 cm LSP top. The “floating” appearance in rejected frames therefore comes from incorrect projected pose/overlap and compositing evidence, not an elevated object origin. `s_018` remains the only fully reviewed background; `s_026` is temporarily acceptable, and the other tested frames remain rejected or provisional.

The scalable flow should run the expensive perception/calibration pass once for each chosen background and save a reusable scene profile: SAM3 masks, one same-frame MoGe map registered to the fixed-camera calibration, fork-prioritized landmarks, object boxes, floor contacts, and confidence/fit residuals. A deterministic batch stage then reuses that profile and the asset library to vary cargo asset, LSP asset/count/arrangement, cargo count, and seed; each output gets its own labels and seed in the sidecar. LLM calls should be limited to initial asset/scenario setup or explicit review changes, not made per image. Before scaling a profile, require a valid fork-axis or fallback multi-landmark fit and a visual overlap gate.

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

## Windows GPU run: s_018, s_026, s_067, s_071 (2026-09-30)

Rebuilt all eight assets on the Windows RTX 3080 workstation (CLI-Anything harness cloned to `tools/CLI-Anything`, currently untracked; `PYTHONUTF8=1` is required because the harness writes non-ASCII into generated scripts under the cp1252 default). SAM3 masks for `forklift`, `LSP`, `cargo`, `SKID` were generated on GPU for all four frames under `work/masks/s_*/`.

Each frame renders scenario V1 (two LSPs in series, cargo on both) with anchors in `work/anchor_s{018,026,067,071}.json`; outputs are `work/out/viol_s*.{jpg,png,json}`.

| Frame | Anchor | Baseline ratio | Notes |
|---|---|---|---|
| s_018 | fork tip/heel | 1.44 | forks toward camera; load correctly in front of the truck. Tip pixel is the weakest estimate. |
| s_026 | fork tip/heel | 1.94 | truck is small and distant; anchor is unreliable, keep only as a stress sample. |
| s_067 | fork tip/heel | 0.99 | best fit; mast and heel reprojections match the photo. |
| s_071 | near-side wheel contacts (`same_side: negative`) | 0.89 | forks point away, so the load is behind the truck: `occluder_masks` (SAM3 `forklift_1`) keeps the real truck in front. |

`synth.composite_anchor_poc` now accepts an optional `occluder_masks` list in the anchor file; pixels inside those masks keep the photograph, so a real object nearer to the camera hides the render. It is a 2D holdout, valid only when the whole masked object is in front of the inserted load.
### GPU data alignment pass (2026-09-30)

Imported the GPU outputs from `work.zip`. The SAM3 forklift mask covers 14,449 pixels in `s_018.jpg`; the same-frame MoGe XYZ map is full resolution. Rechecked the fork-axis points on a magnified crop: heel midpoint `[956,411]`, tip midpoint `[972,433]`. Lifting them through the fixed-camera floor map gives a 1.1019 m baseline against the authored 1.08 m fork axis (ratio 1.0202). A four-endpoint fit was rejected at 29.3 px RMSE because the low-resolution fork edges and floor shadows do not provide consistent endpoint landmarks.

The V6 scene has two LSPs in series, cargo on the first, and no idle SKID. Visual review found that restoring the entire SAM3 forklift mask over the synthetic render incorrectly hid cargo/LSP pixels. The review candidate `work/out/anchor_poc_s018_aligned_v6_frontload.jpg` therefore composites the synthetic objects over the source image; the background remains unchanged outside the rendered support. Its sidecar records the two-LSP violation label and the compositing order.

To assess depth in a common frame, the exact-frame MoGe points were registered to the calibrated clean-background floor map using corresponding floor pixels (similarity fit scale 1.0712; retained-floor residual p50 0.011 m and p90 0.017 m). In calibrated camera depth, the forklift-mask pixels inside the projected LSP-0 box have median depth 16.033 m (n=3,323), versus LSP-0 center 15.986 m: only 0.047 m nearer, too close to call confidently from this geometry. Inside the cargo box, the forklift-mask median is 16.017 m (n=13,871), versus cargo center 15.582 m: cargo is 0.436 m nearer by the median estimate. LSP-1 has no SAM3 forklift-mask overlap. These calculations support putting the cargo in front in its overlap region and show why restoring the full forklift mask was wrong, but they do not establish correct per-pixel occlusion: source-object MoGe depth is noisy and synthetic object depth varies across each box. A proper final composite needs registered source depth and a Blender per-pixel depth pass.

On this viewpoint, V1 with cargo on both LSPs projects the two cargo boxes almost on top of each other. V6 gives the clearer alignment and count preview. Do not compare raw exact-frame MoGe Z directly with Blender camera depth: the frames differ until the MoGe map is registered to the camera calibration. The current front-load image is an alignment review candidate, not a final physically occlusion-correct composite.
# Direction-estimation limitation and fallback

The real-forklift-anchor experiment remains in `synth/composite_anchor_poc.py` for comparison. Across candidate backgrounds, estimating the forklift's travel direction from a partially occluded or poorly oriented real forklift is unreliable; a wrong heading makes the LSP and cargo drift sideways or appear airborne. Keep `s_018` as the only confirmed anchor-oriented example; other sampled views did not pass alignment review.

The fallback is the earlier clean-background workflow: place the calibrated 3D forklift model and its load on the floor, then render valid and violation variants with the camera's saved calibration. `synth/clean_background_poc.py` orchestrates the existing Blender scene renderer without deleting or replacing the anchor implementation. The forklift pose is calibrated explicitly for the camera rather than inferred from a real forklift in each frame. The four-panel JPG at `work/out/clean_background_three_scenes_preview.jpg` compares the configured `data/cam01/background.jpg`, one valid 1-LSP scene, a 2-LSP violation and a 3-LSP violation. The available plate has distant equipment/objects, although the main insertion area is clear; it is not an entirely empty warehouse. Recreate the renders with `.venv/bin/python -m synth.clean_background_poc --samples 16`, or only rebuild the sheet from saved renders with `--reuse-renders`.
