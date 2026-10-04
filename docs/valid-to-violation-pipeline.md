# Valid frame → reviewed violation image

`synth/valid_to_violation.py` accepts one valid CCTV image containing one forklift-loaded LSP and edits that real
image in two stages: add one empty LSP, then add one cargo. Each stage is gated by code first; Astra only scores
realism. Forklift, LSP, SKID and cargo appearance references come from Step A's `work/refs/index.json` by default,
or from `--reference-map` (recommended: `synth.catalogue_reference_map`, view-matched renders of the 3D catalogue).

## Workflow

1. **Anchor.** SAM3 segments forklift, LSP, cargo and floor. The loaded LSP's front edge is measured on the
   photo as the edge of its **top face**: line segments (OpenCV LSD) on the mask's lower outline, slanted edges
   allowed; when the top-face edge and the side face's floor contact both show (two parallel lines ≤ 15 px apart)
   the upper one is kept. Without a measurable edge it falls back to the mask's lowest row. `--contact-edge`
   overrides it (recorded as a manual anchor).
2. **Geometry.** The camera (`--camera`, default `work/camera.json`, principal point from `K_norm`) projects an
   equal second sheet (`lsp_measured_size_m` in `--calibration`) from that edge at top-face height. The source edge
   must measure within ±25 % of the calibrated width.
3. **Empty LSP.** The image model gets the source, the references and the geometry guide. Its raw output is
   composited into the source inside a bounded region only, colour-matched to the source, with SAM3 objects kept
   pixel for pixel and the edit feathered 2 px around them. Gates (all must pass):
   - raw output keeps the frame's aspect ratio and size;
   - raw drift: how much the model changed what it had to keep (outside the region, on protected objects),
     ≤ 8 % pixels and mean RGB delta ≤ 9;
   - no geometry-guide line colours drawn into the photo;
   - composite integrity (outside pixels exactly the source's);
   - SAM3: one new LSP, coverage ≥ 60 %, IoU ≥ 0.5, ≤ 20 % outside the target, ≥ 85 % of the strip along the shared
     edge covered (full width), no cargo yet;
   - Astra on a before/after close-up of the edit: scores 1–5 for edges, lighting, texture and contact shadow plus a
     list of located defects; code accepts only if every score ≥ 3, the mean ≥ 3.5 and Astra accepts.
   Failures (measured and visual) are fed to the next attempt.
4. **Cargo.** Starting from the accepted sheet each retry, the model adds one cargo. Its target box is the projection
   of a 3D box of the reference map's cargo `size_m` standing on the new sheet; without a size, the original load is
   scaled. Same gates, with SAM3 checking one new cargo supported by the new sheet, its height and an unchanged
   exposed deck. Only a passing image is published.

References are **appearance only**: every prompt says so, and sizes and positions come from the geometry above.
All masks, geometry, close-ups, scores and candidate paths are in `work/out/<output-stem>_vision_*/report.json`.
SAM3 masks are cached by input hash under `work/vision_cache`. `--moge` adds a MoGe floor-plane diagnostic to the
report (not a gate).

## Run

The default editor uses a signed-in Codex CLI with `gpt-6-astra` and its `image_gen.imagegen` tool. SAM3 and the
camera calibration must be installed locally. `--editor api` uses `OPENAI_API_KEY` and the imagegen skill CLI.

```bash
python -m synth.catalogue_reference_map --image data/cam01/rtsp_samples/s_003.jpg --seed 3
python -m synth.valid_to_violation data/cam01/rtsp_samples/s_003.jpg \
  --reference-map work/catalogue3d/reference_maps/s_003/reference_map.json \
  --output work/out/s_003_violation.png --max-attempts 3
```

If the source edge is too occluded, give left-to-right endpoints of the top-face edge in source-image pixels with
`--contact-edge 872,471,1033,477` (pass the same edge to `catalogue_reference_map` so its views match).
Existing final outputs are never overwritten. Exit code `2` means the sheet or cargo stage exhausted its attempts,
with no final image. `--reuse-sheet` / `--reuse-cargo` re-verify earlier candidates.

## Verification limits

SAM3 can miss a dark, thin LSP or merge touching sheets into one mask; the numeric gate then fails and the report
shows the measurements. The calibration sets the target sheet size; it cannot prove edges hidden behind the cargo.
Astra's scores are a model's judgement on a close-up, so the hard checks above are all done by code.
