# Valid frame → reviewed violation image

`synth/valid_to_violation.py` accepts one valid **cam01** CCTV image containing
one forklift-loaded LSP. Its default workflow edits that real image in two
stages: add one empty LSP, then add one cargo. Forklift, LSP, SKID and cargo
appearance references come from Step A's `work/refs/index.json` by default. Use
`--reference-map` to supply four selected images from another catalogue.

## Default workflow

1. SAM3 segments forklift, LSP, cargo and floor in the input. It locates the
   central loaded LSP and its exposed front rim. MoGe computes a point map and
   floor-plane diagnostic for the input; the saved cam01 calibration projects
   an equal-size second LSP from the contact edge. A hidden edge is recorded
   as inferred geometry, never claimed to be visibly measured. If the source
   edge cannot be localized, the script stops with `--contact-edge` guidance.
2. Imagegen receives the real source, Step A references and a polygon guide.
   It adds **one empty LSP**. The script preserves original pixels outside a
   bounded edit region and restores original SAM3 objects inside it. SAM3
   checks that one new LSP covers the planned footprint, is not oversized and
   has no new cargo. Astra checks only texture, edges, shadow and lighting.
   Failed candidates receive the measured errors on the next attempt.
3. Only after the empty LSP passes, imagegen adds **one cargo** using the Step A
   cargo crop. Every retry begins from the same accepted empty-LSP image. SAM3
   checks one new cargo, its support on the LSP and exposed LSP geometry. The
   script again restores pixels outside the allowed region and original objects.
   Astra checks visual realism. Only a passing image is published.

All masks, point-map diagnostics, geometry and candidate paths are recorded in
the timestamped `work/out/<output-stem>_vision_*` run directory and its
`report.json`. SAM3 masks and MoGe point maps are cached by input hash under
`work/vision_cache`. A pass means the configured image checks passed; inferred
geometry cannot prove dimensions hidden behind the original cargo.

## Run

The default editor uses a signed-in Codex CLI with `gpt-6-astra` and its
`image_gen.imagegen` tool. SAM3, MoGe, calibration and Step A references must
be installed locally. The optional `--editor api` uses `OPENAI_API_KEY` and the
imagegen skill CLI.

```bash
PYTHONPATH=. .venv/bin/python synth/valid_to_violation.py \
  data/cam01/rtsp_samples/s_003.jpg \
  --output work/out/s_003_violation.png \
  --max-attempts 3
```

The downloaded catalogue in `work/new_reference_catalogue/reference` has
selected individual object images in `work/new_reference_catalogue/selected`.
To use them, add:

```bash
--reference-map work/new_reference_catalogue/selected/reference_map.json
```

If the source edge is too occluded for SAM3, inspect the source and provide
left-to-right endpoints in **source-image pixels**:

```bash
PYTHONPATH=. .venv/bin/python synth/valid_to_violation.py \
  data/cam01/rtsp_samples/s_003.jpg \
  --contact-edge 872,471,1033,477 \
  --output work/out/s_003_violation.png
```

The override is recorded as a manual anchor; it does not assert that the full
edge is visible. Existing final outputs are never overwritten. Exit code `2`
means the sheet or cargo stage exhausted its attempts, with no final image.
`--joint` retains the old one-step workflow and its `--resume` option;
`--legacy-staged` retains the earlier Astra-landmark two-stage workflow.

## Verification limits

SAM3 can miss a dark, thin LSP or combine touching sheets into one mask. In
those cases the numeric gate fails and the report shows the mask measurements.
MoGe supplies an estimated source point map, while the fixed cam01 calibration
sets the target sheet size (1.9 × 1.85 m); neither model proves hidden edges.
The SKID crops in Step A are fragments and are used only as preservation
references. No SKID is added.
