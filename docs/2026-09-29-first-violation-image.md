# First synthetic violation image — cam01 / V1

Generated on 2026-09-29 from the supplied camera background and locally authored 3D assets.

## Result

- `work/out/v_000.jpg`: unannotated 1920×1080 RGB output, with JPEG quantization tables/subsampling taken from the input.
- `work/out/v_000.png`: lossless master; decoded background pixels outside the insertion support and the protected top strip are unchanged.
- `work/out/v_000_review.jpg`: separate review copy with event box and numbered LSP markers.
- `work/out/v_000_detail.png`: cropped native-resolution detail view.
- `work/out/v_000.json`: scenario, source, placements, projected and visible boxes, instance visibility and limitations.
- `work/scenes/v_000.blend`: complete calibrated scene with packed asset textures, lighting and a floor shadow catcher.
- `work/renders/v_000.png`: transparent Cycles render.
- `work/renders/v_000_instance_ids.png` and `v_000_*_visible.png`: actual visible-instance evidence.

The scene contains a forklift facing 60 degrees around +Z, pushing two LSPs end-to-end, with one ULD on each sheet. Asset positions use the existing MoGe camera, floor and metre scale. Both LSPs rest at Z=0 and both ULD bases rest at the currently configured LSP thickness of 0.09 m. The nearest cargo rear face is 5 mm from the fork heels, and the sheet covers the full horizontal blade length. The sheet edges have an 8 mm gap. There are no inserted idle SKIDs in this example.

## Revision 16 — lower seat, 40% saturation and L-section side panels

- Reduced the seat cushion to 0.44 × 0.46 × 0.11 m and lowered its top surface from 1.17 to 0.98 m. The supporting pedestal is smaller and lower; the backrest and armrests follow the new seat position. Roof height remains 1.9725 m, giving 0.9475 m from the cushion surface to the roof underside (19 cm more than revision 15).
- User identified the two outer black body panels and clarified their L orientation: the horizontal face extends from the body into the seat bay, and the vertical face rises at the inner seat edge. Both faces are 0.18 m wide, with 0.032 m thickness. Horizontal faces use dark material; vertical faces retain the real side-panel texture.
- User explicitly requested 40% saturation this time. Applied HSV saturation ×1.4 after the existing selective yellow/black correction, rebuilding all eight assets from original textures. The background is not graded.
- Prior outputs/assets/scripts/configuration are retained in `work/revisions/v_000_before_r16_cabin_saturation/`; this revision uses the `v_000_r16` prefix.
- After the user's clarification, the L orientation places the vertical face at the inner seat edge and the horizontal face between it and the body. Both face widths are 0.18 m. The asset validator checks the inner-edge position and equal widths.
- All assets and the final assembled scene pass validation; both LSPs are visible and the forks remain fully covered. Seven focused tests pass. The r16/r15 comparison is `work/out/v_000_r16_comparison.jpg`.

## Revision 17 — restore flat side panels

- Per user request, reverted only the L-shaped side panels to the flat rectangular panels from revision 15. The lowered seat, 40% saturation, fork/mast/backrest geometry, tire shape and other r16 changes remain.
- Previous state is retained in `work/revisions/v_000_before_r17_side_panel_rollback/`; output uses prefix `v_000_r17`.
- Asset and scene validators pass: two LSPs visible, forks fully covered, rear wheels visible; seven focused tests pass. See `work/out/forklift_r17_comparison.jpg` for the restored panels.

## Revision 18 — lower tail, wider wheel clearance and black cabin details

- Lowered the top of the rear counterweight to 0.98 m, level with the cabin seat surface. The roof remains 1.9725 m.
- Increased front tire-to-arch radial clearance from 3 cm to 7 cm; the arch's outer radius and wheel-well cutter move with it. The wheel positions and diameters remain unchanged.
- Added a visible black footwell panel just ahead of the seat beneath the steering column. Recoloured the four curved guard posts, two side roof rails and two cross rails black to represent the cabin enclosure. The two flat exterior side panels now use body-yellow side texture instead of the black enclosure appearance.
- Prior outputs/assets/scripts are retained in `work/revisions/v_000_before_r18_tail_clearance_frame/`; this revision uses output prefix `v_000_r18`.
- Asset checks confirm the tail is at 0.98 m, roof height remains 1.9725 m, tire-to-arch clearance is 7 cm, the footwell and cabin frame are black, and exterior panels use body yellow. All eight assets pass; the final scene shows 7,771 / 5,581 visible LSP pixels, 4,159 rear-wheel pixels and zero exposed fork pixels. Render and scene validation pass; saved image uses prefix `v_000_r18`.

## Revision 19 — extend the black cabin floor and lower the tail

- Removed both long yellow side sills. Extended the black floor continuously from the rear face of the upright mast to the front of the counterweight, through the steering-column footwell. Added black low side returns along both cabin sides, ending at the counterweight.
- Lowered the counterweight top from 0.98 m to 0.72 m, aligning it with the cabin floor (0.675 m top surface, within 4.5 cm). Lowered the tail vents and lamps with the shell. The roof stays at 1.9725 m.
- Prior outputs/assets/scripts are retained in `work/revisions/v_000_before_r19_cabin_floor_tail/`; this revision uses output prefix `v_000_r19`.
- All eight assets pass validation. Tail top is 0.72 m against the black cabin floor at 0.675 m; the roof remains 1.9725 m and tire clearance remains 7 cm. Scene validation confirms both LSPs visible (7,771 / 5,581 pixels), both forks covered and rear wheels visible. The before/after image is `work/out/v_000_r19_comparison.jpg`.

## Revision 20 — correct the outside frame and yellow components

- The user clarified that the two target bars run vertically along the outside of the forklift body, not inside the cabin. The overhead guard posts and roof rails are now yellow, so they no longer read as black outer bars; the cabin floor and front footwell remain black.
- The seat pedestal/body beneath the seat, roof crossbars and parallel roof bars are yellow. The rear tail remains at 0.72 m near the cabin-floor level.
- The assembled violation image was rerendered with this forklift asset. All eight asset checks pass; the scene validator confirms two visible LSPs, fully occluded fork blades and visible rear wheels. Per-image results are recorded in the r20 sidecar.

## Revision 21 — black rear roof and paired rear supports

- Changed the rear roof panel to a dedicated matte-black material.
- Extended both yellow rear overhead-guard posts from the roof down to the counterweight/tail. Asset validation checks that there are two posts on opposite body sides and that both reach the tail.
- Rebuilt and reviewed the rear-angle preview, then rerendered the violation image. Asset and scene validation pass; the current saved image and sidecar use the r21 prefix.

## Revision 22 — remove the outside body bars and narrow the seat pedestal

- The user clarified that the two bars to remove are the yellow pieces outside along the body, not the rear roof-support posts. Removed both external side panels while retaining the paired yellow rear roof supports.
- Reduced the yellow seat pedestal width from 0.86 m to 0.774 m (90%); its length and height are unchanged.
- Asset validation confirms both side panels are absent and the pedestal width is 90%. Scene validation passes with two visible LSPs, covered fork blades and visible rear wheels. Current outputs use the r22 prefix.

## Scenario set — one valid and two violations

- `work/out/n1_valid.jpg`: N1, forklift pushing exactly one LSP; `is_violation=false`.
- `work/out/v2_three_series.jpg`: V2, three LSPs in series with cargo; `is_violation=true`.
- `work/out/v6_empty_extra.jpg`: V6, two LSPs in series with the extra LSP empty; `is_violation=true`.
- Each image has a lossless PNG, JSON sidecar, review copy and detail crop. All expected LSPs are visible in the ID pass, both fork blades are occluded, and rear wheels remain visible.
- A review found the cargo centre offset had used a forklift-local coordinate as a sheet-relative offset. Corrected it and regenerated all three scenes. `blender/validate_scenario.py` now checks each cargo footprint and height against its matching LSP; all three scenarios pass.

## CCTV realism pass — 2026-09-30

- `synth/composite_violation.py` now uses configured blur/noise, contracts the alpha mask by one pixel before feathering, adapts object chroma gently to the nearby background, rolls off rendered highlights, and applies a configurable Brown radial warp (`lens_k1=-0.035`) before compositing. The distortion coefficient is an initial visual estimate, not a calibration result; it must be tuned against a straight-line reference from this camera.
- `blender/export_asset.py` no longer projects the container's front-label texture onto side/rear shell faces. Those faces use the separate real side crop. Rebuild the reusable cargo asset and rerender to see this face-material correction; the checked-in outputs below were recomposited from the previously rendered assets.
- Re-exported `cargo_0` with the face-specific texture assignment, rerendered `n1_valid`, `v2_three_series`, and `v6_empty_extra`, then recomposited them at 1920×1080. The side now uses its own real crop rather than repeating the front label. The available source crops have blurred lettering, so this cannot recover sharper text detail; a sharper reference of the panel is needed for that.
- Validation: `PYTHONPATH=. .venv/bin/pytest -q tests/test_composite_violation.py tests/test_asset_sources.py tests/test_asset_author.py` (10 passed); each scenario compositor check confirms all expected LSPs, fully hidden fork blades, and visible rear wheels.
- `ffmpeg` 6.1.1 was available after its package-manager lock cleared. Each scenario now produces a five-frame, 5-fps 1920×1080 libx264 clip at CRF 28; the final decoded frame is saved as the `.jpg` (using the source JPEG quantization table). The `.mp4` is retained beside it, and the `.png` remains the pre-encode lossless compositing master.
- `ffprobe` confirms `v2_three_series_h264.mp4` is H.264, 1920×1080, 5 fps, five frames. All three scenario checks and the focused 10-test suite pass after the encode/decode round trip.

## Driver and red U safety light — 2026-09-30

- Added a seated low-detail operator in the forklift cabin, following the dark short-sleeve shirt visible in `rtsp_010.jpg`; the operator and clothing appear as a separate `person` object in each sidecar and instance-ID pass.
- Added a red area-light source mounted under the forklift and a U-shaped emissive floor footprint, open toward the forks. The floor trace is tied to forklift position and heading and is depth-tested, so the visible portion changes with occlusion from the vehicle and load.
- Rerendered and recomposited `n1_valid`, `v2_three_series`, and `v6_empty_extra`. The final ID checks still show every expected LSP, zero exposed fork blades, and visible rear wheels. The person and U-light masks are recorded separately in each JSON sidecar.
- Blender beauty passes used 32 Cycles samples with denoising for this update; existing asset appearance and scene placement are unchanged. Focused compositor/asset tests: 10 passed.

## Revision 15 — no boarding steps and corrected lift assembly

- Removed both boarding steps. Kept the two horizontal fork blades and roof-height upright mast (1.9725 m).
- Rebuilt the load backrest as an open frame spanning the body side panels: 1.244 m wide and 0.98625 m tall, exactly half the mast height. Its rear face sits against the front of the mast/chain assembly. The frame has two side rails, top/bottom cross rails and five internal vertical bars; it starts 0.11 m above the floor.
- Reduced the carriage's former solid plate to a compact mounting crossbar. Existing fork heels, blade dimensions and cargo contact locations are retained.
- Prior output/assets/scripts are retained in `work/revisions/v_000_before_r15_lift_frame/`; this revision uses the `v_000_r15` prefix. Asset metadata records lift dimensions and the mast/backrest gap.
- All eight assets and the assembled scene passed validation. Both LSPs remain visible (7,771 / 5,581 pixels), rear wheels remain visible (4,159 pixels), and horizontal fork blades remain fully covered (zero visible pixels). All seven focused tests passed. `work/out/forklift_r15.png` and `forklift_r15_comparison.jpg` show the unloaded lift frame clearly.

## Revision 14 — corrected colour interpretation and rounded tires

- User clarified that richer colour means stronger yellow and deeper black, not global saturation. Removed the mistaken cumulative saturation boost (1.728 → 1.0) and rebuilt from original source textures. Yellow paint blends 20% toward warm yellow at its existing maximum channel value, with a yellow-pixel mask on photo textures. Black materials use a 0.8 colour gain; texture shadows receive that gain through a smooth mask, preserving bright labels and panels. This is an authored visual correction, not a colour measurement.
- All four tires now have a swept rounded cross-section with 96 circumference segments, 10 subdivisions per shoulder arc, and smooth normals. Outer shoulder radii are 50.112 mm front / 35.2 mm rear. The centre bore has an 8 mm rounded lip. Tire diameters, widths, positions, recessed rims and 3 cm front arch clearance are retained.
- Prior assets/configuration/scripts/output are retained in `work/revisions/v_000_before_r14_color_tires/`; this revision uses the `v_000_r14` prefix.
- Validation passed for all eight assets, rounded tire mesh/smooth normals, dimensions, radial clearance and recessed hardware. The final scene has 8,106 / 5,581 visible LSP pixels, 4,159 visible rear-wheel pixels and zero exposed fork pixels. All seven focused tests passed.
- The current `data/cam01/background.jpg` was modified after the r13 output, outside this revision's changes. R14 uses that existing current background. The comparison image explicitly notes the background difference; the sidecar records its SHA-256. Historical descriptions of objects in the original background should not be assumed to describe this replacement.

## Revision 13 — another 20% colour increase

- Increased model HSV saturation another 20% relative to revision 12. `config.asset_appearance.saturation_multiplier=1.728` (1.44 × 1.2) rebuilds all eight assets from original source textures, preserving hue/value subject to saturation clipping.
- LSP marks remain 3 cm tall, sheet thickness 9 cm, and previous geometry/contact settings are retained.
- Prior assets/configuration/output are retained in `work/revisions/v_000_before_r13_color_boost/`; this revision uses output prefix `v_000_r13`.
- All eight reopened assets and the final scene passed validation. Both LSPs remain visible (8,106 / 5,581 pixels), with zero exposed fork-blade pixels. `work/out/v_000_r13_comparison.jpg` compares the final image with revision 12.

## Revision 12 — 3 cm markings and another 20% colour increase

- `config.lsp_edge_markings.height_m=0.03` reduces all three marks on each vertical LSP face from 4 cm to 3 cm. Their lengths and the 9 cm LSP thickness are retained.
- `config.asset_appearance.saturation_multiplier=1.44` applies another 20% HSV saturation increase relative to revision 11 (1.2 × 1.2), rebuilding from original textures to avoid repeated image quantization. The value is shared by all eight Blender/GLB assets and their previews.
- Prior outputs, assets and configuration are retained in `work/revisions/v_000_before_3cm_marks_color_boost/`. Updated outputs use the `v_000_r12` prefix.
- Validation passed for all eight assets and the reopened scene. The final mask contains 8,106 / 5,581 visible LSP pixels, 4,386 rear-wheel pixels and zero exposed fork-blade pixels. All seven focused tests pass; the before/after image is `work/out/v_000_r12_comparison.jpg`.

## Revision 11 — corrected LSP markings, 9 cm height, softer edges and richer colour

- `config.lsp_thickness_m=0.09` replaces the previous 10 cm estimate. Cargo bases follow the new sheet height. Fork pocket clearance in Z is constrained by the remaining thickness so the physical fork blades remain covered.
- All three edge marks are vertically centred and 4 cm tall (within 1 mm texture raster resolution). The middle mark is twice its revision-10 length; the outer marks are 1.2 times their prior lengths, rounded to the nearest texture pixel. Each vertical face keeps the central group of three.
- Rebuilt all eight reusable assets with small multi-segment bevels and face-weighted normals. Existing broad silhouette curves remain; fitted bounds and contact placements are checked after reopening the assets.
- Interpret “colour 20% stronger” as HSV saturation multiplied by 1.2, preserving hue/value before clipping. The optional colour-mode clarification had no answer at implementation time. Texture changes are baked into separate packed/portable copies under `work/textures/graded`; original source textures and background pixels are preserved. Existing scene exposure and compositing saturation settings remain unchanged.
- Asset validation checks mark widths/height, 9 cm sheet geometry, softened edges and material saturation metadata. The actual-camera mask check still requires visible rear wheels, two visible LSPs, and no exposed horizontal forks.
- Prior outputs/config/assets/textures are retained in `work/revisions/v_000_before_lsp_softness_color_fix/`. Current image is also saved as `work/out/v_000_r11.jpg`, with comparison `v_000_r11_comparison.jpg`.

## Revision 10 — 3 cm radial clearance around front tires

- Interpret the requested outward 10% adjustment as 10% of the 0.26 m tire width: move front wheel centres outward by 0.026 m from revision 9, to ±0.481 m. Hardware moves with the tire, preserving recessed rims and dark bolts.
- User clarified that 3 cm refers to the gap between the tire's outer circumference and the body arch, not lateral protrusion. Curved covers follow the existing chassis side at ±0.565 m and have an inner radius exactly 0.030 m larger than the tire radius. No 3 cm lateral-exposure constraint is applied.
- Geometry validation casts rays from the tire top to the arch underside to measure the 3 cm radial clearance, and checks covered tread versus the outer tire face. Diameter, rear-wheel visibility and cargo contact remain subject to the existing checks.
- Added the user's three light markings to the centre of all four vertical LSP faces, using a packed edge texture and face-specific UV coordinates. Until the orientation question is answered, the arrangement is provisionally three vertical bands in a central group. Top textures and 10 cm thickness remain intact; the stripes do not add protruding geometry or alter the asset dimensions.
- Prior output/assets are preserved in `work/revisions/v_000_before_front_exposure_fix/`; the updated image is saved as `work/out/v_000_r10.jpg`, with comparison `v_000_r10_comparison.jpg`.

## Revision 9 — front wheels inset, recessed rims and darker bolts

- Front wheel centres move from ±0.585 to ±0.455 m: 0.13 m inward per side, half their 0.26 m tire width. Front chassis wheel wells provide clearance. Front/rear diameters remain 0.6264/0.440 m.
- All four tire meshes now have a centre bore instead of being solid cylinders. Rim faces sit 31 mm inside the outer tire face; hubcaps and bolts also remain recessed. This creates real cavity shading rather than hiding the rim inside solid rubber.
- Wheel bolts use dark grey material (linear RGB 0.028/0.031/0.030, roughness 0.88) and rims use subdued dark steel. Portable GLB exports retain the authored base colours for procedural metal/rubber materials.
- Validators check front track, bore geometry, recessed hardware and dark rough bolts; the actual-camera rear-wheel and concealed-fork checks remain required.
- Previous assets and images are preserved in `work/revisions/v_000_before_wheel_inset_rims/`. Current render is also saved as `work/out/v_000_r09.jpg`, with `v_000_r09_comparison.jpg`.

## Revision 8 — front tires reduced 10%, rear wheels visible in CCTV

- Latest user correction is **10%**, superseding the interrupted 20% request. Front tire diameter decreases from 0.696 to 0.6264 m; hub/bolt radii and axle height follow the tires. Rear diameter stays 0.440 m.
- Revision 7's side-view visibility check was insufficient: the high CCTV viewpoint still hid the rear wheels. Rear tire centres now sit at ±0.50 m and their wheel wells open through the side shell. The short rounded tail is retained.
- A separate white ID marks rear tires/hubs/bolts in the actual calibrated camera render. Compositing requires more than 150 visible pixels and merges these pixels back into the forklift instance mask. Saved evidence is `work/renders/v_000_rear_wheels_visible.png`; the sidecar records the visible count.
- Previous output/model files are retained in `work/revisions/v_000_before_front_reduce_rear_reveal/`. Current render is also saved as `work/out/v_000_r08.jpg`, alongside `v_000_r08_comparison.jpg`.

## Revision 7 — smaller enclosed rear wheels and shorter tail

- User accepted the front tire size and requested smaller rear wheels, superseding revision 6's 1.20 ratio. Front diameter stays 0.696 m; rear diameter becomes 0.440 m. Rear axle moves forward 0.14 m, rear tire centres move inboard to ±0.33 m, and tire width becomes 0.18 m; hubs/bolts follow their wheels.
- The rounded tail retains its width but its longitudinal curve radius decreases from 0.58 to 0.38 m, shortening the tail by 0.20 m. Lamps, vents and label follow the shortened shell.
- Rear shell side skirts extend down to Z=0.14 m. Open-bottom wheel wells provide clearance for the complete rear wheels and hubs; wheels remain in the scene and renderable. Side-view ray tests require each rear wheel to be mostly occluded by the actual body.
- These sizes are authoring choices fitted to the user's feedback, not physical measurements. Mast/roof alignment, front tire size, curved cab guard and cargo/fork contact are preserved.
- Previous outputs are retained in `work/revisions/v_000_before_rear_wheel_tuck_fix/`; revision 7 is saved as `work/out/v_000_r07.jpg` with `v_000_r07_comparison.jpg`.

## Revision 6 — mast height, round tail and wheel ratio

- User specified that the upright lifting mast should reach the cab roof. Outer mast, crossbar, inner rails, chains and hydraulic ram now fit within the approximately 1.973 m roof height; the two horizontal forks and cargo contact remain in their prior positions.
- Rear counterweight and lower chassis now have a continuous rounded tail in plan instead of rectangular rear corners. The counterweight arc radius is an authored 0.58 m; tail lamps and ventilation strips follow the curved surface.
- Front tire diameter is 0.696 m and rear tire diameter 0.580 m, making the front exactly 20% larger. Hubs, bolts and axle heights follow the corresponding tire radii; all wheels still touch the floor.
- Reopened-asset validation checks actual mast/roof heights, both wheel ratios, floor contact and the curved tail outline. Prior outputs are preserved in `work/revisions/v_000_before_mast_tail_wheel_fix/`; current render is also saved as `work/out/v_000_r06.jpg` with `v_000_r06_comparison.jpg`.

## Revision 5 — lower roof and curved guard posts

- User clarified that the excessive height comes from the elevated cab roof. Lowered the roof assembly by 0.26 m in the model (about 12% of its previous height); roof top is approximately 1.973 m. This is a visual fitting adjustment, not a measured real-world dimension.
- Replaced four straight guard posts with continuous swept meshes following curved, inclined centrelines. Front posts sweep rearwards into the roof; rear posts have rounded upper bends. Roof rails, grille and headlamp attachments follow the lowered frame.
- Wheel diameters, chassis, seat, mast, fork contact, darker paint and absence of a roof beacon are preserved. The mast remains a straight mechanical guide; curved edits apply to the cab guard posts.
- CLI-Anything authors the blockout; `blender/refine_forklift.py` supplies the curved mesh refinement in detail/final export passes. Both `.blend` and `.glb` include the refined meshes. Validators check roof height and non-straight centrelines.
- Prior assets and image are retained in `work/revisions/v_000_before_roof_frame_fix/`; current render is also saved as `work/out/v_000_r05.jpg`, with comparison `work/out/v_000_r05_comparison.jpg`.

## Revision 4 — correct worker-shirt interpretation and forklift paint

- User confirmed the red patch above the forklift is a worker's shirt, not a roof beacon. Removed both invented beacon and mounting base from the authoring script, reusable forklift assets and generated scene; the model now contains 107 meshes.
- Yellow paint materials use a 0.55 linear colour multiplier and reduced specular strength to make the forklift darker and less washed out. This modifies the material, preserving the actual source texture files. Cargo, LSP and background grading settings are unchanged.
- The GLB exporter drops Blender paint-mix nodes, so the export helper writes the same paint multiplier into standard glTF `baseColorFactor` values. Procedural yellow paint uses its darker average as the GLB fallback; full variation remains in Blender.
- Asset and scene validators reject any remaining beacon object. Previous model/scene/output files are retained in `work/revisions/v_000_before_forklift_color_beacon_fix/`.
- Current output is also saved as `work/out/v_000_r04.jpg`, with before/after comparison `work/out/v_000_r04_comparison.jpg`.

## Revision 3 — source textures and load contact

- Six additional rectified source patches provide separate forklift roof, side sill, counterweight-side and access-panel textures, plus white/dark cargo roofs. Real image patches and their source quadrilaterals are recorded in `work/asset_sources.json`. New textures are packed into the updated `.blend` and `.glb` assets.
- Textured surfaces use rougher, less metallic materials. Painted frame, steel and rubber use restrained procedural wear variations; these are artistic approximations rather than measured surface detail. Existing exposure −0.65 EV and object saturation 0.78 are retained.
- The whole load moves about 1.20 m back from the old fork-tip placement. Contact is computed from the actual fork-heel component bounds; cargo has a 59.25 mm forward offset relative to each sheet centre so its rear face clears the heel by 5 mm.
- Both horizontal fork blades remain present and renderable. The first 10 cm LSP has boolean-cut underside pockets with 3 mm clearance around blade/heel geometry. These pockets are inferred accommodation, not recovered real LSP construction; the reusable standalone LSP assets are unchanged.
- A dedicated cyan ID distinguishes the blades from the forklift in the visibility pass. Compositing now rejects a render if any fully covered pixel of either blade is visible. Both LSPs must still have more than 300 visible pixels each.
- Previous outputs, assets, textures, previews and scene are retained in `work/revisions/v_000_before_texture_contact_fix/`. Revision 3 is also saved as `work/out/v_000_r03.jpg`; `v_000_r03_comparison.jpg` compares it with revision 2.

## Revision 2 — user visual feedback

- The user requested approximately 10 cm thickness instead of the initial 1.5 cm. `config.lsp_thickness_m=0.10` persists the estimate, and all three LSP assets, their GLB exports, previews and the scene are rebuilt from it.
- The inserted objects looked too vivid. Beauty-render exposure is reduced by 0.65 EV and object saturation to 0.78 in linear-space compositing. The ID pass uses neutral exposure, preserving visibility measurements. Source background pixels are not colour-graded.
- The previous image, scene, config and LSP parameters are retained in `work/revisions/v_000_before_color_lsp_fix/`.
- `work/out/v_000.json` contains the current exposure, saturation, thickness, contact target and actual visible-pixel counts. This is still a prototype awaiting realism acceptance.

## Verification

- Reopened the saved scene: all five actors retain their source component count and external bounds. The first sheet has the underside-pocket modification described above.
- Verified cargo-to-sheet height, sheet spacing, cargo-to-heel contact distance, and both blades contained under the first sheet footprint.
- Rendered an independent colour-ID pass to verify both LSPs are visible; current counts are recorded in the sidecar rather than assumed from bounding boxes.
- `lsp_count=2`, `is_violation=true`, and the event class is `Forklift Pushing Multiple Lsps`.
- Seven focused tests pass, including alpha compositing, object-only desaturation, protected-region preservation and the asset/texture tests.
- Camera, SAM3 crops and MoGe calibration inputs were not rewritten. Revision 2 adds the user-requested thickness and colour settings to configuration.

## Reproduce

```bash
blender --background --threads 4 --python-exit-code 1 --python blender/render_violation.py -- config.json
.venv/bin/python -m synth.composite_violation
blender --background --threads 4 --python-exit-code 1 --python blender/validate_violation.py -- work
.venv/bin/python -m pytest -q tests
```

These commands replace the generated `v_000` outputs. Rendering uses CPU Cycles, 64 samples and seed 7. The scene import updates Blender's dependency graph before transforming appended objects and checks their original extents, preventing component transforms from collapsing at the actor origin.

## Review status

This is a **synthetic prototype**, not an accepted training image. The modelled forklift and seated operator remain visibly simpler than the real scene. Materials, dimensions, hidden geometry and ceiling lighting remain approximate. There is a floor shadow catcher but no full rack/conveyor occlusion proxy.

The supplied background already contains real forklifts, people, cargo and idle LSPs. They are retained, and the sidecar describes only the inserted synthetic event; it is not exhaustive annotation of the whole frame. A genuine clean plate and a human realism review are required before treating this as a production dataset output. The supplied image is 1920×1080; it was not resized to the older plan's 960×540 example.

PNG verifies exact preservation outside the insertion support. JPEG recompression can alter decoded pixels anywhere in the frame; it does not provide that bitwise preservation guarantee. Review annotations are confined to the separate `_review.jpg` copy.

## Revision 21 — seated operator and sample-fitted red U safety light

- Aligned the full seated-worker pose to the authored seat centre (local forklift Y = −0.88 m). The pose is shifted 0.36 m rearward; the pelvis enters the 0.98 m-high cushion slightly, and thighs contact the seat instead of floating above it. The dark shirt remains matched to the worker crop in `rtsp_010.jpg`.
- Used the existing sample frame `data/cam01/references/rtsp_010.jpg` and calibrated scene scale; SAM3 and MoGe-2 were not rerun because their masks, camera calibration and floor scale already exist and the new object is a light effect.
- Sample inspection found an approximately 8 px bright red core and a roughly 40 px apparent light spill. Initial metric widths rendered too much like a solid tube, so final projective-fit widths are 0.024 m for the emissive core and 0.14 m for the broader, dimmer band. These are visual-fit estimates, not equipment specifications.
- The rounded U is open toward the forks. Its two side runs sit at local X = ±0.92 m, start at local Y = +1.02 m, and join a rear arc centred at Y = −1.75 m with 0.80 m half-width and 0.28 m depth. Two 35 W red area projectors are mounted to the vehicle sides at Z = 0.72 m, aimed down and out; the curve remains depth-tested so the forklift, cargo and LSP can occlude it from this camera.
- Regenerated `work/out/n1_valid.jpg`, `work/out/v2_three_series.jpg` and `work/out/v6_empty_extra.jpg`, plus their H.264 sidecars and JSON metadata. Compositing confirms all expected LSPs are visible, both fork blades remain fully covered, rear wheels remain visible, and `is_violation` matches each LSP count.
