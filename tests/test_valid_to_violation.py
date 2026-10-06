"""Behavioral checks for the review/retry publication gate."""

from __future__ import annotations

import argparse
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
from PIL import Image

from synth import valid_to_violation as pipeline


SCORED = {"scores": {key: 4 for key in pipeline.SCORES}, "issues": [], "accepted": True, "feedback": "good"}


def args_for(root: Path, source: Path, output: Path, **extra) -> argparse.Namespace:
    values = dict(input=source, output=output, max_attempts=1, editor="codex", model="unused",
                  step_a_index=root / "index.json", image_cli=root / "unused.py", codex_bin="codex",
                  contact_edge="20,50,80,50", reference_map=None, reuse_sheet=None, reuse_cargo=None,
                  camera=root / "camera.json", calibration=root / "calibration.json", aesthetic_review=True)
    values.update(extra)
    return argparse.Namespace(**values)


class ValidToViolationTests(unittest.TestCase):
    def test_selected_reference_map_uses_all_four_local_images(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            entries = {}
            for category in pipeline.CLASSES:
                path = root / f"{category}.png"
                Image.new("RGB", (20, 20), "white").save(path)
                entries[category] = {"path": str(path)}
            entries["cargo"]["size_m"] = [1.2, 1.0, 1.4]
            manifest = root / "reference_map.json"
            manifest.write_text(json.dumps(entries))
            args = argparse.Namespace(reference_map=manifest, step_a_index=root / "unused.json")
            selected = pipeline.selected_references(args)
            self.assertEqual(set(selected), set(pipeline.CLASSES))
            self.assertEqual(selected["LSP"], (root / "LSP.png").resolve())
            self.assertEqual(pipeline.reference_size(args, "cargo"), [1.2, 1.0, 1.4])
            self.assertIsNone(pipeline.reference_size(args, "SKID"))

    def test_aesthetic_gate_is_decided_by_scores(self) -> None:
        self.assertTrue(pipeline.aesthetic_gate(SCORED)[0])
        low = dict(SCORED, scores=dict(SCORED["scores"], edges=2))
        passed, failures = pipeline.aesthetic_gate(low)
        self.assertFalse(passed)
        self.assertTrue(any("edges scored 2/5" in f for f in failures))
        self.assertFalse(pipeline.aesthetic_gate(dict(SCORED, accepted=False))[0])
        self.assertFalse(pipeline.aesthetic_gate({"accepted": True})[0])

    def test_raw_shape_check_rejects_a_reframed_output(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source, wide, square = root / "s.png", root / "w.png", root / "q.png"
            Image.new("RGB", (1920, 1080)).save(source)
            Image.new("RGB", (1536, 864)).save(wide)
            Image.new("RGB", (1024, 1024)).save(square)
            self.assertTrue(pipeline.image_shape_ok(source, wide))
            self.assertFalse(pipeline.image_shape_ok(source, square))

    def test_cargo_box_is_the_skid_footprint_centred_on_the_accepted_sheet(self) -> None:
        # a 1.9 x 1.85 m sheet seen as 60 x 45 px, rear edge on top; a 1.2 x 1.0 m SKID centred on it; 1 m high,
        # 10 px up in this view
        outline = [[20, 50], [80, 50], [80, 95], [20, 95]]
        box = pipeline.cargo_box_on_sheet(outline, [[20, 50], [80, 50]], [1.9, 1.85], [1.2, 1.0], 1.0,
                                          lambda p, h: (p[0], p[1] - 10 * h))
        self.assertEqual(box, [31, 50, 69, 85])

    def test_missing_api_key_fails_before_creating_output(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source = root / "valid.png"
            source.write_bytes(b"input")
            image_cli = root / "image_gen.py"
            image_cli.write_text("# test stub\n")
            args = args_for(root, source, root / "out.png", editor="api", image_cli=image_cli)
            with (mock.patch.object(pipeline, "parse_args", return_value=args),
                  mock.patch.dict(pipeline.os.environ, {"OPENAI_API_KEY": ""})):
                with self.assertRaisesRegex(RuntimeError, "OPENAI_API_KEY"):
                    pipeline.main()
            self.assertFalse((root / "out.png").exists())

    def run_pipeline(self, root: Path, aesthetic: dict | None, drift_passes: bool = True, sheet_geometry=None,
                     max_attempts: int = 1):
        """aesthetic None: run without the realism review (the default of the command line)."""
        source = root / "valid.png"
        Image.new("RGB", (120, 110), "gray").save(source)
        (root / "calibration.json").write_text(json.dumps({"lsp_measured_size_m": [1.9, 1.85]}))
        (root / "config.json").write_text(json.dumps({"lsp_thickness_m": 0.09}))
        (root / "camera.json").write_text(json.dumps({      # straight down from 5 m, 100 px focal length
            "width": 120, "height": 110, "K_norm": [[100 / 120, 0, .5], [0, 100 / 110, .5], [0, 0, 1]],
            "matrix_world": [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 5], [0, 0, 0, 1]]}))
        refs = {}
        for category in pipeline.CLASSES:
            ref = root / f"{category}.png"
            Image.new("RGB", (20, 20), "white").save(ref)
            refs[category] = ref
        output = root / "final.png"
        args = args_for(root, source, output, max_attempts=max_attempts, aesthetic_review=aesthetic is not None)
        review = (mock.patch.object(pipeline, "astra", return_value=aesthetic) if aesthetic is not None else
                  mock.patch.object(pipeline, "astra", side_effect=AssertionError("realism review is switched off")))
        calls = []
        # the sheet the image model actually drew: lower and further right than the projected polygon
        footprint = np.zeros((110, 120), bool)
        footprint[62:106, 34:96] = True

        class FakeSam3:
            def __init__(self, _cache):
                pass

            def get(self, _path, keys):
                return {key: [] for key in keys}

        class FakeMoge:
            def __init__(self, _cache):
                pass

            def get(self, _path):
                return None, None

        def fake_guide(_source, _edge, work, **_paths):
            guide = work / "geometry_guide.png"
            Image.new("RGB", (120, 110), "gray").save(guide)
            (work / "geometry.json").write_text(json.dumps({
                "shared_edge_px": [[20, 50], [80, 50]],
                "new_lsp_near_edge_px": [[20, 95], [80, 95]],
                "original_contact_edge_width_m": 1.9}))
            return guide

        def fake_edit(_args, images, _roles, prompt, raw, _work):
            calls.append((raw.name, images[0].name, prompt, images))
            Image.new("RGB", (120, 110), "gray").save(raw)

        passed = {"passed": True, "failures": []}
        drift = passed if drift_passes else {"passed": False, "failures": ["the image model changed the scene"]}
        with (mock.patch.object(pipeline, "ROOT", root),
              mock.patch.object(pipeline, "parse_args", return_value=args),
              mock.patch.object(pipeline, "step_a_references", return_value=refs),
              mock.patch.object(pipeline, "Sam3Masks", FakeSam3),
              mock.patch.object(pipeline, "make_geometry_guide", side_effect=fake_guide),
              mock.patch.object(pipeline, "guided_edit", side_effect=fake_edit),
              mock.patch.object(pipeline, "MogePoints", FakeMoge),
              mock.patch.object(pipeline, "verify_sheet_geometry", side_effect=sheet_geometry or [passed] * 9),
              mock.patch.object(pipeline, "added_sheet", return_value=(1, footprint)),
              mock.patch.object(pipeline, "verify_cargo_masks", return_value=passed) as cargo_check,
              mock.patch.object(pipeline, "image_shape_ok", return_value=True),
              mock.patch.object(pipeline, "raw_drift", return_value=drift),
              review,
              mock.patch.object(pipeline.shutil, "which", return_value="/usr/bin/codex")):
            code = pipeline.main()
        report = json.loads(next((root / "work/out").glob("final_vision_*/report.json")).read_text())
        self.cargo_check = cargo_check
        return code, output, calls, report

    def test_vision_pipeline_gates_sheet_then_adds_cargo_from_accepted_sheet(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            code, output, calls, report = self.run_pipeline(Path(folder), SCORED)
            self.assertEqual(code, 0)
            self.assertTrue(output.is_file())
            self.assertEqual([name for name, *_ in calls], ["sheet_raw_01.png", "cargo_raw_01.png"])
            self.assertEqual(calls[1][1], "sheet_01.png")
            self.assertIn("appearance only", calls[0][2])
            self.assertTrue((Path(folder) / "work/out").glob("final_vision_*/sheet_zoom_01.png"))
            self.assertTrue(report["stages"]["sheet"][0]["checks"]["composite"]["passed"])
            self.assertEqual(report["calls"], {"image edit - add empty LSP (Astra + imagegen)": 1,
                                               "visual review - LSP realism (Astra)": 1,
                                               "image edit - add cargo (Astra + imagegen)": 1,
                                               "visual review - cargo realism (Astra)": 1})
            with Image.open(report["summary_image"]) as summary:   # input | output side by side, stats below
                self.assertGreater(summary.width, 2 * 120)

    def test_failed_sheet_is_redone_from_the_source_with_its_measurements_and_the_failed_image(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            wrong = {"passed": False, "failures": ["the new LSP is rotated +12.0 deg relative to the original"],
                     "angle_deg": 12.0, "width_ratio": 1.0, "depth_ratio": 1.0, "gap_m": 0.0, "lateral_m": 0.0}
            right = {"passed": True, "failures": []}
            code, output, calls, report = self.run_pipeline(Path(folder), None, sheet_geometry=[wrong, right],
                                                            max_attempts=2)
            self.assertEqual(code, 0)
            self.assertEqual([name for name, *_ in calls], ["sheet_raw_01.png", "sheet_raw_02.png", "cargo_raw_01.png"])
            self.assertEqual(calls[1][1], "valid.png")                     # the edit starts from the source again
            self.assertIn("sheet_01.png", [image.name for image in calls[1][3]])   # and sees what was measured
            self.assertIn("rotated +12.0 deg", calls[1][2])
            self.assertEqual(report["calls"]["image edit - add empty LSP (Astra + imagegen)"], 2)

    def test_cargo_stage_works_on_the_accepted_sheet_not_on_the_projected_polygon(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            _, _, calls, report = self.run_pipeline(Path(folder), None)
            sheet = [[34, 62], [95, 62], [95, 105], [34, 105]]
            self.assertCountEqual(report["accepted_sheet_outline_px"], sheet)
            self.assertCountEqual(self.cargo_check.call_args.args[2], sheet)     # support is checked on the real sheet
            box = report["cargo_target"]["box_px"]
            self.assertEqual(report["cargo_target"]["method"], "skid_footprint_on_sheet")
            self.assertTrue(62 < box[3] <= 105 and 34 < (box[0] + box[2]) / 2 < 95)   # cargo stands on it
            self.assertIn(str(report["accepted_sheet_outline_px"]), calls[1][2])
            self.assertNotIn(str(report["target_polygon_px"]), calls[1][2])

    def test_geometry_alone_decides_without_the_realism_review(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            code, output, calls, report = self.run_pipeline(Path(folder), None)   # astra() would raise if called
            self.assertEqual(code, 0)
            self.assertTrue(output.is_file())
            self.assertEqual(report["calls"], {"image edit - add empty LSP (Astra + imagegen)": 1,
                                               "image edit - add cargo (Astra + imagegen)": 1})
            self.assertIsNone(report["stages"]["sheet"][0]["aesthetic"])
            self.assertTrue(Path(report["summary_image"]).is_file())

    def test_sheet_prompt_gives_the_shared_edge_but_no_target_polygon(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            _, _, calls, report = self.run_pipeline(Path(folder), SCORED)
            prompt = calls[0][2]
            self.assertIn("from [20, 50] to [80, 50]", prompt)              # the measured front edge
            self.assertNotIn(str(report["target_polygon_px"]), prompt)     # depth is the image model's to infer
            self.assertNotIn("[20, 95]", prompt)

    def test_summary_shows_the_candidate_with_the_fewest_failures(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            for name, colour in (("in", "gray"), ("a", "red"), ("b", "blue")):
                Image.new("RGB", (160, 90), colour).save(root / f"{name}.png")
            report = {"input": str(root / "in.png"), "accepted": False, "calls": {}, "stages": {"cargo": [], "sheet": [
                {"candidate": str(root / "a.png"), "failures": ["one"]},
                {"candidate": str(root / "b.png"), "failures": ["one", "two", "three"]}]}}
            with Image.open(pipeline.summary_sheet(report, root / "summary.png", panel_height=90)) as summary:
                red, _, blue = summary.convert("RGB").getpixel((160 + 20 + 80, 40 + 45))
            self.assertGreater(red, blue)

    def test_low_realism_score_stops_without_publishing(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            low = dict(SCORED, scores=dict(SCORED["scores"], lighting=2))
            code, output, calls, report = self.run_pipeline(Path(folder), low)
            self.assertEqual(code, 2)
            self.assertFalse(output.exists())
            self.assertEqual(report["stopped_at"], "sheet_geometry_or_realism")
            self.assertTrue(Path(report["summary_image"]).is_file())   # a failed run still gets its summary

    def test_raw_drift_fails_even_though_the_composite_restores_pixels(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            code, output, _, report = self.run_pipeline(Path(folder), SCORED, drift_passes=False)
            self.assertEqual(code, 2)
            self.assertFalse(output.exists())
            stage = report["stages"]["sheet"][0]
            self.assertTrue(stage["checks"]["composite"]["passed"])
            self.assertIn("the image model changed the scene", stage["failures"])


if __name__ == "__main__":
    unittest.main()
