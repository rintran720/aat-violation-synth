"""Behavioral checks for the review/retry publication gate."""

from __future__ import annotations

import argparse
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from PIL import Image

from synth import valid_to_violation as pipeline


SCORED = {"scores": {key: 4 for key in pipeline.SCORES}, "issues": [], "accepted": True, "feedback": "good"}


def args_for(root: Path, source: Path, output: Path, **extra) -> argparse.Namespace:
    values = dict(input=source, output=output, max_attempts=1, editor="codex", model="unused",
                  step_a_index=root / "index.json", image_cli=root / "unused.py", codex_bin="codex",
                  contact_edge="20,50,80,50", reference_map=None, reuse_sheet=None, reuse_cargo=None,
                  camera=root / "camera.json", calibration=root / "calibration.json", moge=False)
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
            self.assertEqual(pipeline.reference_cargo_size(args), [1.2, 1.0, 1.4])

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

    def test_cargo_box_projects_a_3d_box_on_the_new_sheet(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            camera = Path(folder) / "camera.json"
            # camera 5 m above the floor looking straight down, 1000 px focal length
            camera.write_text(json.dumps({"width": 1000, "height": 1000, "K_norm": [[1, 0, .5], [0, 1, .5], [0, 0, 1]],
                                          "matrix_world": [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 5], [0, 0, 0, 1]]}))
            geometry = {"new_lsp_top_world_m": [[-1, 0, 0], [1, 0, 0], [1, 2, 0], [-1, 2, 0]]}
            box = pipeline.cargo_target_box(geometry, camera, [1.0, 1.0, 1.0])
            # top of the 1 m box is 4 m from the camera: half width 0.5 m -> 125 px; centre at y = 1 m -> 250 px up
            self.assertEqual(box, [375, 125, 625, 400])
            self.assertIsNone(pipeline.cargo_target_box({}, camera, [1, 1, 1]))

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

    def run_pipeline(self, root: Path, aesthetic: dict, drift_passes: bool = True):
        source = root / "valid.png"
        Image.new("RGB", (120, 110), "gray").save(source)
        (root / "calibration.json").write_text(json.dumps({"lsp_measured_size_m": [1.9, 1.85]}))
        refs = {}
        for category in pipeline.CLASSES:
            ref = root / f"{category}.png"
            Image.new("RGB", (20, 20), "white").save(ref)
            refs[category] = ref
        output = root / "final.png"
        args = args_for(root, source, output)
        calls = []

        class FakeSam3:
            def __init__(self, _cache):
                pass

            def get(self, _path, keys):
                return {key: [] for key in keys}

        def fake_guide(_source, _edge, work, **_paths):
            guide = work / "geometry_guide.png"
            Image.new("RGB", (120, 110), "gray").save(guide)
            (work / "geometry.json").write_text(json.dumps({
                "shared_edge_px": [[20, 50], [80, 50]],
                "new_lsp_near_edge_px": [[20, 95], [80, 95]],
                "original_contact_edge_width_m": 1.9}))
            return guide

        def fake_edit(_args, images, _roles, prompt, raw, _work):
            calls.append((raw.name, images[0].name, prompt))
            Image.new("RGB", (120, 110), "gray").save(raw)

        passed = {"passed": True, "failures": []}
        drift = passed if drift_passes else {"passed": False, "failures": ["the image model changed the scene"]}
        with (mock.patch.object(pipeline, "ROOT", root),
              mock.patch.object(pipeline, "parse_args", return_value=args),
              mock.patch.object(pipeline, "step_a_references", return_value=refs),
              mock.patch.object(pipeline, "Sam3Masks", FakeSam3),
              mock.patch.object(pipeline, "make_geometry_guide", side_effect=fake_guide),
              mock.patch.object(pipeline, "guided_edit", side_effect=fake_edit),
              mock.patch.object(pipeline, "verify_sheet_masks", return_value=passed),
              mock.patch.object(pipeline, "verify_cargo_masks", return_value=passed),
              mock.patch.object(pipeline, "image_shape_ok", return_value=True),
              mock.patch.object(pipeline, "raw_drift", return_value=drift),
              mock.patch.object(pipeline, "astra", return_value=aesthetic),
              mock.patch.object(pipeline.shutil, "which", return_value="/usr/bin/codex")):
            code = pipeline.main()
        report = json.loads(next((root / "work/out").glob("final_vision_*/report.json")).read_text())
        return code, output, calls, report

    def test_vision_pipeline_gates_sheet_then_adds_cargo_from_accepted_sheet(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            code, output, calls, report = self.run_pipeline(Path(folder), SCORED)
            self.assertEqual(code, 0)
            self.assertTrue(output.is_file())
            self.assertEqual([name for name, _, _ in calls], ["sheet_raw_01.png", "cargo_raw_01.png"])
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
