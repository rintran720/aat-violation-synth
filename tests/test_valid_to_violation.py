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


class ValidToViolationTests(unittest.TestCase):
    def test_selected_reference_map_uses_all_four_local_images(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            entries = {}
            for category in pipeline.CLASSES:
                path = root / f"{category}.png"
                Image.new("RGB", (20, 20), "white").save(path)
                entries[category] = {"path": str(path)}
            manifest = root / "reference_map.json"
            manifest.write_text(json.dumps(entries))
            args = argparse.Namespace(reference_map=manifest, step_a_index=root / "unused.json")
            selected = pipeline.selected_references(args)
            self.assertEqual(set(selected), set(pipeline.CLASSES))
            self.assertEqual(selected["LSP"], (root / "LSP.png").resolve())

    def test_codex_retry_stays_within_five_image_limit(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            files = {}
            for name in ("source", "previous", *pipeline.CLASSES):
                path = root / f"{name}.png"
                Image.new("RGB", (64, 64), "white").save(path)
                files[name] = path
            candidate = root / "attempt_02.png"
            guide = root / "guide.png"
            Image.new("RGB", (64, 64), "white").save(guide)
            commands = []

            def fake_run(command, *, log):
                commands.append(command)
                Image.new("RGB", (64, 64), "white").save(candidate)

            with (mock.patch.object(pipeline, "ROOT", root),
                  mock.patch.object(pipeline, "run", side_effect=fake_run)):
                pipeline.image_edit_codex("codex", files["source"], files["previous"],
                    {key: files[key] for key in pipeline.CLASSES}, "Fix contact", candidate, root, guide)
            self.assertEqual(commands[0].count("-i"), 5)
            self.assertNotIn(str(files["SKID"]), commands[0])
            self.assertIn(str(guide), commands[0])

    def test_failed_contact_review_retries_and_only_publishes_pass(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source = root / "valid.png"
            Image.new("RGB", (1920, 1080), "gray").save(source)
            refs = {}
            for category in pipeline.CLASSES:
                path = root / f"{category}.png"
                Image.new("RGB", (64, 64), "white").save(path)
                refs[category] = path
            image_cli = root / "image_gen.py"
            image_cli.write_text("# test stub\n")
            output = root / "violation.png"
            args = argparse.Namespace(input=source, output=output, max_attempts=2, resume=None, editor="codex",
                                      joint=True,
                                      model="gpt-image-2.5-sunburst", step_a_index=root / "index.json",
                                      image_cli=image_cli, codex_bin="codex")
            plan = {"valid_source": True, "reason": "one loaded sheet", "source_loaded_lsp_count": 1,
                    "contact_edge_px": [[800, 470], [1000, 478]], "edit_prompt": "Keep the same camera."}
            passed = {key: True for key in pipeline.REVIEW_SCHEMA["properties"]["checks"]["required"]}
            failed = dict(passed, edge_contact=False)
            reviews = [
                {"accepted": True, "checks": failed, "feedback": "visible floor gap", "revised_prompt": "Close the gap."},
                {"accepted": True, "checks": passed, "feedback": "all checks pass", "revised_prompt": ""},
            ]
            prompts: list[str] = []

            def fake_astra(_codex, _images, _schema, _prompt, _work, name):
                return plan if name == "plan" else reviews[int(name.rsplit("_", 1)[1]) - 1]

            def fake_editor(_codex, _source, _previous, _refs, prompt, candidate, _work, _guide):
                prompts.append(prompt)
                Image.new("RGB", (1672, 941), "gray").save(candidate)

            with (mock.patch.object(pipeline, "ROOT", root),
                  mock.patch.object(pipeline, "parse_args", return_value=args),
                  mock.patch.object(pipeline, "step_a_references", return_value=refs),
                  mock.patch.object(pipeline, "make_geometry_guide", return_value=refs["LSP"]),
                  mock.patch.object(pipeline, "verify_pixels_and_geometry", return_value={"passed": True, "failures": []}),
                  mock.patch.object(pipeline, "astra", side_effect=fake_astra),
                  mock.patch.object(pipeline, "image_edit_codex", side_effect=fake_editor),
                  mock.patch.object(pipeline.shutil, "which", return_value="/usr/bin/codex")):
                self.assertEqual(pipeline.main(), 0)

            self.assertTrue(output.is_file())
            with Image.open(output) as result:
                self.assertEqual(result.size, (1920, 1080))
            self.assertEqual(len(prompts), 2)
            self.assertIn("visible floor gap", prompts[1])
            report = json.loads(next((root / "work/out").glob("violation_attempts_*/report.json")).read_text())
            self.assertEqual([item["accepted"] for item in report["attempts"]], [False, True])

    def test_missing_api_key_fails_before_creating_output(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source = root / "valid.png"
            source.write_bytes(b"input")
            image_cli = root / "image_gen.py"
            image_cli.write_text("# test stub\n")
            args = argparse.Namespace(input=source, output=root / "out.png", max_attempts=3, resume=None, editor="api",
                                      joint=True,
                                      model="gpt-image-2.5-sunburst", step_a_index=root / "index.json",
                                      image_cli=image_cli, codex_bin="codex")
            with (mock.patch.object(pipeline, "parse_args", return_value=args),
                  mock.patch.dict(pipeline.os.environ, {"OPENAI_API_KEY": ""})):
                with self.assertRaisesRegex(RuntimeError, "OPENAI_API_KEY"):
                    pipeline.main()
            self.assertFalse((root / "out.png").exists())

    def test_exhausted_review_saves_candidate_without_final(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source = root / "valid.png"
            Image.new("RGB", (1920, 1080), "gray").save(source)
            refs = {}
            for category in pipeline.CLASSES:
                ref = root / f"{category}.png"
                Image.new("RGB", (64, 64), "white").save(ref)
                refs[category] = ref
            output = root / "violation.png"
            args = argparse.Namespace(input=source, output=output, max_attempts=1, resume=None,
                                      joint=True,
                                      editor="codex", model="gpt-image-2.5-sunburst",
                                      step_a_index=root / "index.json", image_cli=root / "unused.py",
                                      codex_bin="codex")
            plan = {"valid_source": True, "reason": "one loaded sheet", "source_loaded_lsp_count": 1,
                    "contact_edge_px": [[800, 470], [1000, 478]], "edit_prompt": ""}
            checks = {key: True for key in pipeline.REVIEW_SCHEMA["properties"]["checks"]["required"]}
            checks["edge_contact"] = False
            review = {"accepted": False, "checks": checks, "feedback": "gap", "revised_prompt": "close gap"}

            def fake_astra(_codex, _images, _schema, _prompt, _work, name):
                return plan if name == "plan" else review

            def fake_editor(_codex, _source, _previous, _refs, _prompt, candidate, _work, _guide):
                Image.new("RGB", (1920, 1080), "gray").save(candidate)

            with (mock.patch.object(pipeline, "ROOT", root),
                  mock.patch.object(pipeline, "parse_args", return_value=args),
                  mock.patch.object(pipeline, "step_a_references", return_value=refs),
                  mock.patch.object(pipeline, "make_geometry_guide", return_value=refs["LSP"]),
                  mock.patch.object(pipeline, "verify_pixels_and_geometry", return_value={"passed": False, "failures": ["gap"]}),
                  mock.patch.object(pipeline, "astra", side_effect=fake_astra),
                  mock.patch.object(pipeline, "image_edit_codex", side_effect=fake_editor),
                  mock.patch.object(pipeline.shutil, "which", return_value="/usr/bin/codex")):
                self.assertEqual(pipeline.main(), 2)
            self.assertFalse(output.exists())
            candidate = root / "violation_needs_review.png"
            self.assertTrue(candidate.is_file())
            report = json.loads(next((root / "work/out").glob("violation_attempts_*/report.json")).read_text())
            self.assertEqual(report["review_candidate"], str(candidate))
            self.assertFalse(report["accepted"])

    def test_numeric_gate_rejects_gap_and_background_changes(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source = root / "source.png"
            candidate = root / "candidate.png"
            Image.new("RGB", (200, 150), "gray").save(source)
            changed = Image.new("RGB", (200, 150), "white")
            changed.save(candidate)
            guide = root / "geometry_guide.png"
            guide.write_bytes(b"unused")
            (root / "geometry.json").write_text(json.dumps({"shared_edge_px": [[80, 70], [100, 70]],
                                                             "new_lsp_near_edge_px": [[80, 100], [100, 100]]}))
            plan = {"contact_edge_px": [[80, 70], [100, 70]]}
            review = {"landmarks": {"original_front_edge": [[80, 70], [100, 70]],
                                    "new_rear_edge": [[80, 90], [100, 90]],
                                    "new_front_edge": [[80, 100], [100, 100]],
                                    "both_junction_ends_visible": True}}
            result = pipeline.verify_pixels_and_geometry(source, candidate, plan, guide, review)
            self.assertFalse(result["passed"])
            self.assertGreater(result["endpoint_error_px"], 10)
            self.assertTrue(any("background" in failure for failure in result["failures"]))

    def test_sequential_pipeline_stops_before_cargo_when_empty_sheet_fails(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source = root / "valid.png"
            Image.new("RGB", (1920, 1080), "gray").save(source)
            refs = {}
            for category in pipeline.CLASSES:
                ref = root / f"{category}.png"
                Image.new("RGB", (64, 64), "white").save(ref)
                refs[category] = ref
            args = argparse.Namespace(input=source, output=root / "final.png", max_attempts=1,
                                      resume=None, editor="codex", model="unused",
                                      step_a_index=root / "index.json", image_cli=root / "unused.py",
                                      codex_bin="codex")
            plan = {"valid_source": True, "reason": "one loaded LSP", "source_loaded_lsp_count": 1,
                    "contact_edge_px": [[800, 470], [1000, 478]], "edit_prompt": ""}
            failed = {"accepted": False, "checks": {}, "landmarks": {},
                      "feedback": "junction hidden", "revised_prompt": "expose both ends"}
            edited = []

            def fake_editor(_codex, _source, previous, _refs, _prompt, candidate, _work, _guide):
                edited.append((previous, candidate.name))
                Image.new("RGB", (1920, 1080), "gray").save(candidate)

            with (mock.patch.object(pipeline, "ROOT", root),
                  mock.patch.object(pipeline, "step_a_references", return_value=refs),
                  mock.patch.object(pipeline, "make_geometry_guide", return_value=refs["LSP"]),
                  mock.patch.object(pipeline, "astra", side_effect=lambda *a: plan if a[-1] == "plan" else failed),
                  mock.patch.object(pipeline, "verify_pixels_and_geometry", return_value={"passed": False, "failures": ["junction hidden"]}),
                  mock.patch.object(pipeline, "image_edit_codex", side_effect=fake_editor),
                  mock.patch.object(pipeline.shutil, "which", return_value="/usr/bin/codex")):
                self.assertEqual(pipeline.sequential_main(args), 2)
            self.assertEqual([name for _, name in edited], ["sheet_01.png"])
            self.assertFalse((root / "final.png").exists())

    def test_vision_pipeline_gates_sheet_then_adds_cargo_from_accepted_sheet(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source = root / "valid.png"
            Image.new("RGB", (120, 110), "gray").save(source)
            calibration = root / "calibration.json"
            calibration.write_text(json.dumps({"lsp_measured_size_m": [1.9, 1.85]}))
            refs = {}
            for category in pipeline.CLASSES:
                ref = root / f"{category}.png"
                Image.new("RGB", (20, 20), "white").save(ref)
                refs[category] = ref
            output = root / "final.png"
            args = argparse.Namespace(input=source, output=output, max_attempts=1,
                                      resume=None, editor="codex", model="unused",
                                      step_a_index=root / "index.json", image_cli=root / "unused.py",
                                      codex_bin="codex", contact_edge="20,50,80,50")
            calls = []

            class FakeSam3:
                def __init__(self, _cache):
                    pass

                def get(self, _path, keys):
                    return {key: [] for key in keys}

            def fake_guide(_source, _edge, work):
                guide = work / "geometry_guide.png"
                Image.new("RGB", (120, 110), "gray").save(guide)
                (work / "geometry.json").write_text(json.dumps({
                    "shared_edge_px": [[20, 50], [80, 50]],
                    "new_lsp_near_edge_px": [[20, 95], [80, 95]],
                    "original_contact_edge_width_m": 1.9}))
                return guide

            def fake_edit(_args, images, _roles, _prompt, raw, _work):
                calls.append((raw.name, images[0].name))
                Image.new("RGB", (120, 110), "gray").save(raw)

            def fake_preserve(_base, raw, candidate, _region, _protected):
                candidate.write_bytes(raw.read_bytes())

            passed = {"passed": True, "failures": []}
            aesthetic = {"accepted": True, "natural_edges": True,
                         "realistic_lighting": True, "feedback": "good"}
            with (mock.patch.object(pipeline, "ROOT", root),
                  mock.patch.object(pipeline, "CALIBRATION", calibration),
                  mock.patch.object(pipeline, "step_a_references", return_value=refs),
                  mock.patch.object(pipeline, "Sam3Masks", FakeSam3),
                  mock.patch.object(pipeline, "moge_floor_diagnostic", return_value={"floor_points": 1000}),
                  mock.patch.object(pipeline, "make_geometry_guide", side_effect=fake_guide),
                  mock.patch.object(pipeline, "guided_edit", side_effect=fake_edit),
                  mock.patch.object(pipeline, "preserve_outside", side_effect=fake_preserve),
                  mock.patch.object(pipeline, "verify_sheet_masks", return_value=passed),
                  mock.patch.object(pipeline, "verify_cargo_masks", return_value=passed),
                  mock.patch.object(pipeline, "image_shape_ok", return_value=True),
                  mock.patch.object(pipeline, "astra", return_value=aesthetic),
                  mock.patch.object(pipeline.shutil, "which", return_value="/usr/bin/codex")):
                self.assertEqual(pipeline.vision_main(args), 0)
            self.assertTrue(output.is_file())
            self.assertEqual([name for name, _ in calls], ["sheet_raw_01.png", "cargo_raw_01.png"])
            self.assertEqual(calls[1][1], "sheet_01.png")


if __name__ == "__main__":
    unittest.main()
