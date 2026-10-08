"""Behavioral checks for the single Astra edit: references, prompt, one output at the frame's size."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from PIL import Image

from synth import astra_violation as astra


class AstraViolationTests(unittest.TestCase):
    def run_main(self, root: Path, *extra: str, edge=None):
        source = root / "valid.png"
        Image.new("RGB", (192, 108), "gray").save(source)
        refs = {}
        for category in astra.REF_CLASSES:
            refs[category] = {"path": str(root / f"{category}.png")}
            Image.new("RGB", (16, 16), "white").save(root / f"{category}.png")
        manifest = root / "map.json"
        manifest.write_text(json.dumps(refs))
        commands = []

        def fake_run(command, **_kwargs):
            commands.append(command)
            raw = Path(command[-1].split("Save the generated result as a nonempty PNG at ")[1].split(". Do not")[0])
            Image.new("RGB", (167, 94), "blue").save(raw)          # imagegen picks its own size
            return mock.Mock(returncode=0)

        built = []

        def fake_build(image, edge_px, out, **kwargs):
            built.append((edge_px, kwargs))
            return manifest

        output = root / "out.png"
        with (mock.patch.object(astra, "ROOT", root),
              mock.patch.object(astra.shutil, "which", return_value="/usr/bin/codex"),
              mock.patch.object(astra.subprocess, "run", side_effect=fake_run),
              mock.patch.object(astra, "measured_edge", return_value=(edge, "measured" if edge else "no sheet")),
              mock.patch("synth.catalogue_reference_map.build_reference_map", side_effect=fake_build)):
            code = astra.main([str(source), "--output", str(output), *extra])
        return code, output, commands, built

    def test_one_output_at_the_frame_size_with_the_edge_in_the_prompt(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            code, output, commands, built = self.run_main(root, "--contact-edge", "40,60,120,62", "--seed", "3")
            self.assertEqual(code, 0)
            with Image.open(output) as image:
                self.assertEqual(image.size, (192, 108))
            self.assertEqual(len(commands), 1)                       # exactly one model call
            self.assertEqual(built[0][0], [[40, 60], [120, 62]])     # references matched at the given edge
            self.assertEqual(built[0][1]["seed"], 3)
            prompt = commands[0][-1]
            self.assertIn("from (40, 60) to (120, 62) in the 192x108 frame", prompt)
            self.assertIn("TWO LSPs", prompt)
            images = [commands[0][i + 1] for i, part in enumerate(commands[0]) if part == "-i"]
            self.assertEqual([Path(p).name for p in images], ["valid.png", "LSP.png", "SKID.png", "cargo.png"])
            run = json.loads((root / "out_run/run.json").read_text())
            self.assertEqual(run["raw_size"], [167, 94])

    def test_unmeasured_edge_matches_at_the_centre_and_names_no_position(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            code, _, commands, built = self.run_main(Path(folder))
            self.assertEqual(code, 0)
            self.assertFalse(built[0][1]["edge_given"])
            self.assertNotIn("front edge runs from", commands[0][-1])

    def test_custom_change_and_ready_map_skip_the_catalogue_build(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            code, _, commands, built = self.run_main(root, "--reference-map", str(root / "map.json"),
                                                     "--change", "add one red cone", edge=[[1, 2], [3, 4]])
            self.assertEqual(code, 0)
            self.assertEqual(built, [])
            self.assertIn("Make this ONE change: add one red cone", commands[0][-1])


if __name__ == "__main__":
    unittest.main()
