"""The reference map built from the 3D catalogue for a frame and a shared edge."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from PIL import Image

from synth import catalogue_reference_map as catalogue


def fake_catalogue(root: Path, skip_render: str | None = None) -> None:
    """A catalogue of one LSP, one SKID and one wrapped cargo with an 8x8 render per view (RGBA, like Blender's)."""
    cat = root / "work/catalogue3d"
    variants = {"lsp_A": {"object": "lsp", "label": "sheet A", "size_m": [1.9, 1.85, 0.09]},
                "skid_B": {"object": "skid", "label": "skid B", "size_m": [1.2, 1.0, 0.13]},
                "cargo_wrap_C_skid": {"object": "cargo_wrap", "label": "wrap C", "size_m": [1.2, 1.0, 1.4]}}
    cat.mkdir(parents=True)
    (cat / "variants.json").write_text(json.dumps(variants))
    for name, v in variants.items():
        for e in catalogue.ELEVATIONS:
            for a in range(0, 360, 45):
                render = cat / "renders" / v["object"] / name / f"e{e:02d}_a{a:03d}.png"
                if render.name == skip_render:
                    continue
                render.parent.mkdir(parents=True, exist_ok=True)
                Image.new("RGBA", (8, 8), (40, 40, 60, 255)).save(render)
    (root / "work/refs").mkdir(parents=True)
    Image.new("RGB", (20, 20), "yellow").save(root / "work/refs/forklift.png")
    (root / "work/refs/index.json").write_text(json.dumps([
        {"class": "forklift", "crop": "work/refs/forklift.png", "score": 0.9, "area_px": 400, "bbox": [0, 0, 20, 20]}]))
    (root / "work/camera.json").write_text(json.dumps({
        "width": 120, "height": 110, "K_norm": [[1, 0, 0.5], [0, 1, 0.5], [0, 0, 1]],
        "matrix_world": [[1, 0, 0, 0], [0, 0.5, -0.866, -4], [0, 0.866, 0.5, 4], [0, 0, 0, 1]]}))
    Image.new("RGB", (120, 110), "gray").save(root / "frame.png")


class CatalogueReferenceMapTests(unittest.TestCase):
    def test_build_reference_map_writes_four_references_matched_to_the_edge(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            fake_catalogue(root)
            with mock.patch.object(catalogue, "ROOT", root), mock.patch.object(catalogue, "CAT", root / "work/catalogue3d"):
                path = catalogue.build_reference_map(root / "frame.png", [[20, 50], [80, 50]], root / "work/out/run/reference_map",
                                                     camera=root / "work/camera.json", step_a_index=root / "work/refs/index.json",
                                                     cargo_kind="cargo_wrap", cargo_skid="yes")
            ref = json.loads(path.read_text())
            self.assertEqual(path.parent, root / "work/out/run/reference_map")
            for category in ("LSP", "SKID", "cargo"):
                self.assertTrue((path.parent / f"{category}.png").is_file())
                self.assertEqual(ref[category]["source"], "3D catalogue")
                self.assertEqual(len(ref[category]["views"]), 3)
            self.assertEqual(ref["cargo"]["variant"], "cargo_wrap_C_skid")
            self.assertEqual(ref["cargo"]["size_m"], [1.2, 1.0, 1.4])
            self.assertEqual(ref["forklift"]["source"], "Step A crop")
            self.assertEqual(ref["view"]["contact_edge_px"], [[20, 50], [80, 50]])
            self.assertTrue(ref["view"]["edge_given"])
            self.assertIn(ref["view"]["render_view"]["elevation_deg"], catalogue.ELEVATIONS)
            self.assertTrue((path.parent / "compare.jpg").is_file())

    def test_missing_catalogue_or_render_is_an_error_not_a_fallback(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            fake_catalogue(root, skip_render="e30_a000.png")
            with mock.patch.object(catalogue, "ROOT", root), mock.patch.object(catalogue, "CAT", root / "work/catalogue3d"):
                with self.assertRaisesRegex(RuntimeError, "render missing"):
                    catalogue.build_reference_map(root / "frame.png", [[20, 50], [80, 50]], root / "work/out/run/reference_map",
                                                  camera=root / "work/camera.json", step_a_index=root / "work/refs/index.json")
                with mock.patch.object(catalogue, "CAT", root / "nowhere"):
                    with self.assertRaisesRegex(RuntimeError, "catalogue is missing"):
                        catalogue.build_reference_map(root / "frame.png", [[20, 50], [80, 50]], root / "work/out/run/reference_map",
                                                      camera=root / "work/camera.json", step_a_index=root / "work/refs/index.json")


if __name__ == "__main__":
    unittest.main()
