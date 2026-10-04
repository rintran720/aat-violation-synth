"""Small synthetic checks for mask-based gates and pixel preservation."""

import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from synth.vision_geometry import (edit_region, polygon_mask, preserve_outside,
                                   verify_cargo_masks, verify_sheet_masks)


class VisionGeometryTests(unittest.TestCase):
    def test_sheet_then_supported_cargo_pass_masks(self):
        size = (120, 110)
        polygon = [[20, 50], [80, 50], [80, 95], [20, 95]]
        deck = polygon_mask(size, polygon)
        empty = np.zeros(deck.shape, bool)
        source = {"LSP": [], "cargo": []}
        sheet = {"LSP": [deck], "cargo": []}
        self.assertTrue(verify_sheet_masks(source, sheet, polygon, size)["passed"])
        cargo = empty.copy()
        cargo[20:90, 32:68] = True
        final = {"LSP": [deck & ~cargo], "cargo": [cargo]}
        self.assertTrue(verify_cargo_masks(sheet, final, polygon, size)["passed"])

    def test_rejects_oversized_sheet_and_unsupported_cargo(self):
        size = (120, 110)
        polygon = [[20, 50], [80, 50], [80, 95], [20, 95]]
        deck = polygon_mask(size, polygon)
        huge = np.zeros(deck.shape, bool)
        huge[45:105, 10:110] = True
        source = {"LSP": [], "cargo": []}
        sheet = {"LSP": [huge], "cargo": []}
        self.assertFalse(verify_sheet_masks(source, sheet, polygon, size)["passed"])
        cargo = np.zeros(deck.shape, bool)
        cargo[20:90, 85:115] = True
        final = {"LSP": [deck], "cargo": [cargo]}
        self.assertFalse(verify_cargo_masks({"LSP": [deck], "cargo": []}, final, polygon, size)["passed"])

    def test_unrelated_parked_lsp_mask_drift_does_not_fail_new_sheet(self):
        size = (120, 110)
        polygon = [[20, 50], [80, 50], [80, 95], [20, 95]]
        deck = polygon_mask(size, polygon)
        parked = np.zeros(deck.shape, bool)
        parked[2:30, 82:118] = True
        source = {"LSP": [], "cargo": []}
        candidate = {"LSP": [deck, parked], "cargo": []}
        measured = verify_sheet_masks(source, candidate, polygon, size)
        self.assertTrue(measured["passed"])
        self.assertEqual(measured["outside_fraction"], 0)

    def test_source_pixels_outside_edit_region_remain_exact(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            base, generated, output = (root / name for name in ("base.png", "generated.png", "out.png"))
            Image.new("RGB", (120, 110), (10, 20, 30)).save(base)
            Image.new("RGB", (120, 110), (250, 240, 230)).save(generated)
            polygon = [[20, 50], [80, 50], [80, 95], [20, 95]]
            mask = edit_region((120, 110), polygon, "sheet")
            protected = np.zeros((110, 120), bool)
            protected[65:70, 45:55] = True
            preserve_outside(base, generated, output, mask, [protected])
            result = np.asarray(Image.open(output))
            self.assertTrue((result[0, 0] == (10, 20, 30)).all())
            self.assertTrue((result[66, 50] == (10, 20, 30)).all())


if __name__ == "__main__":
    unittest.main()
