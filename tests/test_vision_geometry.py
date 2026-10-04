"""Small synthetic checks for mask-based gates and pixel preservation."""

import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from synth.vision_geometry import (edit_region, guide_marks, polygon_mask, preserve_outside, raw_drift,
                                   top_face_edge, verify_cargo_masks, verify_sheet_masks)


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

    def test_rejects_a_sheet_narrower_than_the_shared_edge(self):
        size = (120, 110)
        polygon = [[20, 50], [80, 50], [80, 95], [20, 95]]
        narrow = polygon_mask(size, [[35, 50], [80, 50], [80, 95], [35, 95]])
        measured = verify_sheet_masks({"LSP": [], "cargo": []}, {"LSP": [narrow], "cargo": []}, polygon, size)
        self.assertLess(measured["shared_edge_cover"], .8)
        self.assertTrue(any("shared edge" in f for f in measured["failures"]))

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

    def test_raw_drift_and_guide_marks_measure_the_model_output(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            base, same, shifted, marked = (root / f"{n}.png" for n in ("base", "same", "shifted", "marked"))
            Image.new("RGB", (120, 110), (90, 90, 90)).save(base)
            Image.new("RGB", (120, 110), (90, 90, 90)).save(same)
            Image.new("RGB", (120, 110), (160, 160, 160)).save(shifted)
            im = Image.new("RGB", (120, 110), (90, 90, 90))
            ImageDraw.Draw(im).line([(30, 60), (70, 60)], fill=(0, 255, 70), width=4)
            im.save(marked)
            region = edit_region((120, 110), [[20, 50], [80, 50], [80, 95], [20, 95]], "sheet")
            self.assertTrue(raw_drift(base, same, region, [])["passed"])
            self.assertFalse(raw_drift(base, shifted, region, [])["passed"])
            self.assertTrue(guide_marks(base, same, region)["passed"])
            self.assertFalse(guide_marks(base, marked, region)["passed"])

    def test_top_face_edge_keeps_the_upper_of_two_parallel_lines(self):
        # a slanted dark sheet: top face, then a 6 px side face, on a light floor
        image = Image.new("L", (300, 200), 200)
        draw = ImageDraw.Draw(image)
        top = [(60, 40), (240, 60), (230, 120), (50, 100)]
        draw.polygon(top, fill=60)
        draw.polygon([(50, 100), (230, 120), (230, 126), (50, 106)], fill=25)   # side face below the top edge
        mask = np.zeros((200, 300), bool)
        mask_im = Image.new("L", (300, 200))
        ImageDraw.Draw(mask_im).polygon([(60, 40), (240, 60), (230, 126), (50, 106)], fill=255)
        mask[np.asarray(mask_im) > 0] = True
        edge, info = top_face_edge(mask, np.asarray(image))
        (x0, y0), (x1, y1) = edge
        self.assertEqual(info["method"], "lsd_top_face_edge")
        # the top-face edge runs from (50, 100) to (230, 120); the floor contact is 6 px lower
        expected = lambda x: 100 + (x - 50) * 20 / 180
        self.assertLess(abs(y0 - expected(x0)), 3)
        self.assertLess(abs(y1 - expected(x1)), 3)


if __name__ == "__main__":
    unittest.main()
