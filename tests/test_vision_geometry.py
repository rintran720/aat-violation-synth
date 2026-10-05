"""Small synthetic checks for mask-based gates and pixel preservation."""

import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from synth.vision_geometry import (edit_region, guide_marks, polygon_mask, preserve_outside, raw_drift,
                                   top_face_edge, verify_cargo_masks, verify_sheet_geometry)


SCALE = .02                     # metres per pixel on the synthetic floor
CALIBRATED = [1.9, 1.85]        # LSP width along the shared edge, depth across it
SIZE = (140, 240)               # image width, height
EDGE = [[20, 101], [114, 101]]  # front edge of the original sheet's top face
TOWARD = [67, 190]              # a pixel on the side where the new sheet belongs


def rect(x0, y0, x1, y1):
    return polygon_mask(SIZE, [[x0, y0], [x1, y0], [x1, y1], [x0, y1]])


def scene(old, new):
    """Camera 3 m above a flat floor, looking straight down; sheet pixels are 9 cm higher. Returns the point map,
    its validity, the free floor, and SAM3-like source/candidate masks."""
    height, width = SIZE[1], SIZE[0]
    rows, cols = np.mgrid[0:height, 0:width]
    depth = np.full((height, width), 3.)
    depth[old | new] -= .09
    points = np.dstack([(cols - width / 2) * SCALE, (rows - height / 2) * SCALE, depth])
    valid = np.ones((height, width), bool)
    floor = ~(old | new)
    source = {"LSP": [old], "cargo": []}
    candidate = {"LSP": [old, new], "cargo": []}
    return points, valid, floor, source, candidate


def measure(old, new):
    points, valid, floor, source, candidate = scene(old, new)
    region = Image.new("L", SIZE, 255)
    return verify_sheet_geometry(source, candidate, region, points, valid, floor, EDGE, TOWARD, CALIBRATED)


class VisionGeometryTests(unittest.TestCase):
    def test_sheet_then_supported_cargo_pass_masks(self):
        size = (120, 110)
        polygon = [[20, 50], [80, 50], [80, 95], [20, 95]]
        deck = polygon_mask(size, polygon)
        empty = np.zeros(deck.shape, bool)
        source = {"LSP": [], "cargo": []}
        sheet = {"LSP": [deck], "cargo": []}
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
        cargo = np.zeros(deck.shape, bool)
        cargo[20:90, 85:115] = True
        final = {"LSP": [deck], "cargo": [cargo]}
        self.assertFalse(verify_cargo_masks({"LSP": [deck], "cargo": []}, final, polygon, size)["passed"])

    def test_equal_sheet_touching_the_front_edge_passes(self):
        measured = measure(rect(20, 10, 114, 101), rect(20, 102, 114, 193))
        self.assertTrue(measured["passed"], measured["failures"])
        self.assertLess(abs(measured["angle_deg"]), 1)
        self.assertAlmostEqual(measured["width_ratio"], 1, delta=.05)
        self.assertAlmostEqual(measured["depth_ratio"], 1, delta=.05)
        self.assertLess(abs(measured["gap_m"]), .05)
        self.assertLess(abs(measured["lateral_m"]), .05)
        self.assertEqual(measured["old_depth_source"], "measured")
        self.assertEqual(len(measured["new_rect_px"]), 4)

    def test_narrow_shifted_sheet_reports_width_and_lateral_offset(self):
        measured = measure(rect(20, 10, 114, 101), rect(40, 102, 114, 193))
        self.assertFalse(measured["passed"])
        self.assertAlmostEqual(measured["width_ratio"], .8, delta=.05)
        self.assertAlmostEqual(measured["lateral_m"], .2, delta=.03)
        self.assertTrue(any("width" in f for f in measured["failures"]))
        self.assertTrue(any("toward the right end" in f for f in measured["failures"]))

    def test_gap_between_the_sheets_fails(self):
        measured = measure(rect(20, 10, 114, 101), rect(20, 115, 114, 206))
        self.assertAlmostEqual(measured["gap_m"], .28, delta=.03)
        self.assertTrue(any("gap" in f for f in measured["failures"]))

    def test_rotated_sheet_fails_on_orientation(self):
        c, s = np.cos(np.radians(12)), np.sin(np.radians(12))
        corners = [[-47, -46], [47, -46], [47, 46], [-47, 46]]
        rotated = polygon_mask(SIZE, [[67 + x * c - y * s, 150 + x * s + y * c] for x, y in corners])
        measured = measure(rect(20, 10, 114, 101), rotated)
        self.assertAlmostEqual(abs(measured["angle_deg"]), 12, delta=2)
        self.assertTrue(any("rotated" in f for f in measured["failures"]))

    def test_occluded_original_depth_falls_back_to_calibration(self):
        measured = measure(rect(20, 90, 114, 101), rect(20, 102, 114, 193))
        self.assertEqual(measured["old_depth_source"], "calibration")
        self.assertTrue(measured["passed"], measured["failures"])

    def test_no_new_sheet_has_nothing_to_correct(self):
        points, valid, floor, source, _ = scene(rect(20, 10, 114, 101), np.zeros((240, 140), bool))
        measured = verify_sheet_geometry(source, source, Image.new("L", SIZE, 255), points, valid, floor, EDGE,
                                         TOWARD, CALIBRATED)
        self.assertFalse(measured["passed"])
        self.assertIsNone(measured["new_rect_px"])

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
