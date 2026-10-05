"""Small synthetic checks for mask-based gates and pixel preservation."""

import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from synth.vision_geometry import (edit_region, guide_marks, polygon_mask, preserve_outside, raw_drift,
                                   top_face_edge, verify_cargo_masks, verify_sheet_geometry)


SCALE = .02                     # metres per pixel on the synthetic floor
DEPTH_TO_WIDTH = 1.85 / 1.9     # calibrated LSP depth over the width of the shared edge
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
    return verify_sheet_geometry(source, candidate, region, points, valid, floor, EDGE, TOWARD, DEPTH_TO_WIDTH)


# ---- an exact point map of boxes on a floor seen by an oblique camera, like the CCTV views (22 deg at the sheets)
VIEW = (960, 540)
FOCAL, PITCH, CAMERA_HEIGHT = 900., np.radians(25), 4.
RIGHT = np.array([1., 0, 0])
FORWARD = np.array([0, np.cos(PITCH), -np.sin(PITCH)])
DOWN = np.array([0, -np.sin(PITCH), -np.cos(PITCH)])
EYE = np.array([0, 0, CAMERA_HEIGHT])
OLD_BOX = ((-1.15, 10., 0.), (1.15, 11.85, .09))          # the loaded sheet, front edge at y = 10 m
CARGO_BOX = ((-1.1, 10., .09), (1.1, 11.8, 1.6))          # its load, flush with the front edge


def project(point):
    local = np.array([RIGHT, DOWN, FORWARD]) @ (np.asarray(point, dtype=float) - EYE)
    return [int(round(VIEW[0] / 2 + FOCAL * local[0] / local[2])), int(round(VIEW[1] / 2 + FOCAL * local[1] / local[2]))]


def raycast(boxes):
    """Camera-frame points (x right, y down, z forward, as MoGe gives) and the index of the box each pixel sees
    (-1 for the floor)."""
    ys, xs = np.mgrid[0:VIEW[1], 0:VIEW[0]]
    rays = np.dstack([(xs - VIEW[0] / 2) / FOCAL, (ys - VIEW[1] / 2) / FOCAL, np.ones(xs.shape)])
    world = rays @ np.array([RIGHT, DOWN, FORWARD])
    with np.errstate(divide="ignore", invalid="ignore"):
        best = np.where(world[..., 2] < 0, -EYE[2] / world[..., 2], np.inf)      # the floor
        label = np.full(xs.shape, -1)
        for index, (low, high) in enumerate(boxes):
            t0, t1 = (np.asarray(low) - EYE) / world, (np.asarray(high) - EYE) / world
            near, far = np.minimum(t0, t1).max(axis=2), np.maximum(t0, t1).min(axis=2)
            hit = (near < far) & (near > 0) & (near < best)
            best, label = np.where(hit, near, best), np.where(hit, index, label)
    return rays * best[..., None], np.isfinite(best), label


def oblique(new_box):
    """verify_sheet_geometry on the scene with a new sheet box in front of the loaded one."""
    _, _, before = raycast([OLD_BOX, CARGO_BOX])
    points, valid, after = raycast([OLD_BOX, CARGO_BOX, new_box])
    source = {"LSP": [before == 0], "cargo": [before == 1]}
    candidate = {"LSP": [after == 0, after == 2], "cargo": [after == 1]}
    edge = [project((-1.15, 10, .09)), project((1.15, 10, .09))]
    return verify_sheet_geometry(source, candidate, Image.new("L", VIEW, 255), points, valid,
                                 (before == -1) & (after == -1) & valid, edge, project((0, 9, 0)), 1.85 / 2.3)


class ObliqueViewTests(unittest.TestCase):
    def test_equal_adjoining_sheet_passes_at_a_grazing_view(self):
        measured = oblique(((-1.15, 8.15, 0), (1.15, 10., .09)))
        self.assertTrue(measured["passed"], measured)
        self.assertLess(abs(measured["gap_m"]), .05)
        self.assertLess(abs(measured["lateral_m"]), .05)
        self.assertAlmostEqual(measured["width_ratio"], 1, delta=.04)
        self.assertAlmostEqual(measured["depth_ratio"], 1, delta=.04)
        self.assertAlmostEqual(measured["old_width_m"], 2.3, delta=.05)

    def test_gap_short_and_shifted_sheets_are_measured_in_metres(self):
        gap = oblique(((-1.15, 7.85, 0), (1.15, 9.7, .09)))
        self.assertAlmostEqual(gap["gap_m"], .3, delta=.05)
        self.assertAlmostEqual(gap["depth_ratio"], 1, delta=.05)
        short = oblique(((-1.15, 8.7, 0), (1.15, 10., .09)))
        self.assertAlmostEqual(short["depth_ratio"], .7, delta=.05)
        self.assertLess(abs(short["gap_m"]), .05)
        shifted = oblique(((-.85, 8.15, 0), (1.45, 10., .09)))
        self.assertAlmostEqual(shifted["lateral_m"], .3, delta=.05)
        self.assertAlmostEqual(shifted["width_ratio"], 1, delta=.05)
        for measured in (gap, short, shifted):
            self.assertFalse(measured["passed"])

    def test_depth_may_differ_by_a_fifth_width_by_a_tenth(self):
        # depth along the view is the least certain measure (camera and MoGe differ by ~20% on s_003)
        self.assertTrue(oblique(((-1.15, 8.43, 0), (1.15, 10., .09)))["passed"])        # 85% deep
        self.assertTrue(oblique(((-1.15, 7.87, 0), (1.15, 10., .09)))["passed"])        # 115% deep
        self.assertFalse(oblique(((-1.15, 7.6, 0), (1.15, 10., .09)))["passed"])        # 130% deep
        self.assertFalse(oblique(((-.98, 8.15, 0), (.98, 10., .09)))["passed"])         # 85% wide


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

    def test_original_mask_below_its_top_edge_does_not_shorten_the_new_sheet(self):
        # SAM3's mask of the original sheet also covers its side face and shadow, below the top-face edge; an
        # adjoining new sheet hides those pixels and must still measure full depth with no gap
        measured = measure(rect(20, 10, 114, 112), rect(20, 102, 114, 193))
        self.assertTrue(measured["passed"], measured["failures"])
        self.assertLess(abs(measured["gap_m"]), .05)
        self.assertAlmostEqual(measured["depth_ratio"], 1, delta=.05)

    def test_sheet_merged_with_the_original_by_sam3_is_cut_at_the_shared_edge(self):
        old = rect(20, 10, 114, 101)
        points, valid, floor, source, _ = scene(old, rect(20, 102, 114, 193))
        merged = {"LSP": [rect(20, 10, 114, 193)], "cargo": []}
        measured = verify_sheet_geometry(source, merged, Image.new("L", SIZE, 255), points, valid, floor, EDGE,
                                         TOWARD, DEPTH_TO_WIDTH)
        self.assertTrue(measured["passed"], measured["failures"])
        self.assertAlmostEqual(measured["depth_ratio"], 1, delta=.05)

    def test_calibrated_depth_follows_the_given_depth_to_width_ratio(self):
        # the original is hidden behind its edge; a wide sheet type keeps the calibrated depth, not a scaled one
        points, valid, floor, source, candidate = scene(rect(20, 90, 114, 101), rect(20, 102, 114, 177))
        measured = verify_sheet_geometry(source, candidate, Image.new("L", SIZE, 255), points, valid, floor, EDGE,
                                         TOWARD, 1.5 / 1.88)
        self.assertEqual(measured["old_depth_source"], "calibration")
        self.assertAlmostEqual(measured["old_depth_m"], 1.5, delta=.03)
        self.assertTrue(measured["passed"], measured["failures"])

    def test_edge_end_pixels_off_the_sheet_do_not_move_the_shared_edge(self):
        # at a grazing view the pixel at an end of the edge shows the floor beside the sheet, 0.4 m further away
        # (s_003: an exactly placed sheet measured a 0.23 m gap); the edge is located from pixels on the sheet
        old, new = rect(20, 10, 114, 101), rect(20, 102, 114, 193)
        points, valid, floor, source, candidate = scene(old, new)
        for x, y in EDGE:
            points[y - 4:y + 5, x - 4:x + 5, 1] -= .4
        measured = verify_sheet_geometry(source, candidate, Image.new("L", SIZE, 255), points, valid, floor, EDGE,
                                         TOWARD, DEPTH_TO_WIDTH)
        self.assertLess(abs(measured["gap_m"]), .05)
        self.assertTrue(measured["passed"], measured["failures"])

    def test_no_new_sheet_fails_without_measurements(self):
        points, valid, floor, source, _ = scene(rect(20, 10, 114, 101), np.zeros((240, 140), bool))
        measured = verify_sheet_geometry(source, source, Image.new("L", SIZE, 255), points, valid, floor, EDGE,
                                         TOWARD, DEPTH_TO_WIDTH)
        self.assertFalse(measured["passed"])
        self.assertNotIn("depth_ratio", measured)

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
