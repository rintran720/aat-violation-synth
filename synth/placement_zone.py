"""Stage B4: where inserted objects may stand, per camera.

Run: SYNTH_CONFIG=config/cameras/ch14.json python -m synth.placement_zone
zone = SAM3 floor on the clean plate, minus static objects (racks, parked LSP stacks and cargo found on the plate),
minus an image-border band (lens distortion makes the floor scale unreliable there; cam01 sheets near the bottom
measured ~15-20 % too large) and minus floor farther than MAX_RANGE_M (objects too small, scale least reliable).
Writes <work>/placement_zone.png (255 = allowed) and <work>/placement_zone_overlay.jpg for review.
"""
import json
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import binary_erosion, binary_opening

from synth.common import load_config, write_json
from synth.height_check import floor_depth

BORDER_FRACTION = .06   # of width/height, all sides
MAX_RANGE_M = 22.0      # camera-forward distance on the floor
ERODE_PX = 6            # keep clear of mask edges (walls, kerbs)


def main():
    cfg = load_config(); work = cfg["work"]
    stem = Path(cfg["background_image"]).stem
    masks = work / "masks" / stem
    floor = np.asarray(Image.open(masks / "floor.png").convert("L")) > 127
    H, W = floor.shape
    blocked = np.zeros_like(floor)
    for name in ["rack.png", "LSP.png", "cargo.png", "SKID.png", "forklift.png"]:
        if (masks / name).exists():
            blocked |= np.asarray(Image.open(masks / name).convert("L")) > 127
    border = np.zeros_like(floor)
    by, bx = int(H * BORDER_FRACTION), int(W * BORDER_FRACTION)
    border[:by] = border[-by:] = True; border[:, :bx] = border[:, -bx:] = True
    depth = floor_depth(json.loads((work / "camera.json").read_text()))
    zone = floor & ~blocked & ~border & np.isfinite(depth) & (depth < MAX_RANGE_M)
    zone = binary_opening(binary_erosion(zone, iterations=ERODE_PX), iterations=3)
    Image.fromarray((zone * 255).astype(np.uint8)).save(work / "placement_zone.png")
    bg = np.asarray(Image.open(cfg["background_image"]).convert("RGB")).astype(float)
    bg[zone] = bg[zone] * .55 + np.array([40, 200, 90]) * .45
    Image.fromarray(bg.astype(np.uint8)).save(work / "placement_zone_overlay.jpg", quality=88)
    info = {"camera_id": cfg["camera_id"], "allowed_fraction_of_image": round(float(zone.mean()), 3),
            "border_fraction": BORDER_FRACTION, "max_range_m": MAX_RANGE_M, "blocked_by": "SAM3 masks on the clean plate"}
    write_json(work / "placement_zone.json", info)
    print(json.dumps(info))


if __name__ == "__main__":
    main()
