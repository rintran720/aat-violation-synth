"""Stage B3 result: per-camera apparent-scale correction for rendered objects.

Run: python -m synth.scale_correction
The MoGe-based cameras read heights ~30-40 % low and floor lengths a little short (see scale_check/refine_camera).
Instead of trusting the camera's metric scale, rendered objects are scaled so they look the size real objects of
the same kind look in that camera:
  xy = forklift footprint scale (synth.scale_check),
  z  = median standing-person height / PERSON_HEIGHT_M when >= MIN_PEOPLE people were measured, else the mean of
       that and the forklift height scale (people are many and model-free; the forklift fit leans low).
Writes work/cameras/<ch>/scale_correction.json.
"""
import json
from pathlib import Path

import numpy as np

from synth.common import write_json
from synth.refine_camera import PERSON_HEIGHT_M, Cam, person_points, to_params

MIN_PEOPLE = 5


def people_ratio(cam_id, cam, crops):
    hs = []
    for r in crops:
        if r["camera"] == cam_id and r["class"] == "person" and not r["touches_border"] and r["area_px"] >= 1500:
            foot, head, aspect = person_points(r)
            if aspect >= 2.2:
                hs.append(cam.height(foot, head))
    hs = [h for h in hs if np.isfinite(h)]
    return (float(np.median(hs)) / PERSON_HEIGHT_M if hs else None), len(hs)


def main():
    crops = json.loads(Path("work/library/index.json").read_text())["crops"]
    for cam_dir in sorted(Path("work/cameras").iterdir()):
        if not (cam_dir / "scale_check.json").exists():
            continue
        cam_id = cam_dir.name; fork = json.loads((cam_dir / "scale_check.json").read_text())
        c = json.loads((cam_dir / "camera.json").read_text()); cam = Cam(*to_params(c), c["width"], c["height"])
        person, n = people_ratio(cam_id, cam, crops)
        if person is not None and n >= MIN_PEOPLE:
            z, z_source = person, f"people median ({n})"
        else:
            vals = [v for v in (person, fork["z_scale"]) if v is not None]
            z, z_source = float(np.mean(vals)), f"mean of people ({n}) and forklift"
        out = {"camera_id": cam_id, "xy": fork["xy_scale"], "z": round(z, 3),
               "xy_source": f"forklift silhouette ({len(fork['frames'])} frames)", "z_source": z_source,
               "people_ratio": round(person, 3) if person else None, "forklift_z": fork["z_scale"],
               "meaning": "multiply rendered object sizes by these so they match real objects seen by this camera"}
        write_json(cam_dir / "scale_correction.json", out)
        print(json.dumps(out))


if __name__ == "__main__":
    main()
