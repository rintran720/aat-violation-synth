"""SAM3 text-prompted masks and the per-camera object crop library.

Run: python -m synth.segment                                # clean background + every object frame of the camera
     python -m synth.segment docs/requirements/lsp-reference.png LSP   # another image, selected keys (index untouched)
Writes work/masks/<stem>/<key>.png (union), <key>_<i>.png (instances), source.json,
       work/refs/<stem>_<key>_<rank>.png (top-3 object crops per class and frame),
       work/refs/index.json (class, frame, score, 2D box, area, bottom-centre pixel of every crop)
"""
import sys
from pathlib import Path

import numpy as np
from PIL import Image

from synth.common import load_config, reference_images, write_json

BACKGROUND_KEYS = ["floor", "rack"]  # static scene, segmented on the clean background
OBJECT_KEYS = ["forklift", "LSP", "SKID", "cargo"]  # key objects, segmented on every object frame
MEASURE_KEYS = ["LSP_top"]  # masks only (no crops): flat LSP tops, calibrate.py measures the LSP size on them


def object_record(image, mask):
    """Tight crop of one object (pixels outside the mask filled with the object's mean colour, so the crop
    can serve as a texture) plus its 2D size and position. image: HxWx3 uint8, mask: HxW bool."""
    ys, xs = np.nonzero(mask)
    x0, y0, x1, y1 = xs.min(), ys.min(), xs.max() + 1, ys.max() + 1
    crop = image[y0:y1, x0:x1].copy()
    crop[~mask[y0:y1, x0:x1]] = image[mask].mean(axis=0)
    info = {"bbox": [int(x0), int(y0), int(x1), int(y1)], "area_px": int(mask.sum()),
            "bottom_center_px": [int(round(xs.mean())), int(y1 - 1)]}  # floor contact point
    return crop, info


def load_processor():
    from sam3.model.sam3_image_processor import Sam3Processor
    from sam3.model_builder import build_sam3_image_model

    return Sam3Processor(build_sam3_image_model())


def run(cfg, processor, image_path, keys):
    work, stem = cfg["work"], Path(image_path).stem
    mdir = work / "masks" / stem
    (work / "refs").mkdir(parents=True, exist_ok=True)
    write_json(mdir / "source.json", {"image": str(image_path)})
    image = Image.open(image_path).convert("RGB")
    pixels = np.asarray(image)
    state = processor.set_image(image)
    hsv = np.asarray(image.convert("HSV"), dtype=np.float32) / 255.0
    hue_deg, sat, val = hsv[..., 0] * 360, hsv[..., 1], hsv[..., 2]
    records, unions = [], {}
    for key in keys:
        prompt = cfg["sam3_prompts"][key]
        processor.confidence_threshold = cfg.get("sam3_thresholds", {}).get(key, 0.5)
        masks, scores = [], []
        for p in [prompt] if isinstance(prompt, str) else prompt:  # several prompts: union of instances, duplicates dropped
            out = processor.set_text_prompt(state=state, prompt=p)
            for m, s in zip(out["masks"], out["scores"]):
                m = np.asarray(m.detach().cpu().float()).squeeze() > 0.5
                if all((m & k).sum() < 0.5 * (m | k).sum() for k in masks):
                    masks.append(m)
                    scores.append(float(s))
        filt = cfg.get("sam3_filters", {}).get(key)
        if filt:  # keep instances whose material colour matches and that are not mostly another class
            colour = ((hue_deg >= filt["hue_deg"][0]) & (hue_deg <= filt["hue_deg"][1]) & (sat >= filt["sat"][0])
                      & (sat <= filt["sat"][1]) & (val >= filt["val_min"]))
            keep = [i for i, m in enumerate(masks) if colour[m].mean() >= filt["min_colour_fraction"]
                    and all(unions[k][m].mean() < t for k, t in filt.get("max_overlap", {}).items() if k in unions)]
            masks, scores = [masks[i] for i in keep], [scores[i] for i in keep]
        union = np.zeros((image.height, image.width), bool)
        for i, m in enumerate(masks):
            union |= m
            Image.fromarray(m.astype(np.uint8) * 255).save(mdir / f"{key}_{i}.png")
        Image.fromarray(union.astype(np.uint8) * 255).save(mdir / f"{key}.png")
        unions[key] = union
        print(f"{stem}: {key!r} ({prompt!r}) {len(masks)} instances, scores {[round(s, 2) for s in scores]}")
        if key in BACKGROUND_KEYS or key in MEASURE_KEYS:
            continue
        for rank, idx in enumerate(np.argsort(scores)[::-1][:3]):
            if not masks[idx].any():
                continue
            crop, info = object_record(pixels, masks[idx])
            name = f"{stem}_{key}_{rank}.png"
            Image.fromarray(crop).save(work / "refs" / name)
            records.append({"class": key, "frame": str(image_path), "rank": rank, "score": round(scores[idx], 3),
                            "crop": f"work/refs/{name}", **info})
    return records


if __name__ == "__main__":
    import torch

    torch.autocast("cuda", dtype=torch.bfloat16).__enter__()  # SAM3 runs in bf16 autocast (as in its examples)
    cfg = load_config()
    processor = load_processor()
    if len(sys.argv) > 1:
        run(cfg, processor, sys.argv[1], sys.argv[2:] or OBJECT_KEYS)
    else:
        run(cfg, processor, cfg["background_image"], BACKGROUND_KEYS)
        frames = reference_images(cfg)
        records = [r for f in frames for r in run(cfg, processor, f, OBJECT_KEYS + MEASURE_KEYS)]
        write_json(cfg["work"] / "refs" / "index.json", records)
        print(f"{len(frames)} object frames, {len(records)} object crops -> {cfg['work'] / 'refs'}")
