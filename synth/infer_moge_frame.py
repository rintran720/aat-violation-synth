"""Run MoGe-2 on one image and save its point map for anchor diagnostics."""
import argparse
from pathlib import Path

import numpy as np
from PIL import Image

from synth.common import load_config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("image", help="Input CCTV frame")
    parser.add_argument("--output", help="Point-map .npy path; defaults to work/points_<stem>_moge.npy")
    parser.add_argument("--resolution-level", type=int, default=9)
    args = parser.parse_args()

    import torch
    from moge.model.v2 import MoGeModel

    cfg = load_config()
    image_path = Path(args.image)
    output_path = Path(args.output) if args.output else Path(cfg["work_dir"]) / f"points_{image_path.stem}_moge.npy"
    rgb = np.asarray(Image.open(image_path).convert("RGB"), dtype=np.float32) / 255.0
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = MoGeModel.from_pretrained(cfg["moge_model"]).to(device).eval()
    with torch.inference_mode():
        result = model.infer(
            torch.tensor(rgb, device=device).permute(2, 0, 1),
            resolution_level=args.resolution_level,
            use_fp16=device == "cuda",
        )
    points = result["points"].cpu().numpy()
    valid = result["mask"].cpu().numpy().astype(bool)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, points)
    mask_path = output_path.with_name(output_path.stem + "_valid.npy")
    np.save(mask_path, valid)
    print(f"MoGe device={device} resolution_level={args.resolution_level} shape={points.shape} "
          f"valid={int(valid.sum())} -> {output_path} ({mask_path.name})")


if __name__ == "__main__":
    main()
