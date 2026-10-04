"""Full 3D catalogue sheets: one image per object kind, rows = variant x elevation, columns = azimuth.

Run: python -m synth.catalogue_sheet   (after blender/build_catalogue.py)
Writes work/catalogue3d/catalogue_<kind>.jpg (lsp, skid, cargo_wrap, cargo_wooden, ...).
"""
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

OUT = Path("work/catalogue3d")
TILE, LABEL, HEAD, BG = 140, 250, 70, (128, 130, 126)
AZ = list(range(0, 360, 45)); EL = [10, 30, 55]
TITLES = {"lsp": "LSP", "skid": "SKID", "cargo_wrap": "CARGO - wrapped load", "cargo_carton_stack": "CARGO - unwrapped carton stack",
          "cargo_black_net": "CARGO - black wrap / netted", "cargo_strapped": "CARGO - large strapped crate",
          "cargo_wooden": "CARGO - brown / wooden crate", "cargo_small_carton": "CARGO - small carton"}


def font(size, bold=False):
    try:
        return ImageFont.truetype("arialbd.ttf" if bold else "arial.ttf", size)
    except OSError:
        return ImageFont.load_default()


def main():
    variants = json.loads((OUT / "variants.json").read_text())
    for obj, title in TITLES.items():
        names = [n for n, v in variants.items() if v["object"] == obj]
        if not names:
            continue
        W = LABEL + len(AZ) * TILE; H = HEAD + len(names) * len(EL) * TILE + 30
        img = Image.new("RGB", (W, H), BG); d = ImageDraw.Draw(img)
        d.text((10, 8), f"{title} - {len(names)} variants x 24 views (one model per labelled real object)",
               font=font(22, True), fill=(255, 255, 255))
        for j, az in enumerate(AZ):
            d.text((LABEL + j * TILE + 8, HEAD - 22), f"azimuth {az}°", font=font(14), fill=(255, 255, 255))
        y = HEAD
        for name in names:
            v = variants[name]
            d.line([(0, y), (W, y)], fill=(90, 90, 90), width=2)
            size = " x ".join(f"{s:.2f}" for s in v["size_m"])
            d.multiline_text((8, y + 8), f"{v['label']}\n{size} m\n{name}", font=font(14), fill=(255, 255, 255), spacing=4)
            for i, el in enumerate(EL):
                d.text((8, y + i * TILE + TILE - 22), f"elevation {el}°", font=font(13), fill=(230, 230, 180))
                for j, az in enumerate(AZ):
                    im = Image.open(OUT / "renders" / obj / name / f"e{el:02d}_a{az:03d}.png").convert("RGBA")
                    im = im.resize((TILE, TILE)); bg = Image.new("RGBA", im.size, BG + (255,)); bg.alpha_composite(im)
                    img.paste(bg.convert("RGB"), (LABEL + j * TILE, y + i * TILE))
            y += len(EL) * TILE
        d.text((10, H - 24), "Elevation = camera angle above the floor; azimuth = camera direction around the object.",
               font=font(13), fill=(230, 230, 230))
        img.save(OUT / f"catalogue_{obj}.jpg", quality=88)
        print(obj, len(names), "variants", img.size)


if __name__ == "__main__":
    main()
