"""Object reference sheets from the 3D catalogue: different models of one object, each seen from three angles.

Run: python -m synth.reference_sheet SKID|CARGO [--out references/<ID>/sheet.jpg] [--variant <name> ...]
The sheet has 2 models per row; each model is a block of three tiles (a low side view, a three-quarter view and a
high view), the blocks set apart by a wider gap so the models do not run together. Each tile is the render flattened
on floor grey and trimmed to the object, as synth.catalogue_reference_map does; no text is drawn (an image editor may
copy it). The service gives the sheet as the SKID / CARGO object reference (service.reference_store).
"""
import argparse
import json
from pathlib import Path

from PIL import Image

from synth.catalogue_reference_map import CAT, flatten

ROOT = Path(__file__).resolve().parents[1]
# several different models per object (user, 2026-10-09: six skids, not one; four cargo kinds)
PRESETS = {
    "SKID": ["skid_SK046-1", "skid_SK067-1", "skid_WR058-1",                      # closed deck
             "skid_SK055-1_open", "skid_WR030-1_open", "skid_WR068-1_open"],     # open deck
    # four full-size kinds (user, 2026-10-09: the first wrap and the carton stack taken out; small cartons left
    # out too, as prompts ask for full-size loads)
    "CARGO": ["cargo_wrap_WR096-1_noskid", "cargo_strapped_CA005-1_noskid", "cargo_wooden_CA106-1_noskid",
              "cargo_black_net_WR109-1_noskid"],
}
VIEWS = [(10, 0), (30, 45), (55, 90)]       # (elevation, azimuth) in degrees
TILE, GAP, BLOCK_GAP, PER_ROW = 320, 6, 28, 2
BACKGROUND = (60, 60, 60)


def build_sheet(variants: list[str], out: Path) -> Path:
    """The sheet of these catalogue models (work/catalogue3d/variants.json names), written to out as JPEG."""
    catalogue = json.loads((CAT / "variants.json").read_text())
    unknown = [v for v in variants if v not in catalogue]
    if unknown:
        raise SystemExit(f"not in the catalogue: {', '.join(unknown)}")
    renders = [[CAT / "renders" / catalogue[v]["object"] / v / f"e{e:02d}_a{a:03d}.png" for e, a in VIEWS]
               for v in variants]
    missing = [str(r) for model in renders for r in model if not r.is_file()]
    if missing:
        raise SystemExit(f"missing renders: {', '.join(missing[:3])}")
    block = len(VIEWS) * TILE + (len(VIEWS) - 1) * GAP
    rows = -(-len(variants) // PER_ROW)
    sheet = Image.new("RGB", (PER_ROW * block + (PER_ROW + 1) * BLOCK_GAP, rows * TILE + (rows + 1) * BLOCK_GAP),
                      BACKGROUND)
    for m, model in enumerate(renders):
        x0 = BLOCK_GAP + (m % PER_ROW) * (block + BLOCK_GAP)
        y0 = BLOCK_GAP + (m // PER_ROW) * (TILE + BLOCK_GAP)
        for k, render in enumerate(model):
            sheet.paste(flatten(render, None).resize((TILE, TILE), Image.LANCZOS), (x0 + k * (TILE + GAP), y0))
    out.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out, quality=92)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("object", choices=sorted(PRESETS))
    parser.add_argument("--variant", action="append", help="a catalogue model, repeated (default: the preset six)")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    out = args.out or ROOT / "references" / args.object / "sheet.jpg"
    print(build_sheet(args.variant or PRESETS[args.object], out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
