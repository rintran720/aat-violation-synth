"""Candidate images for hand-labelling texture faces, plus a local labelling page.

Run: python -m synth.face_candidates [--per-variant 8]
Picks, per reviewed object variant (work/library_v2/review), the largest sharp crops that do not touch the frame
border, merges repeats of one object, and cuts each from the ORIGINAL video frame (box + 30 % margin, no SAM3
mask) into work/face_labeling/images/<id>.png. work/face_labeling/index.html labels each real object as a box: object number, kind, and the outline of its
top / front / right faces (4 or more points). "Save" downloads faces.json (one entry per face, grouped by object).
"""
import argparse
import importlib.util
import json
import math
import subprocess
from pathlib import Path

from PIL import Image

REVIEW = Path("work/library_v2/review")
OUT = Path("work/face_labeling")
VIDEOS = Path("data/multicam/video")
# (object, class in ids.json, labels attribute, variant code) -> group shown on the page
GROUPS = {
    "lsp_single": ("lsp", "LSP", "S"), "lsp_stack": ("lsp", "LSP", "K"),
    "skid_single": ("skid", "SKID", "S"), "skid_stack": ("skid", "SKID", "K"), "skid_loaded": ("skid", "SKID", "L"),
    # wrapped loads with and without a visible pallet are one kind (user, 2026-10-03)
    "cargo_wrap": ("wrapped_cargo", "CARGO_WR", "PW"),
    "cargo_carton_stack": ("wrapped_cargo", "CARGO_WR", "T"),
    "cargo_black_net": ("wrapped_cargo", "CARGO_WR", "N"), "cargo_strapped": ("carton", "CARGO_CA", "B"),
    "cargo_wooden": ("carton", "CARGO_CA", "V"), "cargo_small_carton": ("carton", "CARGO_CA", "C"),
}
# Vietnamese help shown on the labelling page (the user labels in Vietnamese, 2026-10-03)
GROUP_VI = {
    "lsp_single": "LSP tấm đơn: một tấm nhựa xám đen phẳng nằm trên sàn, không có gì đè lên.",
    "lsp_stack": "Chồng LSP: chỉ dùng để lấy mặt của MỘT tấm (nóc tấm trên cùng, hoặc mép đứng của một tấm). Không vẽ cả chồng.",
    "skid_single": "SKID đơn: tấm gỗ kê hàng, không có hàng trên.",
    "skid_stack": "Chồng SKID: chỉ dùng để lấy mặt của MỘT chiếc SKID (nóc chiếc trên cùng, hoặc hông của một chiếc). Không vẽ cả chồng.",
    "skid_loaded": "SKID đang đỡ hàng: SKID có kiện hàng đặt lên trên.",
    "cargo_wrap": "Kiện quấn màng co (trong suốt/xám), có hoặc không có SKID bên dưới.",
    "cargo_carton_stack": "Chồng thùng carton không quấn màng, lộ rõ từng thùng và nhãn.",
    "cargo_black_net": "Kiện quấn màng đen và/hoặc phủ lưới.",
    "cargo_strapped": "Kiện lớn, cao, đóng đai/khung, có nhãn dán.",
    "cargo_wooden": "Thùng/kiện màu nâu vàng (gỗ hoặc carton dày), có đai.",
    "cargo_small_carton": "Thùng carton nhỏ để rời (không trên SKID, không quấn màng).",
}
OLD_NAMES = {("skid_single", "S"): "pallet_single", ("skid_stack", "K"): "pallet_stack", ("skid_loaded", "L"): "pallet_loaded",
             ("cargo_wrap", "P"): "cargo_tall_wrap", ("cargo_wrap", "W"): "cargo_plain_wrap"}
# object kinds a labelled box can be (Vietnamese help for the labelling page)
KINDS = {
    # stacks are built by stacking single sheets / skids, so only single objects are labelled (user, 2026-10-03)
    "lsp_single": "LSP (một tấm)", "skid_single": "SKID (một chiếc)",
    "cargo_wrap": "kiện quấn màng", "cargo_carton_stack": "chồng thùng lộ nhãn", "cargo_black_net": "kiện quấn màng đen / lưới",
    "cargo_strapped": "kiện lớn đóng đai", "cargo_wooden": "thùng gỗ / nâu vàng", "cargo_small_carton": "thùng carton nhỏ",
}
# texture names of the first labelling round -> (kind, face)
LEGACY = {
    "lsp_top": ("lsp_single", "top"), "lsp_side": ("lsp_single", "front"), "lsp_stack_side": None,
    "skid_top": ("skid_single", "top"), "skid_side": ("skid_single", "front"), "skid_stack_side": None,
    "wrap_tall_side": ("cargo_wrap", "front"), "wrap_tall_top": ("cargo_wrap", "top"),
    "carton_stack_side": ("cargo_carton_stack", "front"), "carton_stack_top": ("cargo_carton_stack", "top"),
    "black_wrap_side": ("cargo_black_net", "front"), "black_wrap_top": ("cargo_black_net", "top"),
    "strapped_side": ("cargo_strapped", "front"), "strapped_top": ("cargo_strapped", "top"),
    "wooden_crate_side": ("cargo_wooden", "front"), "wooden_crate_top": ("cargo_wooden", "top"),
    "carton_small_side": ("cargo_small_carton", "front"), "carton_small_top": ("cargo_small_carton", "top"),
}


def labels():
    spec = importlib.util.spec_from_file_location("labels", REVIEW / "labels.py")
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod); return mod


def frame(video, t, dest):
    if not dest.exists():
        subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-ss", f"{t:.2f}", "-i", str(VIDEOS / video),
                        "-frames:v", "1", "-q:v", "2", str(dest)], check=True)
    return Image.open(dest).convert("RGB")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--per-variant", type=int, default=8, help="images per variant (a merged group gets one set per variant)")
    args = parser.parse_args()
    lab = labels(); ids = json.loads((REVIEW / "ids.json").read_text())
    (OUT / "images").mkdir(parents=True, exist_ok=True); (OUT / "frames").mkdir(exist_ok=True)
    items = []
    for group, (cls, attr, var) in GROUPS.items():
        picked = []
        for v in var:   # per variant, so a merged group keeps the images of the first labelling round
            rows = [r for r in ids[cls] if getattr(lab, attr).get(r["id"][2:], ("?",))[0] == v and not r["touches_border"]]
            rows.sort(key=lambda r: -(r["box"][2] - r["box"][0]) * (r["box"][3] - r["box"][1]) * min(r["sharpness"], 800))
            n0 = len(picked)
            for r in rows:
                if any(p["camera"] == r["camera"] and p["video"] == r["video"] and abs(p["t_s"] - r["t_s"]) < 30
                       and math.dist(p["bottom_center_px"], r["bottom_center_px"]) < 120 for p in picked[n0:]):
                    continue      # same object again
                picked.append(r)
                if len(picked) - n0 == args.per_variant:
                    break
        for r in picked:
            f = frame(r["video"], r["t_s"], OUT / "frames" / f"{Path(r['video']).stem}_{r['t_s']:07.1f}.jpg")
            x0, y0, x1, y1 = r["box"]; m = .3 * max(x1 - x0, y1 - y0)
            box = (max(0, int(x0 - m)), max(0, int(y0 - m)), min(f.width, int(x1 + m)), min(f.height, int(y1 + m)))
            # file names of the first labelling round are kept, so labels saved in the browser still match
            v = getattr(lab, attr)[r["id"][2:]][0]
            name = f"{OLD_NAMES.get((group, v), group)}_{r['id']}.png"; f.crop(box).save(OUT / "images" / name)
            items.append({"image": f"images/{name}", "group": group, "id": r["id"], "camera": r["camera"],
                          "video": r["video"], "t_s": r["t_s"], "frame_box": list(box)})
        print(group, len(picked))
    (OUT / "candidates.json").write_text(json.dumps(items, indent=1))
    html = (Path(__file__).parent / "face_labeling.html").read_text(encoding="utf-8")
    (OUT / "index.html").write_text(html.replace("/*DATA*/[]", json.dumps(items)).replace("/*KINDS*/{}", json.dumps(KINDS, ensure_ascii=False))
                                    .replace("/*GROUP_VI*/{}", json.dumps(GROUP_VI, ensure_ascii=False))
                                    .replace("/*LEGACY*/{}", json.dumps(LEGACY)),
                                    encoding="utf-8")
    print(f"{len(items)} images -> {OUT / 'index.html'}")


if __name__ == "__main__":
    main()
