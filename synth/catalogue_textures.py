"""Real textures for the 3D catalogue, cut from reviewed library crops.

Run: python -m synth.catalogue_textures
Reads work/library_v2/review/{ids.json,labels.py} (manual variant/view labels) and writes
work/catalogue3d/textures/<name>.png plus textures.json (name -> source crops).
Flat objects (LSP top, SKID deck) are rectified: the crop's 4-corner outline is warped to a rectangle.
Object faces come from the user's box labels (config/catalogue_faces.json, saved by work/face_labeling/index.html):
each labelled object gets its own top / front / right textures in textures/objects/, described in objects.json;
textures_by_object.jpg is the review sheet.
"""
import importlib.util
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

REVIEW = Path("work/library_v2/review")
OUT = Path("work/catalogue3d/textures")
SIZE = 512


def labels():
    spec = importlib.util.spec_from_file_location("labels", REVIEW / "labels.py")
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    ids = json.loads((REVIEW / "ids.json").read_text())
    rows = []
    for cls, lab in [("forklift", mod.FORKLIFT), ("lsp", mod.LSP), ("skid", mod.SKID),
                     ("wrapped_cargo", mod.CARGO_WR), ("carton", mod.CARGO_CA)]:
        for r in ids[cls]:
            v = lab.get(r["id"][2:])
            if v:
                rows.append({**r, "variant": v[0], "view": v[1]})
    return rows


def rgba(path):
    a = np.asarray(Image.open(path).convert("RGBA"))
    return a[..., :3], a[..., 3] > 127


def quad(mask):
    cnt = max(cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0], key=cv2.contourArea)
    hull = cv2.convexHull(cnt)
    for eps in np.linspace(.01, .2, 40) * cv2.arcLength(hull, True):
        q = cv2.approxPolyDP(hull, eps, True)
        if len(q) == 4:
            q = q.reshape(4, 2).astype(np.float32); break
    else:
        q = cv2.boxPoints(cv2.minAreaRect(cnt)).astype(np.float32)
    c = q.mean(0); q = q[np.argsort(np.arctan2(q[:, 1] - c[1], q[:, 0] - c[0]))]   # clockwise in image coords
    return np.roll(q, -int(np.argmin(q.sum(1))), 0)                                  # start top-left


def rectify(path, inset=.06):
    """Flat sheet seen at an angle -> its top face as a square texture (inset drops the sheet's own rim)."""
    rgb, m = rgba(path); q = quad(m); q = q + inset * (q.mean(0) - q)
    dst = np.float32([[0, 0], [SIZE, 0], [SIZE, SIZE], [0, SIZE]])
    return cv2.warpPerspective(rgb, cv2.getPerspectiveTransform(q, dst), (SIZE, SIZE), flags=cv2.INTER_CUBIC)


def inner_rect(path, cover=.985):
    """Largest centred axis-aligned rectangle (shrunk step by step) almost fully inside the mask."""
    rgb, m = rgba(path); ys, xs = np.nonzero(m); x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
    for _ in range(40):
        if m[y0:y1, x0:x1].mean() >= cover:
            break
        dx, dy = max(1, (x1 - x0) // 40), max(1, (y1 - y0) // 40); x0, x1, y0, y1 = x0 + dx, x1 - dx, y0 + dy, y1 - dy
    return cv2.resize(rgb[y0:y1, x0:x1], (SIZE, SIZE), interpolation=cv2.INTER_CUBIC)


def top_band(path, frac=.28):
    """Top part of an upright load (its lid / top film) from a crop that shows the top."""
    rgb, m = rgba(path); ys, xs = np.nonzero(m); y0, y1 = ys.min(), ys.min() + int(frac * (ys.max() - ys.min()))
    row = m[y0:y1]; cols = np.nonzero(row.mean(0) > .9)[0]
    x0, x1 = (cols.min(), cols.max()) if len(cols) > 10 else (xs.min(), xs.max())
    return cv2.resize(rgb[y0:y1, x0:x1], (SIZE, SIZE), interpolation=cv2.INTER_CUBIC)


def best(rows, variant, view=None, cls=None, n=1, min_px=90):
    sel = [r for r in rows if r["variant"] == variant and (view is None or r["view"] == view)
           and (cls is None or r["class"] == cls) and max(r["box"][2] - r["box"][0], r["box"][3] - r["box"][1]) >= min_px]
    return sorted(sel, key=lambda r: -r["quality"])[:n]


# texture name -> (class, variant, view, extractor, count): flat sheets only
SPEC = {
    "lsp_top":       ("lsp", "S", None, rectify, 3),
    "skid_deck":     ("skid", "S", None, rectify, 3),
}
# reviewed on textures_overview.jpg: these picks show a stack, a side view or background, not the face
DROP = {"lsp_top": [1, 2], "skid_deck": [0, 1]}   # lsp_top: user kept only texture 0 (2026-10-03)


def face(crop, quad):
    """Rectify one labelled face. A 4-point outline is the face itself; a longer polygon (e.g. a stack of sheets)
    is rectified by the 4-corner fit of its outline, and pixels outside the polygon are filled from inside it."""
    rgb, _ = rgba(crop); dst = np.float32([[0, 0], [SIZE, 0], [SIZE, SIZE], [0, SIZE]])
    poly = np.float32(quad)
    corners = poly if len(poly) == 4 else _corners(poly)
    H = cv2.getPerspectiveTransform(corners, dst)
    out = cv2.warpPerspective(rgb, H, (SIZE, SIZE), flags=cv2.INTER_CUBIC)
    if len(poly) > 4:
        m = np.zeros(rgb.shape[:2], np.uint8); cv2.fillPoly(m, [np.round(poly).astype(np.int32)], 255)
        m = cv2.warpPerspective(m, H, (SIZE, SIZE), flags=cv2.INTER_NEAREST)
        out = cv2.inpaint(out, (m == 0).astype(np.uint8), 7, cv2.INPAINT_TELEA)
    return out


def _corners(poly):
    """4 corners of a polygon's outline (top-left first, clockwise in image coordinates)."""
    m = np.zeros((int(poly[:, 1].max()) + 2, int(poly[:, 0].max()) + 2), np.uint8)
    cv2.fillPoly(m, [np.round(poly).astype(np.int32)], 1)
    return quad(m)


def orient(q):
    """Clockwise corners with the face's top edge first: the edge whose midpoint is highest in the image."""
    q = np.float32(q)
    if np.dot(q[:, 0], np.roll(q[:, 1], -1)) - np.dot(q[:, 1], np.roll(q[:, 0], -1)) < 0:   # counter-clockwise (y down)
        q = q[::-1].copy()
    k = min(range(4), key=lambda i: (q[i][1] + q[(i + 1) % 4][1]))
    return np.roll(q, -k, 0)


def object_face(image, poly):
    """Rectify one labelled face (4 corners, or a polygon fitted by 4 corners with the outside inpainted).
    Returns the texture and the face's image aspect (top-edge length / side-edge length)."""
    rgb = np.asarray(Image.open(image).convert("RGB")); dst = np.float32([[0, 0], [SIZE, 0], [SIZE, SIZE], [0, SIZE]])
    poly = np.float32(poly); corners = orient(poly if len(poly) == 4 else _corners(poly))
    H = cv2.getPerspectiveTransform(corners, dst)
    out = cv2.warpPerspective(rgb, H, (SIZE, SIZE), flags=cv2.INTER_CUBIC)
    if len(poly) > 4:
        m = np.zeros(rgb.shape[:2], np.uint8); cv2.fillPoly(m, [np.round(poly).astype(np.int32)], 255)
        m = cv2.warpPerspective(m, H, (SIZE, SIZE), flags=cv2.INTER_NEAREST)
        out = cv2.inpaint(out, (m == 0).astype(np.uint8), 7, cv2.INPAINT_TELEA)
    e = lambda a, b: float(np.linalg.norm(corners[a] - corners[b]))
    aspect = (e(0, 1) + e(3, 2)) / max(e(1, 2) + e(0, 3), 1e-6)
    return out, aspect, corners


def objects(faces_path=Path("config/catalogue_faces.json")):
    """One entry per labelled object: its rectified faces (several polygons per face allowed, largest first);
    a missing front/right takes the other one, a missing top is borrowed from another object of the same kind."""
    out_dir = OUT / "objects"; out_dir.mkdir(parents=True, exist_ok=True)
    for old in out_dir.glob("*.png"):
        old.unlink()
    objs = {}
    for f in json.loads(faces_path.read_text())["faces"]:
        if "face" not in f:
            continue
        key = f["object"].replace("#", "-")
        o = objs.setdefault(key, {"kind": f["kind"], "camera": f["camera"], "image": f["image"], "faces": {}, "borrowed": {}})
        tex, aspect, corners = object_face(f["image"], f["quad"])
        area = cv2.contourArea(np.float32(f["quad"]))
        o["faces"].setdefault(f["face"], []).append({"area": area, "aspect": round(aspect, 3), "tex": tex,
                                                     "quad": f["quad"], "corners": corners.round(1).tolist()})
    for key, o in objs.items():
        for face_name, items in o["faces"].items():
            items.sort(key=lambda x: -x["area"])
            for k, it in enumerate(items):
                path = out_dir / f"{key}_{face_name}_{k}.png"; Image.fromarray(it.pop("tex")).save(path)
                it["texture"] = path.as_posix()
    for key, o in objs.items():
        fs = o["faces"]
        for a, b in (("front", "right"), ("right", "front")):
            if a not in fs and b in fs:
                o["borrowed"][a] = f"{key}:{b}"; fs[a] = fs[b]
        for need in ("top", "front"):
            if need not in fs:
                donor = next((k for k, d in objs.items() if d["kind"] == o["kind"] and need in d["faces"]
                              and need not in d["borrowed"]), None)
                if donor:
                    o["borrowed"][need] = f"{donor}:{need}"; fs[need] = objs[donor]["faces"][need]
                    if "right" not in fs:
                        o["borrowed"]["right"] = f"{donor}:{need}"; fs["right"] = fs[need]
    (OUT / "objects.json").write_text(json.dumps(objs, indent=1))
    return objs


def review_sheet(objs, path=Path("work/catalogue3d/textures_by_object.jpg"), T=150):
    """One row per object: the source image with its polygons, then its top / front / right textures."""
    from PIL import ImageDraw
    rows = sorted(objs.items(), key=lambda kv: (kv[1]["kind"], kv[0]))
    sheet = Image.new("RGB", (4 * (T + 10) + 200, len(rows) * (T + 10) + 30), (128, 130, 126)); d = ImageDraw.Draw(sheet)
    for c, name in enumerate(["source + labels", "top", "front", "right"]):
        d.text((200 + c * (T + 10), 8), name, fill=(255, 255, 255))
    colors = {"top": (95, 211, 141), "front": (255, 179, 71), "right": (110, 198, 255)}
    for r, (key, o) in enumerate(rows):
        y = 30 + r * (T + 10)
        d.multiline_text((6, y + 4), "\n".join([key, o["kind"], o["camera"]]), fill=(255, 255, 255))
        src = Image.open(o["image"]).convert("RGB"); dd = ImageDraw.Draw(src)
        for fn, items in o["faces"].items():
            if fn in o["borrowed"]:
                continue
            for it in items:
                dd.polygon([tuple(p) for p in it["quad"]], outline=colors[fn], width=max(2, src.width // 250))
        src.thumbnail((T, T)); sheet.paste(src, (200, y))
        for c, fn in enumerate(["top", "front", "right"], 1):
            if fn in o["faces"]:
                sheet.paste(Image.open(o["faces"][fn][0]["texture"]).resize((T, T)), (200 + c * (T + 10), y))
                if fn in o["borrowed"]:
                    d.text((200 + c * (T + 10) + 4, y + 4), "borrowed", fill=(255, 80, 80))
    sheet.save(path, quality=88)


def main():
    rows = labels(); OUT.mkdir(parents=True, exist_ok=True); manifest = {}
    for name, (cls, var, view, fn, n) in SPEC.items():
        picks = best(rows, var, view, cls, n, min_px=40 if name == "carton_small" else 90)
        if not picks:
            picks = best(rows, var, None, cls, n, min_px=40)
        picks = [r for k, r in enumerate(picks) if k not in DROP.get(name, [])]
        manifest[name] = []
        for old in OUT.glob(f"{name}_*.png"):
            old.unlink()
        for k, r in enumerate(picks):
            path = OUT / f"{name}_{k}.png"; Image.fromarray(fn(r["crop"])).save(path)
            manifest[name].append({"texture": path.as_posix(), "crop": r["crop"], "camera": r["camera"], "id": r["id"]})
        print(name, len(picks))
    # one plank of the reviewed deck photo (no dark gaps), toned down: planks are lit wood, not the overexposed photo
    deck = np.asarray(Image.open(manifest["skid_deck"][0]["texture"]).convert("RGB")).astype(np.float32)
    lum = deck.mean((1, 2)); win = 48
    y0 = int(np.argmax(np.convolve(lum, np.ones(win) / win, "valid")))
    plank = cv2.resize(deck[y0:y0 + win], (SIZE, SIZE), interpolation=cv2.INTER_CUBIC) * np.float32([.78, .70, .52])   # tone of the real pallets (yellow-beige)
    Image.fromarray(np.clip(plank, 0, 255).astype(np.uint8)).save(OUT / "skid_wood_0.png")
    manifest["skid_wood"] = [{"texture": (OUT / "skid_wood_0.png").as_posix(), **{k: v for k, v in manifest["skid_deck"][0].items() if k != "texture"}}]
    (OUT / "textures.json").write_text(json.dumps(manifest, indent=1))
    objs = objects(); review_sheet(objs)
    print(f"{len(objs)} objects, {sum(len(v) for o in objs.values() for k, v in o['faces'].items() if k not in o['borrowed'])} faces")


if __name__ == "__main__":
    main()
