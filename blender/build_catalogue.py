"""3D catalogue: build every LSP / SKID / cargo variant with real textures and render it from all around.

E:/tools/blender-5.2.2-windows-x64/blender.exe -b --python-exit-code 1 --python blender/build_catalogue.py -- [--only NAME] [--size 320]
Textures come from synth.catalogue_textures: every object the user labelled as a box (objects.json) becomes its own
model with its own top / front / right faces; LSP and SKID stacks stack labelled singles. Sizes: config/standards.json
for LSP and SKID; cargo footprints per kind, cargo height from the labelled front face's aspect.
Each variant is saved as work/catalogue3d/blend/<variant>.blend and rendered (Cycles, transparent film, shadow catcher
floor) at 8 azimuths x 3 elevations into work/catalogue3d/renders/<object>/<variant>/e<el>_a<az>.png.
Writes work/catalogue3d/variants.json (object, variant, label, size, textures).
"""
import json
import math
import random
import sys
from pathlib import Path

import bmesh
import bpy

ROOT = Path.cwd()
OUT = ROOT / "work/catalogue3d"
TEX = json.loads((OUT / "textures/textures.json").read_text())
STD = json.loads((ROOT / "config/standards.json").read_text())
AZIMUTHS = list(range(0, 360, 45))
ELEVATIONS = [10, 30, 55]
LSP_W, LSP_D = STD["lsp"]["size_m"]; LSP_T = STD["lsp"]["thickness_m"]
SKID_W, SKID_D = STD["skid"]["size_m"]; SKID_H = STD["skid"]["height_m"]


def tex(name, k=0):
    items = TEX[name]
    return str(ROOT / items[k % len(items)]["texture"])


def clear():
    for o in list(bpy.data.objects):
        bpy.data.objects.remove(o, do_unlink=True)
    for coll in (bpy.data.meshes, bpy.data.materials, bpy.data.images, bpy.data.lights, bpy.data.cameras):
        for b in list(coll):
            if b.users == 0:
                coll.remove(b)


_mats = {}


def material(key, image=None, color=(.5, .5, .5), rough=.6, repeat=(1, 1), rotate=0.):
    if key in _mats:
        return _mats[key]
    m = bpy.data.materials.new(key); m.use_nodes = True
    nt = m.node_tree; bsdf = nt.nodes["Principled BSDF"]; bsdf.inputs["Roughness"].default_value = rough
    if image:
        t = nt.nodes.new("ShaderNodeTexImage"); t.image = bpy.data.images.load(image, check_existing=True)
        mp = nt.nodes.new("ShaderNodeMapping"); uv = nt.nodes.new("ShaderNodeTexCoord")
        mp.inputs["Scale"].default_value = (*repeat, 1); mp.inputs["Rotation"].default_value = (0, 0, rotate)
        nt.links.new(uv.outputs["UV"], mp.inputs["Vector"]); nt.links.new(mp.outputs["Vector"], t.inputs["Vector"])
        nt.links.new(t.outputs["Color"], bsdf.inputs["Base Color"])
    else:
        bsdf.inputs["Base Color"].default_value = (*color, 1)
    _mats[key] = m
    return m


def box(name, size, loc, side, top=None, bevel=0., rot_z=0., right=None):
    """Cuboid with a full 0..1 UV per face. Top/bottom get `top`; the +-Y faces get `side` (the front),
    the +-X faces get `right` (defaults to `side`): back and left repeat front and right."""
    sx, sy, sz = size
    me = bpy.data.meshes.new(name); bm = bmesh.new()
    bmesh.ops.create_cube(bm, size=1.)
    bmesh.ops.scale(bm, vec=(sx, sy, sz), verts=bm.verts)
    uvl = bm.loops.layers.uv.new()
    for f in bm.faces:
        n = f.normal
        f.material_index = 1 if abs(n.z) > .5 else 2 if abs(n.x) > .5 else 0
        # project onto the face plane: u along the face's horizontal axis, v up (or y for top faces)
        for lp in f.loops:
            co = lp.vert.co
            if abs(n.z) > .5:
                u, v = co.x / sx + .5, co.y / sy + .5
            elif abs(n.x) > .5:
                u, v = co.y / sy + .5, co.z / sz + .5
                u = u if n.x > 0 else 1 - u
            else:
                u, v = co.x / sx + .5, co.z / sz + .5
                u = 1 - u if n.y > 0 else u
            lp[uvl].uv = (u, v)
    bm.to_mesh(me); bm.free()
    ob = bpy.data.objects.new(name, me); bpy.context.collection.objects.link(ob)
    for m in (side, top or side, right or side):
        me.materials.append(m)
    ob.location = (loc[0], loc[1], loc[2] + sz / 2); ob.rotation_euler[2] = rot_z
    if bevel:
        mod = ob.modifiers.new("bevel", "BEVEL"); mod.width = bevel; mod.segments = 3; mod.limit_method = "ANGLE"
    for p in me.polygons:
        p.use_smooth = False
    return ob


# ---------------- labelled objects (synth.catalogue_textures -> objects.json) ----------------

OBJ = json.loads((OUT / "textures/objects.json").read_text())


def mats(key, rough):
    """Front / top / right materials of one labelled object."""
    o = OBJ[key]; f = o["faces"]
    m = lambda face: material(f"{key}_{face}", str(ROOT / f[face][0]["texture"]), rough=rough)
    return m("front"), m("top"), m("right")


def of_kind(kind):
    return sorted(k for k, o in OBJ.items() if o["kind"] == kind and {"top", "front", "right"} <= set(o["faces"]))


# ---------------- object builders: each returns the height of its top surface ----------------

def lsp_sheet(z, key=None):
    if key:
        front, top, right = mats(key, .5)
    else:
        front = right = material("lsp_side", str(ROOT / "work/textures/lsp_edge_bands.png"), rough=.55)
        top = material("lsp_top_0", tex("lsp_top", 0), rough=.45)
    ob = box(f"lsp_{z:.2f}", (LSP_W, LSP_D, LSP_T), (0, 0, z), front, top, rot_z=random.uniform(-.03, .03), right=right)
    round_edges(ob, (LSP_W, LSP_D, LSP_T), "lsp")
    return z + LSP_T


def skid(z, key=None, gap=0.):
    """Skid: a top deck of boards on three runners, no bottom deck. gap = 0 gives a closed deck (all labelled
    skids so far); gap > 0 leaves that many metres between boards (open deck; no video of one yet, user 2026-10-03)."""
    if key:
        front, deck, right = mats(key, .8)
    else:
        front = right = deck = material("wood", tex("skid_wood", 0), rough=.8, rotate=math.pi / 2)
    for x in (-SKID_W / 2 + .06, 0, SKID_W / 2 - .06):
        box(f"skid_runner_{x:.2f}_{z:.2f}", (.1, SKID_D, SKID_H - .025), (x, 0, z), front, deck, bevel=.006, right=right)
    n = 8 if not gap else 6; bw = SKID_W / n
    for i in range(n):
        board = box(f"skid_board_{i}_{z:.2f}", (bw - max(gap, .002), SKID_D, .025), (-SKID_W / 2 + (i + .5) * bw, 0, z + SKID_H - .025),
                    front, deck, bevel=.004, right=right)
        uv = board.data.uv_layers.active.data   # each board shows its own strip of the deck / front photos
        for poly in board.data.polygons:
            if poly.material_index in (0, 1):
                for li in poly.loop_indices:
                    uv[li].uv[0] = (i + uv[li].uv[0]) / n
    if gap:   # open deck: three cross bars under the boards, at both ends and in the middle (user, 2026-10-03)
        for y in (-SKID_D / 2 + .05, 0, SKID_D / 2 - .05):
            box(f"skid_bar_{y:.2f}_{z:.2f}", (SKID_W, .1, .04), (0, y, z + SKID_H - .065), right, deck, bevel=.004, right=front)
    return z + SKID_H


CARGO = {   # kind -> footprint (width along the front face, depth) m, default height, roughness
    "cargo_wrap": ((1.2, 1.0), 1.3, .35), "cargo_carton_stack": ((1.2, 1.0), 1.2, .7),
    "cargo_black_net": ((1.2, 1.0), 1.0, .25), "cargo_strapped": ((1.0, .8), 1.5, .6),
    "cargo_wooden": ((1.2, 1.0), .9, .7), "cargo_small_carton": ((.5, .4), .35, .7),
}


# edge rounding as a share of the object's smallest side: (all edges, vertical corners in plan view, smooth shading)
ROUND = {
    "cargo_wrap": (.09, 0, True), "cargo_black_net": (.08, 0, True), "cargo_strapped": (.04, 0, False),
    "cargo_wooden": (.03, 0, False), "cargo_carton_stack": (.03, 0, False), "cargo_small_carton": (.05, 0, False),
    "lsp": (.2, .045, False),   # LSP: edges rounded by 20 % of the 9 cm thickness, plan corners by 4.5 % of the side
}


def round_edges(ob, size, kind):
    """Size-dependent rounding (user, 2026-10-04). Plan-view corners (the 4 vertical edges) get their own, larger
    radius first through edge bevel weights; then every edge is rounded by a share of the smallest side."""
    k_all, k_corner, smooth = ROUND[kind]
    if "bevel" in ob.modifiers:
        ob.modifiers.remove(ob.modifiers["bevel"])
    if k_corner:
        me = ob.data
        w = me.attributes.get("bevel_weight_edge") or me.attributes.new("bevel_weight_edge", "FLOAT", "EDGE")
        for e in me.edges:
            a, b = (me.vertices[i].co for i in e.vertices)
            w.data[e.index].value = 1. if abs(a.x - b.x) < 1e-6 and abs(a.y - b.y) < 1e-6 else 0.
        m = ob.modifiers.new("corners", "BEVEL"); m.limit_method = "WEIGHT"
        m.width = k_corner * min(size[0], size[1]); m.segments = 8
    m = ob.modifiers.new("edges", "BEVEL"); m.limit_method = "ANGLE"
    m.width = min(k_all * min(size), .45 * min(size)); m.segments = 6 if smooth or k_all * min(size) > .03 else 3
    if smooth:
        for p in ob.data.polygons:
            p.use_smooth = True


def cargo_size(key):
    """Footprint and default height from the kind. The labelled front face only adjusts the height by +-25 %:
    its image aspect against the median of its kind (CCTV looks down at an angle, so absolute aspects mislead)."""
    import statistics
    kind = OBJ[key]["kind"]; (w, d), h0, _ = CARGO[kind]
    aspects = [o["faces"]["front"][0]["aspect"] for o in OBJ.values() if o["kind"] == kind and "front" in o["faces"]]
    rel = statistics.median(aspects) / max(OBJ[key]["faces"]["front"][0]["aspect"], 1e-6)
    return (w, d, round(h0 * min(1.25, max(.75, rel)), 2))


def cargo(z, key):
    front, top, right = mats(key, CARGO[OBJ[key]["kind"]][2])
    size = cargo_size(key)
    round_edges(ob := box(f"cargo_{key}", size, (0, 0, z), front, top, right=right), size, OBJ[key]["kind"])
    return z + size[2]


def variants():
    """name -> (object, label, builder): one model per labelled object, stacks of labelled singles,
    each cargo object with and without a SKID (the site uses SKIDs only, no pallets; user, 2026-10-03)."""
    v = {}
    lsps, skids = of_kind("lsp_single"), of_kind("skid_single")
    for k in lsps:
        v[f"lsp_{k}"] = ("lsp", f"Single sheet ({k}, {OBJ[k]['camera']})", lambda k=k: lsp_sheet(0, k))
    for n in (3, 6, 10):
        v[f"lsp_stack_{n}"] = ("lsp", f"Stack of {n} labelled sheets", lambda n=n: [lsp_sheet(i * LSP_T, lsps[i % len(lsps)]) for i in range(n)])
    for k in skids:
        v[f"skid_{k}"] = ("skid", f"Single SKID ({k}, {OBJ[k]['camera']})", lambda k=k: skid(0, k))
    for k in skids:
        v[f"skid_{k}_open"] = ("skid", f"Single SKID, open deck ({k} textures)", lambda k=k: skid(0, k, gap=.06))
    for n in (4, 8):
        v[f"skid_stack_{n}"] = ("skid", f"Stack of {n} labelled SKIDs", lambda n=n: [skid(i * SKID_H, skids[i % len(skids)]) for i in range(n)])
    cargos = sorted(k for k, o in OBJ.items() if o["kind"] in CARGO and {"top", "front", "right"} <= set(o["faces"]))
    for i, k in enumerate(cargos):
        kind = OBJ[k]["kind"]; s = skids[i % len(skids)] if skids else None
        v[f"{kind}_{k}_skid"] = (kind, f"{k} ({OBJ[k]['camera']}) on SKID {s}", lambda k=k, s=s: cargo(skid(0, s), k))
        v[f"{kind}_{k}_noskid"] = (kind, f"{k} ({OBJ[k]['camera']}), no SKID", lambda k=k: cargo(0, k))
    return v


VARIANTS = variants()


def setup_scene(size):
    sc = bpy.context.scene
    sc.render.engine = "CYCLES"; sc.cycles.samples = 48; sc.cycles.use_denoising = True
    try:
        prefs = bpy.context.preferences.addons["cycles"].preferences; prefs.compute_device_type = "OPTIX"
        prefs.get_devices(); [setattr(d, "use", True) for d in prefs.devices]; sc.cycles.device = "GPU"
    except Exception:
        pass
    sc.render.resolution_x = sc.render.resolution_y = size; sc.render.film_transparent = True
    sc.render.image_settings.file_format = "PNG"; sc.render.image_settings.color_mode = "RGBA"
    sc.view_settings.view_transform = "Standard"
    w = bpy.data.worlds.get("World") or bpy.data.worlds.new("World"); sc.world = w; w.use_nodes = True
    w.node_tree.nodes["Background"].inputs["Color"].default_value = (.55, .56, .55, 1)
    w.node_tree.nodes["Background"].inputs["Strength"].default_value = .55


def add_rig():
    floor = bpy.data.meshes.new("floor"); bm = bmesh.new(); bmesh.ops.create_grid(bm, x_segments=1, y_segments=1, size=20)
    bm.to_mesh(floor); bm.free(); fo = bpy.data.objects.new("floor", floor); bpy.context.collection.objects.link(fo)
    fo.is_shadow_catcher = True
    light = bpy.data.lights.new("ceiling", "AREA"); light.energy = 450; light.size = 6
    lo = bpy.data.objects.new("ceiling", light); lo.location = (1.5, -1, 7); bpy.context.collection.objects.link(lo)
    cam = bpy.data.cameras.new("cam"); cam.lens = 50; co = bpy.data.objects.new("cam", cam)
    bpy.context.collection.objects.link(co); bpy.context.scene.camera = co
    return co


def bounds():
    import mathutils
    pts = [o.matrix_world @ mathutils.Vector(c) for o in bpy.data.objects if o.type == "MESH" and o.name != "floor"
           for c in o.bound_box]
    lo = [min(p[i] for p in pts) for i in range(3)]; hi = [max(p[i] for p in pts) for i in range(3)]
    return lo, hi


def render_views(cam, dest):
    import mathutils
    lo, hi = bounds(); c = mathutils.Vector([(a + b) / 2 for a, b in zip(lo, hi)])
    r = .5 * math.dist(lo, hi); fov = 2 * math.atan(18 / cam.data.lens); dist = 1.15 * r / math.sin(fov / 2)
    dest.mkdir(parents=True, exist_ok=True)
    for el in ELEVATIONS:
        for az in AZIMUTHS:
            e, a = math.radians(el), math.radians(az)
            cam.location = c + dist * mathutils.Vector((math.sin(a) * math.cos(e), -math.cos(a) * math.cos(e), math.sin(e)))
            cam.rotation_euler = (c - cam.location).to_track_quat("-Z", "Y").to_euler()
            bpy.context.scene.render.filepath = str(dest / f"e{el:02d}_a{az:03d}.png")
            bpy.ops.render.render(write_still=True)
    return [round(v, 3) for v in (hi[0] - lo[0], hi[1] - lo[1], hi[2] - lo[2])]


def main():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    only = argv[argv.index("--only") + 1] if "--only" in argv else None
    size = int(argv[argv.index("--size") + 1]) if "--size" in argv else 320
    (OUT / "blend").mkdir(parents=True, exist_ok=True)
    manifest_path = OUT / "variants.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    for name, (obj, label, build) in VARIANTS.items():
        if only and only != name:
            continue
        clear(); _mats.clear(); random.seed(7)
        setup_scene(size); cam = add_rig(); build()
        bpy.ops.wm.save_as_mainfile(filepath=str(OUT / "blend" / f"{name}.blend"))
        dims = render_views(cam, OUT / "renders" / obj / name)
        textures = sorted({Path(n.image.filepath).name for m in bpy.data.materials if m.use_nodes
                           for n in m.node_tree.nodes if n.type == "TEX_IMAGE" and n.image})
        manifest[name] = {"object": obj, "label": label, "size_m": dims, "textures": textures}
        print(f"{name}: {dims} m", flush=True)
        manifest_path.write_text(json.dumps(manifest, indent=1))


main()
