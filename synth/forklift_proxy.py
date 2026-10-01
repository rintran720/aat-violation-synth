"""Forklift fitting proxy: a labelled surface point cloud built from catalogue dimensions (config/standards.json).

Used to find a real forklift's floor pose from its silhouette, not for rendering. The frame matches
work/assets/forklift.blend so a fitted pose drives render_anchor_poc directly: +Y forward, floor z=0,
fork heel face at y=FORK_FACE. Labels: 0 body, 1 fork, 2 mast/carriage, 3 front wheel, 4 rear wheel.

Run: python -m synth.forklift_proxy work/forklift_proxy_points.npz [--model sumitomo_quapro_2t5_dual] [--z-scale 0.85]
--z-scale is the camera's height correction from stage B (cam01: 0.85), not a property of the truck.
"""
import argparse
import json
from pathlib import Path

import numpy as np

FORK_FACE = .807  # fork heel face in forklift.blend; keeps fitted poses usable by the renderer
LABELS = ['body', 'fork', 'mast', 'front_wheel', 'rear_wheel']


def load_spec(model, standards='config/standards.json'):
    return json.loads(Path(standards).read_text())['forklift'][model]


def layout(spec):
    """Key longitudinal positions and tyre placements (metres, proxy frame)."""
    front_axle = FORK_FACE - spec['axle_to_fork_face_m']
    rear_axle = front_axle - spec['wheelbase_m']
    body_rear = FORK_FACE - spec['length_to_fork_face_m']
    fw, rw = spec['tyre_front_width_m'], spec['tyre_rear_width_m']
    outer = spec['overall_width_dual_m'] / 2 - fw / 2      # outer front tyre centre
    front_tyres = [(outer, spec['tyre_front_radius_m'], fw), (outer - fw, spec['tyre_front_radius_m'], fw)]
    rear_tyre = (spec['tread_rear_m'] / 2, spec['tyre_rear_radius_m'], rw)
    contacts = {**{f'front_{s}': (s * outer, front_axle) for s in (-1, 1)},
                **{f'front_in_{s}': (s * (outer - fw), front_axle) for s in (-1, 1)},
                **{f'rear_{s}': (s * rear_tyre[0], rear_axle) for s in (-1, 1)}}
    return dict(front_axle=front_axle, rear_axle=rear_axle, body_rear=body_rear,
                front_tyres=front_tyres, rear_tyre=rear_tyre, contacts=contacts)


def _box(lo, hi, rng, density):
    lo, hi = np.array(lo, float), np.array(hi, float); d = hi - lo; pts = []
    for ax in range(3):
        a, b = [i for i in range(3) if i != ax]
        n = max(20, int(d[a] * d[b] * density))
        for side in (lo[ax], hi[ax]):
            p = lo + rng.random((n, 3)) * d; p[:, ax] = side; pts.append(p)
    return np.concatenate(pts)


def _tyre(cx, cy, r, w, rng, density):
    n = max(60, int(2 * np.pi * r * w * density)); t = rng.random(n) * 2 * np.pi
    side = np.stack([cx + (rng.random(n) - .5) * w, cy + r * np.cos(t), r + r * np.sin(t)], 1)
    m = max(30, int(np.pi * r * r * density)); rr = r * np.sqrt(rng.random(m)); tt = rng.random(m) * 2 * np.pi
    caps = [np.stack([np.full(m, cx + s * w / 2), cy + rr * np.cos(tt), r + rr * np.sin(tt)], 1) for s in (-1, 1)]
    return np.concatenate([side, *caps])


def build(spec, z_scale=1.0, seed=0, density=2500):
    """Points (N,3) float32 and labels (N,) int8. Body blocks are coarse boxes sized from the catalogue."""
    g = layout(spec); rng = np.random.default_rng(seed); P, L = [], []
    add = lambda p, label: (P.append(p), L.append(np.full(len(p), label)))
    half = spec['body_width_m'] / 2; roof = spec['overhead_guard_m']
    fa, br = g['front_axle'], g['body_rear']
    add(_box((-half, br, .12), (half, fa + .12, .62), rng, density), 0)                       # chassis
    add(_box((-half, br, .62), (half, br + .75, 1.12), rng, density), 0)                      # counterweight
    add(_box((-.45, br + .75, .62), (.45, fa - .15, 1.02), rng, density), 0)                  # hood / seat base
    roof_front, roof_rear = fa + .05, br + .45
    add(_box((-.55, roof_rear, roof - .06), (.55, roof_front, roof), rng, density), 0)        # overhead guard
    for sx in (-1, 1):
        add(_box((sx * .55 - .04, roof_rear, 1.0), (sx * .55 + .04, roof_rear + .07, roof - .06), rng, density), 0)      # rear posts
        add(_box((sx * .52 - .035, roof_front - .1, .6), (sx * .52 + .035, roof_front - .03, roof - .06), rng, density), 0)  # front posts
        add(_box((sx * .40 - .06, FORK_FACE - .22, .05), (sx * .40 + .06, FORK_FACE - .08, spec['mast_lowered_m']), rng, density), 2)
        for cx, r, w in g['front_tyres']:
            add(_tyre(sx * cx, fa, r, w, rng, density), 3)
        cx, r, w = g['rear_tyre']; add(_tyre(sx * cx, g['rear_axle'], r, w, rng, density), 4)
        add(_box((sx * .30 - .06, FORK_FACE - .05, 0), (sx * .30 + .06, FORK_FACE, .55), rng, density), 1)        # fork heel
        add(_box((sx * .30 - .06, FORK_FACE, 0), (sx * .30 + .06, FORK_FACE + spec['fork_length_m'], .05), rng, density), 1)  # tine
    add(_box((-.535, FORK_FACE - .08, .1), (.535, FORK_FACE - .05, .65), rng, density), 2)   # carriage plate
    P = np.concatenate(P); L = np.concatenate(L); P[:, 2] *= z_scale
    return P.astype(np.float32), L.astype(np.int8)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('out')
    parser.add_argument('--model', default='sumitomo_quapro_2t5_dual')
    parser.add_argument('--z-scale', type=float, default=1.0)
    args = parser.parse_args()
    spec = load_spec(args.model); P, L = build(spec, args.z_scale)
    np.savez(args.out, points=P, labels=L, contacts=json.dumps(layout(spec)['contacts']), model=args.model, z_scale=args.z_scale)
    print(args.out, len(P), 'points', {k: int((L == i).sum()) for i, k in enumerate(LABELS)})


if __name__ == '__main__':
    main()
