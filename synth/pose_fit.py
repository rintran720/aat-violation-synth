"""Silhouette pose fitting of a real forklift on the floor (x, y, yaw) with the catalogue proxy (synth.forklift_proxy).

The proxy's labelled points are projected through the calibrated camera with a z-buffer on the GPU, and poses are
scored by IoU against a SAM3 forklift mask. Fork points are "don't care" (SAM3 rarely includes the tines).
Used by stage B (scale_check) and stage C2 (pose of the truck to anchor loads on).
"""
import math

import numpy as np
import torch
import torch.nn.functional as F

FORK = 1


class Camera:
    """Pinhole camera from camera.json (Blender convention: -Z forward, +Y up)."""
    def __init__(self, cam, device="cuda"):
        self.W, self.H = cam["width"], cam["height"]
        self.K = cam["K_norm"]
        self.M = np.array(cam["matrix_world"])
        self.Mi = torch.tensor(np.linalg.inv(self.M), dtype=torch.float32, device=device)
        self.device = device

    def floor_xy(self, px, py):
        x = (px / self.W - self.K[0][2]) / self.K[0][0]; y = -(py / self.H - self.K[1][2]) / self.K[1][1]
        r = self.M[:3, :3] @ np.array([x, y, -1.]); o = self.M[:3, 3]
        return (o - o[2] / r[2] * r)[:2]


class MaskFit:
    """One frame: SAM3 mask of the forklift, scored at a resolution adapted to its size."""
    def __init__(self, camera, mask, max_cells=120):
        self.cam = camera; dev = camera.device
        A = torch.tensor(mask, device=dev); ys, xs = torch.nonzero(A, as_tuple=True)
        span = max((xs.max() - xs.min()).item(), (ys.max() - ys.min()).item())
        self.ds = ds = max(2, int(math.ceil(span / max_cells)))
        pad = int(.6 * span)
        x0, y0 = max(0, xs.min().item() - pad), max(0, ys.min().item() - pad)
        x1, y1 = min(camera.W, xs.max().item() + pad), min(camera.H, ys.max().item() + pad)
        self.x0, self.y0 = x0 - x0 % ds, y0 - y0 % ds
        self.gw, self.gh = (x1 - self.x0) // ds, (y1 - self.y0) // ds
        crop = A[None, None].float()[..., self.y0:self.y0 + self.gh * ds, self.x0:self.x0 + self.gw * ds]
        self.A = F.avg_pool2d(crop, ds)[0, 0] > .5
        self.init_xy = camera.floor_xy(xs.float().mean().item(), ys.float().quantile(.9).item())
        self.touches_border = bool(xs.min() <= 2 or ys.min() <= 2 or xs.max() >= camera.W - 3 or ys.max() >= camera.H - 3)

    def render(self, poses, P, L):
        """poses (B,3) -> visible per-label masks (B,5,gh,gw) via z-buffered point splats."""
        c, Mi, K = self.cam, self.cam.Mi, self.cam.K; B = len(poses)
        cy, sy = torch.cos(poses[:, 2]), torch.sin(poses[:, 2])
        wx = poses[:, 0, None] + cy[:, None] * P[:, 0] - sy[:, None] * P[:, 1]
        wy = poses[:, 1, None] + sy[:, None] * P[:, 0] + cy[:, None] * P[:, 1]; wz = P[:, 2].expand(B, -1)
        cc = [Mi[i, 0] * wx + Mi[i, 1] * wy + Mi[i, 2] * wz + Mi[i, 3] for i in range(3)]
        px = (-cc[0] / cc[2] * K[0][0] + K[0][2]) * c.W; py = (cc[1] / cc[2] * K[1][1] + K[1][2]) * c.H
        gx = ((px - self.x0) / self.ds).long(); gy = ((py - self.y0) / self.ds).long(); depth = -cc[2]
        ok = (gx >= 0) & (gx < self.gw) & (gy >= 0) & (gy < self.gh) & (depth > 0)
        n = self.gh * self.gw
        pix = torch.where(ok, gy * self.gw + gx, torch.zeros_like(gx)) + torch.arange(B, device=c.device)[:, None] * n
        dep = torch.where(ok, depth, torch.full_like(depth, 1e9))
        zmin = torch.full((B * n,), 1e9, device=c.device).scatter_reduce(0, pix.flatten(), dep.flatten(), "amin")
        vis = ok & (dep <= zmin[pix] + .08)
        out = torch.zeros((B * n * 5,), device=c.device)
        out.index_fill_(0, (pix * 5 + L[None, :]).flatten()[vis.flatten()], 1.)
        return F.max_pool2d(out.view(B, self.gh, self.gw, 5).permute(0, 3, 1, 2), 3, 1, 1) > 0

    def iou(self, poses, P, L):
        m = self.render(poses, P, L)
        body = m[:, [i for i in range(5) if i != FORK]].any(1); fork = m[:, FORK] & ~body
        return (body & self.A).sum((1, 2)).float() / ((body | self.A) & ~fork).sum((1, 2)).float().clamp(min=1)

    def search(self, P, L, xs, ys, yaws, batch=192):
        g = torch.stack(torch.meshgrid(xs, ys, yaws, indexing="ij"), -1).view(-1, 3); best = None
        for i in range(0, len(g), batch):
            s = self.iou(g[i:i + batch], P, L); j = s.argmax()
            if best is None or s[j] > best[0]:
                best = (s[j].item(), g[i + j].clone())
        return best

    def fit(self, P, L, start=None, coarse=True):
        """Best (iou, pose). Full coarse search when no start pose is given, then two local refinements."""
        dev = self.cam.device; ar = lambda a, b, st: torch.arange(a, b + 1e-6, st, device=dev); r = math.radians
        if start is None:
            cx, cy = self.init_xy
            b = self.search(P, L, ar(cx - 1.8, cx + 1.8, .2), ar(cy - 1.8, cy + 1.8, .2), ar(-math.pi, math.pi - 1e-6, r(6)))
        else:
            b = (None, torch.tensor(start, device=dev, dtype=torch.float32))
            if coarse:
                p = b[1]; b = self.search(P, L, ar(p[0] - .5, p[0] + .5, .1), ar(p[1] - .5, p[1] + .5, .1), ar(p[2] - r(10), p[2] + r(10), r(2)))
        for span, st, ya, yst in [(.25, .05, 5, 1), (.06, .015, 1.5, .25)]:
            p = b[1]; b = self.search(P, L, ar(p[0] - span, p[0] + span, st), ar(p[1] - span, p[1] + span, st),
                                      ar(p[2] - r(ya), p[2] + r(ya), r(yst)))
        return b[0], [v.item() for v in b[1]]
