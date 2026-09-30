"""Prepare cam01 asset textures from existing SAM3/MoGe results (no model inference).

Run from the repository root: python -m synth.asset_sources
The source selections are intentionally reviewed cam01 selections, not an automatic
claim that every segmentation mask is a usable texture or a measured 3D surface.
"""
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image

from synth.common import load_config, write_json


def homography(src, dst):
    src, dst = np.asarray(src, float), np.asarray(dst, float)
    if src.shape != (4, 2) or dst.shape != (4, 2):
        raise ValueError("Four 2D corners are required")
    rows = []
    for (x, y), (u, v) in zip(src, dst):
        rows.extend([[x, y, 1, 0, 0, 0, -u*x, -u*y, -u],
                     [0, 0, 0, x, y, 1, -v*x, -v*y, -v]])
    if np.linalg.matrix_rank(rows) < 8:
        raise ValueError("Degenerate texture quadrilateral")
    h = np.linalg.svd(rows)[2][-1].reshape(3, 3)
    return h / h[2, 2]


def rectify(image, corners, size=512):
    """Map TL/TR/BR/BL source corners to an upright square, bilinear sampled."""
    h = homography([[0, 0], [size-1, 0], [size-1, size-1], [0, size-1]], corners)
    yy, xx = np.mgrid[:size, :size]
    p = h @ np.stack([xx.ravel(), yy.ravel(), np.ones(xx.size)])
    x = np.clip(p[0]/p[2], 0, image.shape[1]-1.00001)
    y = np.clip(p[1]/p[2], 0, image.shape[0]-1.00001)
    x0, y0 = x.astype(int), y.astype(int)
    fx, fy = (x-x0)[:, None], (y-y0)[:, None]
    out = (image[y0, x0]*(1-fx)*(1-fy) + image[y0, x0+1]*fx*(1-fy)
           + image[y0+1, x0]*(1-fx)*fy + image[y0+1, x0+1]*fx*fy)
    return out.reshape(size, size, image.shape[2])


def mask_quad(mask):
    y, x = np.nonzero(mask)
    if len(x) < 4:
        raise ValueError("Empty/insufficient LSP mask")
    s, d = x+y, x-y
    indices = [s.argmin(), d.argmax(), s.argmax(), d.argmin()]
    return np.array([[x[i], y[i]] for i in indices], float)


def floor_point(pixel, camera):
    """Intersect a calibrated image ray with z=0; never sample background object depth."""
    k = np.array(camera['K_norm'], float).copy()
    k[0] *= camera['width']; k[1] *= camera['height']
    cv = np.linalg.solve(k, [*pixel, 1])
    matrix = np.array(camera['matrix_world'])
    ray = matrix[:3, :3] @ (cv * [1, -1, -1])
    origin = matrix[:3, 3]
    t = -origin[2] / ray[2]
    if t <= 0:
        raise ValueError('Pixel ray does not meet the floor in front of camera')
    return origin + t*ray


def run():
    cfg = load_config()
    if cfg['camera_id'] != 'cam01':
        raise ValueError('Review source selections for this camera before running')
    work = cfg['work']; dest = work/'textures'; dest.mkdir(exist_ok=True)
    calibration = json.loads((work/'calibration.json').read_text())
    records = json.loads((work/'refs/index.json').read_text())
    camera = json.loads((work/'camera.json').read_text())
    sources = []
    # Three fully visible floor sheets, not large stack masks or the distant outlier.
    selected = ['bg1/LSP_top_5.png', 'rtsp_067/LSP_top_1.png', 'rtsp_072/LSP_top_9.png']
    for i, relative in enumerate(selected):
        mask_path = work/'masks'/relative
        mask = np.array(Image.open(mask_path).convert('L')) > 127
        corners = mask_quad(mask)
        # Move slightly inside the boundary to avoid baking floor pixels into the sheet.
        corners = corners.mean(axis=0) + .965*(corners-corners.mean(axis=0))
        src = Path(json.loads((mask_path.parent/'source.json').read_text())['image'])
        img = np.array(Image.open(src).convert('RGB'), dtype=float)
        output = dest/f'lsp_{i}.png'
        Image.fromarray(np.clip(rectify(img, corners), 0, 255).astype('uint8')).save(output)
        sources.append(dict(texture=str(output), source=str(src), mask=str(mask_path),
                            corners_px=corners.tolist(), method='inset mask quadrilateral'))
    # Surface-only patches. Do not box-project the whole forklift or a whole cargo silhouette.
    patches = {
        'forklift_yellow': ('frame_16_forklift_1.png', [[414,259],[505,302],[501,331],[405,298]]),
        'forklift_body_side': ('rtsp_010_forklift_0.png', [[22,180],[231,232],[232,247],[22,210]]),
        'forklift_counter_side': ('rtsp_010_forklift_0.png', [[25,135],[117,165],[111,195],[16,168]]),
        'forklift_roof': ('frame_16_forklift_1.png', [[418,94],[541,124],[438,196],[307,153]]),
        'forklift_side_panel': ('rtsp_010_forklift_0.png', [[135,182],[213,194],[235,222],[137,204]]),
        'uld_white_roof': ('rtsp_010_cargo_0.png', [[87,24],[259,10],[413,140],[245,177]]),
        'uld_dark_roof': ('rtsp_010_cargo_1.png', [[50,8],[206,21],[292,105],[113,99]]),
        'uld_white_front': ('rtsp_010_cargo_0.png', [[61,43],[230,187],[163,276],[9,149]]),
        'uld_white_side': ('rtsp_010_cargo_0.png', [[247,194],[414,160],[355,237],[185,275]]),
        'uld_dark_front': ('rtsp_010_cargo_1.png', [[30,57],[100,116],[54,238],[9,166]]),
        'carton_front': ('frame_07_cargo_0.png', [[9,32],[83,34],[82,85],[10,86]]),
        'carton_plain': ('frame_07_cargo_0.png', [[54,65],[76,65],[76,80],[54,80]]),
    }
    for name, (filename, corners) in patches.items():
        src = work/'refs'/filename
        image = np.array(Image.open(src).convert('RGBA'), dtype=float)
        result = rectify(image, corners)
        if np.mean(result[..., 3] > 200) < .90:
            raise ValueError(f'{name}: surface patch contains too much non-object area')
        output = dest/f'{name}.png'
        Image.fromarray(np.clip(result[..., :3],0,255).astype('uint8')).save(output)
        sources.append(dict(texture=str(output), source=str(src), corners_px=corners,
                            method='reviewed surface quadrilateral; other sides approximate'))
    measured = calibration['lsp_measured_size_m']
    size = cfg.get('lsp_size_m') or measured
    if not size or min(size) <= 0:
        raise ValueError('No valid LSP size')
    thickness=cfg.get('lsp_thickness_m',.015)
    if not 0<thickness<=.2:raise ValueError('LSP thickness must be in (0, 0.2] metres')
    write_json(work/'lsp_params.json', dict(size_x_m=size[0],size_y_m=size[1],
        thickness_m=thickness,corner_radius_m=.045,roughness=.44,measured_size_m=measured,
        thickness_source='user estimate via config.lsp_thickness_m' if 'lsp_thickness_m' in cfg else 'inferred default',
        notes='XY from existing MoGe calibration, config override takes priority. Thickness follows the user estimate when configured; corner radius remains inferred.'))
    floor_samples = {}
    for crop in ['rtsp_010_forklift_0.png','rtsp_010_cargo_0.png','frame_07_cargo_0.png']:
        rec = next(r for r in records if Path(r['crop']).name == crop)
        floor_samples[crop] = dict(bottom_center_px=rec['bottom_center_px'],
            floor_position_m=floor_point(rec['bottom_center_px'],camera).tolist(),
            caveat='Approximate contact point; full 3D dimensions cannot be inferred from this point alone.')
    inputs = [Path('config.json'), *(work/n for n in ['camera.json','scene_facts.json','calibration.json','points_world.npy','refs/index.json'])]
    manifest = dict(camera_id=cfg['camera_id'],sources=sources,floor_samples=floor_samples,
        input_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs},
        limitations=['SKID masks show fragments only: pallet geometry uses a provisional 1.2 x 1.0 x 0.15 m prior, no claimed real texture.',
                    'Forklift and cargo dimensions are visually fitted priors; MoGe point map describes the background, not these moving objects.',
                    'Unseen sides use shared surface patches; this is an authored reconstruction, not a multi-view scan.',
                    'Background is 1920x1080 and contains existing objects and people; it is not a clean compositing plate.'])
    write_json(work/'asset_sources.json',manifest)
    print(f'Prepared {len(sources)} real-image textures; existing calibration preserved.')


if __name__ == '__main__':
    run()
