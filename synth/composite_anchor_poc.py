"""Composite the anchor-only render onto its photographic forklift background."""
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

from synth.composite_violation import composite


def restore_foreground(result, background, mask):
    """Restore source pixels for a foreground object where it overlaps the render."""
    if result.shape != background.shape or mask.shape != background.shape[:2]:
        raise ValueError('Foreground mask and image dimensions differ')
    restored = result.copy()
    restored[mask] = background[mask]
    return restored


def main(config_path,anchor_path,output_stem,foreground_mask_path=None,moge_points_path=None,render_stem=None):
    root=Path.cwd();cfg=json.loads(Path(config_path).read_text())
    anchor=json.loads(Path(anchor_path).read_text());work=root/cfg['work_dir']
    render_stem=render_stem or output_stem
    background=np.asarray(Image.open(root/anchor['background_image']).convert('RGB'))
    rgba=np.asarray(Image.open(work/'renders'/f'{render_stem}.png').convert('RGBA'))
    if rgba.shape[:2]!=background.shape[:2]:raise ValueError('Render and background dimensions differ')
    support_mask=rgba[...,3]>0
    composite_cfg=cfg['composite'];height,width=background.shape[:2]
    protected=[(0,0,width,max(54,int(height*.05)))]
    result,support=composite(background,rgba,sigma=composite_cfg.get('blur_sigma',.8),
        protected_boxes=protected,saturation=composite_cfg.get('object_saturation',.78),
        object_mask=support_mask,noise_sigma=composite_cfg.get('noise_sigma',.004),
        alpha_erosion_px=composite_cfg.get('alpha_erosion_px',1),
        lens_k1=composite_cfg.get('lens_k1',-.035),
        ambient_strength=composite_cfg.get('ambient_strength',.45))
    dest=work/'out';dest.mkdir(parents=True,exist_ok=True)
    Image.fromarray(result).save(dest/f'{output_stem}.png')
    Image.fromarray(result).save(dest/f'{output_stem}.jpg',quality=92,subsampling=1)
    support_png=(support.astype(np.uint8)*255)
    Image.fromarray(support_png).save(work/'renders'/f'{output_stem}_support.png')
    meta=json.loads((dest/f'{render_stem}.json').read_text())
    meta['output_stem']=output_stem
    foreground_info=None
    if foreground_mask_path:
        if not moge_points_path:
            raise ValueError('--moge-points is required with --foreground-mask to validate occlusion order')
        mask=np.asarray(Image.open(root/foreground_mask_path).convert('L'))>127
        points=np.load(root/moge_points_path)
        if points.shape[:2] != background.shape[:2] or points.shape[2] != 3:
            raise ValueError('MoGe point map dimensions differ from the background')
        valid=mask & np.isfinite(points).all(axis=-1) & (points[...,2]>0)
        forklift_depth=float(np.median(points[...,2][valid])) if valid.any() else None
        camera_cfg=json.loads((root/anchor['source_calibration']).read_text())
        camera_from_world=np.linalg.inv(np.asarray(camera_cfg['matrix_world'],dtype=float))
        cargo_depths=[]
        for obj in meta.get('objects',[]):
            if obj.get('class') != 'cargo':
                continue
            center=np.asarray(obj['location_world_m'],dtype=float)
            center[2]+=float(obj['dimensions_m'][2])/2
            camera_center=camera_from_world @ np.r_[center,1.0]
            cargo_depths.append(float(-camera_center[2]))
        if forklift_depth is None or not cargo_depths:
            raise ValueError('Could not derive valid forklift and cargo depths for occlusion order')
        cargo_depth=float(np.median(cargo_depths))
        if forklift_depth >= cargo_depth:
            raise ValueError(f'Forklift depth {forklift_depth:.3f}m is not in front of cargo depth {cargo_depth:.3f}m')
        result=restore_foreground(result,background,mask)
        foreground_info={
            'source_mask':str(foreground_mask_path),
            'moge_points':str(moge_points_path),
            'foreground_pixels':int(mask.sum()),
            'median_forklift_depth_m':round(forklift_depth,3),
            'median_cargo_depth_m':round(cargo_depth,3),
            'source_forklift_restored_over_render':True,
        }
        Image.fromarray(result).save(dest/f'{output_stem}.png')
        Image.fromarray(result).save(dest/f'{output_stem}.jpg',quality=92,subsampling=1)
    meta['composite']={'output_png':str((dest/f'{output_stem}.png').relative_to(root)),
        'output_jpg':str((dest/f'{output_stem}.jpg').relative_to(root)),
        'source_render':str((work/'renders'/f'{render_stem}.png').relative_to(root)),
        'rendered_support_pixels':int(support.sum()),
        'background_unchanged_outside_support':bool(np.array_equal(result[~support],background[~support])),
        'object_render_contains_forklift':False,
        'layer_order':'synthetic_objects_over_background',
        'foreground_occlusion':foreground_info}
    if foreground_info:
        meta['limitations']=[
            item for item in meta.get('limitations',[])
            if 'No real-forklift mask/depth holdout is applied yet' not in item
        ]
        meta.setdefault('limitations',[]).append(
            'Foreground depth ordering uses median MoGe depth per mask and cargo set, not per-pixel rendered depth; validate coordinate-frame alignment first.'
        )
    else:
        meta['limitations']=[
            item for item in meta.get('limitations',[])
            if 'No real-forklift mask/depth holdout is applied yet' not in item
            and 'Forklift mask compositing uses median MoGe depth' not in item
        ]
        meta.setdefault('limitations',[]).append(
            'Synthetic objects are composited over the source image throughout render support; this candidate does not use per-pixel occlusion.'
        )
        meta.setdefault('limitations',[]).append(
            'Raw same-frame MoGe points and Blender camera depth are different coordinate frames until registered; do not compare their Z values directly.'
        )
    if foreground_info and not np.array_equal(result[mask],background[mask]):
        raise AssertionError('Source forklift pixels were not fully restored over the synthetic render')
    if not meta['composite']['background_unchanged_outside_support']:
        raise AssertionError('Background changed outside render support')
    (dest/f'{output_stem}.json').write_text(json.dumps(meta,indent=2))
    print(json.dumps({'image':str(dest/f'{output_stem}.jpg'),'size':[width,height],
                      **meta['composite']},indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--config',default='config.json')
    parser.add_argument('--anchor',default='work/anchor_poc.json');parser.add_argument('--output-stem',default='anchor_poc_v6')
    parser.add_argument('--foreground-mask',help='SAM3 mask for a real object in front of the inserted render')
    parser.add_argument('--moge-points',help='Same-frame MoGe XYZ map used to validate foreground depth ordering')
    parser.add_argument('--render-stem',help='Render input stem; defaults to --output-stem')
    args=parser.parse_args();main(args.config,args.anchor,args.output_stem,args.foreground_mask,args.moge_points,args.render_stem)
