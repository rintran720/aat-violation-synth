"""Composite the anchor-only render onto its photographic forklift background."""
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

from synth.composite_violation import composite


def main(config_path,anchor_path,output_stem):
    root=Path.cwd();cfg=json.loads(Path(config_path).read_text())
    anchor=json.loads(Path(anchor_path).read_text());work=root/cfg['work_dir']
    background=np.asarray(Image.open(root/anchor['background_image']).convert('RGB'))
    rgba=np.asarray(Image.open(work/'renders'/f'{output_stem}.png').convert('RGBA'))
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
    meta=json.loads((dest/f'{output_stem}.json').read_text())
    meta['composite']={'output_png':str((dest/f'{output_stem}.png').relative_to(root)),
        'output_jpg':str((dest/f'{output_stem}.jpg').relative_to(root)),
        'rendered_support_pixels':int(support.sum()),
        'background_unchanged_outside_support':bool(np.array_equal(result[~support],background[~support])),
        'object_render_contains_forklift':False}
    if not meta['composite']['background_unchanged_outside_support']:
        raise AssertionError('Background changed outside render support')
    (dest/f'{output_stem}.json').write_text(json.dumps(meta,indent=2))
    print(json.dumps({'image':str(dest/f'{output_stem}.jpg'),'size':[width,height],
                      **meta['composite']},indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--config',default='config.json')
    parser.add_argument('--anchor',default='work/anchor_poc.json');parser.add_argument('--output-stem',default='anchor_poc_v6')
    args=parser.parse_args();main(args.config,args.anchor,args.output_stem)
