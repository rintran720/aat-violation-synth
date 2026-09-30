"""Composite the single V1 render, verify both sheets are visible, and make review images."""
import json
import argparse
import io
import subprocess
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont, JpegImagePlugin

from synth.common import load_config


def srgb_to_linear(v):
    return np.where(v<=.04045,v/12.92,((v+.055)/1.055)**2.4)


def linear_to_srgb(v):
    v=np.clip(v,0,1)
    return np.where(v<=.0031308,v*12.92,1.055*v**(1/2.4)-.055)


def blur(array,sigma):
    if sigma<=0:return array.copy()
    radius=max(1,int(np.ceil(3*sigma)))
    x=np.arange(-radius,radius+1);kernel=np.exp(-x*x/(2*sigma*sigma));kernel/=kernel.sum()
    out=array
    for axis in [0,1]:
        padding=[(0,0)]*array.ndim;padding[axis]=(radius,radius)
        padded=np.pad(out,padding,mode='edge');out=np.zeros_like(out)
        for offset,weight in enumerate(kernel):
            slices=[slice(None)]*array.ndim;slices[axis]=slice(offset,offset+array.shape[axis])
            out+=weight*padded[tuple(slices)]
    return out


def radial_warp(array,k1):
    """Sample RGBA through an inverse Brown radial map (negative k1 is barrel)."""
    if not k1:return array.copy()
    h,w=array.shape[:2];yy,xx=np.mgrid[0:h,0:w].astype(np.float32)
    xn=(xx-(w-1)/2)/((w-1)/2);yn=(yy-(h-1)/2)/((h-1)/2)
    radius2=xn*xn+yn*yn;factor=np.maximum(1+k1*radius2,.2)
    # Inverse sample: source points map forward by r_d=r_s(1+k1*r_s^2).
    sx=(w-1)/2+xn/factor*(w-1)/2;sy=(h-1)/2+yn/factor*(h-1)/2
    x0=np.floor(sx).astype(int);y0=np.floor(sy).astype(int);dx=sx-x0;dy=sy-y0
    out=np.zeros_like(array);valid=(x0>=0)&(y0>=0)&(x0+1<w)&(y0+1<h)
    x0c=np.clip(x0,0,w-2);y0c=np.clip(y0,0,h-2)
    for c in range(array.shape[2]):
        a=array[y0c,x0c,c]*(1-dx)+array[y0c,x0c+1,c]*dx
        b=array[y0c+1,x0c,c]*(1-dx)+array[y0c+1,x0c+1,c]*dx
        out[...,c]=np.where(valid,a*(1-dy)+b*dy,0)
    return out


def erode_alpha(alpha,pixels):
    for _ in range(max(0,int(pixels))):
        padded=np.pad(alpha[...,0],1,mode='constant')
        alpha[...,0]=np.minimum.reduce([padded[y:y+alpha.shape[0],x:x+alpha.shape[1]]
                                        for y in range(3) for x in range(3)])
    return alpha


def match_local_illuminant(colour,background,strength=.45,tile=48):
    """Gently inherit the blue/green cast of the nearby CCTV floor/walls."""
    h,w=background.shape[:2];linear_bg=srgb_to_linear(background.astype(np.float32)/255)
    gains=np.ones_like(linear_bg)
    for y in range(0,h,tile):
        for x in range(0,w,tile):
            patch=linear_bg[y:y+tile,x:x+tile]
            mean=patch.mean(axis=(0,1));neutral=mean.mean()
            if neutral>1e-5:
                cast=np.clip(mean/neutral,.72,1.28)
                gains[y:y+tile,x:x+tile]=1+strength*(cast-1)
    return colour*gains


def h264_roundtrip(image,dest,stem,crf=28):
    """Encode a short 5-fps H.264 clip and return its last decoded frame."""
    source=dest/f'.{stem}_h264_source.png';video=dest/f'{stem}_h264.mp4'
    Image.fromarray(image).save(source)
    try:
        subprocess.run(['ffmpeg','-hide_banner','-loglevel','error','-y','-loop','1','-framerate','5',
                        '-i',str(source),'-t','1','-an','-c:v','libx264','-preset','medium',
                        '-crf',str(int(crf)),'-pix_fmt','yuv420p','-movflags','+faststart',str(video)],
                       check=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
        decoded=subprocess.run(['ffmpeg','-hide_banner','-loglevel','error','-i',str(video),
                                '-vf','select=eq(n\\,4)','-vsync','0','-frames:v','1',
                                '-f','image2pipe','-vcodec','png','pipe:1'],check=True,
                              stdout=subprocess.PIPE,stderr=subprocess.PIPE).stdout
        frame=Image.open(io.BytesIO(decoded)).convert('RGB')
        return np.asarray(frame),video
    finally:
        source.unlink(missing_ok=True)


def composite(background,rgba,sigma=.65,protected_boxes=(),saturation=1.0,object_mask=None,
              noise_sigma=.002,alpha_erosion_px=0,lens_k1=0.0,ambient_strength=.45):
    """Blur premultiplied linear colour to prevent dark alpha fringes; preserve other pixels."""
    src=background.astype(np.float32)/255
    rgba=radial_warp(rgba.astype(np.float32),lens_k1)
    alpha=rgba[...,3:4]/255
    alpha[alpha<2/255]=0
    colour=srgb_to_linear(np.clip(rgba[...,:3]/255,0,1))
    if ambient_strength and object_mask is not None:
        adapted=match_local_illuminant(colour,background,ambient_strength)
        colour=np.where(object_mask[...,None],adapted,colour)
    # Roll off hot rendered whites before they clip like a paper-white CG roof.
    highlight=colour/(1+.65*np.maximum(colour-.68,0))
    colour=np.where(object_mask[...,None],highlight,colour) if object_mask is not None else colour
    if not 0<=saturation<=1:raise ValueError('saturation must be between 0 and 1')
    if saturation!=1:
        luminance=np.sum(colour*np.array([.2126,.7152,.0722]),axis=-1,keepdims=True)
        graded=luminance+saturation*(colour-luminance)
        colour=np.where(object_mask[...,None],graded,colour) if object_mask is not None else graded
    alpha=erode_alpha(alpha,alpha_erosion_px)
    premult=blur(colour*alpha,sigma);alpha=blur(alpha,sigma)
    for x0,y0,x1,y1 in protected_boxes:
        alpha[y0:y1,x0:x1]=0;premult[y0:y1,x0:x1]=0
    result=linear_to_srgb(premult+srgb_to_linear(src)*(1-alpha))
    # Low-amplitude seeded sensor noise, only where a rendered object/shadow contributes.
    rng=np.random.default_rng(7)
    if noise_sigma>0:result+=rng.normal(0,noise_sigma,result.shape).astype(np.float32)*alpha
    out=np.clip(np.rint(result*255),0,255).astype('uint8')
    support=alpha[...,0]>1e-5
    out[~support]=background[~support]
    return out,support


def main(output_stem='v_000'):
    cfg=load_config();work=cfg['work'];dest=work/'out'
    source=Image.open(cfg['background_image']);background=np.array(source.convert('RGB'))
    rgba=np.array(Image.open(work/'renders'/f'{output_stem}.png').convert('RGBA'))
    assert rgba.shape[:2]==background.shape[:2], 'Calibrated render/source dimensions mismatch'
    h,w=background.shape[:2]
    meta=json.loads((dest/f'{output_stem}.json').read_text())
    ids=np.array(Image.open(work/'renders'/f'{output_stem}_instance_ids.png').convert('RGBA'))
    assert ids.shape==rgba.shape
    fork_colour=np.array(meta['fork_visibility_color_rgb'])
    fork_mask=(np.max(np.abs(ids[...,:3].astype(int)-fork_colour),axis=-1)<8)&(ids[...,3]>240)
    visible_fork_pixels=int(fork_mask.sum())
    assert visible_fork_pixels==0,f'Fork blades still visible: {visible_fork_pixels} pixels'
    rear_colour=np.array(meta['rear_wheel_visibility_color_rgb'])
    rear_mask=(np.max(np.abs(ids[...,:3].astype(int)-rear_colour),axis=-1)<8)&(ids[...,3]>240)
    rear_pixels=int(rear_mask.sum())
    assert rear_pixels>150,f'Rear wheels hidden in actual CCTV view: {rear_pixels} pixels'
    Image.fromarray(rear_mask.astype('uint8')*255).save(work/'renders'/f'{output_stem}_rear_wheels_visible.png')
    masks={}
    for obj in meta['objects']:
        colour=np.array(obj['mask_color_rgb'])
        mask=(np.max(np.abs(ids[...,:3].astype(int)-colour),axis=-1)<8)&(ids[...,3]>240)
        if obj['name']=='forklift':mask|=rear_mask
        y,x=np.nonzero(mask)
        assert len(x)>50,f"{obj['name']}: invisible object"
        obj['visible_pixels']=len(x);obj['visible_bbox_2d']=[int(x.min()),int(y.min()),int(x.max()+1),int(y.max()+1)]
        masks[obj['name']]=mask
        Image.fromarray(mask.astype('uint8')*255).save(work/'renders'/f"{output_stem}_{obj['name']}_visible.png")
    lsps=[o for o in meta['objects'] if o['class']=='lsp']
    assert len(lsps)==meta['lsp_count'] and all(o['visible_pixels']>300 for o in lsps), 'Every expected LSP must be visible'
    # The old config's OSD boxes were authored for 960x540; protect the full top strip
    # as well, without silently changing that calibration-independent config.
    protected=[tuple(map(int,b)) for b in cfg['osd_boxes']]+[(0,0,w,max(54,int(h*.05)))]
    saturation=cfg['composite'].get('object_saturation',1.0)
    comp=cfg['composite'];sigma=comp.get('blur_sigma',.65)
    noise_sigma=comp.get('noise_sigma')
    if noise_sigma is None:noise_sigma=.002
    erosion=comp.get('alpha_erosion_px',1);lens_k1=comp.get('lens_k1',-.035)
    ambient_strength=comp.get('ambient_strength',.45)
    result,support=composite(background,rgba,sigma=sigma,protected_boxes=protected,
                             saturation=saturation,object_mask=ids[...,3]>0,
                             noise_sigma=noise_sigma,alpha_erosion_px=erosion,lens_k1=lens_k1,
                             ambient_strength=ambient_strength)
    Image.fromarray(result).save(dest/f'{output_stem}.png')
    comp=cfg['composite'];crf=int(comp.get('h264_crf',28))
    encoded_result,h264_path=h264_roundtrip(result,dest,output_stem,crf)
    jpeg_options={'subsampling':JpegImagePlugin.get_sampling(source)}
    if source.quantization:jpeg_options['qtables']=source.quantization
    else:jpeg_options['quality']=95
    Image.fromarray(encoded_result).save(dest/f'{output_stem}.jpg',**jpeg_options)
    Image.fromarray(support.astype('uint8')*255).save(work/'renders'/f'{output_stem}_composite_support.png')
    outside_unchanged=bool(np.array_equal(result[~support],background[~support]))
    osd_unchanged=bool(np.array_equal(result[:54],background[:54]))
    assert outside_unchanged and osd_unchanged
    meta['image_size']=[w,h];meta['composite']={'space':'linear','blur_sigma':sigma,'noise_sigma':noise_sigma,
        'alpha_erosion_px':erosion,'lens_k1':lens_k1,'lens_distortion_calibrated':False,
        'ambient_strength':ambient_strength,
        'object_saturation':saturation,
        'jpeg_quantization_matched':bool(source.quantization),'protected_boxes':protected,
        'background_unchanged_outside_support_in_png':outside_unchanged,'top_strip_unchanged_in_png':osd_unchanged,
        'h264':{'encoded':True,'codec':'libx264','crf':crf,'fps':5,'frames':5,
                'video':str(h264_path),'jpeg_is_decoded_h264_frame':True},
        'note':'Lens distortion is an estimated Brown radial correction, not camera-calibrated. PNG is the lossless pre-encode compositing master; JPEG is decoded from a 5-fps H.264 clip then saved with source JPEG quantization.'}
    meta['verification']={'all_expected_lsps_visible':True,'visible_lsp_pixels':{o['name']:o['visible_pixels'] for o in lsps},
                          'visible_fork_blade_pixels':visible_fork_pixels,'fork_blades_fully_occluded':True,
                          'visible_rear_wheel_pixels':rear_pixels,'rear_wheels_visible_in_cctv':True,
                          'is_violation_matches_lsp_count':meta['is_violation']==(meta['lsp_count']>=2)}
    (dest/f'{output_stem}.json').write_text(json.dumps(meta,indent=2))
    review=Image.fromarray(encoded_result);draw=ImageDraw.Draw(review)
    font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',24)
    box=tuple(meta['event']['bbox_2d']);draw.rectangle(box,outline='#ff5c3a',width=4)
    kind='violation' if meta['is_violation'] else 'valid'
    sheet_word='LSP' if meta['lsp_count']==1 else 'LSPs'
    text=f"{meta['violation_id']} | {kind} | {meta['lsp_count']} {sheet_word} | SYNTHETIC PREVIEW"
    draw.rectangle((box[0],box[1]-36,box[0]+810,box[1]),fill='#161c24')
    draw.text((box[0]+8,box[1]-33),text,font=font,fill='white')
    for i,obj in enumerate(lsps,1):
        y,x=np.nonzero(masks[obj['name']]);xx=int(np.median(x));yy=int(np.median(y))
        draw.ellipse((xx-15,yy-15,xx+15,yy+15),fill='#19a6bc',outline='white',width=2)
        draw.text((xx-8,yy-16),str(i),font=font,fill='white')
    review.save(dest/f'{output_stem}_review.jpg',quality=95)
    # Useful at native pixel density for checking the sheet edges and contact/shadows.
    x0,y0,x1,y1=[int(v) for v in box];pad=45
    Image.fromarray(encoded_result).crop((max(0,x0-pad),max(0,y0-pad),min(w,x1+pad),min(h,y1+pad))).save(dest/f'{output_stem}_detail.png')
    print(json.dumps({'image':str(dest/f'{output_stem}.jpg'),'size':[w,h],**meta['verification']},indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output-stem',default='v_000')
    main(parser.parse_args().output_stem)
