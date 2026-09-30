"""Bake selective model colour corrections; keep source crops intact."""
from pathlib import Path
import bpy
import numpy as np


def saturated(rgb,factor):
    # HSV saturation scaling at constant hue and value, vectorized over RGB.
    maximum=np.max(rgb,axis=-1,keepdims=True)
    return np.clip(maximum+factor*(rgb-maximum),0,1)


YELLOW_MATERIALS={'yellow','tex_yellow','tex_body_side','tex_counter_side'}
BLACK_MATERIALS={'black','roof_black','rubber','seat','steel','wheel_rim','wheel_fastener','tex_side_panel','tex_dark','tex_dark_roof'}


def selective_colour(rgb,material,settings,texture=False):
    rgb=np.asarray(rgb)
    strength=settings.get('yellow_strength',.2)
    gain=settings.get('black_gain',.8)
    maximum=np.max(rgb,axis=-1,keepdims=True)
    result=rgb.copy()
    if material in YELLOW_MATERIALS:
        # Move the paint towards warm yellow, preserving photographed shading.
        target=maximum*np.array([1.,.82,.01])
        weight=1.
        if texture:
            r,g,b=np.moveaxis(rgb,-1,0)
            yellow=(r>b+.04)&(g>b+.04)&(g>.5*r)&(g<1.3*r)
            weight=yellow[...,None].astype(float)
        result=rgb+(target-rgb)*strength*weight
    if texture:
        # Dark pixels receive a smooth shadow mask; bright labels stay bright.
        shadow=np.clip((.45-maximum)/.25,0,1)
        result*=1-(1-gain)*shadow
    elif material in BLACK_MATERIALS:
        result*=gain
    return np.clip(result,0,1)


def grade_materials(meshes,work,factor=1.2,settings=None):
    settings=settings or {}
    selective=settings.get('mode')=='selective_yellow_black'
    def grade(rgb,material,texture=False):
        base=selective_colour(rgb,material,settings,texture) if selective else rgb
        return saturated(base,factor)
    dest=work/'textures/graded';dest.mkdir(exist_ok=True)
    suffix=(f'_yellow20_black20_sat{round(factor*100)}' if selective else f'_sat{round(factor*100)}')
    images={}
    materials={m for obj in meshes for m in obj.data.materials if m}
    for mat in materials:
        for node in mat.node_tree.nodes:
            if node.type=='BSDF_PRINCIPLED':
                value=node.inputs['Base Color'].default_value
                value[:3]=grade(np.array(value[:3]),mat.name).tolist()
            elif node.type=='VALTORGB':
                for element in node.color_ramp.elements:
                    element.color[:3]=grade(np.array(element.color[:3]),mat.name).tolist()
            elif node.type=='TEX_IMAGE' and node.image:
                source=node.image
                key=(source.name,mat.name)
                if key not in images:
                    w,h=source.size;pixels=np.empty(w*h*4,dtype=np.float32)
                    source.pixels.foreach_get(pixels);pixels=pixels.reshape(h,w,4)
                    pixels[:,:,:3]=grade(pixels[:,:,:3],mat.name,texture=True)
                    target=bpy.data.images.new(source.name+suffix,width=w,height=h,alpha=True)
                    target.colorspace_settings.name=source.colorspace_settings.name
                    target.pixels.foreach_set(pixels.ravel())
                    target.filepath_raw=str(dest/(Path(source.filepath).stem+suffix+'.png'))
                    target.file_format='PNG';target.save();target.pack()
                    images[key]=target
                node.image=images[key]
        mat['model_saturation_multiplier']=factor
        mat['model_colour_mode']=settings.get('mode','saturation')
        mat['black_gain']=settings.get('black_gain',1.)
        mat['yellow_strength']=settings.get('yellow_strength',0.)
    return {**settings,'saturation_multiplier':factor,
            'method':'selective warm-yellow mix and dark-tone gain, then HSV saturation' if selective else 'HSV saturation, hue and value preserved',
            'source_images_preserved':True,'graded_textures':[Path(im.filepath).name for im in images.values()]}
