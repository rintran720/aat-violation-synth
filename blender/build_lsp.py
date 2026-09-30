"""Build rounded thin LSP sheets using reviewed real textures and MoGe XY dimensions.

blender --background --threads 4 --python-exit-code 1 --python blender/build_lsp.py -- config.json
"""
import json
import math
import sys
from pathlib import Path

import bpy
import bmesh
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parent))
from export_asset import export_current


def build(cfg_path):
    cfg=json.loads(Path(cfg_path).read_text());work=Path(cfg['work_dir']).resolve()
    p=json.loads((work/'lsp_params.json').read_text())
    sx,sy,t,r=(p[k] for k in ['size_x_m','size_y_m','thickness_m','corner_radius_m'])
    assert 0<r<min(sx,sy)/2 and 0<t<=.2
    ring=[]
    for cx,cy,a0 in [(sx/2-r,sy/2-r,0),(-sx/2+r,sy/2-r,90),
                     (-sx/2+r,-sy/2+r,180),(sx/2-r,-sy/2+r,270)]:
        for i in range(9):
            a=math.radians(a0+90*i/8);ring.append((cx+r*math.cos(a),cy+r*math.sin(a)))
    n=len(ring)
    vertices=[(x,y,t) for x,y in ring]+[(x,y,0) for x,y in ring]
    faces=[list(range(n)),list(range(2*n-1,n-1,-1))]
    faces.extend([[i,n+i,n+(i+1)%n,(i+1)%n] for i in range(n)])
    # User-corrected marks: middle length x2, outer lengths x1.2 relative to r10,
    # configured height (within texture raster resolution), vertically centred.
    width,height=1024,256
    pixels=np.empty((height,width,4),dtype=np.float32)
    pixels[:]=(.143,.215,.293,1)  # sRGB equivalent of the original dark blue edge
    rng=np.random.default_rng(7)
    band_height=cfg.get('lsp_edge_markings',{}).get('height_m',.03)
    assert t>band_height
    y0,y1=round(height*(.5-band_height/(2*t))),round(height*(.5+band_height/(2*t)))
    band_widths=[]
    for centre,factor in zip([.43,.50,.57],[1.2,2.0,1.2]):
        previous_width=round(width*(centre+.018))-round(width*(centre-.018))
        length=round(previous_width*factor);x0=round(width*centre-length/2);x1=x0+length
        band_widths.append(length)
        base=np.array([.60,.61,.56])
        pixels[y0:y1,x0:x1,:3]=np.clip(base+rng.normal(0,.012,(y1-y0,x1-x0,1)),0,1)
    texture_path=work/'textures/lsp_edge_bands.png'
    for i in range(3):
        name=f'lsp_{i}';bpy.ops.wm.read_factory_settings(use_empty=True)
        mesh=bpy.data.meshes.new(name);mesh.from_pydata(vertices,[],faces);mesh.update()
        bm=bmesh.new();bm.from_mesh(mesh);bmesh.ops.recalc_face_normals(bm,faces=bm.faces);bm.to_mesh(mesh);bm.free()
        uv=mesh.uv_layers.new(name='UVMap')
        for loop in mesh.loops:
            co=mesh.vertices[loop.vertex_index].co;uv.data[loop.index].uv=(co.x/sx+.5,co.y/sy+.5)
        for face in mesh.polygons:
            if abs(face.normal.z)>.8:continue
            axis=1 if abs(face.normal.x)>abs(face.normal.y) else 0
            span=sy if axis==1 else sx
            for index in face.loop_indices:
                co=mesh.vertices[mesh.loops[index].vertex_index].co
                uv.data[index].uv=(co[axis]/span+.5,co.z/t)
        mat=bpy.data.materials.new('lsp_surface');mat.use_nodes=True
        shader=mat.node_tree.nodes.get('Principled BSDF');shader.inputs['Roughness'].default_value=p['roughness']
        image=bpy.data.images.load(str(work/'textures'/f'{name}.png'));image.pack()
        tex=mat.node_tree.nodes.new('ShaderNodeTexImage');tex.image=image
        mat.node_tree.links.new(tex.outputs['Color'],shader.inputs['Base Color'])
        edge=bpy.data.materials.new('lsp_edge');edge.use_nodes=True
        edge.node_tree.nodes.get('Principled BSDF').inputs['Base Color'].default_value=(.018,.038,.07,1)
        edge.node_tree.nodes.get('Principled BSDF').inputs['Roughness'].default_value=.6
        band_image=bpy.data.images.new('lsp_edge_bands',width=width,height=height,alpha=True)
        band_image.colorspace_settings.name='sRGB';band_image.pixels.foreach_set(pixels.ravel())
        band_image.filepath_raw=str(texture_path);band_image.file_format='PNG';band_image.save();band_image.pack()
        band_tex=edge.node_tree.nodes.new('ShaderNodeTexImage');band_tex.image=band_image
        edge.node_tree.links.new(band_tex.outputs['Color'],edge.node_tree.nodes.get('Principled BSDF').inputs['Base Color'])
        bottom=bpy.data.materials.new('lsp_bottom');bottom.use_nodes=True
        bottom.node_tree.nodes.get('Principled BSDF').inputs['Base Color'].default_value=(.018,.038,.07,1)
        mesh.materials.append(mat);mesh.materials.append(edge);mesh.materials.append(bottom)
        for face in mesh.polygons:face.material_index=0 if face.normal.z>.8 else (2 if face.normal.z<-.8 else 1)
        obj=bpy.data.objects.new(name,mesh);bpy.context.scene.collection.objects.link(obj)
        obj['edge_band_count_per_face']=3
        obj['edge_band_layout']='three centred rectangles; middle double length; outer 1.2x length; configured height'
        obj['edge_band_height_m']=band_height
        obj['edge_band_widths_px']=band_widths
        bpy.context.scene.world=bpy.data.worlds.new('World')
        export_current(name,3,work)
        metadata_path=work/'assets'/f'{name}.json'
        metadata=json.loads(metadata_path.read_text());metadata['source_project']=None
        metadata['generator']='blender/build_lsp.py'
        metadata['source_textures']=[f'{name}.png','lsp_edge_bands.png']
        metadata['edge_markings']={'bands_per_face':3,'faces':4,'layout':'central group of rectangles',
            'height_m':band_height,'length_factors_from_r10':[1.2,2.0,1.2],
            'widths_px':band_widths,'texture_size':[width,height],'source':'user corrected dimensions'}
        metadata['measurement']=p;metadata_path.write_text(json.dumps(metadata,indent=2))


if __name__=='__main__':build(sys.argv[sys.argv.index('--')+1])
