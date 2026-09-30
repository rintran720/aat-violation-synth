"""Convert a CLI-Anything export into a metre-scale asset and render review views.

Called by synth.build_assets under Blender 5.2.2. Studio floor/camera/lights are
added AFTER saving the reusable asset, so they cannot leak into batch scenes.
"""
import ast
import json
import math
import struct
import sys
from pathlib import Path

import bpy
from mathutils import Vector


def execute_without_render(path):
    tree=ast.parse(Path(path).read_text(),filename=str(path))
    def is_render(node):
        return (isinstance(node,ast.Expr) and isinstance(node.value,ast.Call)
                and ast.unparse(node.value.func)=='bpy.ops.render.render')
    removed=sum(is_render(n) for n in tree.body)
    if removed!=1:raise ValueError(f'Expected exactly one harness render call, got {removed}')
    tree.body=[node for node in tree.body if not is_render(node)]
    exec(compile(tree,str(path),'exec'),{'__name__':'__main__'})


def bounds(objects):
    points=[obj.matrix_world @ Vector(c) for obj in objects for c in obj.bound_box]
    return Vector([min(p[i] for p in points) for i in range(3)]),Vector([max(p[i] for p in points) for i in range(3)])


def normalize_asset(name):
    scene=bpy.context.scene
    meshes=[obj for obj in scene.objects if obj.type=='MESH']
    bpy.context.view_layer.update()
    lo,hi=bounds(meshes)
    shift=Vector(((lo.x+hi.x)/2,(lo.y+hi.y)/2,lo.z))
    collection=bpy.data.collections.new(name)
    scene.collection.children.link(collection)
    for obj in meshes:
        obj.location-=shift
        for parent in list(obj.users_collection):parent.objects.unlink(obj)
        collection.objects.link(obj)
        bpy.ops.object.select_all(action='DESELECT');obj.select_set(True)
        bpy.context.view_layer.objects.active=obj
        # CLI scales a unit cube; apply scale BEFORE bevel so widths are actual metres.
        bpy.ops.object.transform_apply(location=False,rotation=False,scale=True)
        if obj.get('rounded_tire_profile'):
            for face in obj.data.polygons:face.use_smooth=True
        elif 'tire' in obj.name or 'hub' in obj.name:
            for face in obj.data.polygons:face.use_smooth=abs(face.normal.z)<.5
    bpy.context.view_layer.update()
    lo,hi=bounds(meshes)
    collection['asset_name']=name;collection['units']='metres';collection['front']='+Y'
    collection['origin']='footprint centre on floor';collection.asset_mark()
    bpy.ops.object.select_all(action='DESELECT')
    for obj in meshes:obj.select_set(True)
    bpy.context.view_layer.objects.active=meshes[0]
    return meshes,collection,lo,hi


TEXTURES={'tex_yellow':'forklift_yellow.png','tex_white':'uld_white_front.png',
          'tex_white_side':'uld_white_side.png','tex_dark':'uld_dark_front.png',
          'tex_carton':'carton_front.png','tex_carton_plain':'carton_plain.png'}
TEXTURES.update(tex_body_side='forklift_body_side.png',tex_counter_side='forklift_counter_side.png',
                tex_side_panel='forklift_side_panel.png',tex_roof='forklift_roof.png',
                tex_white_roof='uld_white_roof.png',tex_dark_roof='uld_dark_roof.png')

# User-reviewed correction: the real forklift paint is darker and richer than
# the initial reconstruction. Apply in linear material colour, not to the CCTV.
FORKLIFT_PAINT_GAIN=.55
FORKLIFT_PAINT_MATERIALS={'yellow','tex_yellow','tex_body_side','tex_counter_side'}


def set_glb_paint_factors(path):
    """Preserve paint gain in glTF, whose exporter drops Blender colour-mix nodes."""
    raw=path.read_bytes()
    magic,version,total=struct.unpack_from('<III',raw)
    assert magic==0x46546C67 and version==2 and total==len(raw)
    chunks=[];offset=12
    while offset<len(raw):
        length,kind=struct.unpack_from('<II',raw,offset)
        data=raw[offset+8:offset+8+length];offset+=8+length
        if kind==0x4E4F534A:
            document=json.loads(data)
            for mat in document.get('materials',[]):
                name=mat.get('name')
                if name not in FORKLIFT_PAINT_MATERIALS:continue
                pbr=mat.setdefault('pbrMetallicRoughness',{})
                if name=='yellow':
                    base=bpy.data.materials[name].node_tree.nodes.get('Principled BSDF').inputs['Base Color'].default_value
                    pbr['baseColorFactor']=[c*.75 for c in base[:3]]+[1]
                else:
                    assert 'baseColorTexture' in pbr,(name,'Lost source texture')
                    pbr['baseColorFactor']=[FORKLIFT_PAINT_GAIN]*3+[1]
            data=json.dumps(document,separators=(',',':')).encode()
            data+=b' '*((-len(data))%4)
        chunks.append(struct.pack('<II',len(data),kind)+data)
    body=b''.join(chunks)
    path.write_bytes(struct.pack('<III',magic,version,len(body)+12)+body)


def texture_meshes(meshes,work):
    # A decal-bearing side photograph belongs only on the matching vertical face.
    # Keep top/rear body faces on the separate paint patch instead of repeating labels.
    for obj in meshes:
        # The front decal crop contains the container ID. Reusing that material
        # on the shell's side faces stretches and duplicates the printed text.
        # Route side/rear faces to the separate real side crop instead.
        if obj.name=='container_shell' and 'tex_white_side' in bpy.data.materials:
            side=bpy.data.materials['tex_white_side']
            if side.name not in [m.name for m in obj.data.materials]:obj.data.materials.append(side)
            side_index=next(i for i,m in enumerate(obj.data.materials) if m.name==side.name)
            for face in obj.data.polygons:
                if abs(face.normal.x)>.9 or face.normal.y<-.9:
                    face.material_index=side_index
        if obj.name=='rear_counterweight':
            obj.data.materials.append(bpy.data.materials['tex_counter_side'])
            for face in obj.data.polygons:
                if abs(face.normal.x)>.9:face.material_index=len(obj.data.materials)-1
    used=set(mat for obj in meshes for mat in obj.data.materials if mat)
    textured=[]
    for mat in used:
        filename=TEXTURES.get(mat.name)
        if filename:
            image=bpy.data.images.load(str(work/'textures'/filename),check_existing=True);image.pack()
            nodes=mat.node_tree.nodes; shader=next(n for n in nodes if n.type=='BSDF_PRINCIPLED')
            tex=nodes.new('ShaderNodeTexImage');tex.image=image;tex.interpolation='Linear'
            colour=tex.outputs['Color']
            if mat.name in FORKLIFT_PAINT_MATERIALS:
                multiply=nodes.new('ShaderNodeMixRGB');multiply.blend_type='MULTIPLY'
                multiply.label='Darker source-matched forklift paint'
                multiply.inputs[0].default_value=1
                multiply.inputs[2].default_value=(FORKLIFT_PAINT_GAIN,)*3+(1,)
                mat.node_tree.links.new(colour,multiply.inputs[1]);colour=multiply.outputs[0]
                shader.inputs['Specular IOR Level'].default_value=.25
            mat.node_tree.links.new(colour,shader.inputs['Base Color'])
            shader.inputs['Roughness'].default_value=max(.72,shader.inputs['Roughness'].default_value)
            shader.inputs['Metallic'].default_value=min(.12,shader.inputs['Metallic'].default_value)
            textured.append(filename)
        elif mat.name in {'yellow','steel','silver','rubber'}:
            # Small-scale wear breaks up pristine constant paint/metal. These are
            # inferred roughness variations, not claimed recovered surface geometry.
            nodes=mat.node_tree.nodes;links=mat.node_tree.links
            shader=nodes.get('Principled BSDF');base=tuple(shader.inputs['Base Color'].default_value)
            if mat.name=='yellow':
                base=tuple(c*FORKLIFT_PAINT_GAIN for c in base[:3])+(1,)
                shader.inputs['Base Color'].default_value=base
                shader.inputs['Specular IOR Level'].default_value=.25
            noise=nodes.new('ShaderNodeTexNoise');noise.inputs['Scale'].default_value=18
            noise.inputs['Detail'].default_value=3
            ramp=nodes.new('ShaderNodeValToRGB')
            ramp.color_ramp.elements[0].position=.22;ramp.color_ramp.elements[0].color=tuple(c*.48 for c in base[:3])+(1,)
            ramp.color_ramp.elements[1].position=.78;ramp.color_ramp.elements[1].color=base
            links.new(noise.outputs['Fac'],ramp.inputs[0]);links.new(ramp.outputs[0],shader.inputs['Base Color'])
            shader.inputs['Roughness'].default_value=.78
            shader.inputs['Metallic'].default_value=min(.25,shader.inputs['Metallic'].default_value)
        elif mat.name=='wood':
            # Procedural timber, explicitly provisional (SAM3 did not provide a full pallet).
            nodes=mat.node_tree.nodes;links=mat.node_tree.links
            shader=next(n for n in nodes if n.type=='BSDF_PRINCIPLED')
            coord=nodes.new('ShaderNodeTexCoord');scale=nodes.new('ShaderNodeVectorMath');scale.operation='MULTIPLY'
            scale.inputs[1].default_value=(35,1.5,4)
            noise=nodes.new('ShaderNodeTexNoise');noise.inputs['Scale'].default_value=3
            noise.inputs['Detail'].default_value=2
            ramp=nodes.new('ShaderNodeValToRGB')
            ramp.color_ramp.elements[0].color=(.14,.078,.025,1)
            ramp.color_ramp.elements[1].color=(.47,.32,.15,1)
            links.new(coord.outputs['Generated'],scale.inputs[0]);links.new(scale.outputs[0],noise.inputs['Vector'])
            links.new(noise.outputs['Fac'],ramp.inputs[0]);links.new(ramp.outputs[0],shader.inputs['Base Color'])
    for obj in meshes:
        if not any(m and m.name in TEXTURES for m in obj.data.materials):continue
        me=obj.data;uv=me.uv_layers.active or me.uv_layers.new(name='UVMap')
        coords=[v.co for v in me.vertices]
        lo=Vector([min(v[i] for v in coords) for i in range(3)])
        hi=Vector([max(v[i] for v in coords) for i in range(3)])
        span=hi-lo
        for face in me.polygons:
            axis=max(range(3),key=lambda i:abs(face.normal[i]))
            u,v={0:(1,2),1:(0,2),2:(0,1)}[axis]
            for index in face.loop_indices:
                co=me.vertices[me.loops[index].vertex_index].co
                uv_x=(co[u]-lo[u])/max(span[u],1e-8)
                if (axis==1 and face.normal.y>0) or (axis==0 and face.normal.x<0):
                    uv_x=1-uv_x  # readable labels when viewed from outside either face
                uv.data[index].uv=(uv_x,(co[v]-lo[v])/max(span[v],1e-8))
    return sorted(set(textured))


def material(name,color,rough=.6):
    mat=bpy.data.materials.new(name);mat.use_nodes=True
    p=mat.node_tree.nodes.get('Principled BSDF');p.inputs['Base Color'].default_value=(*color,1)
    p.inputs['Roughness'].default_value=rough
    return mat


def preview(meshes,name,stage,work,lo,hi,alternate=False):
    scene=bpy.context.scene
    studio=bpy.data.collections.get('PREVIEW_ONLY')
    if studio:
        for obj in list(studio.objects):bpy.data.objects.remove(obj,do_unlink=True)
        bpy.data.collections.remove(studio)
    studio=bpy.data.collections.new('PREVIEW_ONLY');scene.collection.children.link(studio)
    def add(obj):
        for col in list(obj.users_collection):col.objects.unlink(obj)
        studio.objects.link(obj)
    bpy.ops.mesh.primitive_plane_add(size=200,location=(0,0,-.012));floor=bpy.context.object
    floor.name='preview_floor';floor.data.materials.append(material('preview_floor',(.105,.12,.14)));add(floor)
    target=Vector((0,0,(hi.z-lo.z)*.45))
    pitch=math.radians(json.loads((work/'scene_facts.json').read_text())['camera_pitch_down_deg'])
    span=max(hi.x-lo.x,hi.y-lo.y,hi.z-lo.z)
    # ORTHO review avoids perspective hiding small parts; pitch matches the CCTV camera.
    distance=span*3.4;direction=Vector((-1.0,-1.3,0) if alternate else (1.0,1.4,0)).normalized()
    location=target+direction*distance+Vector((0,0,distance*math.tan(pitch)))
    bpy.ops.object.camera_add(location=location);camera=bpy.context.object;add(camera)
    camera.rotation_euler=(target-camera.location).to_track_quat('-Z','Y').to_euler()
    camera.data.type='ORTHO';scene.camera=camera
    rotation=camera.rotation_euler.to_matrix().transposed()
    projected=[rotation @ (obj.matrix_world @ Vector(c)-target) for obj in meshes for c in obj.bound_box]
    projected_width=max(p.x for p in projected)-min(p.x for p in projected)
    projected_height=max(p.y for p in projected)-min(p.y for p in projected)
    camera.data.ortho_scale=max(projected_width,projected_height*4/3)*1.24
    for key,loc,power,size in [('key',(4,3,7),1100,5),('fill',(-4,1,4),750,4),('rim',(0,-4,6),1000,3)]:
        bpy.ops.object.light_add(type='AREA',location=loc);light=bpy.context.object;add(light)
        light.name='preview_'+key;light.data.energy=power;light.data.shape='DISK';light.data.size=size
        light.rotation_euler=(target-light.location).to_track_quat('-Z','Y').to_euler()
    scene.world.use_nodes=True
    scene.world.node_tree.nodes['Background'].inputs[0].default_value=(.22,.24,.28,1)
    scene.world.node_tree.nodes['Background'].inputs[1].default_value=.4
    scene.render.engine='CYCLES';scene.cycles.device='CPU';scene.cycles.samples=24 if stage==3 else 10
    scene.cycles.use_denoising=True;scene.cycles.seed=7
    scene.render.resolution_x=800 if stage==3 else 480
    scene.render.resolution_y=600 if stage==3 else 360;scene.render.resolution_percentage=100
    scene.render.image_settings.file_format='PNG';scene.render.image_settings.color_mode='RGB'
    scene.render.film_transparent=False;scene.view_settings.view_transform='AgX'
    suffix='_rear' if alternate else ''
    scene.render.filepath=str(work/'previews'/f'{name}_pass{stage}{suffix}.png')
    bpy.ops.render.render(write_still=True)


def export_current(name,stage,work):
    scene=bpy.context.scene;scene.unit_settings.system='METRIC';scene.unit_settings.scale_length=1
    bpy.context.preferences.filepaths.save_version=0
    if name=='forklift' and stage>=2:
        sys.path.insert(0,str(Path(__file__).resolve().parent))
        from refine_forklift import refine_guard, refine_rear, refine_wheels, refine_side_panels
        refine_guard()
        refine_rear()
        refine_wheels()
        refine_side_panels()
    meshes,col,lo,hi=normalize_asset(name)
    edge_settings=[]
    if stage==3:
        sys.path.insert(0,str(Path(__file__).resolve().parent))
        from soften_assets import soften_edges
        edge_settings=soften_edges(meshes,name)
    if stage==3:
        bpy.ops.wm.save_as_mainfile(filepath=str(work/'assets_raw'/f'{name}.blend'))
    textures=texture_meshes(meshes,work) if stage==3 else []
    appearance={}
    if stage==3:
        from grade_assets import grade_materials
        config=json.loads(Path('config.json').read_text())
        factor=config.get('asset_appearance',{}).get('saturation_multiplier',1.2)
        appearance=grade_materials(meshes,work,factor,config.get('asset_appearance',{}))
        textures=sorted({Path(n.image.filepath).name for obj in meshes for mat in obj.data.materials
            if mat for n in mat.node_tree.nodes if n.type=='TEX_IMAGE' and n.image})
    for screen in bpy.data.screens:
        for area in screen.areas:
            if area.type=='VIEW_3D':
                area.spaces.active.region_3d.view_distance=max(hi-lo)*2.1
                area.spaces.active.region_3d.view_location=(lo+hi)*.5
                area.spaces.active.shading.type='MATERIAL'
    if stage==3:
        bpy.ops.wm.save_as_mainfile(filepath=str(work/'assets'/f'{name}.blend'))
        # glTF cannot encode these procedural variations. Preserve their authored
        # base colours instead of allowing the exporter to silently make them white.
        restore=[]
        for mat in {m for obj in meshes for m in obj.data.materials if m}:
            if mat.name not in {'wood','steel','silver','rubber'}:continue
            shader=mat.node_tree.nodes.get('Principled BSDF')
            for link in list(shader.inputs['Base Color'].links):
                restore.append((mat.node_tree,link.from_socket,link.to_socket))
                mat.node_tree.links.remove(link)
        bpy.ops.export_scene.gltf(filepath=str(work/'assets'/f'{name}.glb'),export_format='GLB',
            use_selection=True,export_apply=True,export_yup=True)
        if name=='forklift':set_glb_paint_factors(work/'assets'/f'{name}.glb')
        for tree,source,target in restore:tree.links.new(source,target)
    info=dict(name=name,pass_number=stage,dimensions_m=list(hi-lo),bounds_min=list(lo),bounds_max=list(hi),
        mesh_objects=len(meshes),collection=col.name,units='metres',front='+Y',
        packed_textures=textures,source_project=f'work/cli/{name}_pass{stage}.blend-cli.json',
        caveat='LSP XY is MoGe-derived; other dimensions and unseen surfaces are fitted assumptions. Human realism review pending.')
    info['component_bounds']={obj.name:dict(min=list(bounds([obj])[0]),max=list(bounds([obj])[1]))
                              for obj in meshes if obj.name.startswith(('fork_','carriage'))}
    info['edge_softening']=edge_settings
    info['appearance']=appearance
    if name=='forklift':
        info['paint_linear_gain']=FORKLIFT_PAINT_GAIN
        info['roof_beacon']=False
        roof=next(obj for obj in meshes if obj.name=='roof_rear_panel')
        info['overhead_guard']={'roof_top_m':bounds([roof])[1].z,'roof_lowered_m':.26,
            'curved_posts':2 if stage>=2 else 0,'geometry_source':'visually fitted to source crops, not physical measurements'}
        seat=next(obj for obj in meshes if obj.name=='seat_base')
        seat_back=next(obj for obj in meshes if obj.name=='seat_back')
        info['cabin']={'seat_surface_m':bounds([seat])[1].z,
            'seat_to_roof_clearance_m':bounds([roof])[0].z-bounds([seat])[1].z,
            'seat_back_to_roof_clearance_m':bounds([roof])[0].z-bounds([seat_back])[1].z,
            'seat_surface_lowered_from_r15_m':1.17-bounds([seat])[1].z,
            'side_panel_section':'removed by user request'}
        mast=[obj for obj in meshes if obj.name.startswith('mast_') or obj.name=='hydraulic_ram']
        backrest=[obj for obj in meshes if obj.name.startswith('load_backrest_')]
        mast_lo,mast_hi=bounds(mast)
        backrest_lo,backrest_hi=bounds(backrest)
        body_lo,body_hi=bounds(backrest)
        info['lift_assembly']={'horizontal_forks':2,'boarding_steps':False,
            'mast_height_m':mast_hi.z-mast_lo.z,'mast_top_m':mast_hi.z,
            'backrest_width_m':backrest_hi.x-backrest_lo.x,
            'body_width_m':body_hi.x-body_lo.x,
            'backrest_height_m':backrest_hi.z-backrest_lo.z,
            'backrest_to_mast_height_ratio':(backrest_hi.z-backrest_lo.z)/(mast_hi.z-mast_lo.z),
            'backrest_mast_gap_m':backrest_lo.y-mast_hi.y}
        front=next(obj for obj in meshes if obj.name=='front_tire_1')
        rear=next(obj for obj in meshes if obj.name=='rear_tire_1')
        front_d=bounds([front])[1].z-bounds([front])[0].z
        rear_d=bounds([rear])[1].z-bounds([rear])[0].z
        info['shape_corrections']={'mast_top_m':bounds(mast)[1].z,
            'rear_plan_radius_m':.58 if stage>=2 else None,
            'rear_plan_depth_radius_m':.38 if stage>=2 else None,
            'rear_shortened_m':.20,'rear_tire_track_m':1.0,'rear_skirt_bottom_m':.14,
            'front_diameter_factor_from_r07':.90,'rear_wells_open_to_side':True,
            'front_wheels_moved_inward_m':.104,'front_tire_track_m':.962,
            'front_outward_shift_from_r09_m':.026,
            'front_fender_radial_clearance_m':.07,
            'rear_tail_top_m':bounds([bpy.data.objects['rear_counterweight']])[1].z,
            'cabin_frame_colour':'yellow',
            'cabin_front_footwell_colour':'black',
            'rim_face_recess_m':.031,'wheel_fasteners':'dark grey, roughness 0.88',
            'front_tire_diameter_m':front_d,'rear_tire_diameter_m':rear_d,
            'tire_profile':'rounded shoulders, 96 circumference segments',
            'front_tire_shoulder_radius_m':front.get('shoulder_radius_m'),
            'rear_tire_shoulder_radius_m':rear.get('shoulder_radius_m'),
            'front_to_rear_diameter_ratio':front_d/rear_d}
        info['source_correction']='Red patch above forklift is a worker shirt, not a roof lamp.'
    (work/'assets'/f'{name}.json').write_text(json.dumps(info,indent=2)) if stage==3 else None
    preview(meshes,name,stage,work,lo,hi)
    if stage==3:preview(meshes,name,stage,work,lo,hi,alternate=True)
    print('ASSET_OK '+json.dumps(info))


if __name__=='__main__':
    args=sys.argv[sys.argv.index('--')+1:]
    script,name,stage,folder=args
    execute_without_render(script)
    export_current(name,int(stage),Path(folder))
