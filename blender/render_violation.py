"""Render one calibrated V1 scene and visible-instance masks, using existing assets.

blender --background --threads 4 --python-exit-code 1 --python blender/render_violation.py -- config.json
"""
import json
import argparse
import math
import sys
from pathlib import Path

import bpy
from bpy_extras.object_utils import world_to_camera_view
from mathutils import Matrix, Vector


def projected_bbox(scene, camera, points, width, height):
    ndc=[world_to_camera_view(scene,camera,p) for p in points]
    if any(p.z<=0 for p in ndc):raise ValueError('Object crosses the camera plane')
    pixels=[(p.x*width,(1-p.y)*height) for p in ndc]
    return [max(0,round(min(p[0] for p in pixels),1)),max(0,round(min(p[1] for p in pixels),1)),
            min(width,round(max(p[0] for p in pixels),1)),min(height,round(max(p[1] for p in pixels),1))]


def srgb_to_linear(value):
    value=float(value)
    return value/12.92 if value<=.04045 else ((value+.055)/1.055)**2.4


def scene_material(name,colour,emission=0):
    mat=bpy.data.materials.new(name);mat.diffuse_color=(*colour,1);mat.use_nodes=True
    shader=mat.node_tree.nodes.get('Principled BSDF')
    shader.inputs['Base Color'].default_value=(*colour,1)
    shader.inputs['Roughness'].default_value=.88
    if 'Emission Color' in shader.inputs:
        shader.inputs['Emission Color'].default_value=(*colour,1)
        shader.inputs['Emission Strength'].default_value=emission
    return mat


def add_occupant(forklift_transform):
    """Low-detail seated operator, matched to the dark short-sleeve reference crop."""
    shirt=scene_material('operator_dark_shirt',(.025,.022,.020))
    pants=scene_material('operator_dark_trousers',(.012,.014,.014))
    skin=scene_material('operator_skin',(.19,.105,.061))
    hair=scene_material('operator_hair',(.009,.008,.007))
    objects=[]
    # Seat centre is at local y=-0.88 m; move the complete pose onto its cushion.
    seat_y=-.36
    def point(v):return forklift_transform@Vector((v[0],v[1]+seat_y,v[2]))
    def sphere(name,local,scale,mat):
        bpy.ops.mesh.primitive_uv_sphere_add(segments=16,ring_count=10,location=point(local))
        obj=bpy.context.object;obj.name=name;obj.scale=scale;obj.data.materials.append(mat)
        for face in obj.data.polygons:face.use_smooth=True
        objects.append(obj);return obj
    def limb(name,start,end,radius,mat):
        a,b=point(start),point(end);delta=b-a
        bpy.ops.mesh.primitive_cylinder_add(vertices=12,radius=radius,depth=delta.length,
            location=(a+b)*.5)
        obj=bpy.context.object;obj.name=name;obj.rotation_euler=delta.to_track_quat('Z','Y').to_euler()
        obj.data.materials.append(mat)
        for face in obj.data.polygons:face.use_smooth=True
        objects.append(obj);return obj
    # Torso sits on the existing seat and leans toward the steering wheel (+Y).
    # Lower pelvis onto the 0.98 m seat surface; thighs overlap the cushion so
    # the seated pose has visible contact instead of a gap above the seat.
    sphere('operator_pelvis',(0,-.52,1.045),(.16,.18,.125),pants)
    sphere('operator_torso',(0,-.47,1.36),(.18,.205,.225),shirt)
    sphere('operator_head',(0,-.285,1.675),(.104,.115,.132),skin)
    sphere('operator_hair',(0,-.298,1.765),(.108,.112,.055),hair)
    sphere('operator_neck',(0,-.385,1.565),(.058,.06,.072),skin)
    for side in (-1,1):
        # Bent legs and feet rest in the cabin footwell.
        limb(f'operator_thigh_{side}',(side*.09,-.48,1.04),(side*.105,-.07,.86),.078,pants)
        limb(f'operator_shin_{side}',(side*.105,-.07,.86),(side*.115,.18,.70),.054,pants)
        sphere(f'operator_shoe_{side}',(side*.115,.245,.68),(.072,.13,.05),pants)
        # Short sleeves, forearms and hands reach the steering wheel.
        limb(f'operator_sleeve_{side}',(side*.14,-.42,1.49),(side*.205,-.15,1.42),.068,shirt)
        limb(f'operator_forearm_{side}',(side*.205,-.15,1.42),(side*.145,.075,1.34),.041,skin)
        sphere(f'operator_hand_{side}',(side*.145,.095,1.34),(.044,.05,.041),skin)
    bpy.context.view_layer.update()
    corners=[obj.matrix_world@Vector(c) for obj in objects for c in obj.bound_box]
    return objects,corners


def floor_ribbon(name,path,width,z):
    """Flat strip of constant width along a 2D polyline, with mitred corners; UV v runs across it."""
    verts=[];uvs=[]
    for i,p in enumerate(path):
        a=Vector(path[max(i-1,0)]);b=Vector(path[min(i+1,len(path)-1)])
        tangent=(b-a).normalized();normal=Vector((-tangent.y,tangent.x))
        if 0<i<len(path)-1:
            t_in=(Vector(p)-a).normalized();n_in=Vector((-t_in.y,t_in.x))
            normal=normal/max(normal.dot(n_in),.2)  # mitre: keep the strip width at corners
        for side in (-1,1):
            q=Vector(p)+normal*side*width/2;verts.append((q.x,q.y,z));uvs.append((i/(len(path)-1),(side+1)/2))
    faces=[(2*i,2*i+2,2*i+3,2*i+1) for i in range(len(path)-1)]
    mesh=bpy.data.meshes.new(name);mesh.from_pydata(verts,[],faces);mesh.update()
    uv=mesh.uv_layers.new(name='UVMap')
    for loop in mesh.loops:uv.data[loop.index].uv=uvs[loop.vertex_index]
    return mesh


def floor_light_material(name,colour,strength,falloff):
    """Unlit red emission; alpha fades toward the strip edges, so it reads as light on the floor."""
    mat=bpy.data.materials.new(name);mat.diffuse_color=(*colour,1);mat.use_nodes=True
    nodes=mat.node_tree.nodes;links=mat.node_tree.links;nodes.clear()
    coord=nodes.new('ShaderNodeTexCoord');split=nodes.new('ShaderNodeSeparateXYZ')
    centred=nodes.new('ShaderNodeMath');centred.operation='MULTIPLY_ADD'
    centred.inputs[1].default_value=2;centred.inputs[2].default_value=-1
    distance=nodes.new('ShaderNodeMath');distance.operation='ABSOLUTE'
    profile=nodes.new('ShaderNodeMath');profile.operation='SUBTRACT';profile.inputs[0].default_value=1
    shaped=nodes.new('ShaderNodeMath');shaped.operation='POWER';shaped.inputs[1].default_value=falloff
    emission=nodes.new('ShaderNodeEmission');emission.inputs[0].default_value=(*colour,1)
    emission.inputs[1].default_value=strength
    transparent=nodes.new('ShaderNodeBsdfTransparent');mix=nodes.new('ShaderNodeMixShader')
    output=nodes.new('ShaderNodeOutputMaterial')
    links.new(coord.outputs['UV'],split.inputs[0]);links.new(split.outputs['Y'],centred.inputs[0])
    links.new(centred.outputs[0],distance.inputs[0]);links.new(distance.outputs[0],profile.inputs[1])
    links.new(profile.outputs[0],shaped.inputs[0]);links.new(shaped.outputs[0],mix.inputs[0])
    links.new(transparent.outputs[0],mix.inputs[1]);links.new(emission.outputs[0],mix.inputs[2])
    links.new(mix.outputs[0],output.inputs[0])
    return mat


def add_red_u_light(forklift_transform,settings):
    """Thin red U projected on the floor around the forklift, open toward the forks.

    Flat emissive strips just above the floor plane; the forklift, LSPs and cargo
    occlude them through the normal depth test."""
    outer=settings['side_outer_x_m'];front=settings['open_front_y_m'];rear=settings['rear_y_m']
    # Left arm -> square rear bar -> right arm, forklift-local metres.
    path=[(-outer,front),(-outer,rear),(outer,rear),(outer,front)]
    objects=[]
    for name,width,strength,falloff,z in [
            ('red_u_safety_light_glow',settings['glow_width_m'],settings['glow_emission_strength'],2.0,.002),
            ('red_u_safety_light',settings['core_width_m'],settings['core_emission_strength'],.5,.003)]:
        obj=bpy.data.objects.new(name,floor_ribbon(name,path,width,z))
        bpy.context.scene.collection.objects.link(obj);obj.matrix_world=forklift_transform
        obj.visible_shadow=False
        colour=(1.0,.006,.003) if 'glow' not in name else (1.0,.018,.009)
        obj.data.materials.append(floor_light_material(name+'_emission',colour,strength,falloff));objects.append(obj)
    bpy.context.view_layer.update()
    corners=[obj.matrix_world@Vector(c) for obj in objects for c in obj.bound_box]
    return objects,corners


def main(config_path,scenario='V1',lsp_count=2,arrangement='in_series',output_stem='v_000',samples=64,red_u_light=True):
    cfg=json.loads(Path(config_path).read_text());work=Path(cfg['work_dir']).resolve()
    camera_cfg=json.loads((work/'camera.json').read_text())
    facts=json.loads((work/'scene_facts.json').read_text())
    for directory in ['renders','out','scenes','logs']:(work/directory).mkdir(exist_ok=True)
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.context.preferences.filepaths.save_version=0
    scene=bpy.context.scene;scene.unit_settings.system='METRIC';scene.unit_settings.scale_length=1
    camera=bpy.data.objects.new('calibrated_camera',bpy.data.cameras.new('calibrated_camera'))
    scene.collection.objects.link(camera);scene.camera=camera
    camera.data.sensor_fit='HORIZONTAL';camera.data.sensor_width=camera_cfg['sensor_width_mm']
    camera.data.lens=camera_cfg['lens_mm'];camera.data.shift_x=camera_cfg['shift_x'];camera.data.shift_y=camera_cfg['shift_y']
    camera.data.clip_start=.05;camera.data.clip_end=200;camera.matrix_world=Matrix(camera_cfg['matrix_world'])
    W,H=camera_cfg['width'],camera_cfg['height']
    scene.render.resolution_x=W;scene.render.resolution_y=H;scene.render.resolution_percentage=100
    scene.render.engine='CYCLES';scene.cycles.device='CPU';scene.cycles.samples=samples
    scene.cycles.use_denoising=True;scene.cycles.seed=7
    scene.render.film_transparent=True;scene.render.image_settings.file_format='PNG'
    scene.render.image_settings.color_mode='RGBA';scene.render.image_settings.color_depth='8'
    scene.view_settings.view_transform='Standard';scene.view_settings.look='None'
    scene.view_settings.exposure=cfg['blender'].get('exposure_ev',0)

    world=bpy.data.worlds.new('warehouse_ambient');world.use_nodes=True;scene.world=world
    ambient=world.node_tree.nodes.get('Background')
    ambient.inputs[0].default_value=(*facts['ambient_rgb_linear'],1);ambient.inputs[1].default_value=.9
    for name,location,power,size in [('ceiling_key',(-1,5,7.5),1100,4),('ceiling_fill',(4,9,7),750,3)]:
        data=bpy.data.lights.new(name,'AREA');data.energy=power;data.shape='DISK';data.size=size
        light=bpy.data.objects.new(name,data);scene.collection.objects.link(light);light.location=location
        light.rotation_euler=(Vector((0,7,0))-light.location).to_track_quat('-Z','Y').to_euler()
    bpy.ops.mesh.primitive_plane_add(size=80,location=(0,15,-.001))
    floor=bpy.context.object;floor.name='floor_shadow_catcher';floor.is_shadow_catcher=True
    mat=bpy.data.materials.new('floor_match');mat.use_nodes=True
    bsdf=mat.node_tree.nodes.get('Principled BSDF');bsdf.inputs['Base Color'].default_value=(.26,.27,.245,1)
    bsdf.inputs['Roughness'].default_value=.82;floor.data.materials.append(mat)

    heading=60.0;base=Vector((1.6,6.25,0))
    rotate=Matrix.Rotation(math.radians(heading),4,'Z')
    forklift_meta=json.loads((work/'assets/forklift.json').read_text())
    lsp=json.loads((work/'lsp_params.json').read_text())
    lsp_asset=json.loads((work/'assets/lsp_0.json').read_text())
    parts=forklift_meta['component_bounds']
    heels=[v for k,v in parts.items() if k.startswith('fork_heel')]
    tines=[v for k,v in parts.items() if k.startswith('fork_tine')]
    tine_top=max(v['max'][2] for v in tines)
    assert lsp['thickness_m']>tine_top+.0002,'LSP too thin for existing fork height'
    pocket_z_clearance=min(.003,(lsp['thickness_m']-tine_top)*.3)
    # Seat the load at the fork heels, covering the blades along their full length.
    # Ten-centimetre LSP blocks get inferred underside pockets for the physical forks.
    rear=min(v['min'][1] for v in tines)-.08
    first=rear+lsp['size_y_m']/2
    specs=[('forklift','forklift','forklift',(0,0,0),(1,0,0))]
    lsp_colours=[(0,1,0),(1,1,0),(0,.5,0)]
    cargo_colours=[(0,0,1),(1,0,1),(.5,0,1)]
    cargo_offsets=[]
    for index in range(lsp_count):
        asset=f'lsp_{index%3}';local_x=0.;local_y=first+index*(lsp['size_y_m']+.008)
        if arrangement=='staggered' and index:
            local_x=.36;local_y=first+lsp['size_y_m']*.76+(index-1)*.008
        specs.append((f'lsp_{index}','lsp',asset,(local_x,local_y,0),lsp_colours[index%3]))
        cargo_asset=f'cargo_{index%3}'
        cargo_meta_i=json.loads((work/'assets'/f'{cargo_asset}.json').read_text())
        # Convert the fork-heel contact position to an offset from this sheet's
        # centre. cargo_meta bounds are asset-local, while heel bounds are forklift-local.
        cargo_offset=max(v['max'][1] for v in heels)+.005-cargo_meta_i['bounds_min'][1]-first
        cargo_offsets.append(cargo_offset)
        if index==0 or scenario!='V6':
            specs.append((f'cargo_{index}','cargo',cargo_asset,
                          (local_x,local_y+cargo_offset,lsp['thickness_m']),cargo_colours[index%3]))
    actors=[];all_corners=[]
    for name,cls,asset,local,mask_color in specs:
        location=base+rotate.to_3x3()@Vector(local)
        transform=Matrix.Translation(location)@rotate
        with bpy.data.libraries.load(str(work/'assets'/f'{asset}.blend'),link=False) as (src,dst):
            dst.objects=list(src.objects)
        objects=[]
        for obj in dst.objects:
            if obj.type!='MESH':raise ValueError('Unexpected object in asset')
            scene.collection.objects.link(obj);obj.name=f'{name}__{obj.name}'
            objects.append(obj)
        # Appended datablocks have not been evaluated yet. Their matrix_world can
        # still be identity until linked and updated, collapsing all components.
        bpy.context.view_layer.update()
        local_matrices=[obj.matrix_world.copy() for obj in objects]
        local_corners=[matrix@Vector(c) for obj,matrix in zip(objects,local_matrices) for c in obj.bound_box]
        asset_meta=json.loads((work/'assets'/f'{asset}.json').read_text())
        for axis in range(3):
            extent=max(c[axis] for c in local_corners)-min(c[axis] for c in local_corners)
            assert abs(extent-asset_meta['dimensions_m'][axis])<1e-4,(name,'Asset placement extent mismatch')
        for obj,local_matrix in zip(objects,local_matrices):
            obj.matrix_world=transform@local_matrix
        bpy.context.view_layer.update()
        for obj,local_matrix in zip(objects,local_matrices):
            expected=transform@local_matrix
            assert max(abs(obj.matrix_world[i][j]-expected[i][j]) for i in range(4) for j in range(4))<1e-4
        corners=[obj.matrix_world@Vector(c) for obj in objects for c in obj.bound_box]
        box=projected_bbox(scene,camera,corners,W,H)
        if box[0]<=0 or box[1]<=0 or box[2]>=W or box[3]>=H:raise ValueError(f'{name} clipped by frame')
        all_corners.extend(corners)
        actors.append((objects,dict(name=name,**{'class':cls},asset=asset,location=list(location),
                                   rot_z_deg=heading,bbox_2d=box,mask_color_rgb=[int(c*255) for c in mask_color])))
    sheet=actors[1][0][0]
    forklift_transform=Matrix.Translation(base)@rotate
    for part,bbox in parts.items():
        if not part.startswith(('fork_tine','fork_heel')):continue
        lo,hi=Vector(bbox['min']),Vector(bbox['max'])
        bpy.ops.mesh.primitive_cube_add(size=1)
        cutter=bpy.context.object;cutter.name='temporary_fork_pocket'
        cutter.matrix_world=forklift_transform@Matrix.Translation((lo+hi)/2)
        cutter.scale=hi-lo+Vector((.006,.006,2*pocket_z_clearance))
        bpy.context.view_layer.update()
        modifier=sheet.modifiers.new('fork_clearance_'+part,'BOOLEAN')
        modifier.operation='DIFFERENCE';modifier.solver='EXACT';modifier.object=cutter
        bpy.context.view_layer.objects.active=sheet
        bpy.ops.object.modifier_apply(modifier=modifier.name)
        bpy.data.objects.remove(cutter,do_unlink=True)
    bpy.context.view_layer.update()
    # Add the seated operator and red floor safety-light U in forklift coordinates.
    operator_objects,operator_corners=add_occupant(forklift_transform)
    actors.append((operator_objects,dict(name='operator',**{'class':'person'},asset='procedural_seated_worker',
        location=list(base),rot_z_deg=heading,bbox_2d=projected_bbox(scene,camera,operator_corners,W,H),
        mask_color_rgb=[255,64,64],appearance={'shirt':'dark short-sleeve matching rtsp_010 driver crop'})))
    all_corners.extend(operator_corners)
    if red_u_light:
        red_settings=cfg['red_u_light']
        u_objects,u_corners=add_red_u_light(forklift_transform,red_settings)
        actors.append((u_objects,dict(name='red_u_safety_light',**{'class':'safety_light'},asset='sample_matched_emissive_floor_curve',
            location=list(base),rot_z_deg=heading,bbox_2d=projected_bbox(scene,camera,u_corners,W,H),
            mask_color_rgb=[255,128,64],appearance={'shape':'square-cornered U open toward the forks',**red_settings,
            'source':'flat emissive floor strips (projected line light)','camera_occlusion':'depth-tested against scene geometry'})))
        all_corners.extend(u_corners)
    meta=dict(image=f'{output_stem}.jpg',lossless_image=f'{output_stem}.png',camera_id=cfg['camera_id'],source_image=cfg['background_image'],
              violation_id=scenario,arrangement=arrangement,heading_deg=heading,lsp_count=lsp_count,
              is_violation=lsp_count>=2,skid_count=0,
              seed=7,objects=[record for _,record in actors],
              forklift_appearance={'roof_beacon':forklift_meta.get('roof_beacon'),
                  'paint_linear_gain':forklift_meta.get('paint_linear_gain'),
                  'overhead_guard':forklift_meta.get('overhead_guard')},
              forklift_shape=forklift_meta.get('shape_corrections'),
              model_appearance=forklift_meta.get('appearance'),
              lsp_edge_markings=lsp_asset.get('edge_markings'),
              lsp_thickness_m=lsp['thickness_m'],pushing_contact={'target':'fork_heel','gap_m':.005,
                  'cargo_offset_y_m':cargo_offsets[0],'lsp_rear_y_m':rear,
                  'pocket_clearance_xy_m':.003,'pocket_clearance_z_m':pocket_z_clearance,
                  'underside_fork_pockets':'inferred pockets, vertical clearance constrained by current LSP thickness'},
              render_exposure_ev=scene.view_settings.exposure,
              event={'class':'Forklift Pushing Multiple Lsps' if lsp_count>=2 else 'Forklift Pushing One LSP',
                     'bbox_2d':projected_bbox(scene,camera,all_corners,W,H)},
              scene_file=f'work/scenes/{output_stem}.blend',render_file=f'work/renders/{output_stem}.png',
              status='prototype_for_visual_review',
              limitations=['Existing camera background contains real objects and people; it is not a clean training plate.',
                           'Asset shapes/unseen sides and lighting are approximate; realism gate has not been accepted.',
                           'Event and object bbox_2d are amodal projections; visible_bbox_2d is added from rendered instance masks.',
                           'One open-aisle scene with a floor shadow catcher; no full static occlusion proxy has been built.',
                           'First LSP underside has inferred fork pockets to fit the configured thickness.'])
    bpy.ops.wm.save_as_mainfile(filepath=str(work/'scenes'/f'{output_stem}.blend'))
    if '--mask-only' not in sys.argv:
        scene.render.filepath=str(work/'renders'/f'{output_stem}.png');bpy.ops.render.render(write_still=True)
    # Actual visibility, not just projected boxes: render colour IDs with opaque emission.
    floor.hide_render=True;ambient.inputs[1].default_value=0
    scene.view_settings.exposure=0  # IDs are exact colours, independent of beauty grading.
    scene.cycles.samples=1;scene.cycles.use_denoising=False
    for objects,record in actors:
        material=bpy.data.materials.new('id_'+record['name']);material.use_nodes=True
        nodes=material.node_tree.nodes;nodes.clear()
        shader=nodes.new('ShaderNodeEmission');shader.inputs[0].default_value=(*[srgb_to_linear(v/255) for v in record['mask_color_rgb']],1)
        output=nodes.new('ShaderNodeOutputMaterial');material.node_tree.links.new(shader.outputs[0],output.inputs[0])
        for obj in objects:
            obj.data.materials.clear();obj.data.materials.append(material)
            for polygon in obj.data.polygons:polygon.material_index=0
    # Keep both real blades in the scene; independently measure whether the load
    # actually occludes them, instead of deleting/hiding them to pass visual review.
    diagnostic=bpy.data.materials.new('fork_blade_visibility');diagnostic.use_nodes=True
    nodes=diagnostic.node_tree.nodes;nodes.clear()
    emission=nodes.new('ShaderNodeEmission');emission.inputs[0].default_value=(0,1,1,1)
    output=nodes.new('ShaderNodeOutputMaterial');diagnostic.node_tree.links.new(emission.outputs[0],output.inputs[0])
    for obj in actors[0][0]:
        if 'fork_tine' in obj.name:
            obj.data.materials.clear();obj.data.materials.append(diagnostic)
    meta['fork_visibility_color_rgb']=[0,255,255]
    rear_id=bpy.data.materials.new('rear_wheel_visibility');rear_id.use_nodes=True
    nodes=rear_id.node_tree.nodes;nodes.clear()
    shader=nodes.new('ShaderNodeEmission');shader.inputs[0].default_value=(1,1,1,1)
    output=nodes.new('ShaderNodeOutputMaterial');rear_id.node_tree.links.new(shader.outputs[0],output.inputs[0])
    for obj in actors[0][0]:
        if obj.name.startswith(('forklift__rear_tire','forklift__rear_hub','forklift__rear_bolt')):
            obj.data.materials.clear();obj.data.materials.append(rear_id)
    meta['rear_wheel_visibility_color_rgb']=[255,255,255]
    scene.render.filepath=str(work/'renders'/f'{output_stem}_instance_ids.png');bpy.ops.render.render(write_still=True)
    (work/'out'/f'{output_stem}.json').write_text(json.dumps(meta,indent=2))
    print('SCENARIO_RENDER_OK '+json.dumps(meta))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('config');parser.add_argument('--scenario',default='V1')
    parser.add_argument('--lsp-count',type=int,default=2)
    parser.add_argument('--arrangement',choices=['in_series','staggered'],default='in_series')
    parser.add_argument('--output-stem',default='v_000')
    parser.add_argument('--samples',type=int,default=64)
    parser.add_argument('--no-red-u-light',action='store_true',help='Render without the red U-shaped safety light')
    args=parser.parse_args(sys.argv[sys.argv.index('--')+1:])
    main(args.config,args.scenario,args.lsp_count,args.arrangement,args.output_stem,args.samples,not args.no_red_u_light)
