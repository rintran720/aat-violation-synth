"""Render loads aligned to manually marked landmarks on a real forklift frame."""
import argparse
import json
import math
from pathlib import Path

import bpy
import numpy as np
from bpy_extras.object_utils import world_to_camera_view
from mathutils import Matrix, Vector


def projected_bbox(scene, camera, points, width, height):
    coords=[world_to_camera_view(scene,camera,p) for p in points]
    xs=[p.x*width for p in coords];ys=[(1-p.y)*height for p in coords]
    return [round(min(xs),1),round(min(ys),1),round(max(xs),1),round(max(ys),1)]


def fit_pose(scene, camera, landmarks, width, height, seed):
    """Fit ground-plane translation and yaw from weighted image-space landmarks."""
    def project(params, point):
        x,y,yaw=params
        c,s=math.cos(yaw),math.sin(yaw)
        world=Vector((x+c*point[0]-s*point[1], y+s*point[0]+c*point[1], point[2]))
        p=world_to_camera_view(scene,camera,world)
        return np.array([p.x*width,(1-p.y)*height],dtype=float)
    params=np.array(seed,dtype=float)
    weights=np.repeat(np.array([float(v.get('weight',1)) for v in landmarks]),2)
    for _ in range(40):
        residual=np.concatenate([project(params,v['local'])-np.asarray(v['pixel'],float) for v in landmarks])
        jac=np.empty((len(residual),3),dtype=float)
        for j,step in enumerate((.002,.002,.0005)):
            shifted=params.copy();shifted[j]+=step
            jac[:,j]=(np.concatenate([project(shifted,v['local'])-np.asarray(v['pixel'],float) for v in landmarks])-residual)/step
        root_w=np.sqrt(weights)
        delta=np.linalg.lstsq(jac*root_w[:,None],-residual*root_w,rcond=None)[0]
        params+=delta
        if np.linalg.norm(delta)<1e-7: break
    errors=[(project(params,v['local'])-np.asarray(v['pixel'],float)).tolist() for v in landmarks]
    return params,errors


def load_asset(asset_path, prefix, scene, transform):
    with bpy.data.libraries.load(str(asset_path),link=False) as (src,dst):
        dst.objects=list(src.objects)
    objects=[]
    for obj in dst.objects:
        if obj.type!='MESH':
            raise ValueError(f'Unexpected object in asset {asset_path}: {obj.name}')
        scene.collection.objects.link(obj);obj.name=f'{prefix}__{obj.name}'
        objects.append(obj)
    bpy.context.view_layer.update()
    matrices=[obj.matrix_world.copy() for obj in objects]
    for obj,matrix in zip(objects,matrices):obj.matrix_world=transform@matrix
    bpy.context.view_layer.update()
    corners=[obj.matrix_world@Vector(c) for obj in objects for c in obj.bound_box]
    return objects,corners,matrices


def main(config_path,anchor_path,output_stem):
    root=Path.cwd();cfg=json.loads(Path(config_path).read_text())
    anchor=json.loads(Path(anchor_path).read_text());work=root/cfg['work_dir']
    camera_cfg=json.loads((root/anchor['source_calibration']).read_text())
    facts=json.loads((work/'scene_facts.json').read_text())
    floor_points=np.load(root/anchor['source_floor_points'])
    bg=anchor['foreground_forklift'];rotate=None;landmark_rmse=None;residuals={}
    if 'landmarks' not in bg and 'fork_tip_center_px' not in bg:
        front_px=bg['front_axle_center_px'];rear_px=bg['rear_axle_center_px']
        front=np.asarray(floor_points[front_px[1]//4,front_px[0]//4],dtype=float)
        rear=np.asarray(floor_points[rear_px[1]//4,rear_px[0]//4],dtype=float)
        if not np.isfinite(front).all() or not np.isfinite(rear).all():
            raise ValueError('Anchor wheel pixels have no valid point in the existing MoGe floor map')
        front[2]=0;rear[2]=0;delta=front[:2]-rear[:2]
        heading=math.atan2(-delta[0],delta[1])
        side=1 if bg.get('same_side','positive')=='positive' else -1
        front_local=Vector((side*bg['front_wheel_local_xy_m'][0],bg['front_wheel_local_xy_m'][1],0))
        rear_local=Vector((side*bg['rear_wheel_local_xy_m'][0],bg['rear_wheel_local_xy_m'][1],0))
        rotate=Matrix.Rotation(heading,4,'Z')
        base_front=Vector(front)-rotate.to_3x3()@front_local
        base_rear=Vector(rear)-rotate.to_3x3()@rear_local
        base=(base_front+base_rear)*.5;base.z=0
        model_baseline=abs(bg['front_wheel_local_xy_m'][1]-bg['rear_wheel_local_xy_m'][1])
        anchor_pixels={'front_axle_center':front_px,'rear_axle_center':rear_px}
        anchor_world={'front':front.tolist(),'rear':rear.tolist(),'base':list(base)}
        baseline_name='wheelbase_m'
        anchor_method='manual front/rear wheel ground-contact pixels lifted via existing same-camera MoGe floor map'

    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.context.preferences.filepaths.save_version=0
    scene=bpy.context.scene;scene.unit_settings.system='METRIC';scene.unit_settings.scale_length=1
    camera=bpy.data.objects.new('calibrated_camera',bpy.data.cameras.new('calibrated_camera'))
    scene.collection.objects.link(camera);scene.camera=camera
    camera.data.sensor_fit='HORIZONTAL';camera.data.sensor_width=camera_cfg['sensor_width_mm']
    camera.data.lens=camera_cfg['lens_mm'];camera.data.shift_x=camera_cfg['shift_x'];camera.data.shift_y=camera_cfg['shift_y']
    camera.data.clip_start=.05;camera.data.clip_end=200;camera.matrix_world=Matrix(camera_cfg['matrix_world'])
    width,height=camera_cfg['width'],camera_cfg['height']
    if 'landmarks' in bg:
        # Forks identify the physical front. Their four tip/heel corners dominate;
        # additional trusted chassis points can stabilize a partly hidden fork axis.
        points=bg['landmarks']
        params,errors=fit_pose(scene,camera,points,width,height,bg.get('initial_pose',[0,12,0]))
        base=Vector((float(params[0]),float(params[1]),0));heading=float(params[2])
        rotate=Matrix.Rotation(heading,4,'Z')
        anchor_pixels={v['name']:v['pixel'] for v in points}
        anchor_world={'base':list(base)}
        residuals={v['name']:[round(x,2) for x in e] for v,e in zip(points,errors)}
        errors_px=[float(np.linalg.norm(e)) for e in errors]
        anchor_method='weighted image-space least-squares fit of fork-prioritized landmarks'
        landmark_rmse=float(np.sqrt(np.mean(np.square(errors_px))))
        max_rmse=float(bg.get('max_landmark_rmse_px',12.0))
        print('LANDMARK_FIT '+json.dumps({'rmse_px':round(landmark_rmse,2),'max_rmse_px':max_rmse,
                                          'per_landmark_px':dict(zip((v['name'] for v in points),[round(x,2) for x in errors_px]))}))
        if landmark_rmse>max_rmse:
            raise ValueError(f'Landmark fit rejected: {landmark_rmse:.1f}px RMSE exceeds {max_rmse:.1f}px; check points, camera calibration, or model geometry')
    elif 'fork_tip_center_px' in bg:
        tip_px=bg['fork_tip_center_px'];heel_px=bg['fork_heel_center_px']
        tip=np.asarray(floor_points[tip_px[1]//4,tip_px[0]//4],dtype=float)
        heel=np.asarray(floor_points[heel_px[1]//4,heel_px[0]//4],dtype=float)
        if not np.isfinite(tip).all() or not np.isfinite(heel).all():
            raise ValueError('Fork tip/heel pixels have no valid point in the existing MoGe floor map')
        tip[2]=0;heel[2]=0;delta=tip[:2]-heel[:2]
        heading=math.atan2(-delta[0],delta[1]);rotate=Matrix.Rotation(heading,4,'Z')
        tip_local_y=float(bg.get('fork_tip_local_y_m',1.852));heel_local_y=float(bg.get('fork_heel_local_y_m',.772))
        base_tip=Vector(tip)-rotate.to_3x3()@Vector((0,tip_local_y,0))
        base_heel=Vector(heel)-rotate.to_3x3()@Vector((0,heel_local_y,0))
        base=(base_tip+base_heel)*.5;base.z=0
        model_baseline=abs(tip_local_y-heel_local_y);measured_baseline=float(np.linalg.norm(delta))
        anchor_pixels={'fork_tip_center':tip_px,'fork_heel_center':heel_px}
        anchor_world={'fork_tip':tip.tolist(),'fork_heel':heel.tolist(),'base':list(base)}
        anchor_method='manual visible fork tip/heel centers lifted via existing same-camera MoGe floor map'
    if 'landmarks' not in bg:
        rotate=Matrix.Rotation(heading,4,'Z')
        measured_baseline=float(np.linalg.norm(delta))
    scene.render.resolution_x=width;scene.render.resolution_y=height;scene.render.resolution_percentage=100
    scene.render.engine='CYCLES';scene.cycles.device='CPU';scene.cycles.samples=int(anchor['render']['samples'])
    scene.cycles.seed=int(anchor['render'].get('seed',7));scene.cycles.use_denoising=True
    scene.render.film_transparent=True;scene.render.image_settings.file_format='PNG'
    scene.render.image_settings.color_mode='RGBA';scene.render.image_settings.color_depth='8'
    scene.view_settings.view_transform='Standard';scene.view_settings.look='None'
    scene.view_settings.exposure=cfg['blender'].get('exposure_ev',-.65)
    world=bpy.data.worlds.new('warehouse_ambient');world.use_nodes=True;scene.world=world
    world.node_tree.nodes.get('Background').inputs[0].default_value=(*facts['ambient_rgb_linear'],1)
    world.node_tree.nodes.get('Background').inputs[1].default_value=.9
    for name,location,power,size in [('ceiling_key',(-1,5,7.5),1100,4),('ceiling_fill',(4,9,7),750,3)]:
        light_data=bpy.data.lights.new(name,'AREA');light_data.energy=power;light_data.shape='DISK';light_data.size=size
        light=bpy.data.objects.new(name,light_data);scene.collection.objects.link(light);light.location=location
        light.rotation_euler=(Vector((0,7,0))-light.location).to_track_quat('-Z','Y').to_euler()
    bpy.ops.mesh.primitive_plane_add(size=80,location=(0,15,-.001))
    floor=bpy.context.object;floor.name='floor_shadow_catcher';floor.is_shadow_catcher=True
    floor_mat=bpy.data.materials.new('floor_match');floor_mat.use_nodes=True
    bsdf=floor_mat.node_tree.nodes.get('Principled BSDF');bsdf.inputs['Base Color'].default_value=(.26,.27,.245,1)
    bsdf.inputs['Roughness'].default_value=.82;floor.data.materials.append(floor_mat)

    lsp=json.loads((work/'lsp_params.json').read_text())
    lsp_meta=json.loads((work/'assets/lsp_0.json').read_text())
    forklift_meta=json.loads((work/'assets/forklift.json').read_text())
    # Cargo asset per LSP, cycled; the default keeps the original three so older anchor files render unchanged.
    cargo_assets=anchor['layout'].get('cargo_assets',['cargo_0','cargo_1','cargo_2'])
    lsp_size_y=lsp['size_y_m'];rear_y=min(v['min'][1] for k,v in forklift_meta['component_bounds'].items()
                                           if k.startswith('fork_tine'))-.08
    first_y=rear_y+lsp_size_y/2
    scenario=anchor['layout']['scenario'];lsp_count=int(anchor['layout']['lsp_count'])
    heel_y=max(v['max'][1] for k,v in forklift_meta['component_bounds'].items() if k.startswith('fork_heel'))+.005
    specs=[]
    for i in range(lsp_count):
        local_y=first_y+i*(lsp_size_y+.008)
        specs.append((f'lsp_{i}','lsp',f'lsp_{i%3}',Vector((0,local_y,0))))
        if i==0 or scenario!='V6':
            asset=cargo_assets[i%len(cargo_assets)]
            cargo_meta=json.loads((work/'assets'/f'{asset}.json').read_text())
            cargo_offset=heel_y-cargo_meta['bounds_min'][1]-first_y  # cargo back face against the fork heel
            specs.append((f'cargo_{i}','cargo',asset,Vector((0,local_y+cargo_offset,lsp['thickness_m']))))
    if anchor['layout'].get('include_idle_skid',True):
        skid_xy=anchor['layout']['idle_skid_local_xy_m']
        specs.append(('idle_skid','skid','skid',Vector((skid_xy[0],skid_xy[1],0))))
    objects_meta=[];event_boxes=[]
    for name,kind,asset,local in specs:
        location=base+rotate.to_3x3()@local
        transform=Matrix.Translation(location)@rotate
        objects,corners,_=load_asset(work/'assets'/f'{asset}.blend',name,scene,transform)
        box=projected_bbox(scene,camera,corners,width,height)
        if box[0]<0 or box[1]<0 or box[2]>width or box[3]>height:
            raise ValueError(f'{name} clipped by image: {box}')
        objects_meta.append({'name':name,'class':kind,'asset':asset,'location_world_m':[round(v,4) for v in location],
                             'bbox_2d':box,'dimensions_m':json.loads((work/'assets'/f'{asset}.json').read_text())['dimensions_m']})
        if kind in ('lsp','cargo'):event_boxes.append(box)

    (work/'renders').mkdir(parents=True,exist_ok=True);(work/'scenes').mkdir(parents=True,exist_ok=True)
    render_path=work/'renders'/f'{output_stem}.png';scene.render.filepath=str(render_path)
    scene.render.image_settings.color_mode='RGBA'
    bpy.ops.wm.save_as_mainfile(filepath=str(work/'scenes'/f'{output_stem}.blend'))
    bpy.ops.render.render(write_still=True)
    event_box=[round(min(b[0] for b in event_boxes),1),round(min(b[1] for b in event_boxes),1),
               round(max(b[2] for b in event_boxes),1),round(max(b[3] for b in event_boxes),1)]
    meta={'output_stem':output_stem,'source_background':anchor['background_image'],
          'real_forklift_retained_from_background':True,'forklift_3d_rendered':False,
          'anchor_method':anchor_method,'anchor_pixels':anchor_pixels,'anchor_world_m':anchor_world,
          'pose':{'heading_deg':round(math.degrees(heading),3),
                  'landmark_residuals_px':residuals,
                  'landmark_rmse_px':round(landmark_rmse,2) if landmark_rmse is not None else None,
                  'measured_baseline_m':round(measured_baseline,4) if 'landmarks' not in bg else None,
                  'model_baseline_m':round(model_baseline,4) if 'landmarks' not in bg else None,
                  'baseline_ratio':round(measured_baseline/model_baseline,4) if 'landmarks' not in bg else None},
          'scenario':scenario,'lsp_count':lsp_count,'is_violation':lsp_count>=2,'objects':objects_meta,
          'event':{'class':'Forklift Pushing Multiple Lsps' if lsp_count>=2 else 'Forklift Pushing One LSP',
                   'bbox_2d':event_box},
          'limitations':['Anchor pixels are manually estimated, not SAM3 keypoints.',
                         'MoGe floor map comes from the clean image of the same fixed camera.',
                         'No real-forklift mask/depth holdout is applied yet; load placement is a first visual fit.',
                         'Background forklift and people remain untouched photographic pixels.'],
          'render_file':str(render_path.relative_to(root))}
    out=work/'out';out.mkdir(parents=True,exist_ok=True)
    (out/f'{output_stem}.json').write_text(json.dumps(meta,indent=2))
    print('ANCHOR_POC_RENDER_OK '+json.dumps(meta))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('config');parser.add_argument('--anchor',default='work/anchor_poc.json')
    parser.add_argument('--output-stem',default='anchor_poc_v6')
    argv=__import__('sys').argv
    argv=argv[argv.index('--')+1:] if '--' in argv else argv[1:]
    args=parser.parse_args(argv);main(args.config,args.anchor,args.output_stem)
