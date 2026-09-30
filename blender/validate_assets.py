"""Reopen every .blend and verify its reusable asset contract, including packed images."""
import json
import sys
from pathlib import Path
import bpy
from mathutils import Vector

work=Path(sys.argv[sys.argv.index('--')+1]).resolve()
cfg=json.loads(Path('config.json').read_text())
colour_factor=cfg.get('asset_appearance',{}).get('saturation_multiplier',1.2)
band_height=cfg.get('lsp_edge_markings',{}).get('height_m',.03)
names=['forklift','cargo_0','cargo_1','cargo_2','skid','lsp_0','lsp_1','lsp_2']
results=[]
for name in names:
    path=work/'assets'/f'{name}.blend'
    bpy.ops.wm.open_mainfile(filepath=str(path))
    scene=bpy.context.scene;objects=list(scene.objects)
    assert objects and all(o.type=='MESH' for o in objects),(name,'Non-asset objects')
    assert scene.unit_settings.system=='METRIC' and scene.unit_settings.scale_length==1
    coords=[o.matrix_world@Vector(c) for o in objects for c in o.bound_box]
    lo=[min(p[i] for p in coords) for i in range(3)];hi=[max(p[i] for p in coords) for i in range(3)]
    assert abs(lo[2])<1e-5,(name,'Not on floor',lo[2])
    assert abs(lo[0]+hi[0])<1e-5 and abs(lo[1]+hi[1])<1e-5,(name,'Not centred')
    images=[im for im in bpy.data.images if im.source=='FILE' and im.users>0]
    assert all(im.packed_file for im in images),(name,'Missing packed texture')
    assert bpy.data.collections.get(name) and bpy.data.collections[name].asset_data
    if name!='skid':assert images,(name,'Missing real-image texture')
    if name.startswith('lsp_'):
        params=json.loads((work/'lsp_params.json').read_text())
        assert abs(hi[0]-lo[0]-params['size_x_m'])<1e-5
        assert abs(hi[1]-lo[1]-params['size_y_m'])<1e-5
        assert abs(hi[2]-params['thickness_m'])<1e-5
        assert objects[0].get('edge_band_count_per_face')==3
        bands=next(n.image for n in bpy.data.materials['lsp_edge'].node_tree.nodes if n.type=='TEX_IMAGE');w,h=bands.size
        pixels=list(bands.pixels[:]);row=[pixels[(h//2*w+x)*4]>.4 for x in range(w)]
        runs=sum(value and (i==0 or not row[i-1]) for i,value in enumerate(row))
        assert runs==3,('LSP must have three central edge bands',runs)
        assert bands.packed_file,'LSP edge markings must travel with the asset'
        starts=[i for i,value in enumerate(row) if value and (i==0 or not row[i-1])]
        ends=[i+1 for i,value in enumerate(row) if value and (i==w-1 or not row[i+1])]
        lengths=[b-a for a,b in zip(starts,ends)]
        assert lengths==[44,72,44],('Mark lengths must follow r10 scaling',lengths)
        column=[pixels[(y*w+w//2)*4]>.4 for y in range(h)]
        assert abs(sum(column)/h*params['thickness_m']-band_height)<.001,'Mark height must match configuration'
    if name=='forklift':
        bpy.context.view_layer.update()
        rear_visibility={}
        assert not any('beacon' in o.name for o in objects),'Worker shirt misidentified as roof beacon'
        assert not any(o.name.startswith('step_') for o in objects),'No boarding steps on source forklift'
        assert not any(o.name.startswith('sill_') for o in objects),'Removed external yellow side bars'
        footwell=bpy.data.objects['cabin_front_footwell']
        assert footwell.data.materials[0].name=='black'
        assert max((footwell.matrix_world@Vector(c)).z for c in footwell.bound_box)>.65,'Black footwell must be visible above chassis floor'
        assert not any(o.name.startswith('cabin_side_floor_') for o in objects),'Remove the two black cabin side bars'
        assert bpy.data.objects['engine_cover'].data.materials[0].name=='yellow','Seat pedestal must be yellow'
        pedestal=bpy.data.objects['engine_cover']
        pedestal_width=max((pedestal.matrix_world@Vector(c)).x for c in pedestal.bound_box)-min((pedestal.matrix_world@Vector(c)).x for c in pedestal.bound_box)
        assert abs(pedestal_width-.86*.90)<1e-5,'Yellow seat pedestal width must be 90%'
        for part_name in ['front_guard_-1','front_guard_1',
                     'roof_rail_-1','roof_rail_1','roof_cross_-1.16','roof_cross_-0.07',
                     *[f'roof_grille_{i}' for i in range(7)]]:
            obj=bpy.data.objects[part_name]
            assert all(mat.name=='yellow' for mat in obj.data.materials if mat),(part_name,'Cab frame and roof bars must be yellow')
        assert not any(o.name.startswith('side_panel_') for o in objects),'Both external side bars must be removed'
        assert hi[0]-lo[0]<1.6,'Wheel scaling regression'
        assert sum('tire' in o.name for o in objects)==4
        assert sum('fork_tine' in o.name for o in objects)==2
        roof=bpy.data.objects['roof_rear_panel']
        assert roof.data.materials[0].name=='roof_black','Rear roof panel must be black'
        roof_top=max((roof.matrix_world@Vector(c)).z for c in roof.bound_box)
        assert 1.96<roof_top<2.00,('Cab roof height',roof_top)
        assert abs(roof_top-1.9725)<1e-5,'Seat revision must not raise the roof'
        seat=bpy.data.objects['seat_base']
        seat_points=[seat.matrix_world@Vector(c) for c in seat.bound_box]
        seat_top=max(p.z for p in seat_points)
        roof_bottom=min((roof.matrix_world@Vector(c)).z for c in roof.bound_box)
        assert abs(seat_top-.98)<1e-5,'Seat surface must be lowered 19 cm'
        assert roof_bottom-seat_top>.94,'Seat to roof clearance did not increase'
        assert max(p.x for p in seat_points)-min(p.x for p in seat_points)<.45,'Seat should be smaller'
        mast=[o for o in objects if o.name.startswith('mast_') or o.name=='hydraulic_ram']
        mast_top=max((o.matrix_world@Vector(c)).z for o in mast for c in o.bound_box)
        assert abs(mast_top-roof_top)<1e-5,('Mast must match roof',mast_top,roof_top)
        mast_points=[obj.matrix_world@Vector(c) for obj in mast for c in obj.bound_box]
        backrest=[o for o in objects if o.name.startswith('load_backrest_')]
        assert len(backrest)==9,'Complete open load-backrest frame required'
        points=[o.matrix_world@Vector(c) for o in backrest for c in o.bound_box]
        frame_width=max(p.x for p in points)-min(p.x for p in points)
        frame_height=max(p.z for p in points)-min(p.z for p in points)
        assert abs(frame_width-1.244)<.002,'Backrest width must remain equal to forklift body width'
        assert abs(frame_height-(mast_top-min(p.z for p in mast_points))*.5)<1e-5,'Backrest must be half mast height'
        assert abs(min(p.y for p in points)-max(p.y for p in mast_points))<1e-5,'Backrest must sit against mast'
        for side in [-1,1]:
            diameters=[]
            for axle in ['front','rear']:
                tire=bpy.data.objects[f'{axle}_tire_{side}']
                heights=[(tire.matrix_world@Vector(c)).z for c in tire.bound_box]
                assert abs(min(heights))<1e-5,'Tire must sit on floor'
                diameters.append(max(heights)-min(heights))
            assert abs(diameters[0]-.696*.90)<1e-5,('Front diameter must decrease 10 percent',diameters)
            assert abs(diameters[1]-.44)<1e-5,('Smaller rear size',diameters)
            front=bpy.data.objects[f'front_tire_{side}']
            assert abs(abs(front.matrix_world.translation.x)-.481)<1e-5,'Front wheel must move outwards 0.026 m from revision 9'
            fender=bpy.data.objects[f'front_fender_{side}']
            tire_outer=max(side*(front.matrix_world@Vector(c)).x for c in front.bound_box)
            depsgraph=bpy.context.evaluated_depsgraph_get()
            tyre_top=max((front.matrix_world@Vector(c)).z for c in front.bound_box)
            start=Vector((side*.50,front.matrix_world.translation.y,tyre_top+.001))
            hit,point,_,_,obj,_=scene.ray_cast(depsgraph,start,Vector((0,0,1)))
            assert hit and obj.name==fender.name,'Body arch must cover tire'
            assert abs(point.z-tyre_top-.07)<.001,'Tire circumference to body clearance must be 7 cm'
            for x,expected in [(side*.50,fender.name),(side*(tire_outer-.015),front.name)]:
                hit,_,_,_,obj,_=scene.ray_cast(depsgraph,Vector((x,front.matrix_world.translation.y,3)),Vector((0,0,-1)))
                assert hit and obj.name==expected,('Body coverage/tire visibility check',x,expected,obj.name if hit else None)
            for axle in ['front','rear']:
                tire=bpy.data.objects[f'{axle}_tire_{side}']
                assert tire.get('rim_bore_radius_m',0)>0,'Tire needs a real centre bore'
                assert tire.get('rounded_tire_profile'),'Rounded tire shoulders required'
                assert tire.get('circumference_segments',0)>=96
                assert all(p.use_smooth for p in tire.data.polygons),'Tire shading must be smooth'
                # Actual mesh must curve across its width, not just claim metadata.
                radial_levels={round((v.co.x*v.co.x+v.co.y*v.co.y)**.5,5) for v in tire.data.vertices}
                axial_levels={round(v.co.z,5) for v in tire.data.vertices}
                assert len(radial_levels)>15 and len(axial_levels)>15,'Tire still has square shoulders'
                bore=min((v.co.x*v.co.x+v.co.y*v.co.y)**.5 for v in tire.data.vertices)
                assert abs(bore-tire['rim_bore_radius_m'])<1e-5,'Tire bore geometry mismatch'
                outer=max(side*(tire.matrix_world@Vector(c)).x for c in tire.bound_box)
                for part in ['hub','hubcap',*[f'bolt_{side}_{i}' for i in range(6)]]:
                    part_name=f'{axle}_{part}' if part.startswith('bolt') else f'{axle}_{part}_{side}'
                    obj=bpy.data.objects[part_name]
                    face=max(side*(obj.matrix_world@Vector(c)).x for c in obj.bound_box)
                    assert outer-face>.015,(part_name,'Wheel hardware protrudes beyond tire')
                    if part.startswith('bolt'):
                        shader=obj.data.materials[0].node_tree.nodes.get('Principled BSDF')
                        assert max(shader.inputs['Base Color'].default_value[:3])<.04
                        assert shader.inputs['Roughness'].default_value>.8
            assert bpy.data.objects['chassis'].get('front_wheel_wells')
            tire=bpy.data.objects[f'rear_tire_{side}']
            assert abs(abs(tire.matrix_world.translation.x)-.50)<1e-5,'Rear tires must reach side openings'
            assert not tire.hide_render,'Rear wheels must remain renderable'
            # Orthographic side rays measure physical shell occlusion, independent
            # of whether the far wheel happens to be hidden by camera perspective.
            visible=total=0;centre=tire.matrix_world.translation
            for iy in range(-15,16):
                for iz in range(-15,16):
                    dy=iy*.22/15;dz=iz*.22/15
                    if dy*dy+dz*dz>(.22*.98)**2:continue
                    total+=1
                    origin=Vector((side*2,centre.y+dy,centre.z+dz))
                    hit,_,_,_,obj,_=scene.ray_cast(bpy.context.evaluated_depsgraph_get(),origin,Vector((-side,0,0)))
                    if hit and obj.name.startswith(('rear_tire','rear_hub','rear_bolt')):visible+=1
            rear_visibility[str(side)]=visible/total
            assert visible/total>.55,('Rear wheel opening must expose tire',side,visible/total)
        tail=bpy.data.objects['rear_counterweight']
        tail_top=max((tail.matrix_world@Vector(c)).z for c in tail.bound_box)
        floor=bpy.data.objects['floorboard']
        floor_top=max((floor.matrix_world@Vector(c)).z for c in floor.bound_box)
        assert abs(tail_top-.72)<.002 and abs(tail_top-floor_top-.045)<.003,'Rear top should meet cabin-floor level'
        vertices=[tail.matrix_world@v.co for v in tail.data.vertices]
        back=min(v.y for v in vertices)
        assert max(abs(v.x) for v in vertices if v.y<back+.05)<.31,'Tail is still box shaped'
        assert tail.get('rear_wheel_wells'),'Wheel clearance required'
        assert max(v.y for v in vertices)-back<.62,'Rear shell is still too long'
        assert len({round(v.y,4) for v in vertices})>15,'Tail needs a continuous curved outline'
        guards=[o for o in objects if o.get('source_fitted_curved_guard')]
        assert len(guards)==4 and {g.name for g in guards}=={'front_guard_-1','front_guard_1','rear_guard_-1','rear_guard_1'},'Keep four roof supports; remove external body-side bars'
        rear_guards=[bpy.data.objects[f'rear_guard_{side}'] for side in [-1,1]]
        assert all(abs(min((g.matrix_world@Vector(c)).z for c in g.bound_box)-.72)<.02 for g in rear_guards),'Rear roof supports must reach the tail'
        rear_x=[(g.matrix_world@Vector(c)).x for g in rear_guards for c in g.bound_box]
        assert min(rear_x)<-.49 and max(rear_x)>.49,'Rear roof supports must sit on both sides'
        for guard in guards:
            points=[Vector(p) for p in guard['centreline_points']]
            chord=(points[-1]-points[0]).normalized()
            deviation=max((p-points[0]).cross(chord).length for p in points)
            assert deviation>.05,(guard.name,'Guard was flattened into a straight post')
    for ext in ['glb','json']:assert (work/'assets'/f'{name}.{ext}').is_file()
    assert all(o.get('softened_edges') for o in objects),'Softened edge geometry missing'
    for mat in {m for o in objects for m in o.data.materials if m}:
        assert abs(mat.get('model_saturation_multiplier',0)-colour_factor)<1e-5,'Model saturation must match configuration'
        if cfg.get('asset_appearance',{}).get('mode')=='selective_yellow_black':
            assert mat.get('model_colour_mode')=='selective_yellow_black'
            assert abs(mat.get('black_gain',0)-.8)<1e-5
    for suffix in ['','_rear']:
        preview=work/'previews'/f'{name}_pass3{suffix}.png'
        assert preview.is_file() and preview.stat().st_size>1000
    result=dict(name=name,meshes=len(objects),packed_images=len(images),dimensions_m=[hi[i]-lo[i] for i in range(3)],status='PASS')
    if name=='forklift':result['rear_wheel_visible_side_fraction']=rear_visibility
    results.append(result)
(work/'asset_review'/'validation.json').write_text(json.dumps(results,indent=2))
print('ASSET_VALIDATION: '+json.dumps(results))
