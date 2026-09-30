"""Validate a generated scenario and its scene/sidecar actor agreement."""
import argparse
import json
from pathlib import Path
import sys
import bpy
from mathutils import Matrix, Vector

parser=argparse.ArgumentParser()
parser.add_argument('work')
parser.add_argument('--output-stem',required=True)
args=parser.parse_args(sys.argv[sys.argv.index('--')+1:] if '--' in sys.argv else None)
work=Path(args.work).resolve();stem=args.output_stem
meta=json.loads((work/'out'/f'{stem}.json').read_text())
bpy.ops.wm.open_mainfile(filepath=str(work/'scenes'/f'{stem}.blend'))
bpy.context.view_layer.update()
records={row['name']:row for row in meta['objects']}
lsps=[r for r in records.values() if r['class']=='lsp']
cargos=[r for r in records.values() if r['class']=='cargo']
assert len(lsps)==meta['lsp_count']
assert meta['is_violation']==(meta['lsp_count']>=2)
assert len(cargos)>=1 and records['forklift']['class']=='forklift'
params=json.loads((work/'lsp_params.json').read_text())
rotation=Matrix.Rotation(records['forklift']['rot_z_deg']*3.141592653589793/180,3,'Z')
for cargo in cargos:
    index=int(cargo['name'].split('_')[-1]);sheet=records[f'lsp_{index}']
    delta=rotation.inverted()@(Vector(cargo['location'])-Vector(sheet['location']))
    bounds=json.loads((work/'assets'/f"{cargo['asset']}.json").read_text())
    assert abs(delta.z-params['thickness_m'])<1e-5,(cargo['name'],'cargo must rest on top of the LSP')
    assert delta.x+bounds['bounds_min'][0]>=-params['size_x_m']/2-.001
    assert delta.x+bounds['bounds_max'][0]<= params['size_x_m']/2+.001
    assert delta.y+bounds['bounds_min'][1]>=-params['size_y_m']/2-.001,(cargo['name'],'cargo falls behind LSP')
    assert delta.y+bounds['bounds_max'][1]<= params['size_y_m']/2+.001,(cargo['name'],'cargo falls ahead of LSP')
for name,record in records.items():
    source=json.loads((work/'assets'/f"{record['asset']}.json").read_text())
    transform=Matrix.Translation(Vector(record['location']))@Matrix.Rotation(record['rot_z_deg']*3.141592653589793/180,4,'Z')
    meshes=[o for o in bpy.context.scene.objects if o.name.startswith(name+'__')]
    assert len(meshes)==source['mesh_objects'],(name,'actor mesh count differs from source asset')
    points=[transform.inverted()@o.matrix_world@Vector(c) for o in meshes for c in o.bound_box]
    for axis in range(3):
        assert abs(min(p[axis] for p in points)-source['bounds_min'][axis])<1e-4,(name,'minimum bound mismatch',axis)
        assert abs(max(p[axis] for p in points)-source['bounds_max'][axis])<1e-4,(name,'maximum bound mismatch',axis)
assert records['cargo_0']
assert meta['verification']['is_violation_matches_lsp_count']
print(f"PASS: {stem}: {len(lsps)} LSPs, {len(cargos)} cargo actors, violation={meta['is_violation']}; all actor bounds match assets.")
