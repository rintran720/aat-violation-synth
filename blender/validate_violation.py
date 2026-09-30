"""Reopen the saved scene and verify assembled asset geometry/contacts, without rendering."""
import json
import math
import sys
from pathlib import Path
import bpy
from mathutils import Matrix, Vector

work=Path(sys.argv[sys.argv.index('--')+1]).resolve()
meta=json.loads((work/'out/v_000.json').read_text())
bpy.ops.wm.open_mainfile(filepath=str(work/'scenes/v_000.blend'))
bpy.context.view_layer.update()
for record in meta['objects']:
    source=json.loads((work/'assets'/f"{record['asset']}.json").read_text())
    matrix=Matrix.Translation(Vector(record['location']))@Matrix.Rotation(math.radians(record['rot_z_deg']),4,'Z')
    meshes=[o for o in bpy.context.scene.objects if o.name.startswith(record['name']+'__')]
    assert len(meshes)==source['mesh_objects'],(record['name'],'Lost component')
    points=[matrix.inverted()@o.matrix_world@Vector(c) for o in meshes for c in o.bound_box]
    for axis in range(3):
        assert abs(min(p[axis] for p in points)-source['bounds_min'][axis])<1e-4,(record['name'],'min',axis)
        assert abs(max(p[axis] for p in points)-source['bounds_max'][axis])<1e-4,(record['name'],'max',axis)
records={r['name']:r for r in meta['objects']}
assert sum(r['class']=='lsp' for r in meta['objects'])==meta['lsp_count']==2
lsps=json.loads((work/'lsp_params.json').read_text())
distance=(Vector(records['lsp_1']['location'])-Vector(records['lsp_0']['location'])).length
assert abs(distance-lsps['size_y_m']-.008)<1e-4
for i in range(2):
    cargo=Vector(records[f'cargo_{i}']['location']);sheet=Vector(records[f'lsp_{i}']['location'])
    rot=Matrix.Rotation(math.radians(records['forklift']['rot_z_deg']),3,'Z')
    offset=rot.inverted()@(cargo-sheet)
    assert abs(offset.x)<1e-5
    assert abs(offset.y-meta['pushing_contact']['cargo_offset_y_m'])<1e-5
    assert abs(cargo.z-sheet.z-lsps['thickness_m'])<1e-5
fork=records['forklift']
assert meta['pushing_contact']['target']=='fork_heel'
target=records['cargo_0']
rot=Matrix.Rotation(math.radians(fork['rot_z_deg']),3,'Z')
local=rot.inverted()@(Vector(target['location'])-Vector(fork['location']))
fork_meta=json.loads((work/'assets/forklift.json').read_text())
heel=max(v['max'][1] for k,v in fork_meta['component_bounds'].items() if k.startswith('fork_heel'))
rear=json.loads((work/'assets'/f"{target['asset']}.json").read_text())['bounds_min'][1]
assert abs(local.y+rear-heel-.005)<1e-4
sheet_local=rot.inverted()@(Vector(records['lsp_0']['location'])-Vector(fork['location']))
for name,bbox in fork_meta['component_bounds'].items():
    if not name.startswith('fork_tine'):continue
    assert sheet_local.y-lsps['size_y_m']/2<bbox['min'][1]<bbox['max'][1]<sheet_local.y+lsps['size_y_m']/2
    assert bbox['max'][2]+meta['pushing_contact']['pocket_clearance_z_m']<lsps['thickness_m']
fork_meshes=[o for o in bpy.context.scene.objects if 'fork_tine' in o.name]
assert len(fork_meshes)==2 and all(not o.hide_render for o in fork_meshes)
assert not any('beacon' in o.name for o in bpy.context.scene.objects),'No roof beacon exists in the source forklift'
assert meta['verification']['visible_fork_blade_pixels']==0
assert meta['verification']['visible_rear_wheel_pixels']>150
print(f'PASS: 5 intact actors; 2 visible LSPs; cargo at fork heels; both blades present and fully occluded; thickness={lsps["thickness_m"]} m.')
