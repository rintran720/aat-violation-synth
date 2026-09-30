import json
from pathlib import Path
import pytest
from synth.build_assets import Author, palette, forklift


def test_harness_assigns_material_to_correct_object_and_wheels_use_local_axis(tmp_path):
    for p in ['cli','logs','previews']:(tmp_path/p).mkdir()
    a=Author('forklift',tmp_path,1);palette(a);forklift(a);a.finish()
    project=json.loads(a.project.read_text())
    materials={m['id']:m['name'] for m in project['materials']}
    objects={o['name']:o for o in project['objects']}
    assert materials[objects['rear_counterweight']['material']]=='tex_yellow'
    for side in [-1,1]:
        wheel=objects[f'front_tire_{side}']
        assert materials[wheel['material']]=='rubber'
        assert wheel['rotation']==[0,90,0]
        assert wheel['scale']==[1,1,.26]  # thickness follows local Z, rotated to world X
    assert objects['fork_tine_1']['location'][1]>objects['rear_counterweight']['location'][1]
