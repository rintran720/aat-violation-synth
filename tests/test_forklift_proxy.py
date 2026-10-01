import numpy as np
import pytest

from synth.forklift_proxy import FORK_FACE, build, layout, load_spec


@pytest.fixture
def spec():
    return load_spec('sumitomo_quapro_2t5_dual')


def test_axles_follow_catalogue(spec):
    g = layout(spec)
    assert FORK_FACE - g['front_axle'] == pytest.approx(spec['axle_to_fork_face_m'])
    assert g['front_axle'] - g['rear_axle'] == pytest.approx(spec['wheelbase_m'])


def test_dual_front_tyres_span_overall_width(spec):
    P, L = build(spec)
    front = P[L == 3]
    assert np.abs(front[:, 0]).max() == pytest.approx(spec['overall_width_dual_m'] / 2, abs=.005)
    # two tyres per side
    assert len(layout(spec)['front_tyres']) == 2


def test_z_scale_only_changes_height(spec):
    P1, L1 = build(spec, 1.0)
    P2, L2 = build(spec, .85)
    assert np.array_equal(L1, L2)
    assert np.allclose(P1[:, :2], P2[:, :2])
    assert P2[:, 2].max() == pytest.approx(spec['overhead_guard_m'] * .85, rel=1e-4)
