import numpy as np
import pytest
from synth.asset_sources import homography, rectify, floor_point


def test_projective_texture_preserves_distinct_corners():
    img = np.zeros((30,40,3), float)
    img[:15,:20,0]=255; img[:15,20:,1]=255; img[15:,20:,2]=255
    img[15:,:20]=255
    result=rectify(img,[[0,0],[39,0],[39,29],[0,29]],size=32)
    np.testing.assert_allclose(result[3,3],[255,0,0])
    np.testing.assert_allclose(result[3,-4],[0,255,0])
    np.testing.assert_allclose(result[-4,-4],[0,0,255])
    np.testing.assert_allclose(result[-4,3],[255,255,255])


def test_degenerate_quad_is_rejected():
    with pytest.raises(ValueError, match='Degenerate'):
        homography([[0,0],[0,0],[1,1],[2,2]],[[0,0],[1,0],[1,1],[0,1]])


def test_floor_intersection_uses_camera_conventions():
    cam=dict(width=100,height=100,K_norm=[[1,0,.5],[0,1,.5],[0,0,1]],
             matrix_world=[[1,0,0,0],[0,1,0,0],[0,0,1,5],[0,0,0,1]])
    np.testing.assert_allclose(floor_point([50,50],cam),[0,0,0])
    np.testing.assert_allclose(floor_point([70,60],cam),[1,-.5,0])
