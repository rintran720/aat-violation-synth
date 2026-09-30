import numpy as np
from synth.composite_violation import composite
from synth.composite_anchor_poc import restore_foreground


def test_composite_preserves_protected_and_outside_pixels():
    bg=np.full((20,30,3),120,dtype='uint8')
    rgba=np.zeros((20,30,4),dtype='uint8');rgba[5:15,10:20]=[255,0,0,255]
    result,support=composite(bg,rgba,sigma=.65,protected_boxes=[(0,0,30,8)])
    assert np.array_equal(result[:8],bg[:8])
    assert np.array_equal(result[~support],bg[~support])
    assert result[10,15,0]>240 and result[10,15,1]<10


def test_foreground_mask_restores_source_pixels_only_inside_mask():
    background=np.full((8,9,3),120,dtype='uint8')
    rendered=np.full_like(background,240)
    mask=np.zeros((8,9),dtype=bool);mask[2:6,3:7]=True
    result=restore_foreground(rendered,background,mask)
    assert np.array_equal(result[mask],background[mask])
    assert np.array_equal(result[~mask],rendered[~mask])


def test_transparent_rgb_does_not_create_black_halos():
    bg=np.full((20,20,3),255,dtype='uint8')
    rgba=np.zeros((20,20,4),dtype='uint8');rgba[8:12,8:12]=255
    result,_=composite(bg,rgba,sigma=1)
    assert result.min()>=253


def test_saturation_only_changes_inserted_object_colour():
    bg=np.full((12,12,3),[70,90,130],dtype='uint8')
    rgba=np.zeros((12,12,4),dtype='uint8');rgba[3:9,3:9]=[230,220,20,255]
    mask=rgba[...,3]>0
    original,_=composite(bg,rgba,sigma=0)
    muted,support=composite(bg,rgba,sigma=0,saturation=.5,object_mask=mask)
    assert np.ptp(muted[6,6].astype(int))<np.ptp(original[6,6].astype(int))
    assert np.array_equal(muted[~support],bg[~support])


def test_alpha_is_contracted_before_feathering():
    bg=np.full((32,32,3),40,dtype='uint8')
    rgba=np.zeros((32,32,4),dtype='uint8');rgba[8:24,8:24]=[190,190,190,255]
    _,support=composite(bg,rgba,sigma=0,alpha_erosion_px=1)
    assert not support[8:24,8:24].any() or support[9:23,9:23].all()
    assert not support[8,:].any()


def test_lens_radial_distortion_bends_object_edges():
    bg=np.zeros((64,64,3),dtype='uint8')
    rgba=np.zeros((64,64,4),dtype='uint8')
    rgba[10:14,8:56]=[255,255,255,255]
    _,support=composite(bg,rgba,sigma=0,lens_k1=-.35)
    # Barrel distortion moves the ends of a horizontal line toward image center.
    ys,xs=np.where(support)
    assert xs.min()>8 and xs.max()<56
    assert ys.max()-ys.min()>=3


def test_configurable_noise_is_seeded_and_does_not_touch_background():
    bg=np.full((24,24,3),80,dtype='uint8')
    rgba=np.zeros((24,24,4),dtype='uint8');rgba[8:16,8:16]=[150,150,150,255]
    first,support=composite(bg,rgba,sigma=0,noise_sigma=.03)
    second,_=composite(bg,rgba,sigma=0,noise_sigma=.03)
    assert np.array_equal(first,second)
    assert np.array_equal(first[~support],bg[~support])
    assert np.std(first[8:16,8:16,0])>0
