"""Small geometric bevels with face-weighted normals; preserve fitted dimensions."""
import math


def soften_edges(meshes,name):
    settings=[]
    for obj in meshes:
        if obj.get('rounded_tire_profile'):
            # The swept curved profile supplies its own smooth normals.
            obj['softened_edges']=True
            settings.append({'name':obj.name,'rounded_profile':True,'bevel_widths_m':[]})
            continue
        dims=sorted(float(v) for v in obj.dimensions)
        bevels=[m for m in obj.modifiers if m.type=='BEVEL']
        for bevel in bevels:
            # Keep existing broad silhouette bends; soften previously sharp edges.
            bevel.width=max(bevel.width,min(bevel.width*1.35,dims[0]*.3))
            bevel.segments=max(5,bevel.segments)
            bevel.harden_normals=True
        if not bevels:
            width=.0025 if name.startswith('lsp_') else min(.003,dims[0]*.08)
            if width>.0001:
                bevel=obj.modifiers.new('softened_edges','BEVEL')
                bevel.width=width;bevel.segments=5;bevel.limit_method='ANGLE'
                bevel.angle_limit=math.radians(35);bevel.harden_normals=True
                bevels=[bevel]
        # Flat-face normals prevent bevels from making broad panels look inflated.
        normal=obj.modifiers.new('surface_normals','WEIGHTED_NORMAL')
        normal.keep_sharp=True;normal.weight=40
        obj['softened_edges']=True
        settings.append({'name':obj.name,'bevel_widths_m':[m.width for m in bevels]})
    return settings
