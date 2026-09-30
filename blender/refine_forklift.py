"""Source-fitted overhead-guard bends; coordinates are authored, not measured CAD."""
import bpy
import math
from mathutils import Matrix, Vector


def bezier_points(side, segments, samples=16):
    points=[]
    for segment in segments:
        p=[Vector((side*.51,y,z)) for y,z in segment]
        for i in range(samples+1):
            if points and i==0:continue
            t=i/samples
            points.append((1-t)**3*p[0]+3*(1-t)**2*t*p[1]+3*(1-t)*t*t*p[2]+t**3*p[3])
    return points


def replace_post(name,points):
    obj=bpy.data.objects[name];materials=list(obj.data.materials)
    vertices=[];faces=[];half=.075/2
    for i,point in enumerate(points):
        tangent=(points[min(i+1,len(points)-1)]-points[max(0,i-1)]).normalized()
        normal=Vector((0,tangent.z,-tangent.y))
        for x,v in [(-half,-half),(half,-half),(half,half),(-half,half)]:
            vertices.append(point+Vector((x,0,0))+normal*v)
    for i in range(len(points)-1):
        for j in range(4):faces.append((4*i+j,4*i+(j+1)%4,4*(i+1)+(j+1)%4,4*(i+1)+j))
    faces.extend([(3,2,1,0),tuple(4*(len(points)-1)+j for j in range(4))])
    mesh=bpy.data.meshes.new(name+'_curved');mesh.from_pydata(vertices,[],faces);mesh.update()
    obj.data=mesh;obj.matrix_world=Matrix.Identity(4);obj.modifiers.clear()
    for material in materials:mesh.materials.append(material)
    bevel=obj.modifiers.new('rounded_tube_edges','BEVEL');bevel.width=.006;bevel.segments=3
    obj['source_fitted_curved_guard']=True
    obj['centreline_points']=[list(p) for p in points]


def refine_guard():
    front=[[(.45,.75),(.45,1.12),(.21,1.68),(.10,1.83)],
           [(.10,1.83),(.04,1.91),(-.06,1.91),(-.20,1.91)]]
    rear=[[(-1.10,.72),(-1.12,1.05),(-1.20,1.61),(-1.20,1.76)],
          [(-1.20,1.76),(-1.20,1.86),(-1.16,1.91),(-1.07,1.91)]]
    for side in [-1,1]:
        replace_post(f'front_guard_{side}',bezier_points(side,front))
        replace_post(f'rear_guard_{side}',bezier_points(side,rear))


def install_mesh(name,vertices,faces,bevel_width=0):
    obj=bpy.data.objects[name];materials=list(obj.data.materials)
    mesh=bpy.data.meshes.new(name+'_rounded');mesh.from_pydata(vertices,[],faces);mesh.update()
    obj.data=mesh;obj.matrix_world=Matrix.Identity(4);obj.modifiers.clear()
    for material in materials:mesh.materials.append(material)
    if bevel_width:
        bevel=obj.modifiers.new('rounded_cast_edges','BEVEL');bevel.width=bevel_width;bevel.segments=4
    return obj


def refine_rear():
    # Short elliptical tail with low side skirts enclosing the smaller rear tires.
    for name,front,z0,z1,width in [('rear_counterweight',-.87,.14,.72,.025),
                                  ('chassis',.96,.26,.62,.035)]:
        # Inset the lower casting so overlapping shell faces are not coplanar.
        radius=.58 if name=='rear_counterweight' else .565
        depth=.38 if name=='rear_counterweight' else .365
        outline=[(-radius,front)]
        outline.extend((radius*math.cos(math.pi+i*math.pi/40),
                        -1.10+depth*math.sin(math.pi+i*math.pi/40)) for i in range(41))
        outline.append((radius,front));n=len(outline)
        vertices=[(x,y,z) for z in [z0,z1] for x,y in outline]
        faces=[tuple(reversed(range(n))),tuple(range(n,2*n))]
        faces.extend((i,(i+1)%n,(i+1)%n+n,i+n) for i in range(n))
        obj=install_mesh(name,vertices,faces,width)
        obj['rounded_rear_radius_m']=radius
        obj['rounded_rear_depth_radius_m']=depth
        for face in obj.data.polygons[2:]:face.use_smooth=True
    # Existing vents follow the new shell instead of floating behind its corners.
    for i in range(6):
        name=f'rear_vent_{i}'
        if name not in bpy.data.objects:continue
        z=.30+i*.045;vertices=[];faces=[]
        for j in range(25):
            x=-.32+j*.64/24;y=-1.10-.38*math.sqrt(1-(x/.58)**2)-.008
            vertices.extend([(x,y,z-.0075),(x,y,z+.0075)])
        for j in range(24):faces.append((2*j,2*j+2,2*j+3,2*j+1))
        install_mesh(name,vertices,faces)
    # Keep real wheel geometry inside open-bottom wheel wells, rather than
    # intersecting the tires with a solid casting or hiding wheels from rendering.
    for side in [-1,1]:
        bpy.ops.mesh.primitive_cylinder_add(vertices=48,radius=.245,depth=.40,
            location=(side*.50,-1.10,.22),rotation=(0,math.pi/2,0))
        cutter=bpy.context.object;cutter.name='temporary_rear_wheel_well'
        bpy.context.view_layer.update()
        for name in ['rear_counterweight','chassis']:
            shell=bpy.data.objects[name]
            modifier=shell.modifiers.new('rear_wheel_well','BOOLEAN')
            modifier.operation='DIFFERENCE';modifier.solver='EXACT';modifier.object=cutter
            bpy.context.view_layer.objects.active=shell
            bpy.ops.object.modifier_apply(modifier=modifier.name)
            shell['rear_wheel_wells']=True
        bpy.data.objects.remove(cutter,do_unlink=True)


def refine_side_panels():
    # The source looked like it had two external bars; remove both side pieces.
    for side in [-1,1]:
        obj=bpy.data.objects.get(f'side_panel_{side}')
        if obj:bpy.data.objects.remove(obj,do_unlink=True)


def refine_wheels():
    # Revolve a rounded cross-section: broad tread, curved shoulders, recessed bore.
    for axle,radius in [('front',.348*.90),('rear',.22)]:
        for side in [-1,1]:
            obj=bpy.data.objects[f'{axle}_tire_{side}'];materials=list(obj.data.materials)
            n=96;vertices=[];faces=[]
            width=.26 if axle=='front' else .18
            half=width/2;shoulder=radius*.16;bore=radius*.57;lip=.008
            profile=[]
            # Clockwise rounded rectangle in the radius/axial plane.
            for cr,cz,rounding,start in [
                (radius-shoulder,-half+shoulder,shoulder,-90),
                (radius-shoulder,half-shoulder,shoulder,0),
                (bore+lip,half-lip,lip,90),
                (bore+lip,-half+lip,lip,180)]:
                for j in range(11):
                    a=math.radians(start+90*j/10)
                    profile.append((cr+rounding*math.cos(a),cz+rounding*math.sin(a)))
            for r,z in profile:
                vertices.extend((r*math.cos(2*math.pi*i/n),r*math.sin(2*math.pi*i/n),z/width) for i in range(n))
            for ring in range(len(profile)):
                other=(ring+1)%len(profile)
                for i in range(n):
                    j=(i+1)%n
                    faces.append((ring*n+i,ring*n+j,other*n+j,other*n+i))
            mesh=bpy.data.meshes.new(obj.name+'_annular');mesh.from_pydata(vertices,[],faces);mesh.update()
            obj.data=mesh
            for material in materials:mesh.materials.append(material)
            obj['rim_bore_radius_m']=radius*.57
            obj['rounded_tire_profile']=True
            obj['shoulder_radius_m']=shoulder
            obj['circumference_segments']=n
    # Moving the fronts inwards must not intersect a solid chassis casting.
    for side in [-1,1]:
        bpy.ops.mesh.primitive_cylinder_add(vertices=64,radius=.348*.90+.07,depth=.40,
            location=(side*.481,.58,.348*.90),rotation=(0,math.pi/2,0))
        cutter=bpy.context.object;cutter.name='temporary_front_wheel_well'
        bpy.context.view_layer.update()
        chassis=bpy.data.objects['chassis']
        modifier=chassis.modifiers.new('front_wheel_well','BOOLEAN')
        modifier.operation='DIFFERENCE';modifier.solver='EXACT';modifier.object=cutter
        bpy.context.view_layer.objects.active=chassis
        bpy.ops.object.modifier_apply(modifier=modifier.name)
        bpy.data.objects.remove(cutter,do_unlink=True)
        chassis['front_wheel_wells']=True
        # User's 7 cm is radial clearance around the tire circumference, not
        # lateral protrusion. Cover edge follows the existing .565 m chassis side.
        vertices=[];faces=[];samples=40;radius=.348*.90
        for i in range(samples+1):
            theta=math.pi*i/samples
            for x,r in [(side*.300,radius+.070),(side*.565,radius+.070),
                        (side*.565,radius+.095),(side*.300,radius+.095)]:
                vertices.append((x,.58+r*math.cos(theta),radius+r*math.sin(theta)))
        for i in range(samples):
            for j in range(4):faces.append((4*i+j,4*i+(j+1)%4,4*(i+1)+(j+1)%4,4*(i+1)+j))
        faces.extend([(3,2,1,0),tuple(4*samples+j for j in range(4))])
        if side<0:faces=[tuple(reversed(face)) for face in faces]
        fender=install_mesh(f'front_fender_{side}',vertices,faces,.003)
        fender['radial_clearance_m']=.07
