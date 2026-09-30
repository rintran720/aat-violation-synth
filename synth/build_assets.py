"""Author reusable cam01 models through the installed CLI-Anything Click commands.

python -m synth.build_assets --pass-number 1  # blockout
python -m synth.build_assets --pass-number 2  # component detail
python -m synth.build_assets --pass-number 3  # packed real textures + final assets

No SAM3/MoGe inference, external service, or LLM call is made. Commands are logged.
"""
import argparse
import json
import math
from pathlib import Path
import shlex
import subprocess

from click.testing import CliRunner
from cli_anything.blender import blender_cli
from synth.common import load_config, write_json


def vector(values):
    return ','.join(str(round(float(v),6)) for v in values)


class Author:
    def __init__(self, name, work, stage):
        self.name, self.work, self.stage = name, work, stage
        self.project = work/'cli'/f'{name}_pass{stage}.blend-cli.json'
        self.commands=[]; self.runner=CliRunner(); self.materials={}; self.bevels={}
        blender_cli._session=None
        self.call('scene','new','--name',name,'--engine','CYCLES','--samples','16',
                  '--output',str(self.project))

    def call(self,*args):
        args=['--json',*map(str,args)]
        result=self.runner.invoke(blender_cli.cli,args)
        self.commands.append('cli-anything-blender '+shlex.join(args))
        if result.exit_code:
            raise RuntimeError(f'{self.commands[-1]}\n{result.output}\n{result.exception}')
        # Some errors in the upstream auto-save hook only emit warnings.
        if 'Auto-save failed' in result.output:
            raise RuntimeError(result.output)
        return json.loads(result.output) if result.output.strip() else None

    def material(self,name,color,rough=.5,metal=0):
        result=self.call('material','create','--name',name,'--color',vector([*color,1]),
                         '--roughness',rough,'--metallic',metal)
        self.materials[name]=result['id']

    def part(self,name,loc,size,mat,shape='cube',rotation=(0,0,0),bevel=.01,params=None):
        args=['object','add',shape,'--name',name,'--location='+vector(loc),
              '--scale='+vector(size),'--rotation='+vector(rotation)]
        params=params or ({'size':1} if shape=='cube' else {})
        for k,v in params.items(): args+=['--param',f'{k}={v}']
        obj=self.call(*args)
        self.call('material','assign',self.materials[mat],obj['id'])
        if bevel and shape=='cube':
            self.call('modifier','add','bevel','--object',obj['id'],'--param',f'width={bevel}',
                      '--param','segments=3')
            self.bevels[name]=bevel
        return obj['id']

    def beam(self,name,a,b,width,depth,mat,bevel=.008):
        delta=[b[i]-a[i] for i in range(3)]; length=math.sqrt(sum(x*x for x in delta))
        # All slanted beams here lie in YZ. Cube's local Z follows the beam.
        assert abs(delta[0])<1e-6
        angle=-math.degrees(math.atan2(delta[1],delta[2]))
        return self.part(name,[(a[i]+b[i])/2 for i in range(3)],(width,depth,length),mat,
                         rotation=(angle,0,0),bevel=bevel)

    def finish(self):
        # Persist a portable CLI project plus its own generated bpy export, without rendering.
        self.call('scene','save')
        script=self.work/'cli'/f'{self.name}_pass{self.stage}.py'
        project=json.loads(self.project.read_text())
        from cli_anything.blender.utils.bpy_gen import generate_full_script
        script.write_text(generate_full_script(project,str((self.work/'previews'/'unused.png').resolve())))
        (self.work/'logs'/f'{self.name}_pass{self.stage}_commands.log').write_text('\n'.join(self.commands)+'\n')
        return script


def palette(a):
    for spec in [
        ('tex_yellow',(.73,.68,.015),.48,.12), ('yellow',(.76,.70,.016),.42,.1),
        ('steel',(.055,.065,.07),.43,.72),('black',(.018,.021,.021),.62,.05),
        ('roof_black',(.001,.001,.001),.96,0),
        ('rubber',(.026,.029,.027),.84,0),('silver',(.46,.50,.51),.36,.72),
        ('seat',(.026,.023,.025),.7,0),('roof',(.20,.23,.22),.65,.22),
        ('light',(.86,.88,.77),.25,.25),('red',(.55,.012,.008),.28,0),
        ('tex_white',(.72,.75,.75),.42,.35),('tex_white_side',(.70,.72,.73),.43,.35),
        ('tex_dark',(.075,.10,.15),.48,.3),('tex_carton',(.32,.28,.12),.87,0),
        ('tex_carton_plain',(.30,.26,.115),.9,0),('wood',(.34,.235,.11),.86,0),
        ('tape',(.50,.40,.21),.58,0),('label',(.77,.78,.73),.85,0),
        ('tex_body_side',(.5,.45,.03),.78,.04),
        ('tex_counter_side',(.5,.45,.03),.78,.04),
        ('tex_side_panel',(.04,.04,.03),.85,0),
        ('tex_roof',(.20,.23,.22),.86,.06),
        ('tex_white_roof',(.72,.75,.75),.76,.12),
        ('tex_dark_roof',(.15,.17,.18),.82,.12),
        ('wheel_rim',(.045,.049,.047),.82,.18),
        ('wheel_fastener',(.028,.031,.030),.88,.12),
    ]: a.material(*spec)


def forklift(a):
    # Body length ~2.7m excluding forks. Four wheels, open cab, independent mast and forks.
    p=a.part
    roof_top=1.95+.045/2
    # Latest correction: reduce revision 7 front diameter by 10%, keep rear size.
    front_radius=.348*.90;rear_radius=.22
    axles=[('front',.58,front_radius),('rear',-1.10,rear_radius)]
    p('chassis',(0,-.36,.44),(1.16,2.64,.36),'tex_yellow',bevel=.065)
    p('rear_counterweight',(0,-1.27,.79),(1.16,.80,.68),'tex_yellow',bevel=.16)
    # Smaller, lower seat pedestal frees cabin space without raising the roof.
    p('engine_cover',(0,-.53,.76),(.86*.90,.68,.22),'yellow',bevel=.045)
    # Continuous black cabin floor from the mast face back to the counterweight.
    p('floorboard',(0,-.03,.6575),(.94,1.72,.035),'black',bevel=.008)
    for side in [-1,1]:
        for axle,y,r in axles:
            track,width=(.481,.26) if axle=='front' else (.50,.18)
            p(f'{axle}_tire_{side}',(side*track,y,r),(1,1,width),'rubber','cylinder',
              rotation=(0,90,0),params={'radius':r,'depth':1,'vertices':40},bevel=0)
        p(f'front_fender_{side}',(side*.4325,.58,.6732),(.265,.70,.03),'yellow',bevel=.004)
        # Blockout proxies; export_asset replaces these four posts with swept
        # curved meshes for detail/final passes, matching the source guard bends.
        a.beam(f'front_guard_{side}',(side*.51,.45,.75),(side*.51,-.20,1.91),.075,.075,'yellow')
        a.beam(f'rear_guard_{side}',(side*.51,-1.10,.72),(side*.51,-1.07,1.91),.075,.075,'yellow')
        p(f'roof_rail_{side}',(side*.51,-.62,1.91),(.09,1.10,.08),'yellow')
        p(f'mast_outer_{side}',(side*.41,.92,(roof_top-.12)/2),(.125,.17,roof_top-.12),'steel')
    p('roof_rear_panel',(0,-.89,1.95),(1.09,.72,.045),'roof_black',bevel=.025)
    for y in [-1.16,-.07]: p(f'roof_cross_{y}',(0,y,1.91),(1.1,.075,.08),'yellow')
    p('seat_base',(0,-.52,.925),(.44,.46,.11),'seat',bevel=.045)
    p('seat_back',(0,-.755,1.13),(.45,.105,.40),'seat',rotation=(8,0,0),bevel=.045)
    p('mast_top',(0,.92,roof_top-.06),(.96,.19,.12),'steel')
    # Compact mounting crossbar, rather than a solid plate hiding the open frame.
    p('carriage',(0,1.035,.37),(1.04,.16,.12),'steel',bevel=.012)
    for side in [-1,1]:
        p(f'fork_heel_{side}',(side*.29,1.13,.29),(.11,.07,.51),'steel',bevel=.008)
        p(f'fork_tine_{side}',(side*.29,1.65,.06),(.11,1.12,.055),'steel',bevel=.01)
    p('cabin_front_footwell',(0,.28,.642),(.66,.48,.035),'black',bevel=.008)
    # Third lift assembly: a body-width load backrest seated against the mast.
    # Body width includes the side panels, but excludes the removed boarding steps.
    backrest_width=2*(.606+.032/2)
    backrest_height=roof_top/2
    backrest_bottom=.11;backrest_top=backrest_bottom+backrest_height
    backrest_depth=.06
    backrest_y=1.018+.018/2+backrest_depth/2  # flush with front of mast/chain assembly
    for side in [-1,1]:
        p(f'load_backrest_side_{side}',(side*(backrest_width-.055)/2,backrest_y,(backrest_bottom+backrest_top)/2),
          (.055,backrest_depth,backrest_height),'steel',bevel=.008)
    for label,z in [('bottom',backrest_bottom+.065/2),('top',backrest_top-.065/2)]:
        p(f'load_backrest_cross_{label}',(0,backrest_y,z),(backrest_width,backrest_depth,.065),'steel',bevel=.008)
    for i,x in enumerate([-.4,-.2,0,.2,.4]):
        p(f'load_backrest_bar_{i}',(x,backrest_y,(backrest_bottom+backrest_top)/2),
          (.028,backrest_depth,backrest_height-.13),'steel',bevel=.005)
    if a.stage==1:return
    for side in [-1,1]:
        for axle,y,r in axles:
            hub_x,cap_x,bolt_x=(.571,.579,.583) if axle=='front' else (.550,.558,.562)
            p(f'{axle}_hub_{side}',(side*hub_x,y,r),(1,1,1),'wheel_rim','cylinder',
              rotation=(0,90,0),params={'radius':r*.55,'depth':.018,'vertices':32},bevel=0)
            p(f'{axle}_hubcap_{side}',(side*cap_x,y,r),(1,1,1),'black','cylinder',
              rotation=(0,90,0),params={'radius':r*.23,'depth':.025,'vertices':24},bevel=0)
            for n in range(6):
                angle=2*math.pi*n/6
                p(f'{axle}_bolt_{side}_{n}',(side*bolt_x,y+r*.38*math.cos(angle),r+r*.38*math.sin(angle)),
                  (1,1,1),'wheel_fastener','cylinder',rotation=(0,90,0),
                  params={'radius':.015,'depth':.018,'vertices':6},bevel=0)
        p(f'mast_inner_{side}',(side*.25,.91,(.40+roof_top-.11)/2),(.07,.09,roof_top-.11-.40),'silver')
        p(f'mast_chain_{side}',(side*.30,1.018,(.52+roof_top-.15)/2),(.025,.018,roof_top-.15-.52),'black',bevel=.002)
        p(f'headlamp_{side}',(side*.50,.14,1.71),(.12,.10,.095),'black')
        p(f'headlamp_lens_{side}',(side*.50,.197,1.71),(.091,.012,.066),'light',bevel=.009)
        rear_y=-1.10-.38*math.sqrt(1-(.46/.58)**2)
        rear_angle=side*math.degrees(math.atan2(.46/.58**2,-(rear_y+1.10)/.38**2))
        p(f'taillamp_{side}',(side*.468,rear_y-.008,.61),(.14,.02,.075),'red',rotation=(0,0,rear_angle))
        p(f'armrest_{side}',(side*.30,-.49,1.08),(.065,.33,.07),'seat',bevel=.023)
    for i in range(7):
        p(f'roof_grille_{i}',(-.43+i*.143,-.30,1.923),(.03,.46,.035),'yellow')
    a.beam('steering_column',(0,.25,.62),(0,.12,1.31),.055,.055,'steel')
    p('steering_wheel',(0,.12,1.33),(1,1,1),'black','torus',rotation=(22,0,0),
      params={'major_radius':.16,'minor_radius':.018,'major_segments':32,'minor_segments':8},bevel=0)
    for x in [.22,.28,.34]:
        a.beam(f'control_lever_{x}',(x,.12,.82),(x,.04,1.17),.018,.018,'steel',bevel=.004)
        p(f'lever_knob_{x}',(x,.04,1.19),(.022,.022,.025),'black','sphere',params={'radius':1,'segments':12,'rings':8},bevel=0)
    p('hydraulic_ram',(0,.82,(.355+roof_top-.12)/2),(1,1,1),'steel','cylinder',
      params={'radius':.06,'depth':roof_top-.12-.355,'vertices':24},bevel=0)
    for i in range(6):p(f'rear_vent_{i}',(0,-1.678,.30+i*.045),(.64,.012,.015),'black',bevel=.003)
    p('rear_label',(0,-1.489,.62),(.23,.01,.13),'label',bevel=.004)
    # The red patch above the source forklift is a worker's shirt, not a beacon.


def uld(a,dark=False):
    p=a.part; skin='tex_dark' if dark else 'tex_white'
    p('container_shell',(0,0,.79),(1.52,1.54,1.5),skin,bevel=.045)
    p('container_base',(0,0,.025),(1.58,1.60,.05),'silver',bevel=.015)
    p('container_roof',(0,0,1.55),(1.57,1.59,.055),'tex_dark_roof' if dark else 'tex_white_roof',bevel=.025)
    if a.stage==1:return
    # Surface texture plane spans this face only; roof and rails keep independent materials.
    p('front_panel',(0,.774,.78),(1.43,.014,1.36),skin,bevel=.003)
    if not dark:
        for side in [-1,1]:p(f'side_panel_{side}',(side*.765,0,.78),(.014,1.45,1.36),'tex_white_side',bevel=.003)
    for x in [-.755,.755]:
        for y in [-.765,.765]:p(f'corner_{x}_{y}',(x,y,.79),(.045,.045,1.50),'silver',bevel=.006)
    for z in [.095,1.48]:
        for y in [-.787,.787]:p(f'front_rail_{y}_{z}',(0,y,z),(1.53,.028,.038),'silver')
        for x in [-.778,.778]:p(f'side_rail_{x}_{z}',(x,0,z),(.028,1.56,.038),'silver')
    for x in [-.48,.48]:
        p(f'locking_bar_{x}',(x,.80,.78),(.026,.025,1.26),'silver',bevel=.003)
        p(f'latch_{x}',(x,.825,.69),(.12,.045,.035),'steel')
    for x in [-.64,.64]:
        for z in [.25,.48,.73,1.0,1.26]:
            p(f'rivet_{x}_{z}',(x,.796,z),(.009,.009,.009),'silver','sphere',params={'radius':1,'segments':8,'rings':4},bevel=0)


def carton(a):
    p=a.part
    p('cardboard_box',(0,0,.44),(.90,.78,.88),'tex_carton_plain',bevel=.025)
    if a.stage==1:return
    p('printed_face',(0,.397,.43),(.85,.018,.81),'tex_carton',bevel=.003)
    p('top_tape',(0,0,.883),(.08,.78,.008),'tape',bevel=.002)
    p('top_seam',(0,0,.889),(.003,.76,.002),'black',bevel=0)
    p('rear_tape',(0,-.394,.75),(.08,.01,.25),'tape',bevel=.002)


def skid(a):
    p=a.part
    # A provisional standard block pallet. Real segmentation has no complete deck.
    for x in [-.49,0,.49]:
        p(f'bottom_runner_{x}',(x,0,.013),(.22,1.0,.026),'wood',bevel=.006)
        for y in [-.39,0,.39]:p(f'block_{x}_{y}',(x,y,.075),(.16,.19,.098),'wood',bevel=.008)
    for y in [-.39,0,.39]:p(f'transverse_bearer_{y}',(0,y,.126),(1.2,.18,.024),'wood',bevel=.004)
    for i in range(7):
        x=-.53+i*(1.06/6)
        p(f'deck_board_{i}',(x,0,.144),(.14,1.0,.012),'wood',bevel=.003)
        if a.stage>1:
            for y in [-.38,.0,.38]:
                p(f'nail_{i}_{y}',(x,y,.1505),(.004,.004,.001),'steel','cylinder',params={'radius':1,'depth':1,'vertices':8},bevel=0)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pass-number',type=int,choices=[1,2,3],required=True)
    parser.add_argument('--asset',choices=['forklift','cargo_0','cargo_1','cargo_2','skid'])
    args=parser.parse_args();cfg=load_config();work=cfg['work'].resolve()
    for name in ['cli','logs','assets_raw','assets','previews']:(work/name).mkdir(exist_ok=True)
    builders={'forklift':forklift,'cargo_0':uld,'cargo_1':lambda a:uld(a,True),'cargo_2':carton,'skid':skid}
    for name,builder in builders.items():
        if args.asset and args.asset!=name:continue
        a=Author(name,work,args.pass_number);palette(a);builder(a);script=a.finish()
        command=['blender','--background','--threads','4','--python-exit-code','1','--python',
                 'blender/export_asset.py','--',str(script),name,str(args.pass_number),str(work)]
        log=work/'logs'/f'{name}_pass{args.pass_number}_blender.log'
        print(f'{name} pass {args.pass_number}: exporting and rendering',flush=True)
        with log.open('w') as f:subprocess.run(command,stdout=f,stderr=subprocess.STDOUT,check=True)
        if args.pass_number==3:
            (work/'cli'/f'{name}.blend-cli.json').write_text(a.project.read_text())
        print(f'  done: {work / "previews" / (name+"_pass"+str(args.pass_number)+".png")}',flush=True)


if __name__=='__main__':main()
