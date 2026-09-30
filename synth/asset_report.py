"""Create a local, self-contained review page and an overview contact sheet."""
import base64
import hashlib
import html
import json
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
from synth.common import load_config


def uri(path):
    return 'data:image/png;base64,'+base64.b64encode(Path(path).read_bytes()).decode()


def main():
    work=load_config()['work']; review=work/'asset_review';review.mkdir(exist_ok=True)
    sources=json.loads((work/'asset_sources.json').read_text())
    for path,digest in sources['input_sha256'].items():
        assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==digest,f'Input was changed: {path}'
    names=['forklift','cargo_0','cargo_1','cargo_2','skid','lsp_0','lsp_1','lsp_2']
    labels=['Xe nang / Forklift','ULD sang / White ULD','ULD toi / Dark ULD','Thung carton / Cardboard',
            'SKID - uoc luong / provisional','LSP - texture 1','LSP - texture 2','LSP - texture 3']
    font_path='/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'
    font=ImageFont.truetype(font_path,20);small=ImageFont.truetype(font_path,15)
    sheet=Image.new('RGB',(1600,1000),'#101820');draw=ImageDraw.Draw(sheet)
    draw.text((24,15),'AAT | 3D asset review - cam01',font=font,fill='white')
    draw.text((24,48),'Authored from SAM3 crops + MoGe calibration. Provisional models: human realism review pending.',font=small,fill='#b7c6d1')
    cards=[]
    refs={'forklift':'frame_16_forklift_1.png','cargo_0':'rtsp_010_cargo_0.png',
          'cargo_1':'rtsp_010_cargo_1.png','cargo_2':'frame_07_cargo_0.png','skid':'frame_10_SKID_1.png'}
    for i,(name,label) in enumerate(zip(names,labels)):
        meta=json.loads((work/'assets'/f'{name}.json').read_text())
        dims=' × '.join(f'{v:.3f}' for v in meta['dimensions_m'])+' m'
        im=Image.open(work/'previews'/f'{name}_pass3.png').convert('RGB');im.thumbnail((390,340))
        x=(i%4)*400+5;y=(i//4)*445+90
        sheet.paste(im,(x,y));draw.text((x+8,y+342),label,font=small,fill='white')
        draw.text((x+8,y+367),dims,font=small,fill='#7acbd5')
        draw.text((x+8,y+390),f"{meta['mesh_objects']} meshes | {name}.blend",font=small,fill='#b7c6d1')
        reference=work/'refs'/refs[name] if name in refs else work/'textures'/f'{name}.png'
        buttons=[]
        for suffix,title in [('', 'Góc trước'),('_rear','Góc sau')]:
            buttons.append(f'<button data-src="{uri(work/"previews"/f"{name}_pass3{suffix}.png")}">{title}</button>')
        for stage in [1,2]:
            p=work/'previews'/f'{name}_pass{stage}.png'
            if p.exists():buttons.append(f'<button data-src="{uri(p)}">Lượt {stage}</button>')
        cards.append(f'''<article><h2>{html.escape(label)}</h2><div class="pair">
<figure><img src="{uri(reference)}"><figcaption>Ảnh nguồn / texture thật</figcaption></figure>
<figure><img class="preview" src="{uri(work/'previews'/f'{name}_pass3.png')}"><figcaption>Model 3D đã dựng</figcaption></figure></div>
<div class="buttons">{''.join(buttons)}</div><p>{dims} · {meta['mesh_objects']} mesh</p>
<a href="../assets/{name}.blend">Mở / tải .blend</a> · <a href="../assets/{name}.glb">Tải .glb</a>
<details><summary>Thông số và giới hạn</summary><pre>{html.escape(json.dumps(meta,indent=2,ensure_ascii=False))}</pre></details></article>''')
    sheet.save(review/'contact_sheet.jpg',quality=94)
    limitations=''.join('<li>'+html.escape(item)+'</li>' for item in sources['limitations'])
    page='''<!doctype html><html lang="vi"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>AAT — 3D asset review</title><style>body{margin:0;background:#101820;color:#eef4f7;font:16px system-ui;line-height:1.6}main{max-width:1150px;margin:auto;padding:30px}h1{font-size:32px;margin-bottom:5px}p{color:#bdccd7}article{background:#192731;padding:24px;margin:25px 0;border-radius:14px}h2{margin-top:0}.pair{display:grid;grid-template-columns:1fr 2fr;gap:18px}figure{margin:0}img{width:100%;height:360px;object-fit:contain;background:#263540;border-radius:8px}figcaption{color:#c4d5df}button{background:#314754;color:white;border:1px solid #526874;border-radius:6px;padding:8px 14px;margin:10px 6px 0 0;cursor:pointer}a{color:#78d4de}pre{white-space:pre-wrap;font-size:13px}.notice{border-left:4px solid #edab63;padding:12px 20px;background:#25313a}summary{cursor:pointer;margin-top:16px}@media(max-width:650px){.pair{grid-template-columns:1fr}img{height:270px}main{padding:15px}}</style>
<main><h1>AAT · Thư viện model 3D</h1><p>cam01 · 8 assets · Blender 5.2.2 · Mét · +Z lên trên · +Y hướng trước</p>
<div class="notice">Đây là bản dựng để duyệt, chưa phải dữ liệu tổng hợp đã nghiệm thu. LSP dùng số đo MoGe; các kích thước còn lại là ước lượng từ ảnh. SKID thiếu ảnh đầy đủ nên dùng hình học pallet tham chiếu.</div>'''+''.join(cards)+f'<h2>Giới hạn nguồn dữ liệu</h2><ul>{limitations}</ul></main>'+'''
<script>document.querySelectorAll('button[data-src]').forEach(b=>b.onclick=()=>b.closest('article').querySelector('.preview').src=b.dataset.src)</script></html>'''
    (review/'index.html').write_text(page)
    print(f'Review: {review}/index.html; contact_sheet.jpg; all source input hashes unchanged')


if __name__=='__main__':main()
