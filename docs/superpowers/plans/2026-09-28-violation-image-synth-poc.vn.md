# POC tổng hợp ảnh vi phạm AAT — Kế hoạch triển khai

> Các bước dùng checkbox (`- [ ]`) để theo dõi. Thư mục dự án không phải git repository và kế hoạch này không có bước commit.

**Mục tiêu:** Với 1 camera, từ hai đầu vào riêng biệt, đúng 1 ảnh nền sạch (không có xe nâng, LSP, SKID hay hàng; nền cho mọi output và đầu vào cho bước hiệu chỉnh MoGe-2) và ~10 ảnh tham chiếu của cùng góc nhìn có xe nâng, LSP, SKID và hàng ở nhiều vị trí khác nhau (chỉ dùng để tách đối tượng, không bao giờ dùng làm nền cho output), tạo ~10 ảnh vi phạm tổng hợp "Forklift Pushing Multiple Lsps" và vài ảnh hợp lệ tổng hợp. Ảnh hợp lệ tổng hợp là đề xuất bổ sung của chúng tôi (không có trong yêu cầu của Gary, vốn chỉ yêu cầu ảnh vi phạm): xe nâng, hàng và LSP chỉ được render trong ảnh tổng hợp, nên các ảnh này ngăn model học "vật thể render = vi phạm" và buộc model đếm số tấm LSP bị đẩy. Mỗi ảnh chèn một xe nâng đẩy hàng trên một chuỗi tấm LSP: với ảnh vi phạm, mặc định tổng cộng 2 tấm, ít hơn là 3 tấm (cấu hình được); với ảnh hợp lệ, đúng 1 tấm, và một số ảnh hợp lệ thay vào đó chỉ có một hàng tấm LSP nằm yên, không có xe nâng. Mỗi ảnh còn đặt thêm 0–2 SKID nằm yên (một số có hàng bên trên) làm vật gây nhiễu trong cảnh, cách xa đường đi của xe nâng. Mọi ảnh đúng format ảnh đầu vào, trông hợp lý bằng mắt, và có một file sidecar JSON (`lsp_count`, `is_violation`, box đối tượng, box sự kiện cho ảnh vi phạm). Ticket này chỉ sinh ảnh, sidecar và script rồi bàn giao cho Terry, người sẽ gán nhãn, train và test (xem "Sau POC").

**Camera (khái niệm):** một góc nhìn cố định — một ảnh nền sạch cùng các frame có đối tượng chụp từ đúng góc nhìn đó, với hiệu chỉnh riêng (tỉ lệ, mặt phẳng sàn, ống kính). Khái niệm này không gắn với một thiết bị vật lý: bất kỳ tập ảnh nào chung một góc nhìn đều tính là một camera, và một camera vật lý di chuyển được (ví dụ các preset PTZ) cho ra nhiều camera. `camera_id` đặt tên cho góc nhìn này.

**Kiến trúc:** Cách B. Nền sạch của camera được giữ làm nền. SAM3 phân vùng các đối tượng chính (forklift, LSP, SKID, cargo) trên ~10 ảnh tham chiếu và tạo thư viện ảnh cắt (ảnh cắt, mask, kích thước và vị trí 2D: tham chiếu asset, texture, đối chiếu kích thước, tham chiếu ánh sáng). MoGe-2 chạy trên nền sạch cùng mask sàn của SAM3 hiệu chỉnh camera Blender và một hệ toạ độ sàn theo mét, được đối chiếu với kích thước đo được của các tấm LSP thật; việc hiệu chỉnh này làm một lần cho mỗi camera. Codex (GPT Astra 6) dùng `cli-anything-blender` để dựng scene proxy vô hình của phần hình học tĩnh (shadow catcher, vật che) và đèn, đồng thời dựng lại xe nâng, các loại hàng và SKID thành asset 3D từ ảnh cắt, qua nhiều lượt. Script bpy tất định gắn texture từ ảnh cắt thật lên các asset đó và dựng asset LSP với texture nắn phối cảnh từ LSP thật. Bộ random có seed đặt xe nâng + hàng + N tấm LSP (N = 1 hoặc một hàng LSP nằm yên cho ảnh hợp lệ tổng hợp được đề xuất, N >= 2 cho ảnh vi phạm), cùng 0–2 SKID nằm yên cách xa đường đi của xe nâng, lên sàn. Cycles chỉ render các đối tượng chèn vào cùng bóng của chúng, rồi Python ghép lên nền với bảo vệ OSD, khớp nhiễu và độ mờ, và round-trip H.264 cục bộ. Spec: `docs/superpowers/specs/2026-09-28-violation-image-synth-design.md`.

**Công nghệ:** Máy trạm Ubuntu có GPU NVIDIA và CUDA. Python 3.12, numpy, Pillow, pytest, MoGe-2, SAM3 (bắt buộc có trên máy trạm), Blender ≥ 4.2 (Cycles), `cli-anything-blender` (CLI-Anything), Codex CLI (`codex exec`), ffmpeg.

**Điều kiện tiên quyết (đang chờ: các bên liên quan sẽ cung cấp):** máy trạm Ubuntu có GPU (yêu cầu Ubuntu, GPU NVIDIA, CUDA và SAM3; hiện chưa có), cần từ Task 1; và video gốc không overlay từ camera đích, có nền sạch và các đối tượng chính ở nhiều vị trí, cần từ Task 3.

## Quy ước

- Chạy mọi lệnh từ thư mục gốc dự án `aat-violation-synth/`, với venv đã kích hoạt (`source .venv/bin/activate`).
- Config duy nhất là `config.json`, mỗi camera một file. Mọi đường dẫn khác được suy ra từ `work_dir` (`work/`), thư mục chứa artifact của camera đó; kết quả hiệu chỉnh trong đó được dùng lại cho mọi ảnh của camera. Camera thứ hai (góc nhìn thứ hai) có config riêng với `camera_id`, đầu vào và `work_dir` riêng.
- Kiểm tra mức code: `python -m pytest -q`. Kiểm tra trên output thật: `python -m synth.verify` cộng với bước người duyệt ở Task 15.
- `data/raw/` chứa video gốc **không có overlay**. Không bao giờ dùng `docs/requirements/violation-clip-*.mp4` hoặc các frame của nó làm nền cho output cuối, vì chúng có overlay của detector.

## Cấu trúc file

```
aat-violation-synth/
  AGENTS.md                     # shared agent conventions (Codex reads it automatically)
  config.json                   # the only config (one per camera)
  requirements.txt              # numpy, pillow, pytest
  docs/spike-notes.md           # Task 1 results (CLI-Anything / Codex facts verified on this machine)
  prompts/
    01_proxy_scene.md           # agent: static proxy scene (floor, racks) + lights via cli-anything-blender
    02_propose_variations.md    # agent (optional): variations of one catalogue scenario (POC: V1, V2, N1, N2)
    03_lsp_params.md            # agent: LSP asset spec -> work/lsp_params.json
    04_scenario.md              # agent: scenario template -> work/scenario.json
    05_build_asset.md           # agent: forklift / cargo / SKID 3D asset via cli-anything-blender
  synth/
    __init__.py
    common.py                   # load_config, write_json, apply_lsp_size
    segment.py                  # SAM3 masks + object crop library (crops, 2D box, position)
    calibrate.py                # per camera: MoGe-2 on the clean background + object cross-check
    lsp_texture.py              # rectify real LSP masks into textures
    randomize.py                # scenario -> seeded variants (violation + valid, idle SKIDs)
    composite.py                # alpha-over, OSD mask, blur/noise, H.264/JPEG format match, overlay
    verify.py                   # format / OSD / sidecar checks
    contact_sheet.py            # review grid
  blender/
    cli_to_blend.py             # run CLI-Anything export without rendering -> .blend
    build_lsp.py                # textured rounded thin sheet -> work/assets/lsp_<k>.blend
    texture_asset.py            # real-crop texture on an agent-built asset -> work/assets/<name>.blend
    render_variants.py          # calibrated camera, shadow catchers, render + sidecars
  tests/
    test_segment.py  test_calibrate.py  test_lsp_texture.py  test_randomize.py  test_composite.py  test_verify_sheet.py
  data/raw/  data/frames/  data/input/cam01/background.png  data/input/cam01/objects/obj_*.png
  vendor/                       # CLI-Anything and SAM3 checkouts
  work/                         # all generated artifacts of this camera (safe to delete)
```

---

### Task 1: Cài đặt máy trạm và spike CLI-Anything / Codex

**File:**
- Tạo: `blender/cli_to_blend.py`, `docs/spike-notes.md`

- [ ] **Bước 1: Tạo thư mục và môi trường Python**

```bash
mkdir -p synth blender tests prompts docs data/raw data/frames data/input/cam01/objects vendor work
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader   # Ubuntu workstation: NVIDIA GPU + CUDA
python3.12 -m venv --system-site-packages .venv   # sees the torch + SAM3 already installed on this workstation
source .venv/bin/activate
pip install --upgrade pip
python -c "import torch, sam3" 2>/dev/null || {   # skipped when SAM3 is already installed
  pip install torch==2.10.0 torchvision --index-url https://download.pytorch.org/whl/cu128
  git clone https://github.com/facebookresearch/sam3.git vendor/sam3
  pip install -e vendor/sam3
}
python -c "import torch, sam3; assert torch.cuda.is_available(), 'CUDA not available'; print('torch', torch.__version__, 'CUDA OK, SAM3 OK')"
pip install git+https://github.com/microsoft/MoGe.git
git clone https://github.com/HKUDS/CLI-Anything.git vendor/CLI-Anything
pip install -e vendor/CLI-Anything/blender/agent-harness
pip install numpy pillow pytest
hf auth whoami || hf auth login   # SAM3 checkpoints are gated: request access on the SAM3 Hugging Face page first
```

Kỳ vọng: mọi lệnh cài đặt thành công. CUDA và SAM3 được kỳ vọng có sẵn trên máy trạm Ubuntu do các bên liên quan cung cấp; bước kiểm tra xác nhận điều này và bỏ qua khối cài torch/SAM3 khi import được SAM3, hoặc cài đặt khi còn thiếu. Dù trường hợp nào, dòng kiểm tra vẫn phải in ra `CUDA OK, SAM3 OK`. Nếu SAM3 nằm trong một môi trường khác (ví dụ conda), hãy tạo `.venv` từ Python của môi trường đó. Nếu phiên bản torch giữa SAM3 và MoGe xung đột, tạo venv thứ hai chỉ cho MoGe và chạy `synth/calibrate.py` từ venv đó. Những phần khác giữ nguyên.

- [ ] **Bước 2: Tạo `blender/cli_to_blend.py`** (spike cần file này)

```python
"""Run a bpy script exported by `cli-anything-blender render script` without rendering, save as .blend.

Run: blender --background --python blender/cli_to_blend.py -- EXPORTED.py OUT.blend
"""
import sys

import bpy

src, out = sys.argv[sys.argv.index("--") + 1:][:2]
code = open(src).read().replace("bpy.ops.render.render(", "(lambda **kw: None)(")
exec(compile(code, src, "exec"), {"__name__": "__main__"})
bpy.ops.wm.save_as_mainfile(filepath=out)
print(f"saved {out}: {sorted(o.name for o in bpy.data.objects)}")
```

- [ ] **Bước 3: Chạy các kiểm tra spike và ghi lại mọi kết quả**

```bash
blender --version                                   # S1: need >= 4.2
ffmpeg -version | head -1                           # S1b
mkdir -p work/spike
cli-anything-blender --help                          # S2
cli-anything-blender --json scene new --name spike --output work/spike/spike.blend-cli.json
cli-anything-blender --json --project work/spike/spike.blend-cli.json object add cube --name box -p size=1 --location=-1,2,0.5 --scale=1,1,1
cli-anything-blender --json --project work/spike/spike.blend-cli.json light add sun --name light_sun --rotation=35,0,40 --power 3
cli-anything-blender --json --project work/spike/spike.blend-cli.json object list          # S3: box at (-1, 2, 0.5)?
cli-anything-blender --project work/spike/spike.blend-cli.json render script work/spike/unused.png > work/spike/export.py
python -m py_compile work/spike/export.py && echo "S4 export compiles"                     # S4
blender --background --python blender/cli_to_blend.py -- work/spike/export.py work/spike/spike.blend   # S5: prints ['box', 'light_sun']
codex --version                                                                           # S6
codex login status                                                                        # S6b: expect "Logged in using ChatGPT" (plan sign-in, not an API key)
codex exec --help | grep -E -- "--image|--sandbox|--output-last-message|--skip-git-repo-check|--add-dir"
echo "Reply with the single word OK." | codex exec - -m gpt-astra-6 -s read-only --skip-git-repo-check   # S7: model slug
```

S8 kiểm tra Codex có chạy được Blender, ffmpeg và harness trong chế độ `workspace-write` hay không:

```bash
codex exec - -m gpt-astra-6 -s workspace-write --skip-git-repo-check -C "$PWD" -o work/spike/codex_s8.md <<'EOF'
Run these three commands and report each exit code and the last 5 lines of output verbatim:
1. blender --background --python-expr "import bpy; bpy.ops.wm.save_as_mainfile(filepath='work/spike/codex.blend')"
2. cli-anything-blender --json --project work/spike/spike.blend-cli.json object list
3. ffmpeg -y -loglevel error -f lavfi -i color=c=blue:s=64x64 -frames:v 1 work/spike/codex_ffmpeg.png
EOF
ls -la work/spike/codex.blend work/spike/codex_ffmpeg.png
```

S9 kiểm tra việc đính kèm ảnh và khả năng xem ảnh cục bộ:

```bash
python -c "
from PIL import Image, ImageDraw
for name, text in (('attached', '4172'), ('local', '9386')):
    im = Image.new('RGB', (320, 120), 'white'); ImageDraw.Draw(im).text((20, 40), text, fill='black'); im.save(f'work/spike/{name}.png')"
codex exec - -m gpt-astra-6 -s workspace-write --skip-git-repo-check -C "$PWD" -o work/spike/codex_s9.md -i work/spike/attached.png <<'EOF'
1. What number is written in the attached image?
2. Without it being attached, open the local file work/spike/local.png and tell me the number written in it. If you cannot view local image files, say "CANNOT VIEW LOCAL IMAGES".
EOF
cat work/spike/codex_s9.md
```

- [ ] **Bước 4: Viết `docs/spike-notes.md` với các kết quả quan sát được**

```markdown
# Spike notes (Task 1)

| Check | Command | Result (paste observed output) |
|---|---|---|
| S1 Blender version | `blender --version` | |
| S1b ffmpeg | `ffmpeg -version` | |
| S2 harness help | `cli-anything-blender --help` | |
| S3 negative location parsed | `object list` shows box at -1,2,0.5 | |
| S4 export compiles | `render script` + `py_compile` | |
| S5 cli_to_blend | objects printed | |
| S6 codex flags | `codex exec --help` | |
| S6b Codex sign-in | `codex login status` + the account's plan page | sign-in method, account (personal or team), plan and its usage limits |
| S7 model slug | `-m gpt-astra-6` works? correct slug | |
| S8 sandbox | blender / harness / ffmpeg under workspace-write | |
| S9 images | attached number read? local image viewable? | |

Decisions:
- codex_model: <slug that worked, also written into config.json>
- Codex account: <personal or team>, plan <name>, usage limits <as shown for the plan> -> estimated agent prompt rounds per day (Tasks 11-14)
- SAFETY flags for codex exec: `-s workspace-write` (if S8 passed) or `--dangerously-bypass-approvals-and-sandbox` (only on this dedicated workstation)
- Codex can view local images mid-session: yes/no (if no, the human re-runs prompts with `-i` attachments, as written in Tasks 11-13)
```

Kỳ vọng: S1 ≥ 4.2. S3 hiển thị location `[-1.0, 2.0, 0.5]`; nếu không, đổi quy ước `--location=` trong `AGENTS.md`. S4 và S5 đạt; nếu S4 lỗi vì có dòng thừa, cắt mọi thứ trước `#!/usr/bin/env python3` bằng `sed -n '/^#!\/usr\/bin\/env python3/,$p'` và ghi chú vào AGENTS.md. S7: ghi slug chạy được vào `config.json` → `codex_model`. S6b: `codex login status` phải báo đăng nhập bằng ChatGPT: GPT Astra 6 đi kèm gói đã đăng nhập, không dùng API key, nên không có ngân sách sử dụng nào để đặt và mức dùng bị giới hạn bởi hạn mức của gói. Ghi vào `docs/spike-notes.md` tài khoản Codex đang đăng nhập (cá nhân hay team) và hạn mức sử dụng của gói như hiển thị cho tài khoản đó; các thông tin này dùng để ước lượng số lượt prompt agent (Task 11–14) chạy được mỗi ngày.

---

### Task 2: Khung dự án

**File:**
- Tạo: `config.json`, `requirements.txt`, `synth/__init__.py` (rỗng), `synth/common.py`, `AGENTS.md`

- [ ] **Bước 1: Tạo `config.json`** (đặt `codex_model` là slug từ Task 1)

```json
{
  "camera_id": "cam01",
  "background_image": "data/input/cam01/background.png",
  "object_frames_dir": "data/input/cam01/objects",
  "work_dir": "work",
  "codex_model": "gpt-astra-6",
  "moge_model": "Ruicheng/moge-2-vitl-normal",
  "scale_correction": 1.0,
  "lsp_size_m": null,
  "osd_boxes": [[0, 0, 90, 24], [760, 0, 960, 26]],
  "sam3_prompts": {
    "floor": "floor",
    "forklift": "forklift",
    "LSP": "slip sheet",
    "SKID": "pallet",
    "cargo": "stretch wrapped cargo",
    "rack": "storage rack"
  },
  "n_images": 13,
  "valid_fraction": 0.25,
  "lsp_count_weights": {"2": 3, "3": 1},
  "skid_count_weights": {"0": 1, "1": 2, "2": 1},
  "idle_row_fraction": 0.33,
  "seed": 7,
  "blender": {"samples": 128, "world_strength": 1.0},
  "composite": {"blur_sigma": 0.8, "noise_sigma": null, "gain": [1.0, 1.0, 1.0], "h264_crf": 28}
}
```

`background_image` là ảnh nền sạch duy nhất. `object_frames_dir` là thư mục ảnh tham chiếu: chứa ~10 ảnh tham chiếu (`obj_*.png`) chỉ dùng để tách đối tượng; tên key giữ nguyên.

`n_images` 13 với `valid_fraction` 0.25 cho ra 10 ảnh vi phạm và 3 ảnh hợp lệ tổng hợp (2 ảnh xe nâng đẩy đúng 1 tấm LSP, 1 ảnh hàng LSP nằm yên không có xe nâng, đặt bằng `idle_row_fraction`). Các ảnh hợp lệ này là đề xuất bổ sung của chúng tôi, không có trong yêu cầu của Gary; đặt `valid_fraction` về 0 để bỏ chúng nếu đánh giá A/B của Terry cho thấy không cần. `skid_count_weights` đặt số SKID nằm yên (0, 1 hoặc 2) cho mỗi ảnh.

- [ ] **Bước 2: Tạo `requirements.txt`**

```text
numpy
pillow
pytest
```

- [ ] **Bước 3: Tạo `synth/__init__.py` (rỗng) và `synth/common.py`**

```bash
touch synth/__init__.py
```

```python
import json
from pathlib import Path


def load_config(path="config.json"):
    cfg = json.loads(Path(path).read_text())
    cfg["work"] = Path(cfg["work_dir"])
    return cfg


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2))


def apply_lsp_size(params, size_m):
    """Known LSP size [x, y] in metres (config lsp_size_m) overrides the measured one; None keeps the measurement."""
    if size_m is None:
        return params
    return dict(params, size_x_m=size_m[0], size_y_m=size_m[1],
                measured_size_m=params.get("measured_size_m", [params["size_x_m"], params["size_y_m"]]))
```

- [ ] **Bước 4: Tạo `AGENTS.md`**

````markdown
# AGENTS.md — aat-violation-synth

You are helping synthesize CCTV safety-violation images with Blender. A clean background frame of the real camera (no forklift, LSP, SKID or cargo) stays as the background; Blender only renders the objects we insert (forklift, cargo, LSPs, idle SKIDs) plus their shadows. Always run commands from the project root.

## Violation in scope

"Forklift Pushing Multiple Lsps": a forklift that pushes cargo with MORE THAN ONE LSP at the same time is a violation; exactly one LSP is valid; idle LSPs with no forklift pushing them are not a violation. An LSP is a large thin plastic slip sheet laid flat on the floor (roughly square, rounded corners, ~1-2 cm thick, blue with heavy dark wear, slightly glossy); stretch-wrapped cargo sits on it and the forklift pushes the sheet. Reference: `docs/requirements/lsp-reference.png`. Every output inserts one forklift + cargo + a chain of N LSPs in series ahead of the forks: N = 1 is a synthetic VALID image, N >= 2 a violation; some VALID images show only a row of idle LSPs with no forklift. Both are generated, so a rendered forklift never means "violation" by itself. Every image also gets 0-2 idle SKIDs (pallets, some with cargo on top) as scene distractors, away from the forklift path; they are never part of the violation event.

## Inputs of this camera (`camera_id` in `config.json`)

- `data/input/cam01/background.png`: the clean background, base of every output.
- `data/input/cam01/objects/obj_*.png`: ~10 reference images of the same fixed view with forklifts, LSPs, SKIDs and cargo at different positions, used only for object extraction (SAM3 crops and masks: asset references, textures, size cross-check, lighting and shadow reference). Never used as an output background.

## Coordinate conventions (must follow)

- Units: metres. World Z is up. The real floor is the plane z = 0.
- Origin: the floor point directly below the camera. +Y = camera forward projected on the floor, +X = right.
- `work/camera.json`: calibrated Blender camera (do not create or edit cameras; any camera you add is ignored).
- `work/scene_facts.json`: camera height, pitch, FOV, floor fit quality, ambient colour.
- `work/calibration.json`: per-camera cross-check with the real objects: measured LSP sizes, suggested scale correction, and the floor positions (x, y) where each object class was seen.
- `work/points_world.npy`: shape (H//4, W//4, 3) float32, world XYZ of background pixel (u, v) is `P[v//4, u//4]`, NaN = invalid. The camera does not move, so it also gives the floor point under an object pixel of any reference image. Example:
  `python -c "import numpy as np; P=np.load('work/points_world.npy'); print(P[400//4, 960//4])"`
- `work/masks/background/<key>.png`: SAM3 masks of the clean background (floor, rack). `work/masks/obj_<n>/<key>.png` / `<key>_<i>.png`: union and single-instance masks of the reference images (forklift, LSP, SKID, cargo).
- `work/refs/<frame>_<key>_<rank>.png`: tight crops of real objects (pixels outside the mask filled with the object colour). `work/refs/index.json`: class, frame, 2D box, area and bottom-centre pixel of every crop.
- `work/assets/<name>.blend`: metres, origin at the footprint centre on the floor, front / pushing direction = +Y. `lsp_<k>` are built by `blender/build_lsp.py` (same shape, different real textures); `forklift`, `cargo_<type>` and `skid` are built by you and textured by `blender/texture_asset.py`.

## Blender authoring rules

- Author proxy scenes and 3D assets ONLY with `cli-anything-blender` (read its skill first: `python -c "import cli_anything.blender, os; print(os.path.join(os.path.dirname(cli_anything.blender.__file__), 'skills', 'SKILL.md'))"`). Always pass `--json` and `--project <file>`.
- `cube` and `plane` default to size 2. Always add them with `-p size=1` so `--scale` equals the real dimensions in metres.
- Pass vectors with `=` so negative numbers are not parsed as options: `--location=-1.5,4,0`, `--rotation=0,0,-30`, `--scale=1.2,1,0.02`.
- Rotations are degrees (X, Y, Z). Name every object meaningfully (`proxy_rack_left`, `forklift_mast`, ...).
- Name the materials of asset surfaces that should show the real object's look `tex_<part>` (e.g. `tex_body`, `tex_wrap`): the pipeline projects the real SAM3 crop onto them. Other materials keep your Principled values.
- Convert a CLI project to a .blend with:
  `cli-anything-blender --project <P>.blend-cli.json render script work/unused.png > <P>_export.py`
  `blender --background --python blender/cli_to_blend.py -- <P>_export.py <OUT>.blend`
- Never edit files under `synth/`, `blender/`, `tests/`, `config.json` or `data/`.
````

- [ ] **Bước 5: Kiểm tra**

```bash
python -c "from synth.common import load_config; c = load_config(); print(c['work'], c['osd_boxes'])"
```

Kỳ vọng: `work [[0, 0, 90, 24], [760, 0, 960, 26]]`

---

### Task 3: Đầu vào theo camera: 1 ảnh nền sạch và ~10 ảnh tham chiếu

**File:**
- Đầu vào: `data/raw/cam01.mp4` (video gốc, không overlay; điều kiện tiên quyết, **đang chờ: các bên liên quan sẽ cung cấp**)
- Tạo: `data/frames/cam01_*.png`, `data/input/cam01/background.png`, `data/input/cam01/objects/obj_*.png`

- [ ] **Bước 1: Tách frame ở 5 fps gốc**

```bash
ffprobe -v error -show_entries stream=codec_name,width,height,r_frame_rate -of compact data/raw/cam01.mp4
ffmpeg -loglevel error -i data/raw/cam01.mp4 -vf fps=5 data/frames/cam01_%05d.png
ls data/frames | wc -l
```

Kỳ vọng: `codec_name=h264|width=960|height=540|r_frame_rate=5/1` và mỗi frame một file PNG.

- [ ] **Bước 2: Chọn ảnh nền sạch và các ảnh tham chiếu (người làm)**

Duyệt `data/frames/` và chọn hai tập riêng:
- **đúng một ảnh nền sạch**: nền cho mọi output và đầu vào cho bước hiệu chỉnh MoGe-2; không có xe nâng, LSP, SKID hay hàng trên vùng sàn nơi xe nâng chạy, ánh sáng bình thường, không có người trong vùng đó;
- **~10 ảnh tham chiếu**: xe nâng, LSP, SKID và hàng ở các vị trí khác nhau trong góc nhìn này, chỉ dùng để tách đối tượng (ảnh cắt, mask và kích thước từ SAM3 cho tham chiếu asset, texture, đối chiếu kích thước và tham chiếu ánh sáng), không bao giờ dùng làm nền cho output; tốt nhất gồm cả một xe nâng đẩy hàng trên đúng 1 tấm LSP, các tấm LSP nằm yên và thấy trọn vẹn (cho texture và số đo kích thước tốt nhất), và nhiều loại hàng khác nhau.

Mọi ảnh của một camera phải có cùng một góc nhìn (camera vật lý không được xoay, nghiêng, zoom hay rung giữa các frame này): việc hiệu chỉnh dựa trên việc pixel đối tượng trong ảnh tham chiếu khớp với sàn của nền. Sau đó chạy:

```bash
mkdir -p data/input/cam01/objects
cp data/frames/cam01_00042.png data/input/cam01/background.png   # clean: no forklift, LSP, SKID or cargo
for n in 00120 00355 00610 00987 01240 01502 01777 02031 02315 02640; do   # ~10 reference images: key objects at different positions
  cp data/frames/cam01_$n.png data/input/cam01/objects/obj_$n.png
done
```

- [ ] **Bước 3: Kiểm tra các box OSD trên frame gốc**

```bash
python -c "
from PIL import Image, ImageDraw; import json
c = json.load(open('config.json')); im = Image.open(c['background_image']).convert('RGB'); d = ImageDraw.Draw(im)
[d.rectangle(b, outline='red') for b in c['osd_boxes']]; im.crop((0, 0, 960, 60)).resize((1920, 120)).save('work/osd_check.png')"
```

Kỳ vọng: trong `work/osd_check.png`, các box đỏ bao trọn logo `SENSTAR` và timestamp. Nếu không, sửa `osd_boxes` trong `config.json`.

Phương án tạm, chỉ để phát triển khi chưa có video gốc: `ffmpeg -loglevel error -ss 0 -i docs/requirements/violation-clip-forklift-pushing-2-lsps.mp4 -frames:v 1 data/input/cam01/background.png`, rồi chép cùng file đó thành `data/input/cam01/objects/obj_00000.png`. Frame này không sạch và có overlay, nhưng cho phép chạy Task 4–10. Khi đã có frame gốc, xoá `work/` và làm lại từ Task 4.

---

### Task 4: Phân vùng bằng SAM3 và thư viện ảnh cắt đối tượng

**File:**
- Tạo: `synth/segment.py`, `tests/test_segment.py`

- [ ] **Bước 1: Viết test**

```python
import numpy as np

from synth.segment import object_record


def test_object_record_crop_size_and_position():
    img = np.full((60, 80, 3), 90, np.uint8)
    mask = np.zeros((60, 80), bool)
    mask[20:40, 30:50] = True
    mask[20:25, 30:35] = False  # notch: floor pixels inside the box, not the object
    img[mask] = [200, 40, 40]
    crop, info = object_record(img, mask)
    assert info == {"bbox": [30, 20, 50, 40], "area_px": 375, "bottom_center_px": [40, 39]}
    assert crop.shape == (20, 20, 3) and (crop == [200, 40, 40]).all()  # floor pixels replaced by the object colour
```

- [ ] **Bước 2: Chạy test và xác nhận nó fail**

Chạy: `python -m pytest -q tests/test_segment.py`
Kỳ vọng: FAIL với `ModuleNotFoundError: No module named 'synth.segment'`

- [ ] **Bước 3: Tạo `synth/segment.py`**

```python
"""SAM3 text-prompted masks and the per-camera object crop library.

Run: python -m synth.segment                                # clean background + every object frame of the camera
     python -m synth.segment docs/requirements/lsp-reference.png LSP   # another image, selected keys (index untouched)
Writes work/masks/<stem>/<key>.png (union), <key>_<i>.png (instances), source.json,
       work/refs/<stem>_<key>_<rank>.png (top-3 object crops per class and frame),
       work/refs/index.json (class, frame, score, 2D box, area, bottom-centre pixel of every crop)
"""
import sys
from pathlib import Path

import numpy as np
from PIL import Image

from synth.common import load_config, write_json

BACKGROUND_KEYS = ["floor", "rack"]  # static scene, segmented on the clean background
OBJECT_KEYS = ["forklift", "LSP", "SKID", "cargo"]  # key objects, segmented on every object frame


def object_record(image, mask):
    """Tight crop of one object (pixels outside the mask filled with the object's mean colour, so the crop
    can serve as a texture) plus its 2D size and position. image: HxWx3 uint8, mask: HxW bool."""
    ys, xs = np.nonzero(mask)
    x0, y0, x1, y1 = xs.min(), ys.min(), xs.max() + 1, ys.max() + 1
    crop = image[y0:y1, x0:x1].copy()
    crop[~mask[y0:y1, x0:x1]] = image[mask].mean(axis=0)
    info = {"bbox": [int(x0), int(y0), int(x1), int(y1)], "area_px": int(mask.sum()),
            "bottom_center_px": [int(round(xs.mean())), int(y1 - 1)]}  # floor contact point
    return crop, info


def load_processor():
    from sam3.model.sam3_image_processor import Sam3Processor
    from sam3.model_builder import build_sam3_image_model

    return Sam3Processor(build_sam3_image_model())


def run(cfg, processor, image_path, keys):
    work, stem = cfg["work"], Path(image_path).stem
    mdir = work / "masks" / stem
    (work / "refs").mkdir(parents=True, exist_ok=True)
    write_json(mdir / "source.json", {"image": str(image_path)})
    image = Image.open(image_path).convert("RGB")
    pixels = np.asarray(image)
    state = processor.set_image(image)
    records = []
    for key in keys:
        prompt = cfg["sam3_prompts"][key]
        out = processor.set_text_prompt(state=state, prompt=prompt)
        masks = [np.asarray(m.detach().cpu().float()).squeeze() > 0.5 for m in out["masks"]]
        scores = [float(s) for s in out["scores"]]
        union = np.zeros((image.height, image.width), bool)
        for i, m in enumerate(masks):
            union |= m
            Image.fromarray(m.astype(np.uint8) * 255).save(mdir / f"{key}_{i}.png")
        Image.fromarray(union.astype(np.uint8) * 255).save(mdir / f"{key}.png")
        print(f"{stem}: {key!r} ({prompt!r}) {len(masks)} instances, scores {[round(s, 2) for s in scores]}")
        if key in BACKGROUND_KEYS:
            continue
        for rank, idx in enumerate(np.argsort(scores)[::-1][:3]):
            if not masks[idx].any():
                continue
            crop, info = object_record(pixels, masks[idx])
            name = f"{stem}_{key}_{rank}.png"
            Image.fromarray(crop).save(work / "refs" / name)
            records.append({"class": key, "frame": str(image_path), "rank": rank, "score": round(scores[idx], 3),
                            "crop": f"work/refs/{name}", **info})
    return records


if __name__ == "__main__":
    cfg = load_config()
    processor = load_processor()
    if len(sys.argv) > 1:
        run(cfg, processor, sys.argv[1], sys.argv[2:] or OBJECT_KEYS)
    else:
        run(cfg, processor, cfg["background_image"], BACKGROUND_KEYS)
        frames = sorted(Path(cfg["object_frames_dir"]).glob("obj_*.png"))
        records = [r for f in frames for r in run(cfg, processor, f, OBJECT_KEYS)]
        write_json(cfg["work"] / "refs" / "index.json", records)
        print(f"{len(frames)} object frames, {len(records)} object crops -> {cfg['work'] / 'refs'}")
```

- [ ] **Bước 4: Chạy test, rồi phân vùng nền và mọi ảnh tham chiếu**

```bash
python -m pytest -q tests/test_segment.py
python -m synth.segment
ls work/masks/background work/masks | head -20
python -c "import json, collections; print(collections.Counter(r['class'] for r in json.load(open('work/refs/index.json'))))"
```

Kỳ vọng: `1 passed` cho `python -m pytest -q tests/test_segment.py`. `work/masks/background/floor.png` phủ phần lớn sàn nhìn thấy được và không phủ kệ. Mỗi lớp có mặt trong các ảnh tham chiếu đều có ảnh cắt trong `work/refs/` và mục trong `work/refs/index.json`. Mở vài ảnh cắt: mỗi ảnh chứa trọn một đối tượng. Nếu một prompt không tìm ra gì, thử cách diễn đạt khác trong `config.json` → `sam3_prompts` (với LSP: `"blue plastic sheet"`, `"floor mat"`), chạy lại, rồi ghi prompt chạy được vào `docs/spike-notes.md`.

---

### Task 5: Hiệu chỉnh theo camera từ MoGe-2, đối chiếu với các đối tượng

**File:**
- Tạo: `synth/calibrate.py`, `tests/test_calibrate.py`

- [ ] **Bước 1: Viết test**

```python
import math

import numpy as np

from synth.calibrate import CV_TO_BLENDER_CAM, blender_camera, fit_plane_ransac, footprint_extent, size_ratio, world_from_cv


def test_blender_camera_centered():
    cam = blender_camera([[0.8, 0, 0.5], [0, 0.8 * 1920 / 1080, 0.5], [0, 0, 1]], 1920, 1080)
    assert math.isclose(cam["lens_mm"], 28.8)
    assert cam["shift_x"] == 0 and cam["shift_y"] == 0
    assert math.isclose(cam["hfov_deg"], math.degrees(2 * math.atan(0.5 / 0.8)))


def test_blender_camera_shift_signs():
    # principal point right of / below centre -> negative shift_x, positive shift_y
    cam = blender_camera([[0.8, 0, 0.55], [0, 1.42, 0.6], [0, 0, 1]], 1920, 1080)
    assert math.isclose(cam["shift_x"], -0.05)
    assert math.isclose(cam["shift_y"], 0.1 * 1080 / 1920)


def test_floor_frame_recovers_camera_pose():
    height, pitch = 3.0, math.radians(30)
    # camera at (0, 0, 3) looking along +y, tilted down 30 deg; rows = cv axes in world
    x_c = np.array([1.0, 0, 0])
    z_c = np.array([0, math.cos(pitch), -math.sin(pitch)])
    y_c = np.cross(z_c, x_c)
    r_cw = np.stack([x_c, y_c, z_c])
    rng = np.random.default_rng(1)
    floor_w = np.c_[rng.uniform(-5, 5, 2000), rng.uniform(3, 20, 2000), rng.normal(0, 0.005, 2000)]
    pts_cv = (floor_w - [0, 0, height]) @ r_cw.T

    n, d, ratio = fit_plane_ransac(pts_cv)
    assert abs(d - height) < 0.02 and ratio > 0.95
    t = world_from_cv(n, d)
    back = pts_cv @ t[:3, :3].T + t[:3, 3]
    assert np.abs(back[:, 2]).max() < 0.03  # floor maps to z = 0
    assert np.allclose(t[:3, 3], [0, 0, height], atol=0.02)  # camera above origin
    m = t @ CV_TO_BLENDER_CAM
    fwd = -m[:3, 2]
    assert math.isclose(fwd[2], -math.sin(pitch), abs_tol=0.01)
    assert fwd[1] > 0.8  # looks along +y


def test_footprint_extent_of_rotated_sheet():
    rng = np.random.default_rng(0)
    local = np.c_[rng.uniform(-0.6, 0.6, 4000), rng.uniform(-0.5, 0.5, 4000)]
    a = math.radians(35)
    xy = local @ np.array([[math.cos(a), math.sin(a)], [-math.sin(a), math.cos(a)]]) + [2.0, 7.0]
    long, short = footprint_extent(xy)
    assert abs(long - 1.2) < 0.02 and abs(short - 1.0) < 0.02


def test_size_ratio_skips_partly_covered_sheets():
    # two full sheets measured 10 % small, one half covered by cargo (wrong aspect) -> ignored
    ratio, used = size_ratio([[0.99, 0.99], [1.0, 0.98], [0.99, 0.5]], [1.1, 1.1])
    assert used == 2 and abs(ratio - 1.1 / 0.99) < 0.01
    assert size_ratio([[1.0, 0.4]], [1.1, 1.1]) == (None, 0)
```

- [ ] **Bước 2: Chạy test và xác nhận nó fail**

Chạy: `python -m pytest -q tests/test_calibrate.py`
Kỳ vọng: FAIL với `ModuleNotFoundError: No module named 'synth.calibrate'`

- [ ] **Bước 3: Tạo `synth/calibrate.py`**

```python
"""Per-camera calibration: MoGe-2 on the clean background -> Blender camera + floor-aligned world frame,
cross-checked with the sizes and positions of the SAM3 objects of the same camera. Done once per camera and
reused by every image of that camera.

Run: python -m synth.calibrate
Writes work/camera.json, work/points_world.npy, work/scene_facts.json, work/calibration.json
"""
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image

from synth.common import load_config, write_json

CV_TO_BLENDER_CAM = np.diag([1.0, -1.0, -1.0, 1.0])  # OpenCV cam (x right, y down, z fwd) -> Blender cam (-z fwd, y up)
NOMINAL_LSP_M = [1.1, 1.1]  # ponytail: estimate from the reference sheet (roughly square, 1.0-1.2 m); config lsp_size_m replaces it


def blender_camera(k_norm, width, height, sensor_width=36.0):
    """Normalized intrinsics (fx/W, fy/H, cx/W, cy/H) -> Blender lens/shift with sensor_fit HORIZONTAL."""
    assert width >= height, "shift_y formula assumes a landscape frame"
    fx, fy, cx, cy = k_norm[0][0], k_norm[1][1], k_norm[0][2], k_norm[1][2]
    if abs(fx * width - fy * height) > 0.02 * fx * width:
        print(f"WARNING: non-square pixels fx={fx * width:.1f}px fy={fy * height:.1f}px, using fx")
    return {
        "width": width,
        "height": height,
        "lens_mm": fx * sensor_width,
        "sensor_width_mm": sensor_width,
        "shift_x": 0.5 - cx,
        "shift_y": (cy - 0.5) * height / width,
        "hfov_deg": math.degrees(2 * math.atan(0.5 / fx)),
    }


def fit_plane_ransac(points, iters=300, thresh=0.03, seed=0):
    """Return (n, d, inlier_ratio) with n.p + d = 0, |n| = 1, oriented so the camera (origin) has d > 0."""
    rng = np.random.default_rng(seed)
    best = None
    for _ in range(iters):
        a, b, c = points[rng.choice(len(points), 3, replace=False)]
        n = np.cross(b - a, c - a)
        if np.linalg.norm(n) < 1e-9:
            continue
        n = n / np.linalg.norm(n)
        inliers = np.abs(points @ n - n @ a) < thresh
        if best is None or inliers.sum() > best.sum():
            best = inliers
    p = points[best]
    centroid = p.mean(axis=0)
    n = np.linalg.svd(p - centroid)[2][-1]  # least-squares refine on inliers
    d = -n @ centroid
    if d < 0:
        n, d = -n, -d
    return n, float(d), float(best.mean())


def world_from_cv(n, d):
    """4x4 transform: OpenCV camera coords -> world (z up, floor z=0, origin below camera, +y = camera forward on floor)."""
    z = n
    x = np.array([1.0, 0.0, 0.0]) - n * n[0]  # camera right projected on the floor
    x = x / np.linalg.norm(x)
    y = np.cross(z, x)
    r = np.stack([x, y, z])  # rows = world axes expressed in camera coords
    t = np.eye(4)
    t[:3, :3] = r
    t[:3, 3] = r @ (d * n)  # world = R (p - o), o = -d n
    return t


def footprint_extent(xy):
    """[long, short] sides in metres of the minimum-area rectangle around a flat object's floor footprint (xy: Nx2)."""
    best = None
    for a in np.radians(np.arange(0.0, 90.0, 0.5)):
        e = np.ptp(xy @ np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]]), axis=0)
        if best is None or e[0] * e[1] < best[0] * best[1]:
            best = e
    return sorted(best.tolist(), reverse=True)


def size_ratio(extents, known, tol=0.1):
    """(known / measured size as the median over instances, number of instances used).
    Only instances whose aspect ratio matches the known one count: a sheet partly covered by cargo looks too short."""
    k_long, k_short = sorted(known, reverse=True)
    used = [e for e in extents if abs(e[1] / e[0] - k_short / k_long) < tol]
    if not used:
        return None, 0
    return float(np.median([(k_long + k_short) / (e[0] + e[1]) for e in used])), len(used)


def srgb_to_linear(c):
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def run(cfg):
    import torch
    from moge.model.v2 import MoGeModel

    work = cfg["work"]
    img = Image.open(cfg["background_image"]).convert("RGB")
    w, h = img.size
    rgb = np.asarray(img, dtype=np.float32) / 255.0
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = MoGeModel.from_pretrained(cfg["moge_model"]).to(device).eval()
    with torch.no_grad():
        out = model.infer(torch.tensor(rgb, device=device).permute(2, 0, 1))
    pts = out["points"].cpu().numpy() * cfg["scale_correction"]
    valid = out["mask"].cpu().numpy() > 0
    k_norm = out["intrinsics"].cpu().numpy()
    assert pts.shape[:2] == (h, w), f"MoGe output {pts.shape} != image {(h, w)}"

    floor_mask = work / "masks" / Path(cfg["background_image"]).stem / "floor.png"
    floor = np.asarray(Image.open(floor_mask).convert("L")) > 127
    sel = pts[valid & floor & np.isfinite(pts).all(axis=-1)]
    n, d, inlier_ratio = fit_plane_ransac(sel[:: max(1, len(sel) // 50000)])
    t = world_from_cv(n, d)

    cam = blender_camera(k_norm.tolist(), w, h)
    cam["matrix_world"] = (t @ CV_TO_BLENDER_CAM).tolist()
    cam["K_norm"] = k_norm.tolist()
    write_json(work / "camera.json", cam)

    small = pts[::4, ::4]
    world = small @ t[:3, :3].T + t[:3, 3]
    world[~valid[::4, ::4]] = np.nan
    np.save(work / "points_world.npy", world.astype(np.float32))

    fwd = np.array(cam["matrix_world"])[:3, 2] * -1  # Blender camera looks down -Z
    write_json(work / "scene_facts.json", {
        "image_size": [w, h],
        "camera_height_m": d,
        "camera_pitch_down_deg": math.degrees(math.asin(-fwd[2])),
        "hfov_deg": cam["hfov_deg"],
        "floor_inlier_ratio": inlier_ratio,
        "ambient_rgb_linear": srgb_to_linear(rgb.reshape(-1, 3).mean(axis=0)).tolist(),
        "points_world": "work/points_world.npy shape (H//4, W//4, 3); world xyz of pixel (u, v) = P[v//4, u//4]; NaN = invalid",
    })
    print(f"camera height {d:.2f} m, hfov {cam['hfov_deg']:.1f} deg, floor inliers {inlier_ratio:.0%}")

    # cross-check with this camera's SAM3 objects: the view is fixed, so an object's floor pixels in an object
    # frame map onto the clean background's floor points
    world_full = pts @ t[:3, :3].T + t[:3, 3]
    world_full[~valid] = np.nan
    lsps = []
    for mpath in sorted(work.glob("masks/obj_*/LSP_*.png")):
        xy = world_full[np.asarray(Image.open(mpath).convert("L")) > 127][:, :2]
        xy = xy[np.isfinite(xy).all(axis=1)]
        if len(xy) >= 50:
            lsps.append({"mask": str(mpath), "extent_m": footprint_extent(xy), "centre_m": xy.mean(axis=0).round(2).tolist()})
    known = cfg["lsp_size_m"] or NOMINAL_LSP_M
    ratio, used = size_ratio([o["extent_m"] for o in lsps], known)
    positions = {}
    for rec in json.loads((work / "refs" / "index.json").read_text()):
        u, v = rec["bottom_center_px"]
        p = world_full[v, u]
        if np.isfinite(p).all():
            positions.setdefault(rec["class"], []).append(p[:2].round(2).tolist())
    write_json(work / "calibration.json", {
        "camera_id": cfg["camera_id"],
        "background_image": cfg["background_image"],
        "lsp_known_size_m": known,
        "lsp_instances": lsps,
        "lsp_instances_used": used,
        "suggested_scale_correction": ratio and round(cfg["scale_correction"] * ratio, 3),
        "object_floor_positions_m": positions,
    })
    print(f"LSP size check: {used}/{len(lsps)} sheets used, known/measured = {ratio}")
    if ratio is None or abs(ratio - 1) > 0.1:
        print("WARNING: set scale_correction in config.json to suggested_scale_correction "
              "(work/calibration.json) and re-run, or check the LSP masks")


if __name__ == "__main__":
    run(load_config())
```

- [ ] **Bước 4: Chạy test**

Chạy: `python -m pytest -q tests/test_calibrate.py`
Kỳ vọng: `5 passed`

- [ ] **Bước 5: Hiệu chỉnh camera**

```bash
python -m synth.calibrate
cat work/scene_facts.json work/calibration.json
```

Kỳ vọng: chiều cao camera vài mét (CCTV gắn cao: khoảng 4–10 m), `camera_pitch_down_deg` trong khoảng 15–60, và `floor_inlier_ratio` > 0.6. Trong `work/calibration.json`, có ít nhất một tấm LSP thấy trọn vẹn được dùng (`lsp_instances_used` ≥ 1) và `object_floor_positions_m` liệt kê các vị trí trên sàn nơi đã thấy xe nâng và LSP. Nếu script cảnh báo tỉ số kích thước đã biết / đo được lệch hơn 10 %, đặt `scale_correction` trong `config.json` bằng `suggested_scale_correction` rồi chạy lại bước này. Mốc so sánh là kích thước thật trong `lsp_size_m` khi người dùng đã cung cấp, còn trước đó là giá trị danh nghĩa 1.1 × 1.1 m.

---

### Task 6: Texture LSP từ các tấm LSP thật

**File:**
- Tạo: `synth/lsp_texture.py`, `tests/test_lsp_texture.py`

- [ ] **Bước 1: Viết test**

```python
import numpy as np

from synth.common import apply_lsp_size
from synth.lsp_texture import homography, polygon_area, quad_corners, rectify


def test_config_lsp_size_overrides_measurement():
    measured = {"size_x_m": 1.1, "size_y_m": 1.15, "thickness_m": 0.015}
    assert apply_lsp_size(measured, None) == measured
    p = apply_lsp_size(measured, [1.2, 1.0])
    assert (p["size_x_m"], p["size_y_m"], p["thickness_m"]) == (1.2, 1.0, 0.015)
    assert p["measured_size_m"] == [1.1, 1.15]
    assert apply_lsp_size(p, [1.3, 1.3])["measured_size_m"] == [1.1, 1.15]  # re-applying keeps the measurement


def test_homography_maps_points():
    src = np.array([[0, 0], [10, 0], [10, 10], [0, 10]], float)
    dst = np.array([[5, 3], [40, 8], [35, 30], [2, 25]], float)
    h = homography(src, dst)
    p = h @ np.c_[src, np.ones(4)].T
    assert np.allclose((p[:2] / p[2]).T, dst)


def test_quad_corners_and_rectify_recover_pattern():
    img = np.zeros((200, 300, 3))
    quad = np.array([[60, 40], [240, 50], [220, 170], [80, 160]], float)  # TL TR BR BL
    # paint the quad: left half red, right half green, via the inverse homography
    h = homography(quad, np.array([[0, 0], [1, 0], [1, 1], [0, 1]], float))
    ys, xs = np.mgrid[0:200, 0:300]
    p = h @ np.stack([xs.ravel(), ys.ravel(), np.ones(xs.size)])
    u, v = (p[0] / p[2]).reshape(200, 300), (p[1] / p[2]).reshape(200, 300)
    inside = (u >= 0) & (u <= 1) & (v >= 0) & (v <= 1)
    img[inside & (u < 0.5)] = [1, 0, 0]
    img[inside & (u >= 0.5)] = [0, 1, 0]

    corners = quad_corners(inside)
    assert np.abs(corners - quad).max() <= 2
    assert inside.sum() / polygon_area(corners) > 0.95
    tex = rectify(img, corners, size=64)
    assert tex[32, 10, 0] > 0.9 and tex[32, 54, 1] > 0.9  # left red, right green
```

- [ ] **Bước 2: Chạy test và xác nhận nó fail**

Chạy: `python -m pytest -q tests/test_lsp_texture.py`
Kỳ vọng: FAIL với `ModuleNotFoundError: No module named 'synth.lsp_texture'`

- [ ] **Bước 3: Tạo `synth/lsp_texture.py`**

```python
"""Perspective-rectify SAM3 masks of real LSPs into square textures for the LSP asset.

Run: python -m synth.lsp_texture
Reads work/masks/*/LSP_<i>.png (+ source.json), writes work/textures/lsp_<k>.png (512x512)
"""
import json
from pathlib import Path

import numpy as np
from PIL import Image

from synth.common import load_config

SIZE = 512
MIN_AREA_PX = 1500
MIN_FILL = 0.85  # mask area / quad area; lower = sheet partly covered (e.g. cargo on top) -> skip


def quad_corners(mask):
    """Corners TL, TR, BR, BL of a roughly quadrilateral mask (extremes of x+y and x-y)."""
    ys, xs = np.nonzero(mask)
    s, d = xs + ys, xs - ys
    idx = [s.argmin(), d.argmax(), s.argmax(), d.argmin()]
    return np.array([[xs[i], ys[i]] for i in idx], dtype=float)


def polygon_area(p):
    x, y = p[:, 0], p[:, 1]
    return 0.5 * abs(x @ np.roll(y, -1) - y @ np.roll(x, -1))


def homography(src, dst):
    """3x3 H with dst ~ H @ src for 4 point pairs (DLT)."""
    rows = []
    for (x, y), (u, v) in zip(src, dst):
        rows.append([x, y, 1, 0, 0, 0, -u * x, -u * y, -u])
        rows.append([0, 0, 0, x, y, 1, -v * x, -v * y, -v])
    h = np.linalg.svd(np.array(rows))[2][-1].reshape(3, 3)
    return h / h[2, 2]


def rectify(img, corners, size=SIZE):
    """Bilinear-sample the quad (TL, TR, BR, BL) of img (HxWx3 float) into a size x size texture."""
    square = np.array([[0, 0], [size - 1, 0], [size - 1, size - 1], [0, size - 1]], dtype=float)
    h = homography(square, corners)
    v, u = np.mgrid[0:size, 0:size]
    p = h @ np.stack([u.ravel(), v.ravel(), np.ones(u.size)])
    x = np.clip(p[0] / p[2], 0, img.shape[1] - 1.001)
    y = np.clip(p[1] / p[2], 0, img.shape[0] - 1.001)
    x0, y0 = x.astype(int), y.astype(int)
    fx, fy = (x - x0)[:, None], (y - y0)[:, None]
    out = (img[y0, x0] * (1 - fx) * (1 - fy) + img[y0, x0 + 1] * fx * (1 - fy)
           + img[y0 + 1, x0] * (1 - fx) * fy + img[y0 + 1, x0 + 1] * fx * fy)
    return out.reshape(size, size, 3)


def run(cfg):
    dest = cfg["work"] / "textures"
    dest.mkdir(parents=True, exist_ok=True)
    k = 0
    for mpath in sorted(cfg["work"].glob("masks/*/LSP_*.png")):
        mask = np.asarray(Image.open(mpath)) > 127
        area = mask.sum()
        corners = quad_corners(mask) if area else None
        if area < MIN_AREA_PX or area / max(polygon_area(corners), 1) < MIN_FILL:
            print(f"skip {mpath} (area {area}px)")
            continue
        src = json.loads((mpath.parent / "source.json").read_text())["image"]
        img = np.asarray(Image.open(src).convert("RGB"), dtype=np.float32) / 255.0
        tex = rectify(img, corners)
        Image.fromarray((tex * 255 + 0.5).astype(np.uint8)).save(dest / f"lsp_{k}.png")
        print(f"{mpath} -> {dest / f'lsp_{k}.png'}")
        k += 1
    print(f"{k} textures")


if __name__ == "__main__":
    run(load_config())
```

- [ ] **Bước 4: Chạy test, rồi tạo texture**

```bash
python -m pytest -q tests/test_lsp_texture.py
python -m synth.lsp_texture
```

Kỳ vọng: `3 passed`, sau đó có ít nhất 3 texture trong `work/textures/`. Mở ra xem: mỗi texture phải là một tấm màu xanh, có vết mòn, phủ kín hình vuông, không lẫn sàn hay hàng ở mép. Xoá texture xấu và đánh số lại các file còn lại theo thứ tự `lsp_0..lsp_k`. Nếu các ảnh tham chiếu cho ít hơn 3 texture tốt, chạy thêm `python -m synth.segment docs/requirements/lsp-reference.png LSP` rồi tạo lại (đây là camera khác, nên kiểm tra màu).

---

### Task 7: Bộ random kịch bản

**File:**
- Tạo: `synth/randomize.py`, `tests/test_randomize.py`

- [ ] **Bước 1: Viết test**

```python
import math

import pytest

from synth.randomize import make_variants

SCENARIO = {
    "violation_id": "forklift_pushing_multiple_lsps",
    "assets": {"forklift": ["forklift"], "cargo": ["cargo_0", "cargo_1"], "lsp": ["lsp_0", "lsp_1"], "skid": ["skid"]},
    "lsp_thickness_m": 0.015,
    "floor_region": [[-4.0, 4.0], [4.0, 16.0]],
    "heading_deg": [60.0, 120.0],
    "forks_to_lsp_m": 1.9,
    "arrangements": {"in_series": [0.0, 1.25, 0.0], "stacked": [0.0, 0.0, 0.015]},
    "jitter": {"xy_m": 0.03, "rot_deg": 3.0, "texture_turns": 0},
    "skid_height_m": 0.15,
    "skid_clearance_m": 2.0,
    "idle_row_lsps": [2, 5],
}
WEIGHTS = {"2": 3, "3": 1}  # total LSP count of a violation -> relative weight (config lsp_count_weights)


def test_seeded_valid_and_violation_mix():
    a = make_variants(SCENARIO, 40, 7, WEIGHTS, 0.25)
    assert a == make_variants(SCENARIO, 40, 7, WEIGHTS, 0.25)
    assert a != make_variants(SCENARIO, 40, 8, WEIGHTS, 0.25)
    assert a[0]["id"] == "v_000"
    valid = [v for v in a if not v["is_violation"]]
    assert len(valid) == 10 and all(v["lsp_count"] == 1 for v in valid)  # synthetic valid: exactly 1 LSP pushed
    counts = [v["lsp_count"] for v in a if v["is_violation"]]
    assert set(counts) == {2, 3} and counts.count(2) > counts.count(3)  # 2 LSPs primary, 3 less frequent
    (x0, y0), (x1, y1) = SCENARIO["floor_region"]
    for v in a:
        classes = [o["class"] for o in v["objects"]]
        assert classes.count("forklift") == 1 and classes.count("cargo") == 1 and classes.count("LSP") == v["lsp_count"]
        assert v["is_violation"] == (v["lsp_count"] >= 2) and 60 <= v["heading_deg"] <= 120
        assert all(x0 <= o["location"][0] <= x1 and y0 <= o["location"][1] <= y1 for o in v["objects"])
    assert {v["arrangement"] for v in a} == {"in_series", "stacked"}


def test_lsp_chain_follows_forklift_heading():
    one = dict(SCENARIO, heading_deg=[90.0, 90.0], arrangements={"in_series": [0.0, 1.25, 0.0]},
               jitter={"xy_m": 0.0, "rot_deg": 0.0, "texture_turns": 0})
    (v,) = make_variants(one, 1, 0, {"3": 1}, 0.0)
    fork, *lsps, cargo = v["objects"]
    fx, fy, _ = fork["location"]
    for k, o in enumerate(lsps):
        # heading 90 deg: local +y (pushing direction) is world -x
        assert math.isclose(o["location"][0], fx - (1.9 + 1.25 * k)) and math.isclose(o["location"][1], fy)
        assert o["rot_z_deg"] == 90.0 and o["location"][2] == 0.0
    assert cargo["location"] == [lsps[0]["location"][0], lsps[0]["location"][1], 0.015]


def test_stacked_cargo_rests_on_top_sheet():
    stacked = dict(SCENARIO, arrangements={"stacked": [0.0, 0.0, 0.015]})
    (v,) = make_variants(stacked, 1, 0, {"3": 1}, 0.0)
    assert v["lsp_count"] == 3 and math.isclose(v["objects"][-1]["location"][2], 3 * 0.015)


def test_no_room_in_floor_region_raises():
    tiny = dict(SCENARIO, floor_region=[[0.0, 0.0], [1.0, 1.0]])
    with pytest.raises(ValueError, match="floor_region"):
        make_variants(tiny, 1, 0, {"3": 1}, 0.0)


def test_idle_skids_are_distractors_beside_the_forklift_lane():
    a = make_variants(SCENARIO, 40, 7, WEIGHTS, 0.25, {"0": 1, "1": 2, "2": 1})
    assert {v["skid_count"] for v in a} == {0, 1, 2}
    (x0, y0), (x1, y1) = SCENARIO["floor_region"]
    gap, on_skid = SCENARIO["skid_clearance_m"], 0
    for v in a:
        event = [o for o in v["objects"] if not o["name"].startswith("skid_")]
        skids = [o for o in v["objects"] if o["class"] == "skid"]
        assert len(skids) == v["skid_count"] and [o["class"] for o in event].count("cargo") == 1
        h = math.radians(v["heading_deg"])
        fx, fy, _ = event[0]["location"]
        lane = [math.cos(h) * (o["location"][0] - fx) + math.sin(h) * (o["location"][1] - fy) for o in event]
        for s in skids:
            sx, sy, sz = s["location"]
            side = math.cos(h) * (sx - fx) + math.sin(h) * (sy - fy)
            assert sz == 0.0 and x0 <= sx <= x1 and y0 <= sy <= y1
            assert min(abs(side - d) for d in lane) >= gap - 0.01  # beside the forklift path, never on the LSP chain
        assert all(math.dist(p["location"][:2], q["location"][:2]) >= gap for p, q in zip(skids, skids[1:]))
        for c in (o for o in v["objects"] if o["name"].endswith("_cargo")):
            skid = next(s for s in skids if c["name"] == s["name"] + "_cargo")
            assert c["location"] == skid["location"][:2] + [SCENARIO["skid_height_m"]]
            on_skid += 1
    assert 0 < on_skid < sum(v["skid_count"] for v in a)  # some SKIDs carry cargo, some are empty


def test_idle_lsp_row_without_forklift_is_valid():
    a = make_variants(SCENARIO, 40, 7, WEIGHTS, 0.25, None, 0.5)
    rows = [v for v in a if v["arrangement"] == "idle_row"]
    assert len(rows) == 5 and all(not v["is_violation"] and v["lsp_count"] == 0 for v in rows)  # scenario N2
    assert sum(v["lsp_count"] == 1 for v in a) == 5  # the other valid images: forklift pushing 1 LSP (N1)
    for v in rows:
        classes = [o["class"] for o in v["objects"]]
        assert "forklift" not in classes and "cargo" not in classes and 2 <= classes.count("LSP") <= 5
```

- [ ] **Bước 2: Chạy test và xác nhận nó fail**

Chạy: `python -m pytest -q tests/test_randomize.py`
Kỳ vọng: FAIL với `ModuleNotFoundError: No module named 'synth.randomize'`

- [ ] **Bước 3: Tạo `synth/randomize.py`**

```python
"""Scenario template -> N seeded placements of forklift + LSP chain + cargo, plus idle SKIDs, on the clean floor
(no LLM per image).

Run: python -m synth.randomize [N]
Reads work/scenario.json, writes work/variants.json
"""
import json
import math
import random
import sys

from synth.common import load_config, write_json


def to_world(origin, heading_deg, local):
    """Event frame (forklift origin on the floor, +Y = forklift heading) -> world xy."""
    c, s = math.cos(math.radians(heading_deg)), math.sin(math.radians(heading_deg))
    return [origin[0] + c * local[0] - s * local[1], origin[1] + s * local[0] + c * local[1]]


def to_local(origin, heading_deg, xy):
    """World xy -> event frame (inverse of to_world)."""
    c, s = math.cos(math.radians(heading_deg)), math.sin(math.radians(heading_deg))
    dx, dy = xy[0] - origin[0], xy[1] - origin[1]
    return [c * dx + s * dy, -s * dx + c * dy]


def place_skids(rng, scenario, count, origin, heading, chain, max_tries):
    """Idle SKIDs as scene distractors (never part of the event), about half with cargo on top. Each lies inside
    floor_region, at least skid_clearance_m beside the forklift's lane (the line through the forklift and its LSP
    chain) and from the other SKIDs. chain: world xy of the forklift origin and the LSP centres."""
    (x0, y0), (x1, y1) = scenario["floor_region"]
    gap, assets, j = scenario["skid_clearance_m"], scenario["assets"], scenario["jitter"]
    lane = [to_local(origin, heading, p)[0] for p in chain]
    lo, hi = min(lane) - gap, max(lane) + gap
    placed, objects = [], []
    for _ in range(count):
        for _ in range(max_tries):
            p = [rng.uniform(x0, x1), rng.uniform(y0, y1)]
            if not lo < to_local(origin, heading, p)[0] < hi and all(math.dist(p, q) >= gap for q in placed):
                break
        else:
            continue  # ponytail: no free floor left -> fewer distractors (skid_count records it)
        name = f"skid_{len(placed)}"
        placed.append(p)
        rot = heading + rng.choice([0, 90]) + rng.uniform(-j["rot_deg"], j["rot_deg"])
        objects.append({"name": name, "asset": rng.choice(assets["skid"]), "class": "skid",
                        "location": p + [0.0], "rot_z_deg": rot})
        if assets["cargo"] and rng.random() < 0.5:
            objects.append({"name": f"{name}_cargo", "asset": rng.choice(assets["cargo"]), "class": "cargo",
                            "location": p + [scenario["skid_height_m"]], "rot_z_deg": rot})
    return objects


def make_variants(scenario, n, seed, count_weights, valid_fraction, skid_weights=None, idle_row_fraction=0.0,
                  max_tries=200):
    """Each variant: 1 forklift, a chain of LSPs ahead of its forks along its heading, cargo on the chain.
    round(n * valid_fraction) variants are valid (exactly 1 LSP, same pipeline, so "rendered forklift" is no
    violation cue); the others are violations whose total LSP count is drawn from count_weights
    ({total: relative weight}, config lsp_count_weights, e.g. {"2": 3, "3": 1}). skid_weights (config
    skid_count_weights, e.g. {"0": 1, "1": 2, "2": 1}) draws the number of idle SKIDs per variant; None = no SKIDs.
    round(valid * idle_row_fraction) of the valid variants are an idle LSP row with no forklift and no cargo
    (scenario N2: idle_row_lsps [min, max] sheets in series, lsp_count 0 = no LSP pushed)."""
    rng = random.Random(seed)
    totals = sorted(count_weights, key=int)
    weights = [count_weights[t] for t in totals]
    skid_totals = sorted(skid_weights or {}, key=int)
    skid_w = [skid_weights[t] for t in skid_totals]
    valid_ids = set(rng.sample(range(n), round(n * valid_fraction)))
    idle_ids = set(rng.sample(sorted(valid_ids), round(len(valid_ids) * idle_row_fraction))) if idle_row_fraction else set()
    (x0, y0), (x1, y1) = scenario["floor_region"]
    assets, j = scenario["assets"], scenario["jitter"]
    variants = []
    for i in range(n):
        idle_row = i in idle_ids
        lsp_count = 0 if idle_row else 1 if i in valid_ids else int(rng.choices(totals, weights)[0])
        name = "idle_row" if idle_row else rng.choice(sorted(scenario["arrangements"]))
        dx, dy, dz = scenario["arrangements"]["in_series" if idle_row else name]
        sheets = rng.randint(*scenario["idle_row_lsps"]) if idle_row else lsp_count
        first = 0.0 if idle_row else scenario["forks_to_lsp_m"]  # idle row: origin = centre of the first sheet
        for _ in range(max_tries):
            origin = [rng.uniform(x0, x1), rng.uniform(y0, y1)]
            heading = rng.uniform(*scenario["heading_deg"])
            lsps = []
            for k in range(sheets):
                local = [k * dx + rng.uniform(-j["xy_m"], j["xy_m"]),
                         first + k * dy + rng.uniform(-j["xy_m"], j["xy_m"])]
                lsps.append(to_world(origin, heading, local) + [k * dz])
            if all(x0 <= x <= x1 and y0 <= y <= y1 for x, y, _ in lsps):
                break
        else:
            raise ValueError(f"variant {i}: no placement inside floor_region after {max_tries} tries")
        # ponytail: only the forklift origin and the LSP centres are kept inside floor_region, not whole footprints
        objects = [] if idle_row else [{"name": "forklift", "asset": rng.choice(assets["forklift"]), "class": "forklift",
                                        "location": origin + [0.0], "rot_z_deg": heading}]
        for k, loc in enumerate(lsps):
            rot = heading + rng.uniform(-j["rot_deg"], j["rot_deg"]) + rng.choice([0, 90, 180, 270]) * j["texture_turns"]
            objects.append({"name": f"idle_lsp_{k}" if idle_row else f"lsp_{k}", "asset": rng.choice(assets["lsp"]), "class": "LSP",
                            "location": loc, "rot_z_deg": rot})
        if assets["cargo"] and not idle_row:
            # ponytail: cargo sits above the first LSP, on the highest sheet; arrangements mixing dy and dz make it float
            top = max(loc[2] for loc in lsps) + scenario["lsp_thickness_m"]
            objects.append({"name": "cargo", "asset": rng.choice(assets["cargo"]), "class": "cargo",
                            "location": [lsps[0][0], lsps[0][1], top], "rot_z_deg": heading})
        n_skids = int(rng.choices(skid_totals, skid_w)[0]) if skid_totals else 0
        chain = ([] if idle_row else [origin]) + [loc[:2] for loc in lsps]
        objects += place_skids(rng, scenario, n_skids, origin, heading, chain, max_tries)
        variants.append({"id": f"v_{i:03d}", "violation_id": scenario["violation_id"], "arrangement": name,
                         "heading_deg": round(heading, 2), "lsp_count": lsp_count, "is_violation": lsp_count >= 2,
                         "skid_count": sum(o["class"] == "skid" for o in objects), "objects": objects})
    return variants


if __name__ == "__main__":
    cfg = load_config()
    n = int(sys.argv[1]) if len(sys.argv) > 1 else cfg["n_images"]
    scenario = json.loads((cfg["work"] / "scenario.json").read_text())
    variants = make_variants(scenario, n, cfg["seed"], cfg["lsp_count_weights"], cfg["valid_fraction"],
                             cfg["skid_count_weights"], cfg["idle_row_fraction"])
    write_json(cfg["work"] / "variants.json", variants)
    n_viol = sum(v["is_violation"] for v in variants)
    print(f"wrote {n} variants ({n_viol} violation, {n - n_viol} valid)")
```

- [ ] **Bước 4: Chạy test**

Chạy: `python -m pytest -q tests/test_randomize.py`
Kỳ vọng: `6 passed`

---

### Task 8: Ghép ảnh và khớp format

**File:**
- Tạo: `synth/composite.py`, `tests/test_composite.py`

- [ ] **Bước 1: Viết test**

```python
import shutil

import numpy as np
import pytest
from PIL import Image

from synth.composite import composite, estimate_noise, gblur, h264_roundtrip, mask_osd, save_like


def test_transparent_render_leaves_background_untouched():
    rng = np.random.default_rng(0)
    bg = rng.uniform(0, 1, (40, 60, 3))
    out = composite(bg, np.zeros((40, 60, 4)), 0.8, 0.02, [1, 1, 1], rng)
    assert np.allclose(out, bg)


def test_opaque_object_and_shadow():
    bg = np.full((40, 60, 3), 0.5)
    fg = np.zeros((40, 60, 4))
    fg[10:20, 10:20] = [1, 0, 0, 1]  # opaque red object
    fg[25:35, 10:20, 3] = 0.6  # shadow: black, 60% alpha
    out = composite(bg, fg, 0, 0, [1, 1, 1], np.random.default_rng(0))
    assert np.allclose(out[15, 15], [1, 0, 0])
    assert np.allclose(out[30, 15], 0.5 * 0.4)


def test_osd_is_never_covered():
    fg = np.ones((40, 60, 4))
    out = mask_osd(fg, [[0, 0, 20, 10]])
    assert out[:10, :20, 3].max() == 0 and out[20, 30, 3] == 1 and fg[0, 0, 3] == 1


def test_blur_preserves_constant_and_noise_estimate():
    assert np.allclose(gblur(np.full((20, 20, 3), 0.3), 1.5), 0.3)
    rng = np.random.default_rng(0)
    noisy = np.clip(0.5 + rng.normal(0, 0.02, (200, 200, 3)), 0, 1)
    assert 0.01 < estimate_noise(noisy) < 0.03


def test_save_like_copies_jpeg_format(tmp_path):
    src = tmp_path / "src.jpg"
    Image.fromarray(np.full((30, 50, 3), 128, np.uint8)).save(src, quality=63, subsampling=2)
    out = tmp_path / "out.jpg"
    save_like(src, np.full((30, 50, 3), 0.25), out)
    a, b = Image.open(src), Image.open(out)
    assert b.size == a.size and b.mode == a.mode
    assert b.quantization == a.quantization


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")
def test_h264_roundtrip_keeps_size_and_content():
    rng = np.random.default_rng(0)
    img = gblur(rng.uniform(0, 1, (54, 96, 3)), 2)
    out = h264_roundtrip(img, 28)
    assert out.shape == img.shape
    assert np.abs(out - img).mean() < 0.05
```

- [ ] **Bước 2: Chạy test và xác nhận nó fail**

Chạy: `python -m pytest -q tests/test_composite.py`
Kỳ vọng: FAIL với `ModuleNotFoundError: No module named 'synth.composite'`

- [ ] **Bước 3: Tạo `synth/composite.py`**

```python
"""Composite Blender RGBA renders over the camera's clean background and save in its exact format.

Run: python -m synth.composite all
     python -m synth.composite overlay work/debug/proxy.png work/debug/proxy_overlay.jpg
"""
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image, JpegImagePlugin

from synth.common import load_config


def gblur(x, sigma):
    """Separable Gaussian blur over axes 0 and 1 (edge padded)."""
    if sigma <= 0:
        return x
    r = max(1, int(3 * sigma + 0.5))
    k = np.exp(-np.arange(-r, r + 1) ** 2 / (2 * sigma * sigma))
    k /= k.sum()
    for ax in (0, 1):
        pad = [(r, r) if i == ax else (0, 0) for i in range(x.ndim)]
        p = np.pad(x, pad, mode="edge")
        n = x.shape[ax]
        x = sum(w * np.take(p, np.arange(i, i + n), axis=ax) for i, w in enumerate(k))
    return x


def estimate_noise(bg):
    """Robust sensor-noise sigma (0..1 units) from the high-pass residual of the luminance."""
    g = bg.mean(axis=2)
    res = g - gblur(g, 1.0)
    return float(1.4826 * np.median(np.abs(res - np.median(res))))


def mask_osd(fg, boxes):
    """Rendered pixels never cover the burned-in camera OSD (logo, timestamp)."""
    fg = fg.copy()
    for x0, y0, x1, y1 in boxes:
        fg[y0:y1, x0:x1, 3] = 0
    return fg


def composite(bg, fg_rgba, blur_sigma, noise_sigma, gain, rng):
    """bg: HxWx3 float 0..1; fg_rgba: HxWx4 float 0..1 with straight alpha (Blender PNG)."""
    a = fg_rgba[..., 3:]
    rgb = np.clip(fg_rgba[..., :3] * np.asarray(gain), 0, 1)
    pa = gblur(rgb * a, blur_sigma)
    a = gblur(a, blur_sigma)
    out = pa + bg * (1 - a) + rng.normal(0, noise_sigma, bg.shape) * a
    return np.clip(out, 0, 1)


def to_u8(arr):
    return (np.clip(arr, 0, 1) * 255 + 0.5).astype(np.uint8)


def h264_roundtrip(arr, crf):
    """Encode one frame with libx264 and decode it back, to get CCTV-like block/chroma artifacts."""
    with tempfile.TemporaryDirectory() as d:
        Image.fromarray(to_u8(arr)).save(f"{d}/in.png")
        ff = ["ffmpeg", "-y", "-loglevel", "error"]
        subprocess.run(ff + ["-i", f"{d}/in.png", "-c:v", "libx264", "-crf", str(crf), "-pix_fmt", "yuv420p", f"{d}/f.mp4"], check=True)
        subprocess.run(ff + ["-i", f"{d}/f.mp4", "-frames:v", "1", f"{d}/out.png"], check=True)
        return load(f"{d}/out.png", "RGB")


def save_like(src_path, arr, out_path):
    """Save float image with the same format, mode, JPEG tables, subsampling, EXIF and ICC as src_path."""
    src = Image.open(src_path)
    img = Image.fromarray(to_u8(arr)).convert(src.mode)
    kw = {k: src.info[k] for k in ("exif", "icc_profile") if src.info.get(k)}
    if src.format == "JPEG":
        sub = JpegImagePlugin.get_sampling(src)
        img.save(out_path, "JPEG", qtables=src.quantization, subsampling=sub if sub != -1 else 0, **kw)
    else:
        img.save(out_path, src.format, **kw)


def load(path, mode):
    return np.asarray(Image.open(path).convert(mode), dtype=np.float32) / 255.0


def run_all(cfg):
    work, src = cfg["work"], Path(cfg["background_image"])
    c = cfg["composite"]
    bg = load(src, "RGB")
    is_jpeg = Image.open(src).format == "JPEG"
    noise = c["noise_sigma"] if c["noise_sigma"] is not None else estimate_noise(bg)
    print(f"noise sigma {noise:.4f}")
    variants = json.loads((work / "variants.json").read_text())
    for i, v in enumerate(variants):
        fg = mask_osd(load(work / "renders" / f"{v['id']}.png", "RGBA"), cfg["osd_boxes"])
        rng = np.random.default_rng(cfg["seed"] * 1000 + i)
        out = composite(bg, fg, c["blur_sigma"], noise, c["gain"], rng)
        if not is_jpeg and c["h264_crf"]:
            # video frame source: H.264 artifacts only around the inserted pixels, background stays bit-exact
            near = gblur(fg[..., 3:], 4) > 0.01
            out = np.where(near, h264_roundtrip(out, c["h264_crf"]), out)
        save_like(src, out, work / "out" / f"{v['id']}{src.suffix}")
    print(f"composited {len(variants)} images into {work / 'out'}")


def overlay(cfg, render_path, out_path):
    """50% blend of a debug render over the clean background, for proxy alignment checks."""
    bg = load(cfg["background_image"], "RGB")
    fg = load(render_path, "RGBA")
    a = fg[..., 3:] * 0.5
    Image.fromarray(to_u8(fg[..., :3] * a + bg * (1 - a))).save(out_path, quality=95)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    cfg = load_config()
    if sys.argv[1] == "all":
        run_all(cfg)
    else:
        overlay(cfg, sys.argv[2], sys.argv[3])
```

- [ ] **Bước 4: Chạy test**

Chạy: `python -m pytest -q tests/test_composite.py`
Kỳ vọng: `6 passed` (test H.264 cần ffmpeg có trong PATH)

---

### Task 9: Kiểm tra tự động và contact sheet

**File:**
- Tạo: `synth/verify.py`, `synth/contact_sheet.py`, `tests/test_verify_sheet.py`

- [ ] **Bước 1: Viết test**

```python
import json

import numpy as np
from PIL import Image

from synth.contact_sheet import build
from synth.verify import check

META = {"lsp_count": 2, "is_violation": True, "objects": [{"bbox_2d": [100, 60, 140, 70]}], "event": {"bbox_2d": [80, 30, 150, 75]}}
VALID = {"lsp_count": 1, "is_violation": False, "objects": [{"bbox_2d": [100, 60, 140, 70]}], "event": None}


def test_verify_and_contact_sheet(tmp_path):
    src = tmp_path / "cam.png"
    frame = np.full((90, 160, 3), 100, np.uint8)
    Image.fromarray(frame).save(src)
    out = tmp_path / "out"
    out.mkdir()
    good = frame.copy()
    good[60:70, 100:140] = 30  # inserted object away from the OSD
    Image.fromarray(good).save(out / "v_000.png")
    (out / "v_000.json").write_text(json.dumps(META))
    bad = frame.copy()
    bad[0:10, 0:30] = 255  # paints over the OSD, no sidecar
    Image.fromarray(bad).save(out / "v_001.png")
    Image.fromarray(good).save(out / "v_002.png")  # synthetic valid image: 1 LSP, no event box
    (out / "v_002.json").write_text(json.dumps(VALID))

    n, errors = check(src, out, [[0, 0, 30, 10]])
    assert n == 3
    assert any("v_001.png: OSD region" in e for e in errors)
    assert any("v_001.png: missing sidecar" in e for e in errors)
    assert not any(e.startswith(("v_000", "v_002")) for e in errors)
    assert build(src, out, tmp_path / "sheet.jpg") == 3
    assert Image.open(tmp_path / "sheet.jpg").size == (4 * 640, 360)
```

- [ ] **Bước 2: Chạy test và xác nhận nó fail**

Chạy: `python -m pytest -q tests/test_verify_sheet.py`
Kỳ vọng: FAIL với `ModuleNotFoundError: No module named 'synth.contact_sheet'`

- [ ] **Bước 3: Tạo `synth/verify.py`**

```python
"""Check every output image matches the clean background's format, keeps the OSD intact and has a usable sidecar.

Run: python -m synth.verify   (exit code 1 on any failure)
"""
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, JpegImagePlugin

from synth.common import load_config


def inside(box, w, h):
    x0, y0, x1, y1 = box
    return 0 <= x0 < x1 <= w and 0 <= y0 < y1 <= h


def check(src_path, out_dir, osd_boxes):
    src = Image.open(src_path)
    src_px = np.asarray(src.convert("RGB"), dtype=np.int16)
    w, h = src.size
    errors = []
    images = sorted(Path(out_dir).glob(f"v_*{Path(src_path).suffix}"))
    if not images:
        errors.append(f"no images in {out_dir}")
    for p in images:
        img = Image.open(p)
        if (img.size, img.mode, img.format) != (src.size, src.mode, src.format):
            errors.append(f"{p.name}: {img.size} {img.mode} {img.format} != {src.size} {src.mode} {src.format}")
            continue
        if src.format == "JPEG" and (
            img.quantization != src.quantization
            or JpegImagePlugin.get_sampling(img) != JpegImagePlugin.get_sampling(src)
        ):
            errors.append(f"{p.name}: JPEG tables/subsampling differ from input")
        px = np.asarray(img.convert("RGB"), dtype=np.int16)
        for x0, y0, x1, y1 in osd_boxes:
            if np.abs(px[y0:y1, x0:x1] - src_px[y0:y1, x0:x1]).mean() > 2:
                errors.append(f"{p.name}: OSD region {[x0, y0, x1, y1]} changed")
        side = p.with_suffix(".json")
        if not side.exists():
            errors.append(f"{p.name}: missing sidecar")
            continue
        meta = json.loads(side.read_text())
        if meta["is_violation"] != (meta["lsp_count"] >= 2):
            errors.append(f"{p.name}: is_violation inconsistent with lsp_count")
        if not any(o["bbox_2d"] and inside(o["bbox_2d"], w, h) for o in meta["objects"]):
            errors.append(f"{p.name}: no added object box inside the image")
        event = meta["event"]
        if meta["is_violation"] and not (event and event["bbox_2d"] and inside(event["bbox_2d"], w, h)):
            errors.append(f"{p.name}: violation without event box")
        if not meta["is_violation"] and event:
            errors.append(f"{p.name}: valid image with an event box")
    return len(images), errors


if __name__ == "__main__":
    cfg = load_config()
    n, errors = check(cfg["background_image"], cfg["work"] / "out", cfg["osd_boxes"])
    print("\n".join(errors) or f"OK: {n} images match the input format")
    sys.exit(1 if errors else 0)
```

- [ ] **Bước 4: Tạo `synth/contact_sheet.py`**

```python
"""Clean background + all outputs in one grid, with projected sidecar boxes drawn
(red = forklift, cargo and LSPs, cyan = idle SKIDs and their cargo, yellow = event box of violation images).

Run: python -m synth.contact_sheet   -> work/contact_sheet.jpg
"""
import json
from pathlib import Path

from PIL import Image, ImageDraw

from synth.common import load_config

COLS, THUMB_W = 4, 640


def build(src_path, out_dir, dest):
    tiles = [(Image.open(src_path).convert("RGB"), "BACKGROUND", [])]
    for p in sorted(Path(out_dir).glob(f"v_*{Path(src_path).suffix}")):
        side = p.with_suffix(".json")
        meta = json.loads(side.read_text()) if side.exists() else {"objects": [], "lsp_count": "?", "event": None}
        boxes = [(o["bbox_2d"], "cyan" if o.get("name", "").startswith("skid_") else "red")
                 for o in meta["objects"] if o["bbox_2d"]]
        if meta["event"] and meta["event"]["bbox_2d"]:
            boxes.append((meta["event"]["bbox_2d"], "yellow"))
        tiles.append((Image.open(p).convert("RGB"), f"{p.stem} LSP x{meta['lsp_count']}", boxes))
    w, h = tiles[0][0].size
    s = THUMB_W / w
    th = int(h * s)
    rows = (len(tiles) + COLS - 1) // COLS
    sheet = Image.new("RGB", (COLS * THUMB_W, rows * th), "white")
    for i, (img, label, boxes) in enumerate(tiles):
        t = img.resize((THUMB_W, th))
        d = ImageDraw.Draw(t)
        for (x0, y0, x1, y1), color in boxes:
            d.rectangle([x0 * s, y0 * s, x1 * s, y1 * s], outline=color, width=1)
        d.rectangle([0, th - 16, 8 * len(label) + 8, th], fill="black")
        d.text((4, th - 14), label, fill="white")
        sheet.paste(t, ((i % COLS) * THUMB_W, (i // COLS) * th))
    sheet.save(dest, quality=90)
    return len(tiles) - 1


if __name__ == "__main__":
    cfg = load_config()
    n = build(cfg["background_image"], cfg["work"] / "out", cfg["work"] / "contact_sheet.jpg")
    print(f"contact sheet with {n} outputs -> {cfg['work'] / 'contact_sheet.jpg'}")
```

- [ ] **Bước 5: Chạy toàn bộ test**

Chạy: `python -m pytest -q`
Kỳ vọng: `22 passed`

---

### Task 10: Các script Blender và một lần render smoke end-to-end

**File:**
- Tạo: `blender/build_lsp.py`, `blender/texture_asset.py`, `blender/render_variants.py`
- Dùng: `blender/cli_to_blend.py` (Task 1), `work/camera.json` và `work/scene_facts.json` (Task 5)

- [ ] **Bước 1: Tạo `blender/build_lsp.py`**

```python
"""Build the LSP asset: thin rounded-corner sheet with a real rectified texture, one .blend per texture.

Run: blender --background --python blender/build_lsp.py -- config.json
Reads work/lsp_params.json, work/textures/lsp_*.png; writes work/assets/lsp_<k>.blend
Asset convention: metres, origin at footprint centre on the floor, +Y = pushing direction.
"""
import json
import math
import sys
from pathlib import Path

import bmesh
import bpy

cfg = json.loads(Path(sys.argv[sys.argv.index("--") + 1]).read_text())
work = Path(cfg["work_dir"]).resolve()
p = json.loads((work / "lsp_params.json").read_text())
sx, sy, t, r = p["size_x_m"], p["size_y_m"], p["thickness_m"], p["corner_radius_m"]


def rounded_rect(sx, sy, r, seg=8):
    pts = []
    for cx, cy, a0 in ((sx / 2 - r, sy / 2 - r, 0), (-sx / 2 + r, sy / 2 - r, 90),
                       (-sx / 2 + r, -sy / 2 + r, 180), (sx / 2 - r, -sy / 2 + r, 270)):
        for i in range(seg + 1):
            a = math.radians(a0 + 90 * i / seg)
            pts.append((cx + r * math.cos(a), cy + r * math.sin(a)))
    return pts


pts = rounded_rect(sx, sy, r)
n = len(pts)
verts = [(x, y, t) for x, y in pts] + [(x, y, 0.0) for x, y in pts]
faces = [list(range(n)), list(range(2 * n - 1, n - 1, -1))]
faces += [[i, n + i, n + (i + 1) % n, (i + 1) % n] for i in range(n)]

(work / "assets").mkdir(parents=True, exist_ok=True)
for k, tex_path in enumerate(sorted((work / "textures").glob("lsp_*.png"))):
    bpy.ops.wm.read_factory_settings(use_empty=True)
    me = bpy.data.meshes.new("lsp")
    me.from_pydata(verts, [], faces)
    bm = bmesh.new()
    bm.from_mesh(me)
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    bm.to_mesh(me)
    bm.free()
    uv = me.uv_layers.new(name="UVMap")
    for loop in me.loops:
        co = me.vertices[loop.vertex_index].co
        uv.data[loop.index].uv = (co.x / sx + 0.5, co.y / sy + 0.5)

    mat = bpy.data.materials.new("lsp")
    try:
        mat.use_nodes = True  # deprecated/always-on in newer Blender
    except AttributeError:
        pass
    bsdf = next(nd for nd in mat.node_tree.nodes if nd.type == "BSDF_PRINCIPLED")
    tex = mat.node_tree.nodes.new("ShaderNodeTexImage")
    tex.image = bpy.data.images.load(str(tex_path))
    tex.image.pack()
    mat.node_tree.links.new(tex.outputs["Color"], bsdf.inputs["Base Color"])
    bsdf.inputs["Roughness"].default_value = p["roughness"]
    me.materials.append(mat)

    obj = bpy.data.objects.new("lsp", me)
    bpy.context.scene.collection.objects.link(obj)
    out = work / "assets" / f"lsp_{k}.blend"
    bpy.ops.wm.save_as_mainfile(filepath=str(out))
    print(f"{tex_path.name} -> {out}")
```

- [ ] **Bước 2: Tạo `blender/texture_asset.py`.** Harness CLI-Anything không thêm được texture ảnh, nên script này gắn một ảnh cắt SAM3 thật lên các material mà agent đặt tên `tex_*`.

```python
"""Put a real SAM3 object crop on an agent-built asset (the CLI-Anything harness has no image textures).

Run: blender --background --python blender/texture_asset.py -- config.json NAME CROP.png
Reads work/assets_raw/NAME.blend; writes work/assets/NAME.blend
Every material named tex_* gets CROP box-projected into its Base Color; other materials stay as authored.
"""
import json
import sys
from pathlib import Path

import bpy

args = sys.argv[sys.argv.index("--") + 1:]
cfg = json.loads(Path(args[0]).read_text())
name, crop = args[1], Path(args[2]).resolve()
work = Path(cfg["work_dir"]).resolve()
bpy.ops.wm.open_mainfile(filepath=str(work / "assets_raw" / f"{name}.blend"))
img = bpy.data.images.load(str(crop))
img.pack()
done = []
for mat in bpy.data.materials:
    if not mat.name.startswith("tex_"):
        continue
    try:
        mat.use_nodes = True  # deprecated/always-on in newer Blender
    except AttributeError:
        pass
    nt = mat.node_tree
    bsdf = next(nd for nd in nt.nodes if nd.type == "BSDF_PRINCIPLED")
    coord = nt.nodes.new("ShaderNodeTexCoord")
    tex = nt.nodes.new("ShaderNodeTexImage")
    tex.image, tex.projection, tex.projection_blend = img, "BOX", 0.2
    nt.links.new(coord.outputs["Generated"], tex.inputs["Vector"])
    nt.links.new(tex.outputs["Color"], bsdf.inputs["Base Color"])
    done.append(mat.name)
assert done, f"{name}: no tex_* material; name the parts that should carry the real texture tex_<part>"
(work / "assets").mkdir(parents=True, exist_ok=True)
out = work / "assets" / f"{name}.blend"
bpy.ops.wm.save_as_mainfile(filepath=str(out))
print(f"{name}: {crop.name} on {done} -> {out}")
```

- [ ] **Bước 3: Tạo `blender/render_variants.py`**

```python
"""Render the inserted forklift, cargo, LSPs and idle SKIDs + caught shadows (transparent film) with the calibrated camera, and dump sidecars.

Run: blender --background --python blender/render_variants.py -- config.json               # all variants
     blender --background --python blender/render_variants.py -- config.json --debug-proxy # proxy alignment render
Reads work/camera.json, work/scene_facts.json, work/proxy.blend, work/assets/*.blend, work/variants.json
Writes work/renders/<id>.png, work/out/<id>.json   (debug: work/debug/proxy.png)
"""
import json
import math
import sys
from pathlib import Path

import bpy
from bpy_extras.object_utils import world_to_camera_view
from mathutils import Matrix, Vector

args = sys.argv[sys.argv.index("--") + 1:]
cfg = json.loads(Path(args[0]).read_text())
debug = "--debug-proxy" in args
work = Path(cfg["work_dir"]).resolve()
cam_cfg = json.loads((work / "camera.json").read_text())
facts = json.loads((work / "scene_facts.json").read_text())
src_suffix = Path(cfg["background_image"]).suffix

bpy.ops.wm.read_factory_settings(use_empty=True)
scene = bpy.context.scene


def append_objects(path):
    """Append every object of a .blend; drop cameras (the calibrated camera is the only camera)."""
    with bpy.data.libraries.load(str(path), link=False) as (src, dst):
        dst.objects = list(src.objects)
    kept = []
    for o in dst.objects:
        if o is None:
            continue
        if o.type == "CAMERA":
            bpy.data.objects.remove(o)
        else:
            kept.append(o)
    return kept


# camera
cam_data = bpy.data.cameras.new("calibrated_cam")
cam = bpy.data.objects.new("calibrated_cam", cam_data)
scene.collection.objects.link(cam)
scene.camera = cam
cam_data.sensor_fit = "HORIZONTAL"
cam_data.sensor_width = cam_cfg["sensor_width_mm"]
cam_data.lens = cam_cfg["lens_mm"]
cam_data.shift_x = cam_cfg["shift_x"]
cam_data.shift_y = cam_cfg["shift_y"]
cam_data.clip_start, cam_data.clip_end = 0.05, 500.0
cam.matrix_world = Matrix(cam_cfg["matrix_world"])
W, H = cam_cfg["width"], cam_cfg["height"]

# render settings
r = scene.render
r.resolution_x, r.resolution_y, r.resolution_percentage = W, H, 100
r.film_transparent = True
r.image_settings.file_format = "PNG"
r.image_settings.color_mode = "RGBA"
r.image_settings.color_depth = "8"
scene.view_settings.view_transform = "Standard"
scene.view_settings.look = "None"

# proxy scene (floor + static structures of the clean background): meshes become shadow catchers
# (invisible, catch shadows, occlude); lights are kept
proxies = append_objects(work / "proxy.blend")
for o in proxies:
    scene.collection.objects.link(o)
    if o.type == "MESH":
        o.is_shadow_catcher = not debug
        o.color = (0.0, 1.0, 1.0, 1.0) if "floor" in o.name else (1.0, 0.0, 1.0, 1.0)  # debug colours

if debug:
    r.engine = "BLENDER_WORKBENCH"
    scene.display.shading.light = "FLAT"
    scene.display.shading.color_type = "OBJECT"
    r.filepath = str(work / "debug" / "proxy.png")
    bpy.ops.render.render(write_still=True)
    sys.exit(0)

r.engine = "CYCLES"
scene.cycles.samples = cfg["blender"]["samples"]
world = bpy.data.worlds.new("photo_ambient")
scene.world = world
try:
    world.use_nodes = True  # deprecated/always-on in newer Blender
except AttributeError:
    pass
bg = next(n for n in world.node_tree.nodes if n.type == "BACKGROUND")
bg.inputs[0].default_value = (*facts["ambient_rgb_linear"], 1.0)
bg.inputs[1].default_value = cfg["blender"]["world_strength"]

# assets: one collection per .blend (not linked to the scene), instanced per placement
assets = {}
for path in sorted((work / "assets").glob("*.blend")):
    col = bpy.data.collections.new(f"asset_{path.stem}")
    corners = []
    for o in append_objects(path):
        if o.type == "LIGHT":
            bpy.data.objects.remove(o)
            continue
        col.objects.link(o)
        if o.type == "MESH":
            corners += [o.matrix_world @ Vector(c) for c in o.bound_box]
    lo = Vector([min(c[i] for c in corners) for i in range(3)])
    hi = Vector([max(c[i] for c in corners) for i in range(3)])
    box = [Vector((x, y, z)) for x in (lo.x, hi.x) for y in (lo.y, hi.y) for z in (lo.z, hi.z)]
    assets[path.stem] = (col, box, list(hi - lo))


def bbox_2d(points):
    """Projected pixel box [x0, y0, x1, y1] clipped to the image; None if behind camera or off-screen."""
    ndc = [world_to_camera_view(scene, cam, p) for p in points]
    if any(p.z <= 0 for p in ndc):
        return None
    xs = [min(max(p.x * W, 0), W) for p in ndc]
    ys = [min(max((1 - p.y) * H, 0), H) for p in ndc]
    box = [round(min(xs), 1), round(min(ys), 1), round(max(xs), 1), round(max(ys), 1)]
    return box if box[2] > box[0] and box[3] > box[1] else None


variants = json.loads((work / "variants.json").read_text())
(work / "out").mkdir(parents=True, exist_ok=True)
for v in variants:
    made = []
    for it in v["objects"]:
        e = bpy.data.objects.new(it["name"], None)
        e.instance_type = "COLLECTION"
        e.instance_collection = assets[it["asset"]][0]
        e.location = it["location"]
        e.rotation_euler = (0.0, 0.0, math.radians(it["rot_z_deg"]))
        scene.collection.objects.link(e)
        made.append((e, it))
    bpy.context.view_layer.update()
    r.filepath = str(work / "renders" / f"{v['id']}.png")
    bpy.ops.render.render(write_still=True)
    meta = {
        "image": f"{v['id']}{src_suffix}",
        "camera_id": cfg["camera_id"],
        "source_image": cfg["background_image"],
        "violation_id": v["violation_id"],
        "arrangement": v["arrangement"],
        "heading_deg": v["heading_deg"],
        "lsp_count": v["lsp_count"],
        "is_violation": v["is_violation"],
        "skid_count": v.get("skid_count", 0),
        "objects": [
            {
                "name": it["name"],
                "class": it["class"],
                "asset": it["asset"],
                "location": [round(c, 3) for c in it["location"]],
                "rot_z_deg": round(it["rot_z_deg"], 2),
                "size_m": [round(s, 3) for s in assets[it["asset"]][2]],
                "bbox_2d": bbox_2d([e.matrix_world @ c for c in assets[it["asset"]][1]]),
            }
            for e, it in made
        ],
        # future-label hint: one box per violation event (forklift + cargo + all LSPs), occlusion ignored;
        # synthetic valid images (1 LSP) have no event; idle SKIDs (skid_*) and their cargo are never in it
        "event": {"class": "Forklift Pushing Multiple Lsps",
                  "bbox_2d": bbox_2d([e.matrix_world @ c for e, it in made if not it["name"].startswith("skid_")
                                      for c in assets[it["asset"]][1]])}
        if v["is_violation"] else None,
    }
    (work / "out" / f"{v['id']}.json").write_text(json.dumps(meta, indent=2))
    for e, _ in made:
        bpy.data.objects.remove(e, do_unlink=True)
print(f"rendered {len(variants)} variants")
```

- [ ] **Bước 4: Dựng scene smoke.** Scene gồm một proxy sàn và một đèn sun dựng bằng CLI-Anything, một tấm thử dày 0.3 m đặt tại điểm trục quang học chạm sàn để kiểm tra phần toán camera ngay trong Blender, và một khối lập phương có material `tex_body` để chạy thử `texture_asset.py`.

```bash
mkdir -p work_smoke/cli work_smoke/textures work_smoke/assets_raw
cp work/camera.json work/scene_facts.json work_smoke/
python - <<'EOF'
import json, math
from PIL import Image
cfg = json.load(open("config.json")); cfg["work_dir"] = "work_smoke"
json.dump(cfg, open("work_smoke/config.json", "w"), indent=2)
f = json.load(open("work_smoke/scene_facts.json"))
y = f["camera_height_m"] / math.tan(math.radians(f["camera_pitch_down_deg"]))
json.dump({"size_x_m": 1.2, "size_y_m": 1.2, "thickness_m": 0.3, "corner_radius_m": 0.06, "roughness": 0.4},
          open("work_smoke/lsp_params.json", "w"))
json.dump([{"id": "v_000", "violation_id": "smoke", "arrangement": "smoke", "heading_deg": 30, "lsp_count": 2, "is_violation": True,
            "objects": [{"name": "lsp_extra_1", "asset": "lsp_0", "class": "LSP", "location": [0, y, 0], "rot_z_deg": 30}]}],
          open("work_smoke/variants.json", "w"))
Image.new("RGB", (512, 512), (40, 70, 160)).save("work_smoke/textures/lsp_0.png")
print(f"test sheet at y = {y:.2f} m")
EOF
Y=$(python -c "import json, math; f = json.load(open('work_smoke/scene_facts.json')); print(round(f['camera_height_m'] / math.tan(math.radians(f['camera_pitch_down_deg'])), 3))")
cli-anything-blender --json scene new --name smoke --output work_smoke/cli/proxy.blend-cli.json
cli-anything-blender --json --project work_smoke/cli/proxy.blend-cli.json object add plane --name proxy_floor -p size=1 --location=0,$Y,0 --scale=30,30,1
cli-anything-blender --json --project work_smoke/cli/proxy.blend-cli.json light add sun --name light_sun --rotation=35,0,40 --power 3
cli-anything-blender --project work_smoke/cli/proxy.blend-cli.json render script work_smoke/unused.png > work_smoke/cli/proxy_export.py
blender --background --python blender/cli_to_blend.py -- work_smoke/cli/proxy_export.py work_smoke/proxy.blend
blender --background --python blender/build_lsp.py -- work_smoke/config.json
blender --background --python-expr "import bpy; bpy.ops.wm.read_factory_settings(use_empty=True); bpy.ops.mesh.primitive_cube_add(); bpy.context.object.data.materials.append(bpy.data.materials.new('tex_body')); bpy.ops.wm.save_as_mainfile(filepath='work_smoke/assets_raw/box.blend')"
blender --background --python blender/texture_asset.py -- work_smoke/config.json box work_smoke/textures/lsp_0.png
blender --background --python blender/render_variants.py -- work_smoke/config.json
blender --background --python blender/render_variants.py -- work_smoke/config.json --debug-proxy
```

Kỳ vọng: các dòng cuối là `saved work_smoke/proxy.blend: ['light_sun', 'proxy_floor']`, `lsp_0.png -> .../work_smoke/assets/lsp_0.blend`, `box: lsp_0.png on ['tex_body'] -> .../work_smoke/assets/box.blend` và `rendered 1 variants`, không có Python traceback.

- [ ] **Bước 5: Kiểm tra ảnh render smoke**

```bash
python - <<'EOF'
import json
import numpy as np
from PIL import Image
a = np.asarray(Image.open("work_smoke/renders/v_000.png"))
alpha, rgb_max = a[..., 3], a[..., :3].max(-1)
opaque, shadow = alpha == 255, (alpha > 20) & (alpha < 235) & (rgb_max < 30)
print("opaque px", opaque.sum(), "shadow px", shadow.sum())
assert opaque.sum() > 200, "sheet not rendered"
assert shadow.sum() > 50, "no caught shadow (check shadow catcher / sun)"
cam = json.load(open("work_smoke/camera.json"))
cx, cy = cam["K_norm"][0][2] * cam["width"], cam["K_norm"][1][2] * cam["height"]
ys, xs = np.nonzero(opaque)
print("sheet centroid", xs.mean(), ys.mean(), "principal point", cx, cy)
assert abs(xs.mean() - cx) < 0.08 * cam["width"] and abs(ys.mean() - cy) < 0.08 * cam["width"], "camera shift/pose wrong"
meta = json.load(open("work_smoke/out/v_000.json"))
x0, y0, x1, y1 = meta["objects"][0]["bbox_2d"]
assert x0 - 3 <= xs.min() and xs.max() <= x1 + 3 and y0 - 3 <= ys.min() and ys.max() <= y1 + 3, "bbox_2d misses the sheet"
assert meta["event"]["bbox_2d"] and meta["lsp_count"] == 2 and meta["is_violation"]
assert (np.asarray(Image.open("work_smoke/debug/proxy.png"))[..., 3] > 0).mean() > 0.2, "debug proxy render empty"
print("SMOKE OK")
EOF
```

Kỳ vọng: `SMOKE OK`. Nếu kiểm tra trọng tâm thất bại thì dấu của `shift_x`/`shift_y` hoặc pose đang sai: kiểm tra lại `blender_camera` với phiên bản Blender đang dùng trước khi làm tiếp. Sau đó chạy: `rm -rf work_smoke`.

---

### Task 11: Agent — scene proxy và cổng kiểm tra căn chỉnh

**File:**
- Tạo: `prompts/01_proxy_scene.md`
- Sinh ra: `work/cli/proxy.blend-cli.json`, `work/proxy.blend`, `work/debug/proxy_overlay.jpg`

- [ ] **Bước 1: Tạo `prompts/01_proxy_scene.md`**

````markdown
# Task: build the proxy scene for this camera

Read AGENTS.md first. The attached images are the clean background (`data/input/cam01/background.png`), the reference images of the same view (`data/input/cam01/objects/obj_*.png`, object extraction only, never a background), and possibly `work/debug/proxy_overlay.jpg` from a previous iteration.

Goal: a PROXY scene of simple boxes/planes that sits exactly on top of the static geometry of the clean background (floor, racks, walls, columns), plus lights that reproduce the lighting of the real frames. Proxies are never visible in the final images: they only catch the shadows of the objects we insert (forklift, cargo, LSPs, SKIDs) and hide inserted objects that are behind real static things. Accuracy matters most on the open floor where forklifts are really seen (`object_floor_positions_m` in `work/calibration.json`) and within about 4 m of it.

Steps:
1. Read `work/scene_facts.json`, `work/camera.json` and `work/calibration.json`. Use `work/points_world.npy` and `work/masks/background/*.png` to measure world positions and sizes (e.g. take the mask pixels of a rack, look up their world points, compute extents with numpy).
2. Create (or, if it exists, edit) the CLI project `work/cli/proxy.blend-cli.json`:
   `cli-anything-blender --json scene new --name proxy --output work/cli/proxy.blend-cli.json`
3. Add proxies (all with `-p size=1`, scale = dimensions in metres):
   - `proxy_floor`: a plane at z = 0 covering all visible floor where forklifts drive (at least 12 m x 12 m).
   - Racks, walls, columns and other static objects within 6 m of that floor: boxes named `proxy_<what>` (they must occlude inserted objects placed behind them).
   - No proxies for forklifts, LSPs, SKIDs or cargo: the background is clean and those objects are inserted later.
4. Lights: study the real shadows of the forklifts, cargo and LSPs in the reference images (direction, softness, strength). Indoors with ceiling lights, use 1-3 `area` lights above the scene; with sunlight through doors/windows, use a `sun`. Name them `light_<what>`. Set power so an inserted grey object would be about as bright as similar real objects in the object frames.
5. Export to `work/proxy.blend` (see AGENTS.md), then render the alignment check:
   `blender --background --python blender/render_variants.py -- config.json --debug-proxy`
   `python -m synth.composite overlay work/debug/proxy.png work/debug/proxy_overlay.jpg`
6. If you can view local images, open `work/debug/proxy_overlay.jpg`: magenta proxies must cover the corresponding real static objects (the cyan floor should cover the visible floor), and floor edges/rack edges must line up. Ignore the camera OSD (logo top-left, timestamp top-right). Fix and repeat steps 3-6 at most 5 times.

Final message: a table of proxies (name, location, size), the lights with the real shadows they were matched to, and any region you believe is still misaligned.
````

- [ ] **Bước 2: Chạy agent.** Phần thiết lập shell dùng chung cho Task 11–14 và 16. `-` (prompt đọc từ stdin) đặt trước các option và `--image` đặt cuối cùng, vì `--image` nhận một hoặc nhiều giá trị.

```bash
source .venv/bin/activate
MODEL=$(python -c "import json; print(json.load(open('config.json'))['codex_model'])")
SAFETY="-s workspace-write"   # or the flag recorded in docs/spike-notes.md (S8)
OBJ=$(ls data/input/cam01/objects/obj_*.png | paste -sd, -)   # reference images of this camera
mkdir -p work/logs work/cli
codex exec - -m "$MODEL" $SAFETY --skip-git-repo-check -C "$PWD" -o work/logs/01_proxy.md \
  -i data/input/cam01/background.png,$OBJ < prompts/01_proxy_scene.md
```

- [ ] **Bước 3: Cổng kiểm tra căn chỉnh (người làm).** Mở `work/debug/proxy_overlay.jpg` cạnh `data/input/cam01/background.png`. Tiêu chí đạt:
  - các proxy magenta phủ lên kệ, tường và cột thật, lệch không quá vài pixel;
  - sàn cyan phủ vùng sàn nhìn thấy được nơi xe nâng chạy;
  - không có proxy nào phủ lên vùng sàn trống nơi sẽ chèn đối tượng.

  Nếu không đạt, chạy lại với overlay đính kèm (tối đa 3 lần):

```bash
codex exec - -m "$MODEL" $SAFETY --skip-git-repo-check -C "$PWD" -o work/logs/01_proxy_iter.md \
  -i data/input/cam01/background.png,$OBJ,work/debug/proxy_overlay.jpg < prompts/01_proxy_scene.md
```

Kỳ vọng: `work/proxy.blend` tồn tại và overlay qua được cổng kiểm tra. Ghi lại trong `work/logs/01_proxy.md` những vùng còn yếu.

---

### Task 12: Agent — thông số asset LSP, rồi dựng các asset LSP

**File:**
- Tạo: `prompts/03_lsp_params.md`
- Sinh ra: `work/lsp_params.json`, `work/assets/lsp_<k>.blend`

- [ ] **Bước 1: Tạo `prompts/03_lsp_params.md`**

````markdown
# Task: specify the LSP asset

Read AGENTS.md first. Attached: `docs/requirements/lsp-reference.png` and SAM3 crops of real LSPs from this camera (`work/refs/*_LSP_*.png`).

1. Size: `lsp_instances` in `work/calibration.json` holds the measured footprint (`extent_m`: long and short side) of every LSP seen in the reference images. Sheets partly covered by cargo or the forklift measure too short: rely on the fully visible ones, combine them with `docs/requirements/lsp-reference.png` (LSPs are roughly square) and state your assumptions. You can re-measure a sheet with its mask `work/masks/obj_<n>/LSP_<i>.png` and `work/points_world.npy`.
   If `lsp_size_m` in `config.json` is not null, that is the known real size: still measure (as a cross-check) and report both; the pipeline replaces your size with the configured one afterwards.
2. Estimate thickness (expected 0.01-0.02 m), corner radius, and roughness (slightly glossy plastic: 0.3-0.5) from the crops.
3. Write `work/lsp_params.json` exactly in this shape:
   {"size_x_m": 1.2, "size_y_m": 1.2, "thickness_m": 0.015, "corner_radius_m": 0.06, "roughness": 0.4,
    "notes": "how each value was measured"}

Do not modify any other file. Final message: the JSON and your confidence per value.
````

- [ ] **Bước 2: Chạy agent**

```bash
codex exec - -m "$MODEL" $SAFETY --skip-git-repo-check -C "$PWD" -o work/logs/03_lsp_params.md \
  -i docs/requirements/lsp-reference.png,$(ls work/refs/*_LSP_0.png | head -8 | paste -sd, -) < prompts/03_lsp_params.md
python -c "
import json; from synth.common import apply_lsp_size, load_config, write_json
c = load_config(); p = c['work'] / 'lsp_params.json'
write_json(p, apply_lsp_size(json.loads(p.read_text()), c['lsp_size_m']))"   # config lsp_size_m (if set) wins
cat work/lsp_params.json
```

Kỳ vọng: JSON hợp lệ với `size_*` trong khoảng 0.9–1.4 (bằng `lsp_size_m` khi giá trị này được đặt trong `config.json`; khi đó giá trị đo được giữ trong `measured_size_m`) và `thickness_m` trong khoảng 0.01–0.02.

- [ ] **Bước 3: Dựng các asset**

```bash
blender --background --python blender/build_lsp.py -- config.json
ls work/assets
```

Kỳ vọng: mỗi texture trong `work/textures/` có một file `lsp_<k>.blend`.

---

### Task 13: Agent — dựng lại xe nâng, các loại hàng và SKID thành asset 3D

**File:**
- Tạo: `prompts/05_build_asset.md`
- Sinh ra: `work/cli/<name>.blend-cli.json`, `work/assets_raw/<name>.blend`, `work/assets/forklift.blend`, `work/assets/cargo_<type>.blend`, `work/assets/skid.blend`, `work/previews/*.png`

- [ ] **Bước 1: Tạo `prompts/05_build_asset.md`**

````markdown
# Task: rebuild one real object as a 3D asset with cli-anything-blender

Read AGENTS.md first. The first line of this prompt says `ASSET: <name>` (`forklift`, `cargo_<type>` or `skid`; SAM3 class `forklift`, `cargo` or `SKID`). The attached images are tight SAM3 crops of the real object from this camera's reference images (`work/refs/*_<class>_*.png`), and possibly preview renders of a previous attempt.

Requirements:
- Real-world size in metres. Measure it: the object's masks (`work/masks/obj_<n>/<class>_<i>.png`) with `work/points_world.npy` give its floor footprint; the height follows from the mask height and the camera geometry in `work/scene_facts.json`. State the measured dimensions and compare them with typical real sizes (a counterbalance forklift is about 2.5-3.5 m long without forks, 1.1-1.3 m wide, 2.1-2.3 m high at the overhead guard; a SKID is about 1.0-1.2 m x 0.8-1.2 m and 0.12-0.16 m high).
- Origin: centre of the footprint on the floor (bottom at z = 0). Front / pushing direction faces +Y (forklift: the forks / push plate; SKID: a fork-entry side).
- Build ONLY from `cli-anything-blender` primitives, modifiers (bevel, array, boolean, solidify, mirror, ...) and Principled materials (colour, metallic, roughness). CCTV resolution hides tiny details, so prioritise silhouette, proportions, thickness, edges and colour.
- Name the materials of the large surfaces that should show the real object's look `tex_<part>` (forklift: body and counterweight; cargo: the wrap; SKID: the deck boards). The pipeline projects the real crop onto them afterwards; still give them a base colour close to the crop. Small parts (mast, forks, wheels, overhead guard) keep plain Principled materials.
- Forklift: counterbalance body, overhead guard, mast, forks or push plate at the front (+Y), wheels. Cargo: stretch-wrapped goods, a box stack with slightly bulged, bevelled sides and a light grey, glossy, semi-reflective wrap; do not model an LSP under it (the pipeline places the LSPs). SKID: a pallet with top deck boards, blocks or stringers and bottom boards, with the gaps between the boards visible at the camera pitch; do not put cargo on it (the pipeline places cargo on some SKIDs).
- Project file: `work/cli/<name>.blend-cli.json`. Output: `work/assets_raw/<name>.blend` (convert as in AGENTS.md).

Work in 3 passes, rendering a preview after each pass:
1. Blockout: correct overall dimensions.
2. Detail: parts, edges, lips, holes, ribs visible at CCTV distance.
3. Materials: colour, roughness, metallic, `tex_*` names.

Preview: add a camera looking at the asset from the same pitch as the real camera (`camera_pitch_down_deg` in `work/scene_facts.json`) plus one `sun` light to the CLI project (they are ignored by the pipeline), then run
`cli-anything-blender --json --project work/cli/<name>.blend-cli.json render execute work/previews/<name>_pass<N>.png`
and run the generated `work/previews/_render_script.py` with `blender --background --python work/previews/_render_script.py`. If you can view local images, compare each preview with the reference crops before the next pass.

Final message: measured dimensions, parts list, which materials are `tex_*`, and what still differs from the reference.
````

- [ ] **Bước 2: Dựng xe nâng, một asset cho mỗi loại hàng và SKID.** Astra 6 tạo hình học bằng harness; sau đó `blender/texture_asset.py` gắn ảnh cắt thật lớn nhất lên các material `tex_*`. SKID là bắt buộc: mọi ảnh POC đều đặt SKID nằm yên.

```bash
build_asset() {   # NAME CROP_GLOB [EXTRA_IMAGES]: Astra 6 rebuilds the object from its crops; the largest crop becomes its texture
  (echo "ASSET: $1"; cat prompts/05_build_asset.md) | codex exec - -m "$MODEL" $SAFETY --skip-git-repo-check -C "$PWD" \
    -o work/logs/05_$1.md -i $(ls -S $2 | head -6 | paste -sd, -)${3:+,$3}
  blender --background --python blender/texture_asset.py -- config.json $1 $(ls -S $2 | head -1)
}
mkdir -p work/assets_raw work/previews
build_asset forklift "work/refs/*_forklift_*.png"
build_asset cargo_0 "work/refs/*_cargo_*.png"
# more cargo types: pass the crops of that cargo only, e.g. build_asset cargo_1 "work/refs/obj_00610_cargo_*.png"
build_asset skid "work/refs/*_SKID_*.png"   # required: idle SKIDs appear in every POC image
ls work/assets
```

Kỳ vọng: có `work/assets/forklift.blend`, mỗi loại hàng một file `work/assets/cargo_<type>.blend` và `work/assets/skid.blend`, mỗi asset có một dòng `... on ['tex_...'] -> ...`, và kích thước đo được trong `work/logs/05_<name>.md` nằm trong khoảng điển hình.

- [ ] **Bước 3: Cổng kiểm tra chi tiết (người làm; prompt nhiều lần cho tới khi đủ chi tiết).** So sánh `work/previews/<name>_pass3.png` với các ảnh cắt trong `work/refs/`: đường bao, tỉ lệ, trụ nâng, khung bảo vệ trên đầu, càng nâng, hình dạng và màu hàng, ván mặt và chân khối của SKID phải đọc ra đúng là vật thật ở tỉ lệ CCTV. Nếu không, chạy lại với các ảnh preview đính kèm để Astra 6 tinh chỉnh lần trước (tối đa 3 lần mỗi asset), ví dụ với xe nâng:

```bash
build_asset forklift "work/refs/*_forklift_*.png" $(ls work/previews/forklift_pass*.png | paste -sd, -)
```

Kỳ vọng: mọi asset qua được cổng kiểm tra. Vẻ ngoài cuối cùng được đánh giá lại trên các ảnh ghép kiểm tra ở Task 14.

---

### Task 14: Agent — template kịch bản

**File:**
- Tạo: `prompts/04_scenario.md`
- Sinh ra: `work/scenario.json` (cùng 3 ảnh kiểm tra, sẽ bị ghi đè ở Task 15)

- [ ] **Bước 1: Tạo `prompts/04_scenario.md`**

````markdown
# Task: write the scenario template for "Forklift Pushing Multiple Lsps"

Read AGENTS.md first. Attached: the clean background, the reference images of this camera and `work/debug/proxy_overlay.jpg`. Inputs: `work/calibration.json` (floor positions where forklifts and LSPs were really seen), `work/lsp_params.json` (LSP size), the assets in `work/assets/` (`forklift`, `cargo_<type>`, `lsp_<k>`, `skid`).

Every generated image shows one forklift pushing cargo on a chain of N LSPs laid end to end ahead of its forks, along its heading. N = 1 is a synthetic VALID image, N >= 2 a violation; `valid_fraction` and `lsp_count_weights` in `config.json` set the mix (2 LSPs primary, 3 less often), so you do not set N. In the real violation clip the forklift pushes 2 LSPs in series under/ahead of the cargo; at this camera distance an LSP is only a thin strip of a few tens of pixels, partly hidden by the forklift and cargo. Every image also gets 0-2 idle SKIDs (`skid_count_weights` in `config.json`, so you do not set their number either), some with cargo on top: scene distractors, never part of the event.

1. `floor_region`: an axis-aligned world rectangle [[x_min, y_min], [x_max, y_max]] of open, visible floor where this camera really sees forklifts drive (use `object_floor_positions_m`). The forklift origin and every LSP centre are kept inside it; it must not include racks, walls or other static proxies.
2. `heading_deg`: [min, max] forklift heading in degrees about +Z (0 = +X, 90 = +Y), matching the real driving directions in the reference images (e.g. [80, 100] along a +Y aisle). One range covers one direction; if both directions are seen, pick the more frequent one for the POC.
3. `forks_to_lsp_m`: distance along the heading from the forklift origin to the centre of the first LSP (front of the forklift asset + half the LSP length + a small gap). The cargo stands on that first LSP, against the forks.
4. `arrangements`: per-extra-LSP step in the forklift's local frame (metres, +Y = heading). `in_series` (primary): [0, size_y + gap, 0], gap 0.00-0.05. Add `offset_series` (e.g. [0.15, size_y + gap, 0]) and `stacked` ([0, 0, thickness]) only if they are plausible here.
5. Idle SKIDs: `skid_height_m` is the height of the `skid` asset (cargo on an idle SKID stands at that height; see `work/logs/05_skid.md`). `skid_clearance_m` is the minimum sideways distance of a SKID from the forklift's lane (the line through the forklift and its LSP chain) and from the other SKIDs: at least half the forklift width + half the SKID diagonal + 0.3 m (about 2.0). SKIDs are kept inside `floor_region`, so it must leave room for them beside the lane.
   `idle_row_lsps`: [min, max] number of idle LSPs in a row for the valid scene without a forklift (`idle_row_fraction` in `config.json` sets how often); the row uses the `in_series` step and must fit inside `floor_region` (`docs/requirements/lsp-reference.png` shows 5).
6. Write `work/scenario.json` exactly in this shape (list every asset that exists in `work/assets/`):
   {"violation_id": "forklift_pushing_multiple_lsps",
    "assets": {"forklift": ["forklift"], "cargo": ["cargo_0"], "lsp": ["lsp_0", "lsp_1"], "skid": ["skid"]},
    "lsp_thickness_m": 0.015,
    "floor_region": [[x_min, y_min], [x_max, y_max]],
    "heading_deg": [min, max],
    "forks_to_lsp_m": 1.9,
    "arrangements": {"in_series": [0, 1.25, 0]},
    "skid_height_m": 0.15, "skid_clearance_m": 2.0, "idle_row_lsps": [2, 5],
    "jitter": {"xy_m": 0.05, "rot_deg": 4.0, "texture_turns": 1}}
   (`texture_turns` 1 allows random 90-degree turns of the LSP texture, only for square sheets; otherwise 0. If 3 LSPs in a row do not fit this floor, say so in your final message.)
7. Check it end to end with 3 samples (one of them valid):
   `python -m synth.randomize 3`
   `blender --background --python blender/render_variants.py -- config.json`
   `python -m synth.composite all`
   If you can view local images, open `work/out/v_000.png` .. `v_002.png` next to the reference images: forklift, cargo and LSPs must stand/lie on the floor at the same scale as the real ones, the LSPs must be in series ahead of the forks, idle SKIDs must lie on the floor clear of the forklift's path, nothing may intersect static objects, and inserted objects must be hidden where racks are in front of them. Adjust `scenario.json` and repeat at most 5 times.

Final message: the final scenario.json and one sentence per value explaining why it fits this camera.
````

- [ ] **Bước 2: Chạy agent**

```bash
codex exec - -m "$MODEL" $SAFETY --skip-git-repo-check -C "$PWD" -o work/logs/04_scenario.md \
  -i data/input/cam01/background.png,$OBJ,work/debug/proxy_overlay.jpg < prompts/04_scenario.md
python -c "import json; s = json.load(open('work/scenario.json')); print(s['floor_region'], s['heading_deg'], s['arrangements'])"
```

- [ ] **Bước 3: Cổng kiểm tra 3 ảnh (người làm).** Xem `work/out/v_000.png`..`v_002.png` ở mức zoom 100 % và 200 %, đặt cạnh các ảnh tham chiếu. Xe nâng phải đứng trên sàn với càng hướng về phía hàng, các tấm LSP phải nằm phẳng và nối tiếp phía trước càng, cùng tỉ lệ với LSP thật, các SKID nằm yên phải nằm trên sàn, tránh xa đường đi của xe nâng, đối tượng phải bị che ở chỗ kệ nằm phía trước, và không có gì đè lên OSD. Nếu không đạt, chạy lại Bước 2 với `-i data/input/cam01/background.png,$OBJ,work/out/v_000.png,work/out/v_001.png` (tối đa 3 lần); nếu chính xe nâng hoặc hàng trông sai, quay lại Task 13 Bước 3.

---

### Task 15: Sinh ~10 ảnh vi phạm + 3 ảnh hợp lệ, kiểm tra và duyệt (nghiệm thu)

**File:**
- Sinh ra: `work/variants.json`, `work/renders/*.png`, `work/out/v_*.png` + `v_*.json`, `work/contact_sheet.jpg`, `work/review.md`

- [ ] **Bước 1: Sinh ảnh**

```bash
rm -rf work/renders work/out
python -m synth.randomize
blender --background --python blender/render_variants.py -- config.json
python -m synth.composite all
```

Kỳ vọng: `wrote 13 variants (10 violation, 3 valid)`, `rendered 13 variants`, `composited 13 images into work/out`.

- [ ] **Bước 2: Kiểm tra tự động**

```bash
python -m pytest -q
python -m synth.verify
python -m synth.contact_sheet
```

Kỳ vọng: `22 passed`, `OK: 13 images match the input format`, và `contact sheet with 13 outputs -> work/contact_sheet.jpg`.

- [ ] **Bước 3: Tính tái lập**

```bash
cp work/variants.json work/variants.first.json && python -m synth.randomize && cmp work/variants.json work/variants.first.json && echo REPRODUCIBLE
```

- [ ] **Bước 4: Người duyệt.** Mở `work/contact_sheet.jpg` (đỏ = box xe nâng, hàng và LSP, xanh lơ = SKID nằm yên và hàng trên đó, vàng = box sự kiện của ảnh vi phạm), rồi mở từng `work/out/v_*.png` ở 100 % và 200 % cạnh nền và các ảnh tham chiếu. Điền vào `work/review.md`:

```markdown
# Review — Forklift Pushing Multiple Lsps POC

Reviewer:            Date:

| Image | Valid / violation | Scale OK | On floor / no float | Occlusion OK | Forklift & cargo look real | Colour, light & shadows match reference images | Edges/noise/compression match | OSD intact | Idle SKIDs on floor, off path | LSP count readable | Pass |
|---|---|---|---|---|---|---|---|---|---|---|---|
| v_000 | | | | | | | | | | | |
| v_001 | | | | | | | | | | | |
| v_002 | | | | | | | | | | | |
| v_003 | | | | | | | | | | | |
| v_004 | | | | | | | | | | | |
| v_005 | | | | | | | | | | | |
| v_006 | | | | | | | | | | | |
| v_007 | | | | | | | | | | | |
| v_008 | | | | | | | | | | | |
| v_009 | | | | | | | | | | | |
| v_010 | | | | | | | | | | | |
| v_011 | | | | | | | | | | | |
| v_012 | | | | | | | | | | | |

Knob changes made (config.json / lights / assets):
Notes:
```

- [ ] **Bước 5: Tinh chỉnh nếu cần, rồi lặp lại Bước 1–4.** Mỗi lần chỉ đổi một núm chỉnh:

| Triệu chứng | Núm chỉnh |
|---|---|
| Đối tượng chèn vào quá sắc / quá mờ | `composite.blur_sigma` |
| Quá sạch / quá nhiễu | `composite.noise_sigma` (null = tự động) |
| Quá sạch / quá vỡ khối so với xung quanh | `composite.h264_crf` (cao hơn = nhiều artifact hơn) |
| Lệch màu hoặc sai độ sáng so với các ảnh tham chiếu | `composite.gain`, `blender.world_strength`, công suất đèn trong `work/cli/proxy.blend-cli.json` (sau đó export lại theo AGENTS.md) |
| Xe nâng, hàng hoặc SKID trông giả / sai hình dạng | Task 13 Bước 3 (thêm một lượt Astra 6 với ảnh preview), hoặc dùng ảnh cắt khác cho `texture_asset.py` |
| Kích thước LSP lệch so với LSP thật | `lsp_size_m` trong `config.json` khi đã biết kích thước thật (chạy lại Task 12 Bước 2–3), nếu không thì kích thước trong `work/lsp_params.json` (dựng lại asset) hoặc `scale_correction` (làm lại từ Task 5) |
| Tỉ lệ ảnh hợp lệ so với ảnh vi phạm | `valid_fraction` trong `config.json` |
| Tỉ lệ ảnh hàng LSP nằm yên không có xe nâng trong số ảnh hợp lệ | `idle_row_fraction` trong `config.json`; độ dài hàng `idle_row_lsps` trong `work/scenario.json` |
| Tỉ lệ ảnh 2 tấm LSP so với 3 tấm LSP | `lsp_count_weights` trong `config.json` (tổng số tấm LSP → trọng số tương đối) |
| Số SKID nằm yên mỗi ảnh, hoặc SKID quá gần xe nâng | `skid_count_weights` trong `config.json`; `skid_clearance_m` trong `work/scenario.json` |
| Sai vị trí, hướng hoặc chồng lấn | `work/scenario.json` (`floor_region`, `heading_deg`, `forks_to_lsp_m`, bước sắp xếp, jitter) |
| Đốm sáng / hạt trong render | `blender.samples` |

**Tiêu chí nghiệm thu POC:**
1. `python -m synth.verify` trả exit code 0 với 13 ảnh: cùng kích thước (960×540), mode và format như nền, vùng OSD không đổi, mọi sidecar có `is_violation` khớp với `lsp_count ≥ 2` và một box đối tượng chèn vào nằm trong ảnh, mọi sidecar vi phạm có box sự kiện, và không sidecar hợp lệ nào có box sự kiện.
2. `python -m pytest -q` đạt.
3. `variants.json` tái lập được từ `scenario.json` + `seed` (Bước 3 in ra `REPRODUCIBLE`).
4. Người duyệt: **≥ 7 / 10** ảnh vi phạm và **≥ 2 / 3** ảnh hợp lệ đạt mọi cột trong `work/review.md`, và trong **≥ 8 / 10** ảnh vi phạm, một người duyệt biết loại vi phạm này nhận ra được có hơn một tấm LSP đang bị đẩy.
5. Contact sheet (nền + 13 output) được đính kèm vào báo cáo POC cùng `work/review.md` và `docs/spike-notes.md`.

Nghiệm thu độ thật của POC là phần của chúng tôi và khép lại POC của ticket này. Đây là điều kiện cần nhưng chưa đủ cho dự án: ảnh có cải thiện model trên dữ liệu thật giữ riêng hay không là phần kiểm tra của Terry, nằm ngoài ticket này (xem "Sau POC").

---

### Task 16 (tuỳ chọn, không cần cho nghiệm thu): đề xuất biến thể cho các kịch bản trong danh mục

**File:**
- Tạo: `prompts/02_propose_variations.md`

- [ ] **Bước 1: Tạo `prompts/02_propose_variations.md`**

````markdown
# Task (optional): propose variations of one "Forklift Pushing Multiple Lsps" scenario

Read AGENTS.md first. The first line of this prompt says `SCENARIO: <id>`, a row of the scenario catalogue in `docs/superpowers/specs/2026-09-28-violation-image-synth-design.md` (section 1.1, "Scenario catalogue: Forklift Pushing Multiple Lsps"): `V1`-`V8` are violation scenarios (the forklift pushes 2 or more LSPs), `N1`-`N5` are look-alike valid scenes (hard negatives). Do not invent scenarios or other violation types: generate VARIATIONS of that scenario only. The POC runs `V1`, `V2`, `N1` and `N2`. Some rules are still to be confirmed with Terry: whether `V3` (stacked) and `V6` (empty extra LSP) are violations, and where the boundary is when a forklift touches an idle LSP (`N3`/`N4`). For those scenarios, if the decision does not follow the `SCENARIO` line, stop and say which rule is missing.

The attached images are the clean background, the reference images of this camera, `docs/requirements/lsp-reference.png` (5 idle LSPs: NOT a violation, the look of `N2`) and `docs/requirements/violation-clip-frames/zoom-t05-t06-t08.jpg` (a real `V1`: the forklift pushes 2 LSPs in series; the coloured overlays come from an existing detector and are not part of the scene).

Propose 4-8 physically plausible VARIATIONS of the scenario for this camera, along these axes: forklift heading and position in the view, LSP gap and offset, cargo type and height (from the cargo seen in the reference images), SKID distractors, lighting. Reject variations that are physically impossible for this forklift and camera view, or where the LSPs that decide valid vs. violation would be fully hidden.

Write `work/variations/<id>.json`: a JSON list of objects with keys `scenario`, `id`, `title`, `is_violation`, `lsp_total`, `arrangement`, `step_local_m` ([dx, dy, dz] offset per extra LSP in the forklift's local frame, +Y = heading), `heading_deg` ([min, max]), `cargo_type`, `skid_distractors`, `lighting`, `visibility` (how much of the LSPs the camera would see), `plausibility_notes`. Use null for keys that do not apply.

Do not modify any other file. Final message: the list as a markdown table.
````

- [ ] **Bước 2: Chạy prompt đề xuất biến thể cho các kịch bản POC V1, V2, N1 và N2, rồi duyệt kết quả (người làm)**

```bash
mkdir -p work/variations
for S in V1 V2 N1 N2; do   # POC scenarios of the catalogue
  (echo "SCENARIO: $S"; cat prompts/02_propose_variations.md) | codex exec - -m "$MODEL" $SAFETY --skip-git-repo-check -C "$PWD" \
    -o work/logs/02_variations_$S.md -i data/input/cam01/background.png,$OBJ,docs/requirements/lsp-reference.png,docs/requirements/violation-clip-frames/zoom-t05-t06-t08.jpg
done
```

Các biến thể được duyệt trở thành mục trong `work/scenario.json` (`arrangements`, `heading_deg`) hoặc asset hàng mới (Task 13 Bước 2); các tổng số tấm LSP khác đưa vào `lsp_count_weights` trong `config.json`, còn độ dài hàng LSP nằm yên đưa vào `idle_row_lsps`. Chạy lại Task 15. Các kịch bản còn lại (V3–V8, N3–N5) không có task code mới trong kế hoạch này: chúng chờ Terry quyết định quy tắc (spec mục 1.1).

---

## Sau POC

Kế hoạch này và ticket này chỉ giới hạn ở việc sinh ảnh. Sau khi nghiệm thu, chúng tôi bàn giao cho Terry một gói gồm:

- các ảnh: `work/out/v_*.png` (ảnh vi phạm và ảnh hợp lệ tổng hợp);
- các file JSON sidecar `v_*.json` nằm cạnh ảnh, làm gợi ý gán nhãn (box đối tượng, `lsp_count`, `is_violation`, box sự kiện cho ảnh vi phạm);
- các script sinh ảnh và cấu hình (`synth/`, `blender/`, `prompts/`, `AGENTS.md`, `config.json`, `work/scenario.json`), để có thể sinh lại các lô mới;
- ghi chú duyệt: `work/review.md`, `work/contact_sheet.jpg`, `docs/spike-notes.md`.

Terry gán nhãn cho ảnh, chuyển sang format nhãn anh ấy cần, train model và test. Phần đánh giá khuyến nghị nằm ở spec mục 10: tập test thật giữ riêng, A/B giữa model chỉ dùng dữ liệu thật và model dùng thật + tổng hợp, chạy nhanh một A/B với ~200 ảnh trước khi chúng tôi sinh 1.000 ảnh, sau đó là bài test RTSP trực tiếp (bước cuối; Gate 3 trong kế hoạch dự án). Nếu Terry báo không có cải thiện, chúng tôi tinh chỉnh độ thật hoặc tỉ lệ kịch bản rồi sinh lại ảnh. Kế hoạch này không có task cho phần việc của Terry.

---

## Độ phủ spec (tự rà soát)

| Mục trong spec | Task |
|---|---|
| Điều kiện tiên quyết về video gốc, tách frame bằng ffmpeg, 1 ảnh nền sạch + ~10 ảnh tham chiếu theo camera | 3 |
| Các điểm chưa xác minh về CLI-Anything / Codex → spike | 1 |
| Mask SAM3, thư viện ảnh cắt đối tượng (ảnh cắt, mask, box 2D, vị trí) | 4 |
| Hiệu chỉnh theo camera: MoGe-2 trên nền sạch, mặt phẳng sàn, tỉ lệ mét đối chiếu với kích thước LSP thật, dùng lại cho mọi ảnh của camera | 2, 5 |
| Texture LSP từ ảnh cắt thật đã nắn phối cảnh; tấm mỏng bo góc tham số hoá | 6, 10, 12 |
| Xe nâng, các loại hàng và SKID do Astra 6 dựng lại qua nhiều lượt; texture từ ảnh cắt thật | 10, 13 |
| Scene proxy của hình học tĩnh (shadow catcher, vật che, đèn) qua cli-anything-blender + cổng kiểm tra overlay | 10, 11 |
| Vi phạm cố định; xe nâng + hàng + chuỗi LSP theo hướng xe; `lsp_count_weights` (2 tấm là chính, 3 tấm ít hơn) / arrangements / jitter | 2, 7, 14 |
| Ảnh hợp lệ tổng hợp: N1 đẩy đúng 1 tấm LSP, N2 hàng LSP nằm yên không có xe nâng (`valid_fraction`, `idle_row_fraction`) | 2, 7, 9, 14, 15 |
| SKID nằm yên làm vật gây nhiễu (0–2 mỗi ảnh, `skid_count_weights`), tránh đường đi của xe nâng, nằm ngoài box sự kiện | 2, 7, 9, 10, 14 |
| Kích thước LSP đã biết ghi đè giá trị đo (`lsp_size_m`) | 2, 5, 6, 12 |
| Chỉ render đối tượng chèn vào + bóng (film transparent, shadow catcher) | 10, 15 |
| Khớp format: kích thước, mode, bảo vệ OSD, nhiễu, độ mờ, H.264 / JPEG | 8, 9, 15 |
| Sidecar: đối tượng, `lsp_count`, `is_violation` (`lsp_count >= 2`), box sự kiện "Forklift Pushing Multiple Lsps" chỉ cho ảnh vi phạm | 7, 9, 10 |
| Duyệt trực quan + nghiệm thu | 15 |
| Tuỳ chọn: đề xuất biến thể cho từng kịch bản trong danh mục (POC: V1, V2, N1, N2) | 16 |
