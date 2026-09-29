# Tổng hợp ảnh vi phạm AAT — Đặc tả thiết kế (POC)

- Ngày: 2026-09-28
- Trạng thái: Bản nháp. Hướng đi đã duyệt (Cách B; vi phạm cố định "Forklift Pushing Multiple Lsps"; đầu vào theo camera: 1 ảnh nền sạch + ~10 ảnh tham chiếu). Cập nhật 2026-09-29: asset SKID là bắt buộc và được đặt làm vật gây nhiễu nằm yên; danh mục kịch bản (mục 1.1); thu hẹp phạm vi: ticket này sinh ảnh, sidecar và script cho Terry, người sẽ gán nhãn, train và test (mục 1 và 10)
- Thư mục dự án: `aat-violation-synth/`
- Yêu cầu và quyết định: [`docs/requirements/2026-09-28-gary-request.md`](../../requirements/2026-09-28-gary-request.md) (xem mục "LSP reference" và "Violation clip reference")
- Kế hoạch đi kèm: [`docs/superpowers/plans/2026-09-28-violation-image-synth-poc.md`](../plans/2026-09-28-violation-image-synth-poc.md)

## 1. Mục tiêu

Khách hàng AAT phát hiện vi phạm an toàn trong kho từ CCTV (RTSP). Các đối tượng chính: **forklift** (xe nâng), **LSP**, **SKID** (pallet), **cargo** (hàng hoá). Ảnh vi phạm thật quá ít để train detector, nên chúng ta **tổng hợp ảnh vi phạm** trông như do chính camera thật quay được. Ticket này sinh các ảnh đó (vi phạm và hợp lệ) cùng file JSON sidecar và các script sinh ảnh, rồi bàn giao cho Terry. Terry gán nhãn, chuyển sang format nhãn anh ấy cần, train model và test.

**Vi phạm trong phạm vi (cố định): "Forklift Pushing Multiple Lsps".** Xe nâng đẩy hàng bằng **nhiều hơn một tấm LSP cùng lúc** là vi phạm: tối đa một tấm LSP là hợp lệ, từ hai tấm trở lên là vi phạm. LSP là tấm nhựa lót (slip sheet) lớn, mỏng, đặt phẳng trên sàn. Hàng quấn màng co đặt lên trên, và xe nâng đẩy tấm để di chuyển hàng. Các tấm LSP nằm yên, không có xe nâng đẩy thì **không** phải vi phạm. Tham chiếu: `docs/requirements/lsp-reference.png` (5 tấm LSP nằm yên, xếp nối tiếp) và `docs/requirements/violation-clip-frames/` (một vi phạm thật, xe nâng đẩy 2 tấm LSP nối tiếp).

**Camera (khái niệm):** một góc nhìn cố định — đúng một ảnh nền sạch cùng ~10 ảnh tham chiếu có các đối tượng chụp từ đúng góc nhìn đó, với hiệu chỉnh riêng (tỉ lệ, mặt phẳng sàn, ống kính). Khái niệm này không gắn với một thiết bị vật lý: bất kỳ tập ảnh nào chung một góc nhìn đều tính là một camera, và một camera vật lý di chuyển được (ví dụ các preset PTZ) cho ra nhiều camera. `camera_id` đặt tên cho góc nhìn này.

**Mục tiêu POC:** lấy **1 camera** với hai đầu vào riêng biệt: đúng **1 ảnh nền sạch** (không có xe nâng, LSP, SKID hay hàng), là nền cho mọi output và là đầu vào cho bước hiệu chỉnh MoGe-2, và **~10 ảnh tham chiếu** của cùng góc nhìn có xe nâng, LSP, SKID và hàng ở nhiều vị trí khác nhau. Ảnh tham chiếu chỉ dùng để tách đối tượng và không bao giờ được dùng làm nền cho output. SAM3 tách các đối tượng từ các ảnh tham chiếu vào một thư viện ảnh cắt (tham chiếu asset, texture, đối chiếu kích thước, tham chiếu ánh sáng), MoGe-2 hiệu chỉnh camera một lần từ nền sạch, và Astra 6 dựng lại xe nâng, các loại hàng và SKID thành asset 3D. Mỗi output chèn lên nền sạch **một xe nâng đẩy hàng trên một chuỗi tấm LSP nối tiếp** nằm phía trước càng nâng, theo hướng của xe: tổng cộng **2 tấm LSP** (trường hợp chính, như trong clip thật) hoặc, ít hơn, **3 tấm**, tỉ lệ theo `lsp_count_weights` trong `config.json`. Như một **đề xuất bổ sung (không có trong yêu cầu của Gary, vốn chỉ yêu cầu ảnh vi phạm)**, cùng pipeline đó cũng tạo **ảnh hợp lệ tổng hợp**: xe nâng đẩy đúng 1 tấm LSP (`valid_fraction` trong `config.json`) và một số ảnh hàng LSP nằm yên không có xe nâng (`idle_row_fraction`). Xe nâng, hàng và LSP chỉ được render trong ảnh tổng hợp, còn ảnh hợp lệ thật không có chúng, nên nếu thiếu các ảnh này model có thể học lối tắt "vật thể render = vi phạm"; có chúng, model học đếm số tấm LSP bị đẩy thay vì nhận ra vật thể render. Đánh giá A/B của Terry (mục 10) sẽ cho thấy có cần chúng hay không; có thể bỏ chúng bằng cách đặt cả hai tỉ lệ về 0. Mọi output còn đặt thêm **0–2 SKID nằm yên** (có hoặc không có hàng bên trên) làm vật gây nhiễu trong cảnh, nằm trong `floor_region` và cách xa đường đi của xe nâng (`skid_count_weights` trong `config.json`); chúng không bao giờ thuộc về sự kiện. Trong danh mục kịch bản (mục 1.1), đây là V1, V2, N1 và N2. Tạo ra **~10 ảnh vi phạm và vài (3) ảnh hợp lệ** thoả mãn:

1. khớp chính xác format frame đầu vào: frame 960×540 từ luồng H.264 5 fps, cùng định dạng file và mode, có artifact nén kiểu H.264, nhiễu và độ mờ CCTV, và giữ nguyên OSD in sẵn trên hình (logo `SENSTAR` góc trên trái, timestamp góc trên phải);
2. trông hợp lý ở tỉ lệ CCTV khi đặt cạnh các ảnh tham chiếu thật. Xe nâng, hàng và SKID chèn vào cần đúng kích thước, hình dạng, màu và sắc độ, phải đứng trên sàn và đổ bóng khớp với ảnh thật. Một tấm LSP chỉ là một dải mỏng rộng vài chục pixel, bị xe nâng và hàng che một phần, nên vi phạm rất khó thấy: mỗi tấm LSP cần đúng tỉ lệ, nằm phẳng trên sàn, nối tiếp phía trước càng nâng, bị che đúng, và khớp màu, vết mòn, độ bóng của tấm thật;
3. mỗi ảnh có một file JSON sidecar chứa: các đối tượng chèn vào (class, pose 3D, box 2D chiếu; SKID nằm yên có class `skid`), `lsp_count`, `is_violation` (`lsp_count >= 2`), và, chỉ với ảnh vi phạm, một **box sự kiện** 2D class `Forklift Pushing Multiple Lsps` (tên model/class đích) bao xe nâng, hàng và mọi tấm LSP, không bao giờ bao các SKID nằm yên. Detector hiện có gán một box cho mỗi sự kiện, nên box này là gợi ý gán nhãn cho Terry. Ảnh hợp lệ không có box sự kiện.

Ngoài phạm vi POC: thêm camera (góc nhìn), sinh hàng loạt (1000+ ảnh), chuỗi video, mô hình méo ống kính (chỉ kiểm tra), các loại vi phạm khác, và các kịch bản trong danh mục ngoài V1, V2, N1 và N2 (mục 1.1). Nằm ngoài ticket này (phần việc của Terry): gán nhãn, chuyển sang format nhãn, train, đánh giá A/B và test RTSP trực tiếp.

**Thước đo thành công:** với ticket này, ảnh trông thật, đúng format đầu vào và đạt bước người duyệt (Task 15 của kế hoạch). Với dự án, độ thật của ảnh là điều kiện cần nhưng chưa đủ: ảnh tổng hợp chỉ có giá trị khi model train có chúng phát hiện vi phạm thật tốt hơn model baseline chỉ train bằng dữ liệu thật, đo trên footage thật được giữ riêng. Terry làm bước kiểm tra đó (mục 10); nếu anh ấy báo không có cải thiện, chúng tôi tinh chỉnh độ thật hoặc tỉ lệ kịch bản rồi sinh lại ảnh.

**Điều kiện tiên quyết (đang chờ: các bên liên quan sẽ cung cấp):** **video gốc không có overlay** từ camera đích, gồm hai tập riêng: đúng 1 ảnh nền sạch (không có xe nâng, LSP, SKID hay hàng) và ~10 ảnh tham chiếu có xe nâng, LSP, SKID và hàng ở các vị trí khác nhau (chỉ để tách đối tượng), quay trong lúc camera không xê dịch. Clip vi phạm được cung cấp là output đã chú thích của một detector có sẵn (mask, box, mũi tên, nhãn `Multiple LSPs ID: 5`). Clip này chỉ để tham khảo và không bao giờ được dùng làm nền hay ảnh train. Trong lúc chờ video gốc, một frame từ clip đó chỉ được dùng để phát triển và smoke-test phần code tất định.

**Điều kiện tiên quyết (đang chờ: các bên liên quan sẽ cung cấp):** **máy trạm Ubuntu có GPU** (mục 2). Hiện máy này chưa có.

### 1.1 Danh mục kịch bản: Forklift Pushing Multiple Lsps

Danh mục chỉ xoay quanh đúng một vi phạm này. Astra 6 sinh biến thể cho từng kịch bản đã duyệt theo các trục: hướng và vị trí xe nâng trong khung hình, khoảng hở và độ lệch giữa các tấm LSP, loại hàng và chiều cao hàng, SKID gây nhiễu, ánh sáng (kế hoạch `prompts/02_propose_variations.md`). POC gồm V1 và V2 (vi phạm) cùng N1 và N2 (hợp lệ).

Kịch bản vi phạm (`is_violation = true`, số tấm được đẩy `lsp_count >= 2`):

| # | Kịch bản | Ảnh thể hiện gì | Ưu tiên |
|---|---|---|---|
| V1 | Nối tiếp, 2 tấm LSP | Xe nâng đẩy hàng trên 2 tấm LSP nối đầu nhau (như trong clip thật) | POC |
| V2 | Nối tiếp, 3 tấm LSP | Tương tự, với 3 tấm LSP | POC |
| V3 | LSP xếp chồng | Từ 2 tấm LSP trở lên chồng lên nhau dưới hàng | Tiếp theo |
| V4 | So le hoặc chồng mép | Tấm LSP thêm bị lệch sang bên hoặc chồng một phần lên tấm đầu | Tiếp theo |
| V5 | Cạnh nhau | 2 tấm LSP nằm cạnh nhau, được đẩy cùng lúc như một khối hàng rộng | Tiếp theo |
| V6 | Tấm LSP thêm để trống | Hàng chỉ nằm trên tấm LSP đầu; tấm thêm để trống | Tiếp theo |
| V7 | Bị che nhiều | Tấm LSP thêm bị hàng, xe nâng, kệ hoặc SKID che gần hết | Tiếp theo |
| V8 | Vừa đẩy vừa rẽ | Chuỗi LSP bị lệch góc khi xe nâng rẽ | Sau |

Cảnh hợp lệ dễ nhầm (hard negative, `is_violation = false`). Ảnh hợp lệ tổng hợp là đề xuất bổ sung của chúng tôi, không thuộc yêu cầu của Gary (vốn chỉ yêu cầu ảnh vi phạm): chúng ngăn model học "vật thể render = vi phạm" và buộc model đếm số tấm LSP bị đẩy (xem mục tiêu POC):

| # | Cảnh | Ảnh thể hiện gì | Ưu tiên |
|---|---|---|---|
| N1 | Một tấm LSP | Xe nâng đẩy hàng trên đúng 1 tấm LSP | POC |
| N2 | Hàng LSP nằm yên | Các tấm LSP nằm yên thành hàng, không có xe nâng (như `lsp-reference.png`) | POC |
| N3 | Xe nâng gần LSP nằm yên | Xe nâng chạy ngang qua hoặc đỗ cạnh các tấm LSP nằm yên, không đẩy chúng | Tiếp theo |
| N4 | Một tấm LSP kèm vật gây nhiễu | Xe nâng đẩy 1 tấm LSP, gần đó có SKID hoặc tấm LSP nằm yên | Tiếp theo |
| N5 | SKID trên càng | Xe nâng chở SKID trên càng, không có LSP | Sau |

Các quy tắc cần xác nhận với Terry trước khi sinh ảnh: V3 (xếp chồng) có phải vi phạm không? V6 (tấm LSP thêm để trống) có phải vi phạm không? Ranh giới nằm ở đâu khi xe nâng chạm vào một tấm LSP nằm yên (N3/N4)?

Các loại vi phạm khác (chắn lối đi, xếp hàng quá cao, người đi bộ, …) nằm ngoài phạm vi.

## 2. Máy và công cụ

Máy trạm (đang chờ: các bên liên quan sẽ cung cấp; hiện chưa có). Yêu cầu: Ubuntu, GPU NVIDIA, CUDA và SAM3.

Một lần kiểm tra tính năng Blender 5.2.2 ban đầu đã chạy trên một máy Windows và đạt; Ubuntu chưa được kiểm tra.

| Công cụ | Vai trò | Ghi chú |
|---|---|---|
| Blender (>= 4.2, theo yêu cầu của harness Blender của CLI-Anything) | Scene proxy, asset xe nâng / hàng / SKID / LSP, render Cycles với shadow catcher | Chạy headless: `blender --background --python script.py -- args` |
| Codex CLI, model "GPT Astra 6" | Agent: dựng scene proxy, dựng lại asset 3D từ ảnh cắt thật (xe nâng, các loại hàng, SKID) qua nhiều lượt, thông số asset LSP, template kịch bản, đề xuất biến thể tuỳ chọn cho từng kịch bản trong danh mục (mục 1.1) | Chạy không tương tác: `codex exec`. Đăng nhập Codex với gói có sẵn GPT Astra 6; mức dùng bị giới hạn bởi hạn mức của gói (không dùng API key, không có ngân sách nào để đặt) |
| Harness Blender của CLI-Anything (`cli-anything-blender`) | CLI để agent soạn scene Blender dưới dạng project JSON | Khả năng và giới hạn đã xác minh ở mục 6 |
| MoGe-2 (`Ruicheng/moge-2-vitl-normal`) | Point map theo mét và intrinsics chuẩn hoá từ nền sạch (một lần cho mỗi camera) | Hiệu chỉnh camera, mặt phẳng sàn, tỉ lệ mét |
| SAM3 | Mask theo prompt văn bản: sàn và kệ trên nền sạch (fit mặt phẳng, proxy); xe nâng, LSP, SKID và hàng trên các ảnh tham chiếu (thư viện ảnh cắt, đối chiếu kích thước và vị trí, texture) | Bắt buộc có trên máy trạm (GPU CUDA) |
| ffmpeg | Tách frame từ video gốc; round-trip H.264 cho vùng pixel chèn vào | Bắt buộc có trên máy trạm |
| Python 3.12 + numpy + Pillow + pytest | Phần keo tất định: thư viện ảnh cắt đối tượng, hiệu chỉnh, nắn texture LSP, random, ghép ảnh và khớp format, kiểm tra, contact sheet | Không dùng framework |

## 3. Cách B (đã chọn): giữ nền sạch thật, chỉ render phần chèn vào

Frame nền sạch của camera là **nền của mọi output**. Blender chỉ render **các đối tượng chèn vào (xe nâng, hàng, LSP, SKID nằm yên) và bóng của chúng** trên film trong suốt. Scene proxy chỉ bao phần hình học tĩnh của nền (sàn, kệ, tường, cột) và không bao giờ hiện ra. Nó chỉ để (a) hứng bóng của vật chèn vào và (b) che vật chèn vào khi chúng nằm sau vật tĩnh thật. Các vật chèn vào tự che nhau ngay trong ảnh render (xe nâng và hàng che một phần các tấm LSP). Ảnh render được ghép lên nền sạch rồi khớp format, lấy các ảnh tham chiếu làm chuẩn tham chiếu cho ánh sáng, màu, bóng, nhiễu, độ mờ và nén.

### Vì sao không render lại toàn bộ (Cách A)

Render lại toàn bộ sẽ đưa cả bức ảnh qua khoảng lệch domain: texture CG, ánh sáng CG và nhiễu CG ở khắp nơi. Detector sẽ học "ảnh render = vi phạm" và thất bại trên frame RTSP thật. Với Cách B:

- gần như mọi pixel đến từ camera thật (texture, ánh sáng, nhiễu, nén, OSD thật);
- chỉ các vật chèn vào là tổng hợp, và công sức dồn vào việc khớp chúng: asset dựng lại từ ảnh cắt thật với texture từ ảnh cắt thật, texture nắn từ LSP thật, đèn suy ra từ bóng thật, độ mờ, nhiễu, gain, nén lại H.264/JPEG;
- hình học proxy có thể chỉ là các hộp thô, vì nó chỉ cần đặt đúng chỗ;
- ảnh hợp lệ tổng hợp (đề xuất bổ sung của chúng tôi, không có trong yêu cầu của Gary) đi qua cùng pipeline, nên riêng việc render không phân biệt được hai lớp và model phải đếm số tấm LSP bị đẩy thay vì nhận ra vật thể render.

Cái giá là cần camera chính xác (intrinsics, pose, tỉ lệ mét), đèn cho bóng khớp với các frame thật, và asset xe nâng, hàng, SKID trông như thật ở tỉ lệ CCTV (mục 8, rủi ro lớn nhất). MoGe-2 cung cấp camera, một lần cho mỗi camera, có đối chiếu với kích thước vật thật. Agent thiết lập đèn từ bóng thật trong các ảnh tham chiếu, rồi agent và người cùng duyệt.

## 4. Kiến trúc

```
  data/raw/cam01.mp4 (RAW, no overlays) --ffmpeg fps=5--> data/frames/*.png --human picks, per camera-->
      data/input/cam01/background.png      (clean: no forklift / LSP / SKID / cargo)
      data/input/cam01/objects/obj_*.png   (~10 reference images: forklift, LSP, SKID, cargo at different positions; object extraction only, never a background)
                           |
                           v
  [SAM3] synth/segment.py
    background -> work/masks/background/{floor,rack}.png
    objects    -> work/masks/obj_*/*.png, work/refs/<frame>_<class>_<rank>.png + index.json
                  (object crop library: crop, mask, 2D box, floor-contact pixel)
                           |
          +----------------+-------------------------------+---------------------------------+
          v                                                v                                 |
  [MoGe-2] synth/calibrate.py   (once per camera)    synth/lsp_texture.py                    |
    clean background -> K, floor plane, metric scale (homography rectify LSP instances)      |
    cross-check: real LSP sizes, object positions    -> work/textures/lsp_<k>.png            |
    -> work/camera.json, points_world.npy,                     |                             |
       scene_facts.json, calibration.json                      |                             |
          |                                                    |                             |
          v                                                    |                             |
  [Codex exec + GPT Astra 6 + cli-anything-blender]  (agent, once per camera)                |
    prompts/01_proxy_scene.md -> work/proxy.blend (static proxies: floor, racks + lights)    |
        ^ loop: render_variants.py --debug-proxy + composite overlay (agent + human)         |
    prompts/03_lsp_params.md  -> work/lsp_params.json (size, thickness, corners, gloss)      |
    prompts/05_build_asset.md -> work/assets_raw/{forklift,cargo_<type>,skid}.blend <- crops +
        ^ loop: 3 passes + previews, re-prompted until the details show (agent + human)
                           |                                   |
                           v                                   v
    blender/build_lsp.py      -> work/assets/lsp_<k>.blend  (rounded thin sheet + real LSP texture)
    blender/texture_asset.py  -> work/assets/{forklift,cargo_<type>,skid}.blend  (real crop on tex_* materials)
    prompts/04_scenario.md    -> work/scenario.json  (floor region, headings, forks-to-LSP, in-series step, jitter)
    [optional] prompts/02_propose_variations.md -> work/variations/<scenario>.json  (catalogue scenario, section 1.1)
                           |
                           v   (deterministic, no LLM per image)
    synth/randomize.py         -> work/variants.json      (seeded forklift + cargo + N LSPs; N=1 valid, N>=2 violation; idle LSP rows; 0-2 idle SKIDs)
    blender/render_variants.py -> work/renders/<id>.png   (RGBA: forklift, cargo, LSPs, SKIDs + caught shadows)
                               -> work/out/<id>.json      (sidecar; event box "Forklift Pushing Multiple Lsps" if violation)
    synth/composite.py         -> work/out/<id>.png       (on the clean background: OSD protected, alpha-over, blur, noise, local H.264)
    synth/verify.py            -> format / OSD / sidecar checks (exit code)
    synth/contact_sheet.py     -> work/contact_sheet.jpg  (background + outputs, object and event boxes)
                           |
                           v
                  HUMAN visual review (acceptance)
```

## 5. Luồng dữ liệu và quy ước

- **Đầu vào theo camera** (`config.json`: `camera_id`, `background_image`, `object_frames_dir`): hai đầu vào riêng biệt của cùng một góc nhìn cố định. `background_image` là đúng 1 ảnh nền sạch (không có xe nâng, LSP, SKID hay hàng): nền cho mọi output và đầu vào cho bước hiệu chỉnh MoGe-2. `object_frames_dir` là thư mục ảnh tham chiếu: ~10 ảnh tham chiếu có xe nâng, LSP, SKID và hàng ở nhiều vị trí khác nhau, chỉ dùng để tách đối tượng (ảnh cắt, mask và kích thước từ SAM3 làm tham chiếu asset, texture, đối chiếu kích thước và tham chiếu ánh sáng), không bao giờ dùng làm nền cho output. Mỗi camera một `work_dir`; kết quả hiệu chỉnh và asset trong đó được dùng lại cho mọi ảnh của camera.
- **Thư viện đối tượng** (`synth/segment.py`): SAM3 tạo mask sàn và kệ trên nền sạch, và mask xe nâng, LSP, SKID, hàng trên từng ảnh tham chiếu. Với 3 instance điểm cao nhất của mỗi lớp trong mỗi ảnh tham chiếu, nó lưu một ảnh cắt sát (pixel ngoài mask được tô bằng màu trung bình của đối tượng, để ảnh cắt dùng được làm texture), và `work/refs/index.json` liệt kê lớp, frame, điểm, box 2D, diện tích và pixel tâm đáy (điểm chạm sàn).
- **Hệ toạ độ thế giới:** đơn vị mét, trục Z hướng lên, sàn là `z = 0`, gốc toạ độ là điểm trên sàn ngay dưới camera, +Y là hướng nhìn của camera chiếu xuống sàn, +X là bên phải.
- **Camera** (`work/camera.json`, từ MoGe-2 trên nền sạch, một lần cho mỗi camera): `width`, `height`, `lens_mm`, `sensor_width_mm` (36, sensor fit ngang), `shift_x`, `shift_y`, `matrix_world` (ma trận 4×4 theo quy ước camera Blender), `hfov_deg`, `K_norm`. Intrinsics của MoGe đã chuẩn hoá (`fx/W`, `fy/H`, `cx/W`, `cy/H`):
  - `lens_mm = fx_n · 36`
  - `shift_x = 0.5 − cx_n`
  - `shift_y = (cy_n − 0.5) · H / W` (đúng khi `W ≥ H`)
- **Pose:** fit mặt phẳng bằng RANSAC trên các điểm MoGe nằm trong mask sàn SAM3 của nền sạch. Pháp tuyến hướng về phía camera trở thành +Z thế giới, và chiều cao camera là khoảng cách tới mặt phẳng.
- **Tỉ lệ mét:** MoGe-2 cho ra đơn vị mét, và bước hiệu chỉnh đối chiếu nó với các đối tượng SAM3. Mọi ảnh của một camera có cùng một góc nhìn, nên các pixel mask LSP của một ảnh tham chiếu khớp với các điểm sàn của nền. Dấu chân của mọi tấm LSP thấy trọn vẹn (hình chữ nhật diện tích nhỏ nhất; tấm bị che một phần bị loại nhờ tỉ lệ cạnh) được so với kích thước LSP đã biết (`lsp_size_m`, giá trị danh nghĩa 1.1 × 1.1 m cho tới khi có số thật). `work/calibration.json` ghi lại các kích thước đo được, một `suggested_scale_correction` và các vị trí trên sàn nơi đã thấy từng lớp đối tượng. `scale_correction` trong `config.json` là núm hiệu chỉnh.
- **Pixel → thế giới** cho agent: `work/points_world.npy` có shape `(H//4, W//4, 3)`. Toạ độ XYZ thế giới của pixel `(u, v)` là `P[v//4, u//4]`.
- **Asset LSP:** dựng tất định bằng `blender/build_lsp.py` từ `work/lsp_params.json` (`size_x_m`, `size_y_m`, `thickness_m` ≈ 0.01–0.02, `corner_radius_m`, `roughness`), dưới dạng tấm mỏng bo góc với UV phẳng. Base colour là **texture LSP thật**: mask instance LSP của SAM3 được nắn phối cảnh bằng homography 4 góc thành ảnh 512×512. Các instance có tỉ lệ lấp đầy mask thấp, tức bị hàng che một phần, được bỏ qua. Mỗi texture một file `.blend` (`lsp_<k>`), và bộ random chọn texture cho từng vị trí đặt. Điều đó, cộng với dịch và xoay nhỏ và xoay 90° tuỳ chọn, tạo sự đa dạng về bề ngoài ("vết mòn"). Gốc toạ độ ở tâm đáy tấm trên sàn, +Y là hướng đẩy. Kích thước LSP chính xác sẽ do người dùng cung cấp: khi đó `lsp_size_m` trong `config.json` (`[x, y]` theo mét, `null` = dùng giá trị đo được) ghi đè `size_x_m`/`size_y_m` đo được trong `work/lsp_params.json`, và giá trị đo được giữ lại trong `measured_size_m`.
- **Asset xe nâng, hàng và SKID:** Astra 6 dựng lại từng đối tượng từ ảnh cắt của nó bằng harness (primitive, modifier, material Principled) qua 3 lượt có render preview, và được prompt lại kèm các preview cho tới khi đủ chi tiết: `forklift`, mỗi loại hàng một `cargo_<type>`, và `skid` (bắt buộc: SKID nằm yên xuất hiện trong mọi ảnh POC). Output: `work/assets_raw/<name>.blend`. Sau đó `blender/texture_asset.py` chiếu kiểu box ảnh cắt thật lớn nhất lên các material agent đặt tên `tex_*` và lưu `work/assets/<name>.blend`. Cùng quy ước với LSP: đơn vị mét, gốc toạ độ ở tâm dấu chân trên sàn, +Y = phía trước / hướng đẩy.
- **Proxy** (`work/proxy.blend`), do agent soạn bằng CLI-Anything. Mọi mesh trở thành **shadow catcher** khi render: vô hình trong pass Combined nhưng vẫn chặn tia từ camera, nên vừa hứng bóng vừa che khuất. Proxy chỉ chứa phần hình học tĩnh của nền sạch: `proxy_floor` (bắt buộc) và các hộp `proxy_<what>` cho kệ, tường và cột. Không có proxy xe nâng, hàng, LSP hay SKID. Đèn trong proxy là đèn của scene. Camera trong các file soạn thảo bị loại bỏ.
- **Template kịch bản** (`work/scenario.json`): `assets` (danh sách `forklift`, `cargo`, `lsp`, `skid`), `lsp_thickness_m`, `floor_region` (hình chữ nhật trong hệ toạ độ thế giới, là vùng sàn trống nơi thực sự thấy xe nâng chạy), `heading_deg` ([min, max]), `forks_to_lsp_m` (từ gốc toạ độ xe nâng tới tâm tấm LSP đầu tiên), `arrangements`, `jitter`, `skid_height_m`, `skid_clearance_m`, `idle_row_lsps`. Chuỗi LSP nằm phía trước càng nâng theo hướng xe: tấm thứ k nằm ở `forks_to_lsp_m` + k · bước dịch trong hệ toạ độ cục bộ của xe nâng, với `in_series` = `[0, size_y + gap, 0]` là chính. Hàng đặt trên tấm LSP đầu tiên. Vị trí đặt nào có gốc xe nâng hoặc tâm tấm LSP ra ngoài `floor_region` thì được bốc lại. Số tấm LSP không nằm trong template: `valid_fraction` trong `config.json` đặt tỉ lệ ảnh hợp lệ (đúng 1 tấm LSP), còn `lsp_count_weights` ánh xạ tổng số tấm LSP của ảnh vi phạm sang trọng số tương đối (mặc định `{"2": 3, "3": 1}`: 2 tấm là chính, 3 tấm ít hơn). `idle_row_fraction` đặt tỉ lệ ảnh hợp lệ chỉ có một hàng tấm LSP nằm yên (`idle_row_lsps` tấm nối tiếp, không có xe nâng, không có hàng; kịch bản N2). SKID nằm yên: `skid_count_weights` trong `config.json` ánh xạ 0, 1 hoặc 2 SKID mỗi ảnh sang trọng số tương đối (mặc định `{"0": 1, "1": 2, "2": 1}`). Mỗi SKID nằm trong `floor_region`, cách làn của xe nâng (đường thẳng đi qua xe nâng và chuỗi LSP của nó) và các SKID khác ít nhất `skid_clearance_m` theo phương ngang; khoảng một nửa số SKID có hàng bên trên (ở độ cao `skid_height_m`). Chúng là vật gây nhiễu trong cả ảnh hợp lệ lẫn ảnh vi phạm, nên không phải dấu hiệu phân lớp, và không bao giờ nằm trong box sự kiện.
- **Variants** (`work/variants.json`): các vị trí đặt có seed (xe nâng + hàng + N tấm LSP hoặc một hàng LSP nằm yên, cùng 0–2 SKID nằm yên, `heading_deg`, `lsp_count`, `is_violation`, `skid_count`). Đây là thứ giúp việc sinh hàng loạt về sau không cần LLM.
- **Render:** Cycles, film transparent, view transform `Standard`, PNG RGBA 8-bit đúng độ phân giải đầu vào. Bóng xuất hiện dưới dạng màu đen với alpha một phần (hành vi shadow catcher của Blender ≥ 3.0).
- **Ghép ảnh:** alpha của render bị đặt về 0 trong `osd_boxes`. Sau đó `out = blur(rgb·a) + bg·(1 − blur(a)) + noise·blur(a)`, với σ nhiễu ước lượng từ nền sạch.
  - Đầu vào là frame video (PNG): các pixel gần vật chèn vào được thay bằng bản đã round-trip qua libx264 (`h264_crf`) để có artifact khối và chroma. Nền giữ nguyên từng bit.
  - Đầu vào JPEG: output được lưu với bảng lượng tử, subsampling, EXIF và ICC của ảnh đầu vào.
- **Sidecar metadata** (`work/out/<id>.json`): `image`, `camera_id`, `source_image` (nền sạch), `violation_id`, `arrangement`, `heading_deg`, `lsp_count` (số tấm LSP xe nâng đẩy, 0 với hàng LSP nằm yên), `is_violation` (`lsp_count >= 2`), `skid_count`, `objects[]` (mọi đối tượng chèn vào, kể cả SKID nằm yên với class `skid` tên `skid_<k>`, hàng trên chúng tên `skid_<k>_cargo`: `name`, `class`, `asset`, `location`, `rot_z_deg`, `size_m`, `bbox_2d`), `event` (ảnh vi phạm: `class: "Forklift Pushing Multiple Lsps"`, `bbox_2d` = hình chiếu của xe nâng, hàng và mọi tấm LSP, không bao giờ gồm SKID nằm yên hay hàng trên chúng; ảnh hợp lệ: `null`). Các box bỏ qua việc bị che. File JSON sidecar là output của POC và là gợi ý gán nhãn; việc chuyển sang format nhãn Terry cần là phần việc của Terry, nằm ngoài ticket này.

## 6. CLI-Anything: các điểm đã xác minh

Nguồn, đọc từ nhánh `main` ngày 2026-09-28: README <https://github.com/HKUDS/CLI-Anything>; mã nguồn harness <https://github.com/HKUDS/CLI-Anything/tree/main/blender/agent-harness> (`cli_anything/blender/blender_cli.py`, `utils/bpy_gen.py`, `utils/blender_backend.py`, `core/*.py`, `setup.py`, `skills/SKILL.md`); Codex skill <https://github.com/HKUDS/CLI-Anything/tree/main/codex-skill>.

Đã xác minh từ mã nguồn:

1. Package `cli-anything-blender` (Python ≥ 3.10; phụ thuộc `click`, `prompt-toolkit`), entry point `cli-anything-blender`. Cài từ mã nguồn: `cd CLI-Anything/blender/agent-harness && pip install -e .`. README cũng hướng dẫn `pip install cli-anything-hub`, `cli-hub install blender`.
2. Harness **lưu trạng thái trong một file project JSON** (`*.blend-cli.json`), không phải một phiên Blender đang chạy. Các option toàn cục là `--json`, `--project PATH` và `--dry-run`. Lệnh chạy một lần tự lưu. Chạy không kèm subcommand sẽ mở REPL.
3. Nhóm lệnh:
   - `scene`: `new`, `open`, `save`, `info`, `profiles`, `json`
   - `object`: `add`, `remove`, `duplicate`, `transform`, `set`, `list`, `get`
   - `material`: `create`, `assign`, `set`, `list`, `get`
   - `modifier`: `list-available`, `info`, `add`, `remove`, `set`, `list`
   - `camera`: `add`, `set`, `set-active`, `list`
   - `light`: `add`, `set`, `list`
   - `animation`
   - `render`: `settings`, `info`, `presets`, `execute`, `script`
   - `preview`: `recipes`, `capture`, `latest`, `live ...`
   - `session`: `status`, `undo`, `redo`, `history`
4. `object add` chỉ nhận primitive (`cube, sphere, cylinder, cone, plane, torus, monkey, empty`) với `--location/-l x,y,z`, `--rotation/-r` (độ), `--scale/-s` và `--param/-p key=value`. `cube` và `plane` mặc định `size=2.0`.
5. Tính năng được hỗ trợ:
   - Modifier: `subdivision_surface, mirror, array, bevel, solidify, decimate, boolean, smooth`.
   - Material: Principled với màu, metallic, roughness, specular.
   - Đèn: `point, sun, spot, area`.
   - Thuộc tính camera: `location, rotation, focal_length, sensor_width, clip_start, clip_end, type, name, dof_*`. **Không có lens shift và sensor fit**.
6. Render hoạt động bằng cách sinh một script bpy hoàn chỉnh từ JSON. Script xoá mọi object ở đầu và kết thúc bằng `bpy.ops.render.render(...)`, và được thiết kế để chạy bằng `blender --background --python script.py`.
   - `render script OUTPUT` in script ra stdout.
   - `render execute OUTPUT` ghi `_render_script.py` cạnh `OUTPUT`. Trong mã nguồn hiện tại lệnh này **không** tự khởi chạy Blender.
   - `render settings --transparent` bật film transparent.
7. **Không hỗ trợ:**
   - chạy Python tuỳ ý
   - import mesh (OBJ/FBX/glTF)
   - sửa vertex
   - texture ảnh
   - cờ shadow catcher / holdout
   - lens shift của camera
   - thiết lập compositor
   - option `--collection` của `object add` (không được đưa vào script sinh ra)
8. `codex-skill/` (cài bằng `bash CLI-Anything/codex-skill/scripts/install.sh` vào `${CODEX_HOME:-~/.codex}/skills/cli-anything`) dùng để **xây dựng hoặc tinh chỉnh harness**, không phải để dùng harness. Harness Blender có skill hướng dẫn sử dụng riêng tại `cli_anything/blender/skills/SKILL.md`.

Hệ quả cho thiết kế:

- Agent dùng `cli-anything-blender` để **soạn** scene proxy và hình học của các asset 3D (xe nâng, các loại hàng, SKID), đặt tên `tex_*` cho các material cần mang vẻ ngoài thật.
- Script bpy export ra được chạy bằng `blender/cli_to_blend.py` của chúng ta, với lời gọi render đã vô hiệu hoá, để lưu thành `.blend`.
- Những việc harness không làm được nằm trong các script bpy tất định nhỏ do chúng ta sở hữu: LSP có texture ảnh (`blender/build_lsp.py`), texture từ ảnh cắt thật trên asset của agent (`blender/texture_asset.py`), camera đã hiệu chỉnh có lens shift, shadow catcher, đặt vị trí, thiết lập màu, instancing và metadata (`blender/render_variants.py`).
- Nếu harness quá hạn chế, chế độ "Refine" của CLI-Anything có thể thêm lệnh (shadow catcher, lens shift, texture ảnh). POC không cần việc này.

## 7. Codex CLI: các điểm đã xác minh

Nguồn, đọc ngày 2026-09-28:

- Trang tham chiếu <https://developers.openai.com/codex/cli/reference>, chuyển hướng tới <https://learn.chatgpt.com/docs/developer-commands?surface=cli>.
- Mã nguồn <https://github.com/openai/codex/blob/main/codex-rs/exec/src/cli.rs> và `codex-rs/utils/cli/src/shared_options.rs`.

Các điểm đã xác minh:

- `codex exec` (alias `codex e`) chạy không tương tác. Các option:
  - `--model/-m`
  - `--sandbox/-s {read-only|workspace-write|danger-full-access}`
  - `--cd/-C DIR`, `--add-dir DIR`
  - `--image/-i FILE[,FILE...]` (`num_args = 1..`, phân tách bằng dấu phẩy)
  - `--json`, `--output-last-message/-o FILE`, `--output-schema FILE`
  - `--skip-git-repo-check`, `--ephemeral`, `-c key=value`
  - `--dangerously-bypass-approvals-and-sandbox`
- `PROMPT` có thể là `-` (hoặc bỏ trống) để đọc từ stdin. Nếu stdin được pipe vào và có cả prompt, stdin được nối thêm thành một khối `<stdin>`.
- `codex exec resume [SESSION_ID] [--last]` nhận `--image`.
- Vì `--image` nhận 1 giá trị trở lên, kế hoạch đặt prompt `-` **trước** các option và `--image` **cuối cùng**.
- Skill được nạp từ `~/.codex/skills`. `AGENTS.md` ở thư mục gốc dự án chứa hướng dẫn cho dự án, và kế hoạch đặt các quy ước dùng chung vào đó một lần.

## 8. Rủi ro và cách giảm thiểu

| Rủi ro | Hậu quả | Giảm thiểu trong POC |
|---|---|---|
| Xe nâng, hàng và SKID render chưa đủ thật (rủi ro lớn nhất: giờ đây chúng là vùng tổng hợp lớn nhất) | Lệch domain; detector học "xe nâng render = vi phạm" và thất bại trên RTSP | Asset dựng lại từ ảnh cắt thật của cùng camera, với texture từ ảnh cắt thật. Nhiều lượt Astra 6 có preview và cổng kiểm tra chi tiết do người làm (Task 13 của kế hoạch). Ảnh hợp lệ tổng hợp từ cùng pipeline (đề xuất bổ sung, không có trong yêu cầu của Gary), nên việc render không phải dấu hiệu phân lớp. Người duyệt so với các ảnh tham chiếu. A/B của Terry (mục 10) là thước đo cuối cùng |
| Không có video gốc (chỉ có clip đã chú thích) | Không tạo được output hợp lệ | Điều kiện tiên quyết bắt buộc, đang chờ: các bên liên quan sẽ cung cấp. Code tất định được phát triển trên một frame của clip, còn output cuối cần ảnh nền sạch gốc và ảnh tham chiếu gốc |
| Không có frame thật sự sạch (sàn không bao giờ trống) | Không có nền, hoặc mọi output đều còn sót vật | Chọn frame trống nhất và để vật còn sót nằm ngoài `floor_region`. Nền tạo bằng median theo thời gian là việc tương lai |
| Camera vật lý xê dịch giữa các ảnh tham chiếu và nền (PTZ, rung), nên chúng không còn chung một góc nhìn | Đối chiếu kích thước và vị trí trên sàn bị sai | Chọn mọi frame trong cùng một khoảng thời gian cố định; frame từ một preset PTZ khác tạo thành một camera riêng. Đối chiếu các vị trí trong `work/calibration.json` với overlay |
| Mục tiêu khó thấy: LSP là dải mỏng vài chục px, bị che một phần | Sai lệch pose nhỏ khiến tấm bị nổi, biến mất hoặc xuyên qua hàng | Xe nâng, hàng và các tấm LSP được render cùng nhau, nên việc chúng che nhau là chính xác. Bước nối tiếp lấy từ kích thước LSP đo được, hướng xe và khoảng cách từ càng tới LSP lấy từ kịch bản. Người duyệt xem ảnh cắt ở độ phân giải đầy đủ |
| SKID nằm yên bị đặt trên đường đi của xe nâng hoặc lên chuỗi LSP | Cảnh không hợp lý; SKID nằm trong sự kiện có thể dạy model dấu hiệu sai | Bộ random giữ mọi SKID cách làn của xe nâng và các SKID khác ít nhất `skid_clearance_m`, bên trong `floor_region`. SKID không bao giờ nằm trong box sự kiện và xuất hiện như nhau ở ảnh hợp lệ lẫn ảnh vi phạm. Người duyệt các ảnh kiểm tra (Task 14 của kế hoạch) |
| Proxy không chính xác (phạm vi sàn, kệ) | Che khuất sai, bóng bị cắt | Vòng lặp overlay debug, `points_world.npy` để tra toạ độ theo mét, chỉ đặt vật trong `floor_region` (vùng sàn trống nơi thực sự thấy xe nâng chạy) |
| MoGe sai tỉ lệ hoặc sai sàn; camera ở xa các đối tượng | Đối tượng sai kích thước | Tỉ lệ inlier RANSAC trong `scene_facts.json`, tự động đối chiếu với kích thước đo được của các tấm LSP thật trong ảnh tham chiếu (`work/calibration.json`, `suggested_scale_correction`), núm `scale_correction`, `lsp_size_m` khi người dùng cung cấp kích thước thật |
| Ánh sáng không khớp | Vật chèn vào quá sáng hoặc quá tối, bóng sai | Màu world = màu tuyến tính trung bình của nền sạch. Agent suy ra đèn từ bóng thật trong các ảnh tham chiếu. Núm `gain`. Texture lấy từ ảnh cắt thật nên albedo và vết mòn gần đúng ngay từ đầu |
| Nguồn texture khác (camera khác, ánh sáng khác) | Ám màu | Ảnh cắt lấy từ các ảnh tham chiếu của cùng camera; nguồn khác (`lsp-reference.png`) chỉ là phương án dự phòng. Núm `gain`. Có nhiều texture nên không texture xấu nào chiếm ưu thế |
| Méo ống kính (CCTV góc rộng, thấy rõ trong clip) | Proxy thẳng không khớp cạnh thật bị cong gần mép ảnh | `floor_region` giữ các vật chèn vào tránh xa mép ảnh. Overlay cho thấy độ méo. Bước khử méo/tạo lại méo là việc tương lai |
| Nén không khớp | Vùng chèn quá sạch hoặc quá vỡ khối | Round-trip libx264 cục bộ với `h264_crf` điều chỉnh được, so sánh bằng mắt ở crop 100 % |
| Mọi ảnh dùng chung một nền | Detector ghi nhớ nền | Ảnh hợp lệ và ảnh vi phạm cũng dùng chung nền đó, nên nền không phải dấu hiệu phân lớp. Khi sinh hàng loạt sẽ thêm nền và camera (góc nhìn) |
| OSD bị ghi đè | Artifact lộ liễu, detector học lối tắt | Mask alpha `osd_boxes` trong bước ghép, được `synth/verify.py` kiểm tra |
| Sandbox của Codex chặn Blender | Agent không render được | Spike (Task 1 của kế hoạch). Phương án dự phòng là `--dangerously-bypass-approvals-and-sandbox` trên máy trạm chuyên dụng |
| Chạm hạn mức sử dụng của gói Codex giữa chừng khi đang dựng asset | Các lượt prompt dựng asset và scene bị tạm dừng; giai đoạn agent một lần (Task 11–14 của kế hoạch) có thể kéo dài thêm 1–2 ngày | Lưu mọi output của agent ra file (`work/`) để khi chạy lại không phải gọi lại agent. Gộp yêu cầu vào ít prompt hơn. Dùng `codex exec resume` để giữ ngữ cảnh thay vì mở phiên mới. Khi bị giới hạn, hoãn lượt prompt kế tiếp tới khi hạn mức được làm mới. Việc sinh ảnh không bị ảnh hưởng: phần này chạy bằng script và không gọi LLM nào |
| Ghép trong không gian sRGB, bóng chỉ màu đen | Sai tông nhẹ | Chấp nhận cho POC. Hướng nâng cấp: pass Shadow Catcher (EXR), `bg × pass`, rồi alpha-over |

## 9. Việc tương lai (không thuộc POC)

- Gợi ý gán nhãn chi tiết hơn trong sidecar, nếu Terry cần:
  - mask chính xác cho từng đối tượng từ pass object-index của Blender;
  - box có xét che khuất.
- SAM3 làm bước QA độ thật.
- Sinh hàng loạt (~1000 ảnh): cùng chuỗi xử lý với `n_images: 1000`, thêm nền sạch và thêm camera, tức là góc nhìn (hiệu chỉnh, scene proxy và kiểm tra asset một lần cho mỗi camera; mọi ảnh của camera chạy bằng script), cộng đề xuất biến thể tuỳ chọn cho từng kịch bản trong danh mục (`prompts/02`). Khuyến nghị: Terry chạy nhanh một A/B với ~200 ảnh trước khi chúng tôi sinh 1.000 ảnh (mục 10).
- Các kịch bản trong danh mục ngoài POC (V3–V8, N3–N5, mục 1.1) sau khi Terry xác nhận quy tắc; nền tạo bằng median theo thời gian khi không có frame trống, mô hình méo ống kính, ghép bằng pass Shadow Catcher.
- Nhất quán theo thời gian cho clip tổng hợp ngắn (camera chạy 5 fps).

## 10. Đánh giá khuyến nghị (do Terry thực hiện)

Ảnh trông thật không phải là đích cuối. Dữ liệu tổng hợp phải làm model sau khi train tốt hơn trên footage thật. Phần đánh giá này là việc của Terry, nằm ngoài ticket này; được giữ ở đây dưới dạng khuyến nghị.

- **Tập test thật giữ riêng.** Không bao giờ dùng để train hay làm nền cho ảnh tổng hợp. Gồm:
  - clip vi phạm gốc (xe nâng đẩy ≥ 2 tấm LSP);
  - frame hợp lệ gốc (xe nâng đẩy đúng 1 tấm LSP);
  - frame có LSP nằm yên (không có xe nâng đẩy).

  Để tránh rò rỉ dữ liệu (leakage), loại mọi frame thuộc cùng clip hoặc cùng khoảng thời gian với bất kỳ frame nào đã dùng làm nền sạch hoặc ảnh tham chiếu. Vi phạm thật hiếm, nên được phép dùng vi phạm dàn dựng quay bằng camera và phát lại các clip đã quay.
- **So sánh A/B** với thiết lập train giống hệt nhau (cùng kiến trúc, hyperparameter, số epoch và seed):
  - Model A: chỉ dữ liệu thật (baseline);
  - Model B: dữ liệu thật + dữ liệu tổng hợp (ảnh vi phạm và ảnh hợp lệ).
- **Dose-response (tuỳ chọn):** train B với 0 / 250 / 500 / 1000 ảnh tổng hợp để xem thêm dữ liệu tổng hợp có giúp ích, bão hoà hay gây hại.
- **Metric** trên class `Forklift Pushing Multiple Lsps`: recall, precision và mAP@0.5, cộng số báo động giả trên frame hợp lệ 1 tấm LSP và frame LSP nằm yên (model không được báo khi chỉ đẩy đúng 1 tấm LSP). Báo cáo theo từng camera. Số báo động giả của B trên frame hợp lệ thật không được tăng: ảnh hợp lệ tổng hợp (đề xuất bổ sung của chúng tôi) có mặt chính là vì điều này. Train B có và không có chúng (đặt cả hai tỉ lệ về 0) sẽ cho thấy có cần chúng hay không.
- **Tín hiệu sớm (khuyến nghị):** Terry chạy nhanh một A/B với ~200 ảnh tổng hợp trước khi chúng tôi sinh 1.000 ảnh. Nếu B không tốt hơn, chúng tôi quay lại tinh chỉnh độ thật thay vì tăng số lượng.
- **Tiêu chí đạt:** B hơn A về recall theo một biên độ đã thống nhất, không giảm precision và không tăng báo động giả. Các ngưỡng do Terry quyết định.
- **Sau đó** bài test RTSP trực tiếp của Terry (bước cuối; Gate 3 trong kế hoạch dự án) xác nhận kết quả trên luồng live.
- **Phần của chúng tôi:** khi Terry báo không có cải thiện (hoặc báo động giả tăng), chúng tôi tinh chỉnh độ thật (asset, ánh sáng, `gain`, `h264_crf`) hoặc tỉ lệ kịch bản (`lsp_count_weights`, `valid_fraction`, `idle_row_fraction`, `skid_count_weights`), sinh lại ảnh và bàn giao lô mới.

## 11. Câu hỏi còn mở

Đầu vào đang chờ (các bên liên quan sẽ cung cấp; không còn là câu hỏi mở):

- Ảnh gốc không overlay theo từng camera (POC: camera này), gồm hai tập riêng: đúng 1 ảnh nền sạch (không có xe nâng, LSP, SKID hay hàng) cùng ~10 ảnh tham chiếu có xe nâng, LSP, SKID và hàng ở nhiều vị trí (gồm cả xe nâng đẩy đúng 1 tấm LSP và các tấm LSP nằm yên thấy trọn vẹn), chỉ dùng để tách đối tượng; quay trong lúc camera không xê dịch. Là điều kiện tiên quyết cho Task 3 của kế hoạch.
- Máy trạm Ubuntu có GPU (mục 2): Ubuntu, GPU NVIDIA, CUDA và SAM3. Là điều kiện tiên quyết cho Task 1 của kế hoạch.
- Kích thước LSP chính xác. Trong lúc chờ, kích thước được đo (agent + MoGe); khi có, giá trị được đặt vào `lsp_size_m` trong `config.json`.

Do Terry quyết định (nằm ngoài ticket này):

- Format nhãn (YOLO hoặc COCO), một box mỗi sự kiện (như hệ thống hiện tại) hay mỗi đối tượng, và việc chuyển đổi từ file JSON sidecar của chúng tôi.
- Tập test thật giữ riêng, ngưỡng đạt A/B và tiêu chí đạt khi test RTSP (mục 10).

Còn mở:

1. Ngoài kiểu "nối tiếp", nhiều tấm LSP được xếp thế nào trong vi phạm thật? Chồng lên nhau? Cạnh nhau? (Các kịch bản V3–V5 ở mục 1.1.) (Số tấm đã quyết: mặc định tổng cộng 2 tấm, 3 tấm ít hơn.)
2. Tấm thật có mép gập hay không?
3. Slug model Codex cho "GPT Astra 6". Kế hoạch dùng `gpt-astra-6`, và Task 1 xác nhận hoặc sửa lại. Task 1 cũng ghi lại tài khoản Codex đang đăng nhập (cá nhân hay team) và hạn mức sử dụng của gói (Codex chạy bằng gói đã đăng nhập, không dùng API key, nên không có ngân sách nào để đặt).
4. Sandbox `workspace-write` của Codex có cho chạy `blender --background` và `ffmpeg` không? Codex có xem được ảnh cục bộ giữa phiên không? (Spike ở Task 1. Kế hoạch đính kèm lại ảnh bằng `--image`, cách này luôn chạy được.)
5. `cli-anything-blender` có trên PyPI không? (SKILL.md của nó ghi `pip install cli-anything-blender`. Kế hoạch cài từ mã nguồn.)
6. `cli-anything-blender render script` có in gì khác ngoài script không? (Spike kiểm tra output có compile được hay không.)
7. Blender 5.x: `material.use_nodes` / `world.use_nodes` còn gán được không? (Các script chấp nhận cả hai trường hợp.)
8. Prompt văn bản SAM3 nào nhận diện tốt LSP (`"slip sheet"` trong `config.json`; phương án khác là `"blue plastic sheet"` và `"floor mat"`). Spike ghi lại prompt phân vùng ổn định.
9. Toạ độ box OSD. Đã đo trên clip chú thích là `[0, 0, 90, 24]` và `[760, 0, 960, 26]`, và cần kiểm tra lại trên video gốc.
10. Mỗi camera có frame thật sự trống không (không có xe nâng, LSP, SKID hay hàng trên sàn)? Nếu không, cần nền tạo bằng median theo thời gian.
11. Tên material đặt bằng `material create` có được giữ trong script bpy export ra không (cần cho `tex_*`)? `blender/texture_asset.py` dừng với lỗi nếu không có material `tex_*` nào.
12. Mỗi camera có những mẫu xe nâng, loại hàng và loại SKID nào (mỗi loại một asset)?
13. Các quy tắc kịch bản cần xác nhận với Terry trước khi sinh ảnh (mục 1.1): V3 (xếp chồng) có phải vi phạm không? V6 (tấm LSP thêm để trống) có phải vi phạm không? Ranh giới nằm ở đâu khi xe nâng chạm vào một tấm LSP nằm yên (N3/N4)?
