# Ảnh vi phạm tổng hợp cho AAT — Yêu cầu gốc & Các quyết định

Nguồn: tin nhắn của Gary Ng gửi Paul (ảnh chụp: [gary-request.png](gary-request.png)).

## Yêu cầu gốc (giữ nguyên tiếng Anh)

> The other one is related to 3D augmented data generation with AAT project that we will need to build an agent pipeline using Blender and GPT Astra 6 to recreating the data for training. Currently I am using Qwen 3.8 which is not having good result but Astra 6 is very strong in 3D modeling. We need to use that to produce good quality data especially for AAT project as it is not having enough violation data base to train the model. Please assign someone to work with me on this.

> The workflow should be first data collection for the background and key object like forklift, LSP, SKID and cargo in different camera and different position for those object in the camera. Then using Sam3 to extract the segmentation for each object.
>
> A clean background will be selected as the bases, then use MOGE to evaluate the depth and cross check with the size and position for each detected object from Sam3 to calibrate the depth and size for each camera.
>
> Feed the extract key objects into Astra 6 and ask it to reproduce it in 3D first like forklift, LSP, SKID and different cargo types. Prompt it few times to make sure the detail can be showed.
>
> Then ask Astra 6 to come up with all the violation scenario and randomly reproduce 3D model with volation events in different angle and direction and place the generated 3D model back to each camera background with photorealistic treatment based on the background with some earlier extracted frames to match.
>
> Readjust the prompt and make it a python model which can generate 1000 images for retraining.
>
> Train the augment model and test against the RTSP.

## Tóm tắt yêu cầu

1. Thu thập nền và các đối tượng chính (forklift, LSP, SKID, cargo) từ nhiều camera, nhiều vị trí; dùng SAM3 tách segmentation từng đối tượng.
2. Chọn nền sạch; dùng MoGe ước lượng depth, đối chiếu kích thước/vị trí đối tượng từ SAM3 để hiệu chỉnh depth và tỉ lệ cho từng camera.
3. Đưa đối tượng đã tách vào Astra 6 để dựng lại 3D; prompt nhiều lần cho đủ chi tiết.
4. Astra 6 đề xuất các kịch bản vi phạm, dựng ngẫu nhiên nhiều góc/hướng, đặt lại lên nền camera với xử lý photorealistic.
5. Đóng gói thành pipeline Python sinh ~1000 ảnh để train lại.
6. Train model và test trên RTSP.

## Thuật ngữ

**Camera (khái niệm):** một góc nhìn cố định — đúng một ảnh nền sạch cùng ~10 ảnh tham chiếu có các đối tượng chụp từ đúng góc nhìn đó, với hiệu chỉnh riêng (tỉ lệ, mặt phẳng sàn, ống kính). Khái niệm này không gắn với một thiết bị vật lý: bất kỳ tập ảnh nào chung một góc nhìn đều tính là một camera, và một camera vật lý di chuyển được (ví dụ các preset PTZ) cho ra nhiều camera. `camera_id` đặt tên cho góc nhìn này.

## Các quyết định (2026-09-28)

| Chủ đề | Quyết định |
|---|---|
| Loại vi phạm (POC) | **Forklift Pushing Multiple Lsps** (tên model đích): xe nâng dùng nhiều hơn 1 tấm LSP cùng lúc để đẩy hàng là vi phạm; đúng 1 tấm LSP là hợp lệ. Astra 6 chỉ đề xuất các biến thể của loại này (số tấm LSP, cách xếp, hướng và vị trí xe nâng, loại hàng), được liệt kê trong danh mục kịch bản (các quyết định 2026-09-29 bên dưới). Các loại vi phạm khác nằm ngoài phạm vi |
| Quy tắc số tấm | Hợp lệ: tối đa 1 tấm LSP. Vi phạm: từ 2 tấm trở lên (`is_violation = lsp_count >= 2`); clip thật có 2 tấm. Mặc định khi sinh ảnh: tổng cộng 2 tấm (chính), 3 tấm là biến thể ít gặp hơn; tỉ lệ cấu hình được (`lsp_count_weights` trong `config.json`) |
| Mô hình đầu vào (theo camera) | Mỗi camera có hai đầu vào riêng biệt: (1) đúng **1 ảnh nền sạch** không có xe nâng, LSP, SKID hay hàng: làm nền cho mọi output và là đầu vào cho bước hiệu chỉnh MoGe; (2) **~10 ảnh tham chiếu** chụp từ cùng góc nhìn, có xe nâng, LSP, SKID và hàng ở nhiều vị trí khác nhau. Ảnh tham chiếu chỉ dùng để tách đối tượng: ảnh cắt, mask, kích thước và vị trí 2D từ SAM3 được đưa vào thư viện đối tượng của camera (tham chiếu asset, texture, đối chiếu kích thước, tham chiếu ánh sáng). Chúng không bao giờ được dùng làm nền cho output |
| Hiệu chỉnh | Một lần cho mỗi camera: MoGe chạy trên nền sạch cho depth và hình học (camera, mặt phẳng sàn, tỉ lệ mét), đối chiếu với kích thước và vị trí của các đối tượng SAM3 phát hiện (kích thước thật đã biết, ví dụ LSP). Mọi ảnh của camera đó dùng lại kết quả hiệu chỉnh này |
| Tên class | Tên model/class đích chính xác là `Forklift Pushing Multiple Lsps`, và box sự kiện trong sidecar dùng tên này. "Multiple LSPs" chỉ là chữ overlay của detector có sẵn |
| Kích thước LSP | Người dùng sẽ cung cấp kích thước chính xác sau. Trong lúc chờ, kích thước được đo (agent + MoGe); khi có, giá trị được đặt vào `lsp_size_m` trong `config.json` và ghi đè giá trị đo (được thay thế ngày 2026-09-29, xem bên dưới) |
| Máy trạm | Đang chờ: các bên liên quan sẽ cung cấp. Yêu cầu: Ubuntu, GPU NVIDIA, CUDA và SAM3 |
| Cách dựng 3D | Astra 6 (qua Codex + CLI-Anything điều khiển Blender) dựng lại các đối tượng chính thành asset 3D từ ảnh cắt SAM3: forklift, LSP, SKID và các loại cargo, prompt qua nhiều lượt cho tới khi đủ chi tiết; ảnh cắt thật cũng được dùng làm texture. Astra 6 cũng dựng scene proxy và sinh biến thể cho từng kịch bản đã duyệt trong danh mục |
| Công cụ | Một máy có Blender + Codex CLI (model GPT Astra 6), điều khiển Blender qua [CLI-Anything](https://github.com/HKUDS/CLI-Anything). Đăng nhập Codex với gói có sẵn GPT Astra 6; mức dùng bị giới hạn bởi hạn mức của gói (không dùng API key, không có ngân sách nào để đặt) |
| Cách render | **Cách B**: giữ nền sạch thật của camera làm nền; scene Blender chỉ là proxy (định vị, che khuất, đổ bóng); chỉ render các đối tượng chèn vào (xe nâng + hàng + LSP + SKID nằm yên) + bóng rồi ghép lên nền, với xử lý photorealistic khớp theo các ảnh tham chiếu (ánh sáng, màu, nhiễu, độ mờ, nén, bóng). Giữ nguyên góc camera và format ảnh gốc. |
| Vì sao không render toàn cảnh | Ảnh render toàn bộ trông "giả" → lệch domain; model học "ảnh render = vi phạm" và thất bại trên RTSP |
| Ảnh hợp lệ tổng hợp | **Đề xuất bổ sung (không có trong yêu cầu của Gary):** Gary chỉ yêu cầu ảnh vi phạm. Xe nâng, hàng và LSP chỉ được render trong ảnh tổng hợp, còn ảnh hợp lệ thật không có chúng, nên nếu dữ liệu tổng hợp chỉ có ảnh vi phạm thì model có thể học lối tắt "vật thể render = vi phạm". Vì vậy cùng pipeline đó sinh thêm ảnh hợp lệ tổng hợp (N1: xe nâng đẩy đúng 1 tấm LSP; N2: hàng LSP nằm yên không có xe nâng; sidecar `is_violation: false`), để model học đếm số tấm LSP bị đẩy thay vì nhận ra vật thể render. Đánh giá A/B của Terry sẽ cho thấy có cần chúng hay không. Tỉ lệ cấu hình được (`valid_fraction` trong `config.json`) |
| Ưu tiên | Tạo ảnh vi phạm trước; gán nhãn là bước của Terry, nằm ngoài ticket này |
| Nhãn | Do Terry phụ trách, nằm ngoài ticket này (các quyết định 2026-09-29 bên dưới): Terry gán nhãn cho ảnh và chuyển sang format nhãn anh ấy cần. Ticket này bàn giao ảnh cùng file JSON sidecar; sidecar (box đối tượng chiếu từ Blender, box sự kiện cho ảnh vi phạm) là gợi ý gán nhãn, không phải nhãn cuối cùng. Chúng tôi vẫn có thể dùng SAM3 để QA độ thật của ảnh render |
| Model đích | Model "Forklift Pushing Multiple Lsps", do Terry train bằng ảnh thật + ảnh vi phạm và ảnh hợp lệ do chúng tôi sinh ra. Việc train nằm ngoài ticket này |
| Thước đo thành công | Ticket này: ảnh trông thật, đúng format đầu vào, đạt bước người duyệt (Task 15 của kế hoạch). Mức cải thiện của model trên dữ liệu thật giữ riêng (A/B so với baseline chỉ dùng dữ liệu thật, spec mục 10) và bài test RTSP trực tiếp là phần kiểm tra của Terry, nằm ngoài ticket này; nếu Terry báo không có cải thiện, chúng tôi tinh chỉnh độ thật hoặc tỉ lệ kịch bản rồi sinh lại ảnh |
| Hướng mở rộng | Agent dựng asset + template kịch bản một lần cho mỗi camera, lưu thành script tái chạy được; sinh hàng loạt (~1000 ảnh) bằng script + tham số ngẫu nhiên, không gọi LLM cho từng ảnh |
| Phạm vi POC | 1 camera: 1 ảnh nền sạch (nền cho mọi output, đầu vào MoGe) + ~10 ảnh tham chiếu (chỉ để tách đối tượng); vi phạm "Forklift Pushing Multiple Lsps"; output ~10 ảnh vi phạm (V1, V2) + vài (3) ảnh hợp lệ tổng hợp (N1, N2); mỗi ảnh còn có 0–2 SKID nằm yên |

## Các quyết định (2026-09-29)

| Chủ đề | Quyết định |
|---|---|
| Phạm vi ticket | Ticket này chỉ sinh ảnh vi phạm và ảnh hợp lệ tổng hợp, cùng file JSON sidecar và các script sinh ảnh, rồi bàn giao cho Terry. Terry làm phần gán nhãn, chuyển sang format nhãn anh ấy cần, train và test (A/B so với baseline chỉ dùng dữ liệu thật, test RTSP trực tiếp) |
| SKID trong POC | Asset SKID là bắt buộc, không còn tuỳ chọn: Astra 6 dựng nó từ ảnh cắt SKID của SAM3 bằng harness, giống xe nâng và hàng, với texture từ ảnh cắt thật (`blender/texture_asset.py`) và cùng cổng duyệt preview do người làm. Mọi ảnh POC đặt 0–2 SKID nằm yên (có hoặc không có hàng bên trên) làm vật gây nhiễu trong cảnh, nằm trong vùng sàn và cách xa đường đi của xe nâng (`skid_count_weights` trong `config.json`, mặc định `{"0": 1, "1": 2, "2": 1}`). SKID không bao giờ nằm trong box sự kiện; sidecar liệt kê chúng trong `objects[]` với class `skid` |
| Danh mục kịch bản | [Spec mục 1.1](../superpowers/specs/2026-09-28-violation-image-synth-design.md) "Danh mục kịch bản: Forklift Pushing Multiple Lsps" chỉ liệt kê các kịch bản của vi phạm đích: kịch bản vi phạm V1–V8 và cảnh hợp lệ dễ nhầm (hard negative) N1–N5. POC gồm V1, V2 (2 và 3 tấm LSP nối tiếp) cùng N1, N2 (đẩy đúng 1 tấm LSP; một hàng LSP nằm yên không có xe nâng). Astra 6 sinh biến thể cho từng kịch bản đã duyệt (hướng và vị trí xe nâng, khoảng hở và độ lệch giữa các tấm LSP, loại hàng và chiều cao hàng, SKID gây nhiễu, ánh sáng). Cần xác nhận với Terry trước khi sinh ảnh: V3 (xếp chồng) và V6 (tấm LSP thêm để trống) có phải vi phạm không, và ranh giới khi xe nâng chạm vào một tấm LSP nằm yên (N3/N4). Các loại vi phạm khác (chắn lối đi, xếp hàng quá cao, người đi bộ, …) nằm ngoài phạm vi |
| Truy cập Codex | GPT Astra 6 được dùng qua Codex CLI đăng nhập bằng một gói, không qua API key, nên không đặt được ngân sách sử dụng: mức dùng bị giới hạn bởi hạn mức của gói. Thiết kế giữ mức dùng thấp: agent chỉ chạy ở vài bước một lần (scene proxy, asset, template kịch bản), việc sinh 1.000 ảnh chạy bằng script, không gọi LLM cho từng ảnh, và không có code nào gọi Astra 6 qua API. Task 1 của kế hoạch ghi lại tài khoản đang đăng nhập (cá nhân hay team) và hạn mức sử dụng của gói |
| Kích thước đối tượng | Không có số đo thật của LSP hay các đối tượng khác, nên dùng nguyên tỉ lệ mét của MoGe-2 (`scale_correction` 1.0, `lsp_size_m` null). `synth/calibrate.py` đo LSP ở dạng 3D từ các tấm đơn nằm trên sàn thấy trọn vẹn (key SAM3 `LSP_top`; mặt trên của các chồng đo ra lớn hơn một cách đều đặn nên không dùng) và ghi `lsp_measured_size_m` vào `work/calibration.json` (cam01: 1.90 × 1.85 m, trung vị của 5 tấm). Nếu sau này biết kích thước thật, giá trị đó được đặt vào `lsp_size_m` và khi đó chỉ dùng để đối chiếu |

## Tham chiếu LSP

![Tham chiếu LSP](lsp-reference.png)

- LSP = **tấm nhựa** lớn, mỏng, đặt phẳng trên sàn; hàng hóa (quấn màng co) đặt lên trên; xe nâng đẩy tấm để di chuyển hàng.
- Hình dạng (từ frame CCTV): gần vuông, bo góc, mỏng (~1–2 cm), màu xanh dương với nhiều vết mòn/bẩn đen, hơi bóng.
- Frame tham chiếu có 5 tấm LSP xếp nối tiếp thành một hàng.
- Ý tưởng asset: khối mỏng bo cạnh; texture lấy từ ảnh cắt SAM3 của LSP thật (thật hơn material procedural).

## Tham chiếu clip vi phạm

Clip: [violation-clip-forklift-pushing-2-lsps.mp4](violation-clip-forklift-pushing-2-lsps.mp4) — frame trong [violation-clip-frames/](violation-clip-frames/) (1 fps, `t01` = giây 0).

- Format: H.264, **960×540, 5 fps**, 8.2 giây. OSD camera in cứng vào hình: logo `SENSTAR` góc trên trái, timestamp góc trên phải.
- Xe nâng chạy ra xa camera, đẩy hàng trên LSP; từ khoảng giây 4 xe đẩy **2 tấm LSP nối tiếp** → bị gắn cờ "Multiple LSPs".
- Các tấm LSP hiện thành một dải mỏng cạnh/trước xe nâng, nhỏ (vài chục pixel) và bị xe + hàng che một phần — vi phạm khá khó thấy bằng mắt.
- **Clip này là output đã được vẽ đè** của một detector có sẵn (mask trên xe/hàng/LSP, bbox, mũi tên hướng, vệt sàn, nhãn `Multiple LSPs ID: 5`). Dùng tốt để tham khảo vi phạm trông thế nào, nhưng overlay khiến nó không dùng được làm nền hay ảnh train. Cần footage gốc không overlay (đang chờ: các bên liên quan sẽ cung cấp).
- Hộp đỏ sự kiện của hệ thống hiện tại bao cả xe nâng + hàng + LSP → nhiều khả năng câu trả lời cho "bbox theo sự kiện": một box cho mỗi sự kiện vi phạm. Box sự kiện trong sidecar của chúng ta dùng class đích `Forklift Pushing Multiple Lsps`.

## Đầu vào đang chờ (các bên liên quan sẽ cung cấp)

- Ảnh gốc (không overlay) của từng camera đích (POC: camera này), gồm hai tập riêng: (1) đúng 1 ảnh nền sạch không có xe nâng, LSP, SKID hay hàng; (2) ~10 ảnh tham chiếu có xe nâng, LSP, SKID và hàng ở nhiều vị trí trong khung hình (tốt nhất gồm cả xe nâng đẩy đúng 1 tấm LSP, các tấm LSP nằm yên thấy trọn vẹn và nhiều loại hàng), chỉ dùng để tách đối tượng. Ảnh tham chiếu phải đa dạng: lấy ở các thời điểm khác nhau, mỗi ảnh thêm một góc nhìn mới (hướng xe nâng, vị trí gần và xa, loại hàng, tấm LSP đơn thấy trọn vẹn, SKID nhìn rõ); các ảnh gần trùng nhau như frame liên tiếp không tạo ra giá trị và không đưa vào. Camera không được xê dịch giữa các ảnh. Là điều kiện tiên quyết cho Task 3 của kế hoạch.
- Máy trạm Ubuntu có GPU (hiện chưa có). Yêu cầu: Ubuntu, GPU NVIDIA, CUDA và SAM3. Là điều kiện tiên quyết cho Task 1 của kế hoạch.
- Đăng nhập Codex trên máy trạm đó với gói có sẵn GPT Astra 6 (gói đã đăng nhập, không phải API key; mức dùng bị giới hạn bởi hạn mức của gói, không có ngân sách nào để đặt). Task 1 của kế hoạch ghi lại đó là tài khoản nào (cá nhân hay team) và hạn mức sử dụng của gói. Là điều kiện tiên quyết cho Task 1 của kế hoạch.
- Kích thước LSP chính xác: không có (quyết định 2026-09-29, kích thước đối tượng); dùng số đo của MoGe-2. Nếu sau này biết, giá trị được đặt vào `lsp_size_m` trong `config.json`.

## Phần việc của Terry (nằm ngoài ticket này)

- Gán nhãn, format nhãn (YOLO / COCO) và việc chuyển đổi từ sidecar của chúng tôi, bbox theo từng đối tượng hay theo sự kiện vi phạm, và kiến trúc detector đích.
- Train, tập test thật giữ riêng, ngưỡng đạt A/B và tiêu chí đạt khi test RTSP.

## Câu hỏi còn mở

- Các quy tắc kịch bản, cần xác nhận với Terry trước khi sinh ảnh: V3 (LSP xếp chồng) có phải vi phạm không? V6 (tấm LSP thêm để trống) có phải vi phạm không? Ranh giới nằm ở đâu khi xe nâng chạm vào một tấm LSP nằm yên (N3/N4)?
