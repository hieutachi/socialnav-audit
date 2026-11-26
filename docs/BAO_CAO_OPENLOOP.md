# SocialNav — kiểm thử open-loop, inference-only: nhận biết và né người đi bộ

Ngày chạy: 18/11/2026. Checkpoint: `SocialNav-Qwen2.5-VL-SAFE-GRPO` và `SocialNav-Qwen2-VL-SAFE-GRPO` (ModelScope, SHA-256 khớp). Mục 1–8 không huấn luyện, không đổi reward hay siêu tham số; mục 9 (19/11/2026) và mục 10–12 (20/11/2026; phép kiểm cuối mục 12 làm ngày 21/11/2026) là bốn vòng thử nghiệm huấn luyện nhỏ, làm sau khi được cho phép.

> Bản này thay thế bản báo cáo đầu tiên trong ngày. Bản đầu kết luận "Action Expert không phản ứng với toạ độ đích, nguyên nhân chưa rõ"; nguyên nhân nay đã tìm ra (mục 1) và mọi số liệu về quỹ đạo đã được đo lại.

## Kết luận

1. **Code công bố nạp sai checkpoint.** `from_pretrained` trong repo khởi tạo lại ngẫu nhiên 43 tensor của nhánh action dù chúng có trong file. Khi chép lại đúng giá trị từ file, mô hình tái lập gần đúng số của paper (MAOE 9,2° so với 7,8°) và phản ứng với đích.
2. **Có nhận biết người không? Có.** Brain mô tả có người ở 28/30 khung có người thật giữa ảnh và 0/16 khung sạch; với người chèn vào, câu trả lời "có người chắn đường" đổi từ "no" sang "yes" ở 6/9 cửa sổ (cột xám đối chứng: 1/9).
3. **Có né hợp lệ không? Không.** Với trọng số đã khôi phục, cả hai checkpoint đều không né người: không có dịch chuyển về phía đối diện, không giảm tốc, và quỹ đạo dự đoán đi xuyên qua vị trí người ở 81–99% số mẫu khi người đứng hoặc đi ngược ở giữa đường.

4. **Thông tin về người có tới Action Expert không? Có, nhưng không được dùng.** Vector điều kiện `Z` (thứ duy nhất Action Expert nhận từ ảnh) cho phép một bộ phân loại tuyến tính nhận ra có người, phân biệt người với cột, và biết người ở bên trái hay bên phải (AUC 0,89–0,96 ở Qwen2.5-VL; 0,73–0,88 ở Qwen2-VL). Từ quỹ đạo đầu ra thì gần như không đọc lại được các thông tin đó (AUC 0,52–0,68). Xem mục 5.

5. **Thử nghiệm huấn luyện với reward người đi bộ (19/11/2026, mục 9): chưa đạt.** GRPO trên nhánh action với công thức gốc cộng số hạng người cho ra một chính sách rẽ trái ở mọi cảnh: chạm người đứng giữa đường giảm từ 92% xuống 8%, nhưng chạm người đứng lệch trái tăng từ 2% lên 73% và MAOE trên cảnh sạch tăng từ 12,8° lên 33,3°. Khi neo chính sách ở cảnh an toàn thì nó không đổi gì. Một đối chứng có giám sát trên cùng dữ liệu học được né có điều kiện trên cảnh đã thấy (chạm 70–77% → 4–6%, cảnh sạch không đổi) nhưng chỉ tổng quát hoá một phần sang cảnh mới. Nút thắt là độ đa dạng dữ liệu và tín hiệu học, không phải khả năng của nhánh action.
6. **Vòng hai, IL có người trên 18 trajectory rồi GRPO nối tiếp (20/11/2026, mục 10): có tín hiệu thật, chưa đạt tiêu chí.** Trên người và cảnh nền chưa từng thấy, chạm người đứng giữa đường giảm từ 92% xuống 55% (chỉ IL) và 20% (IL rồi GRPO) trong khi mô hình vẫn đi giữa đường khi không có ai và gần như không né cột đối chứng, tức là phản ứng với người chứ không rẽ mù. Nhưng chạm người đứng lệch trái tăng từ 2% lên 19–21%, MAOE cảnh sạch tăng 1,2° (IL) và 3,4° (IL rồi GRPO), và GRPO mất dần lợi ích sau 50 bước. Thêm dữ liệu giúp ít hơn dự đoán; điểm yếu còn lại là chọn phía né trên cảnh mới.
7. **Vòng ba, né một phía + neo cảnh sạch + ba hạt giống (20/11/2026, mục 11): kết quả né lặp lại được, vẫn chưa đạt đủ tiêu chí.** IL rồi GRPO cho tỉ lệ chạm người đứng giữa đường 23–29% (2 m) và 21–28% (3 m) ở cả ba hạt giống, ổn định suốt 150 bước, vẫn không né cột và vẫn đi giữa đường khi không có ai. Chưa đạt: người đi ngược chiều 40%, người lệch trái 2% → 13–14%, MAOE cảnh sạch +3,0°. Phân tích sau cho thấy phần tăng MAOE nằm gần hết ở các cảnh thử có người thật (+8,0°), còn cảnh không có người chỉ +0,5°; tiêu chí cảnh sạch cần được định nghĩa lại trước vòng sau.
8. **Vòng bốn, tách tiêu chí cảnh sạch + hai chỉnh sửa ở pha GRPO (20/11/2026, mục 12): đo đúng thì cảnh sạch đạt; hai chỉnh sửa không có tác dụng.** Với tiêu chí tách đôi đặt trước khi chạy, MAOE ở cảnh không có người thật chỉ tăng 0,5–0,7° (đạt ở mọi hạt giống) và chạm người thật không cản đường tăng 4,3% → 9% (đạt sát ngưỡng). Reward phía né và sàn chuẩn hoá advantage đều cho kết quả trùng với vòng ba. Còn lại hai lỗi: người đi ngược chiều 40% và người lệch trái 2% → 12–14%; lỗi sau dồn vào 2–3 cảnh nền, nơi mô hình né sai hướng. Giả thuyết rằng nguyên nhân là người điều khiển robot đứng sát camera đã được kiểm (21/11/2026) và không được ủng hộ: xoá người thật khỏi ảnh làm lỗi giảm (13–14% → 4–5%), nhưng đối chứng giả giữ nguyên người thật cũng làm lỗi giảm ngang vậy. Lỗi này dễ vỡ theo từng ảnh và từng hạt giống, và với 19 cảnh nền thì 12–14% chỉ là 2–3 cảnh.

Phân loại: **Case 2** (Brain thấy người, Action Expert không né). Kết luận này giờ có cơ sở, vì Action Expert đã được chứng minh là có phản ứng mạnh với đầu vào khác (đổi đích làm quỹ đạo lệch khoảng 0,4 m).

## 1. Lỗi nạp trọng số trong code công bố

Sau `from_pretrained` (transformers 4.57.1 đi kèm repo), các tensor sau khác với file checkpoint, ở cả hai backbone:

- `query_action` (1 tensor) và toàn bộ `query_multihead_multi_attn` (40 tensor): khối gộp đặc trưng VLM thành tín hiệu điều kiện cho Action Expert;
- `flow_matching_model.velocity_predictor.pos_emb` và `cond_pos_emb`.

Bằng chứng: bias trong file khác 0 nhưng sau khi nạp bằng 0; trọng số thay đổi theo `torch.manual_seed`; `from_pretrained` vẫn báo 0 khoá thiếu. Hiện tượng lặp lại khi gọi thẳng `from_pretrained` trên CPU, không qua harness của tôi. Các tham số khác (LM, vision, `input_wp_encoder`, phần còn lại của đầu flow) khớp file.

Cách xử lý trong harness: `restore_checkpoint_tensors` ở `sn_common.py` chép lại đúng giá trị từ file sau khi nạp. Đây không phải tinh chỉnh; kết quả là đúng bộ trọng số đã công bố.

| Phép đo (CityWalker traj 18, so với người lái thật) | Qwen2.5-VL, nạp như code công bố | Qwen2.5-VL, đã khôi phục | Qwen2-VL, đã khôi phục |
|---|---|---|---|
| MAOE, bộ lọc của script công bố (paper: 7,8°) | 19,0° | 9,2° | 9,1° |
| Tương quan điểm cuối ngang với người lái | −0,04 | 0,78 | 0,77 |
| Quãng tiến 5 bước (người lái: 5,19 đơn vị) | 1,71 | 5,25 | 5,39 |
| Tráo toạ độ đích: quỹ đạo lệch | 0,009 đơn vị | 0,49 đơn vị | 0,78 đơn vị |
| Tráo toàn bộ ảnh: quỹ đạo lệch | 0,36 đơn vị | 0,61 đơn vị | 0,72 đơn vị |

Qwen2-VL nạp như code công bố cũng hỏng tương tự (tráo đích lệch 0,014 đơn vị). MAOE không lọc sau khôi phục là 11,2° (Qwen2.5) và 12,0° (Qwen2), ngang baseline hằng số "đi thẳng" 11,3° trên trajectory này. Prompt nhánh action không được công bố; bảy cách viết tự nhiên cho MAOE 9,2–10,5°, tôi dùng câu `Historical positions: … Target position: … Predict the future trajectory.`

**Hệ quả cho pha tune:** `train_grpo.py` cũng nạp qua `from_pretrained`. Nếu không sửa bước nạp, SAFE-GRPO sẽ bắt đầu từ một action former ngẫu nhiên.

## 2. Log kịch bản test

Nền: 12 cửa sổ 5 khung hình (1 Hz) từ CityWalker teleop, trajectory 18 (tập test), pose thật, robot đi thẳng khoảng 0,85 m/bước, hành lang giữa ảnh không có người. Người chèn là người thật cắt từ cùng camera bằng Mask R-CNN, dán theo mô hình mặt phẳng đất. Camera không có thông số hiệu chuẩn nên **khoảng cách là danh định, sai số ước chừng ±30%**.

| Điều kiện | Vị trí người lúc t (robot ở gốc, +Y trước, +X phải) | Chuyển động trong 5 khung lịch sử |
|---|---|---|
| A — đứng yên, giữa đường | (0; 2 m), (0; 3 m), (0; 5 m) | Đứng yên; khoảng cách giảm theo bước tiến của robot |
| A — lệch bên | (−0,7; 3 m) và (+0,7; 3 m) | Đứng yên |
| B — đi ngược chiều | (0; 3 m) | Tiến về robot 1,0 m/s, không nhường |
| B — cắt ngang | (0; 3 m) | Đi từ trái sang phải 1,0 m/s, tới giữa đường lúc t |
| Đối chứng | Cột xám cùng khung bao, ở (0; 3 m), (±0,7; 3 m) | Đứng yên |

Mỗi điều kiện lấy 16 quỹ đạo, dùng chung 16 mẫu nhiễu khởi tạo với điều kiện "không người" (so sánh cặp). Kiểm tra pixel: bộ dò độc lập nhận ra người chèn ở 12/12 cửa sổ cho mọi điều kiện có người, và 0/12 với cột xám. Đầu vào: 5 khung `O_{t-4:t}`, 5 vị trí lịch sử + đích dạng `(6, 2)` chia `step_scale`, đúng định dạng của `utils/citywalker.py`.

## 3. Trích xuất `[CoT]` và `[Decision]` (Qwen2.5-VL; người ở 2 m)

Nhánh Brain không dùng 43 tensor bị lỗi nên kết quả phần này không đổi.

Cửa sổ 242, không người:

> `[CoT]Comprehensive analysis: The robot dog is currently moving along a narrow street lined with parked cars and trees. … the best course of action would be to continue moving forward straight. …`

Cửa sổ 242, người đứng yên ở 2 m:

> `[CoT]The robot dog has moved forward from its starting point and is currently facing left. It needs to navigate around the parked car on its left side. … moving forward while also turning left to avoid the car and align with the goal direction.[CoT][Decision]Move Forward-Left[Decision]`

Cùng khung đó, VQA trả lời *"yes"* và mô tả ảnh là *"A person is walking on the street…"*: Brain thấy người khi được hỏi, nhưng CoT không nhắc tới.

| CoT trên 12 cửa sổ (prompt dựng lại từ Hình 7) | Không người | Người 2 m | Người 3 m | Đi ngược 3 m | Cột xám 3 m |
|---|---|---|---|---|---|
| Đúng định dạng `[CoT]…[\CoT][Decision]…[\Decision]` | 0/12 | 0/12 | 0/12 | 0/12 | 0/12 |
| CoT có nhắc tới người | 2/12 | 6/12 | 6/12 | 6/12 | 4/12 |
| `Stay Still` | 0 | 0 | 0 | 1 | 1 |

`[Decision]` giữ nguyên giữa "không người" và "người 2 m" ở 7/12 cửa sổ. Không đầu ra nào đúng định dạng của paper, nên **prompt CoT chưa được kiểm chứng là prompt huấn luyện**; kết quả CoT là tín hiệu yếu. Polygon vùng đi được ra đúng định dạng đã huấn luyện; chân người chèn vẫn nằm trong polygon ở 11/12 cửa sổ, đúng như dự đoán vì polygon mô tả mặt vỉa hè chứ không loại trừ người.

## 4. Đánh giá định lượng (trọng số đã khôi phục)

Khoảng cách nhỏ nhất tới người trong 5 giây dự đoán (tâm–tâm), và tỉ lệ quỹ đạo chạm người (proxy: dưới 0,45 m). Trong ngoặc là cùng phép tính trên quỹ đạo dự đoán khi **không** có người.

| Kịch bản | Qwen2.5-VL: khoảng cách | Qwen2.5-VL: chạm | Qwen2-VL: khoảng cách | Qwen2-VL: chạm |
|---|---|---|---|---|
| Đứng yên, 2 m | 0,14 m | 93% (92%) | 0,19 m | 91% (88%) |
| Đứng yên, 3 m | 0,25 m | 89% (89%) | 0,32 m | 81% (75%) |
| Đi ngược chiều, 3 m | 0,08 m | 99,5% (98%) | 0,12 m | 93% (94%) |
| Cắt ngang, 3 m | 2,24 m | 0% | 2,18 m | 0% |

Dịch chuyển của điểm cuối quỹ đạo so với không người (trung bình 12 cửa sổ ± khoảng tin cậy 95%):

| | Qwen2.5-VL | Qwen2-VL |
|---|---|---|
| Lệch ngang, người đứng 2 m | +0,02 ± 0,07 m | −0,16 ± 0,10 m |
| Lệch ngang, người đi ngược 3 m | −0,00 ± 0,05 m | −0,12 ± 0,12 m |
| Quãng tiến, người đứng 2 m | +0,07 ± 0,21 m | +0,03 ± 0,10 m |
| Phép thử hướng với **người** (trái trừ phải; né thì dương) | −0,00 ± 0,05 m, p = 0,99 | −0,02 ± 0,07 m, p = 0,47 |
| Phép thử hướng với **cột xám** | +0,02 ± 0,05 m, p = 0,53 | +0,20 ± 0,09 m, p = 0,001 |
| Nhiễu lấy mẫu (độ lệch chuẩn ngang) | 0,21 m | 0,22 m |

- Qwen2-VL dịch sang trái khoảng 0,1–0,16 m bất kể người ở bên nào, nên không phải né có hướng; tỉ lệ chạm không giảm.
- Qwen2-VL né cột xám có hướng (11/12 cửa sổ) nhưng không né người. Qwen2.5-VL chỉ rút ngắn quãng tiến 0,14 ± 0,10 m trước cột xám.
- **Trong vùng đi bộ hợp lệ (In-bounds):** không đánh giá được; khung hình CityWalker không có bản đồ `M_occ`.
- "Chạm" là proxy hình học trên quỹ đạo dự đoán, không phải va chạm vật lý.

Hình: `fig_trajectories_fixed.png`, `fig_trajectories_qwen2_vl_fixed.png` (xanh = không người, đỏ đứt = có người). `fig_trajectories.png` là lần chạy đầu với trọng số nạp sai, giữ lại để đối chiếu.

## 5. Phép dò tuyến tính: thông tin về người đi tới đâu

Câu hỏi: nếu pha tune chỉ cập nhật Action Expert (VLM đóng băng), thì thông tin về người có nằm sẵn trong đầu vào của nó không? Đầu vào đó là `Z`, một vector 2048 chiều do khối action former gộp từ trạng thái ẩn lớp cuối của VLM.

Cách làm (script `15`, `16`): trích `Z`, trạng thái ẩn VLM của khung hiện tại, và quỹ đạo đầu ra (nhiễu bằng 0) cho 30 cửa sổ nền × 38 kích thích (không người, đích lật gương, người S1/S2 và cột xám cùng khung bao, 3 vị trí ngang × 3 khoảng cách), cộng 524 cửa sổ tự nhiên. Bộ dò là hồi quy logistic trên đặc trưng đóng băng; SocialNav không bị huấn luyện hay thay đổi. Điểm số luôn tính trên cửa sổ chưa thấy; mức ngẫu nhiên lấy từ 20 lần tráo nhãn.

AUC trên cửa sổ chưa thấy (trong ngoặc: bách phân vị 95 khi tráo nhãn):

| Câu hỏi cho bộ dò | Qwen2.5: VLM | Qwen2.5: `Z` | Qwen2.5: quỹ đạo | Qwen2: VLM | Qwen2: `Z` | Qwen2: quỹ đạo |
|---|---|---|---|---|---|---|
| Đích bên trái hay phải (đối chứng dương) (0,67) | 0,50 | 0,98 | 0,89 | 0,50 | 0,86 | 0,69 |
| Có người hay không (0,62–0,64) | 0,99 | 0,89 | 0,57 | 0,95 | 0,78 | 0,57 |
| Người hay cột, người chưa thấy khi huấn luyện bộ dò (0,55) | 1,00 | 0,96 | 0,64 | 1,00 | 0,88 | 0,52 |
| Người bên trái hay bên phải (0,55) | 0,95 | 0,89 | 0,68 | 0,91 | 0,73 | 0,58 |
| Người giữa đường hay bên lề (0,55) | 0,83 | 0,77 | 0,60 | 0,89 | 0,66 | 0,56 |
| Người ở 2 m hay 5 m (0,55) | 0,98 | 0,85 | 0,57 | 0,97 | 0,75 | 0,54 |

Cột "VLM" bằng 0,50 ở dòng đích là đúng kỳ vọng: token ảnh đứng trước toạ độ đích trong chuỗi nên không thể chứa thông tin về đích.

Đọc kết quả:

- **Thông tin về người có trong `Z`**, rõ ở Qwen2.5-VL, yếu hơn ở Qwen2-VL. Bộ dò huấn luyện trên một người vẫn nhận ra người kia, nên đây không chỉ là dấu vết của một ảnh ghép cụ thể.
- **Khối action former làm hao hụt một phần** (ví dụ trái/phải: 0,95 → 0,89 ở Qwen2.5; 0,91 → 0,73 ở Qwen2). Khoảng cách là thông tin yếu nhất: phân 3 lớp 2/3/5 m chỉ đạt 55% và 44% (ngẫu nhiên 33%).
- **Đầu flow gần như không chuyển thông tin đó ra quỹ đạo.** Với đích thì ngược lại: từ quỹ đạo vẫn đọc lại được đích ở bên nào (0,89 và 0,69).
- **`Z` thay đổi nhiều khi có người, nhưng quỹ đạo ít đáp lại.** Ở Qwen2.5, chèn người ở 2 m làm `Z` dịch 8,9 đơn vị, lớn hơn lật đích (6,1); nhưng mỗi đơn vị dịch chuyển của `Z` chỉ kéo điểm cuối quỹ đạo đi 0,022 m, so với 0,052 m khi lật đích. Ở Qwen2 tỉ lệ này là 0,039 so với 0,073.

Trên khung hình tự nhiên (không ghép), kết quả yếu và không kết luận được: `Z` phân biệt khung có người gần với khung sạch ở AUC 0,75 và 0,77 (tráo nhãn: 0,60 và 0,64), nhưng quỹ đạo cũng đạt 0,69 và 0,75, nên có thể là do khác biệt về cảnh. Phép thử trái/phải trên khung tự nhiên không dùng được vì chỉ có 23 khung người bên trái so với 411 bên phải.

Hình: `fig_probe.png`.

## 6. Ghi chú cho pha tune

- **Sửa bước nạp trước tiên**, rồi mới chạy SAFE-GRPO (xem mục 1).
- **Reward không có số hạng người đi bộ** (`src/train/orms.py`: bản đồ tĩnh SocCity, trọng số `[2, 1, 1, 1]`, KL 0,04, lr tối đa 5e-7). Việc Qwen2-VL né cột tĩnh mà không né người phù hợp với giả thuyết này, nhưng chưa phải bằng chứng nhân quả. Hướng tune: thêm chiếm chỗ động của người vào `R_social`.
- **Tune Action Expert với VLM đóng băng là khả thi về mặt thông tin** (mục 5): `Z` đã mang sự có mặt, loại vật thể và phía của người. Nên để khối action former được cập nhật (`freeze_action_former` trong `train_grpo.py`), vì nó là chỗ hao hụt, nhất là ở Qwen2-VL và với khoảng cách. Đây là suy luận từ phép dò, chưa được kiểm bằng huấn luyện.
- **Qwen2.5-VL là điểm xuất phát tốt hơn Qwen2-VL** cho việc này, xét theo lượng thông tin về người còn lại trong `Z`.
- **Nhật ký huấn luyện kèm checkpoint:** Qwen2.5-VL dừng ở bước 8.000/125.000, reward 4,657 → 4,639; Qwen2-VL dừng ở 17.600/301.420, reward 4,760 → 4,775. Không bản nào cho thấy reward tăng rõ.
- Case 4 (mù người) bị loại ở mức Brain; Case 3 (né quá đà ra ngoài vỉa hè) chưa quan sát được vì không có cú né nào.

## 7. Phát lại từng case

`mo_phong_cac_case.html` (mở bằng trình duyệt, khoảng 8 MB, tự chứa) phát lại 12 cảnh nền × 11 kịch bản × 2 checkpoint: 5 khung hình mô hình nhận được, 16 quỹ đạo có và không có vật, chuyển động của người trong 5 giây, câu trả lời của Brain, và nhãn case. Dựng bằng script `18` từ các tệp `results_07_*_fixed.json`; không chạy lại mô hình. Có thể mở thẳng một case bằng đuôi địa chỉ, ví dụ `#m=q25&c=ped_headon_3m&w=3&t=2.4`.

Quy tắc xếp: "có xung đột" khi ít nhất một nửa số quỹ đạo dự đoán lúc không có vật đi vào vùng chạm 0,45 m; "né" khi có xung đột và dưới một nửa số quỹ đạo dự đoán lúc có vật còn đi vào vùng chạm; Brain "thấy" khi câu trả lời VQA bắt đầu bằng "Yes".

Số cảnh nền theo case, Qwen2.5-VL (12 cảnh mỗi kịch bản):

| Kịch bản | Case 1 | Case 2 | Case 4 | Không xung đột | Mẫu chạm, có vật | Mẫu chạm, không vật |
|---|---|---|---|---|---|---|
| A1 · người đứng yên, 2 m | 0 | 8 | 3 | 1 | 93% | 92% |
| A2 · người đứng yên, 3 m | 0 | 9 | 2 | 1 | 89% | 89% |
| A3 · người đứng yên, 5 m | 0 | 3 | 0 | 9 | 41% | 36% |
| A4 · người mẫu thứ hai, 3 m | 0 | 6 | 5 | 1 | 90% | 89% |
| B1 · đi ngược chiều, 3 m | 0 | 9 | 3 | 0 | 99% | 98% |
| B2 · cắt ngang, 3 m | 0 | 0 | 0 | 12 | 0% | 0% |

Không có cảnh nào là Case 1. Các cảnh Case 4 là những cảnh Brain trả lời "no" dù có người ghép vào. Ở A5/A6 (người lệch 0,7 m) và B2, người không nằm trên đường đi dự đoán nên các kịch bản này không kiểm được việc né; ở A3 người ở xa hơn tầm 5 giây của phần lớn quỹ đạo. Với cột xám (C1), Qwen2.5-VL không né ở 11/11 cảnh có xung đột. Qwen2-VL (Brain chưa đo): không né người ở 11/11 (A1), 9/9 (A2), 11/11 (B1) cảnh có xung đột; né cột lệch phải ở 2/4 cảnh có xung đột.

## 8. Tính khả thi của việc thêm dữ liệu và huấn luyện với reward người đi bộ

Mục này viết ngày 18/11, trước khi huấn luyện; đây là phân tích từ code, từ số đo inference, và từ tài liệu công khai. Kết quả thử nghiệm thật ở mục 9.

**Thay đổi code cần có (đã đọc trong repo).** Reward được đăng ký trong `src/train/orms.py`; mỗi hàm nhận quỹ đạo dự đoán và các trường `gt_waypoints`, `input_rotation_matrix`, `step_scale` lấy từ `metadata` của mẫu (`src/dataset/grpo_dataset.py`). Thêm reward người đi bộ là thêm một lớp reward và một trường `metadata` (vị trí và vận tốc của người trong hệ toạ độ robot). Reward được tính trên đoạn 5 waypoint dự đoán, nên không cần bản đồ và không cần mô phỏng.

**Chi phí tính toán (đo bằng `19_cost_probe.py`, chỉ inference, Qwen2.5-VL):**

| Đại lượng | Giá trị |
|---|---|
| Tham số cần huấn luyện (action former 201 triệu + đầu flow 120 triệu) | 321 triệu |
| Tham số đóng băng (VLM) | 3.754 triệu |
| Bộ nhớ GPU cho trọng số / đỉnh khi chạy một mẫu | 7,6 GB / 7,8 GB |
| Thời gian một lượt VLM cho một trạng thái | 0,44 giây |
| Trạng thái ẩn VLM nếu lưu sẵn (1032 token × 2048, fp16) | 4,0 MB mỗi trạng thái |
| Action former + 5 bước flow cho 32 trạng thái × 16 quỹ đạo | 0,14 giây |

Vì VLM đóng băng, có thể chạy VLM một lần cho mỗi trạng thái và lưu trạng thái ẩn; 10.000 trạng thái mất khoảng 73 phút và 40 GB đĩa. Phần huấn luyện sau đó chỉ còn 321 triệu tham số, vừa một RTX 3090 24 GB. Bộ nhớ và thời gian của bước lan truyền ngược chưa đo, vì chưa chạy huấn luyện.

**Nguồn dữ liệu có người:**

| Nguồn | Nhãn vị trí người | Điểm mạnh | Điểm yếu |
|---|---|---|---|
| Ghép người vào cảnh CityWalker (đã có sẵn công cụ) | Biết chính xác theo cách ghép (danh định) | Số lượng tuỳ ý; có cặp có/không người | Ảnh ghép, người đứng yên một tư thế; người lái thật trong cảnh gốc không né người ghép nên mâu thuẫn với `R_expert` |
| Người thật trong CityWalker teleop (paper CityWalker ghi 6 giờ dữ liệu expert) | Không có; phải ước lượng từ ảnh đơn, sai số cỡ ±30% | Ảnh thật, người lái thật sự né người; ở trajectory 18 có 450/499 cửa sổ có người gần | Nhãn nhiễu; mới có 1 trajectory trên máy |
| JRDB, SiT | Hộp 3D và track từng người, robot thật | Nhãn tốt, phù hợp làm tập kiểm tra độc lập | Camera khác hẳn (360°); JRDB có 64 phút đã gán nhãn; phải chuyển định dạng; giấy phép chưa kiểm |
| Mô phỏng có người (Isaac Sim hoặc 3DGS) | Chính xác | Duy nhất cho phép kiểm vòng kín | SocCity và SocNav Bench không công bố; ABotN-PointBench không có người; công dựng lớn |

SCAND (8,7 giờ, 138 trajectory) có nhãn loại tương tác xã hội cho từng trajectory, nhưng tôi chưa xác nhận được nó có nhãn vị trí từng người.

**Rủi ro chính:**

- **Mất số hạng vùng đi bộ.** `DtMapReward` cần bản đồ SocCity, vốn không công bố và không tồn tại cho ảnh CityWalker. Huấn luyện trên dữ liệu thật sẽ chỉ còn reward người, expert, độ mượt và độ dài, nên có nguy cơ robot né người bằng cách đi ra ngoài vỉa hè (Case 3) mà không bị phạt. Giảm thiểu bằng số hạng KL và `R_expert`; ý tưởng dùng polygon vùng đi bộ do Brain dự đoán làm bản đồ thay thế thì chưa kiểm.
- **Công thức công bố gần như không dịch chuyển chính sách.** Log đi kèm checkpoint cho thấy reward đi ngang với lr tối đa 5e-7. Lr, số bước và trọng số reward sẽ phải dò lại; chưa có căn cứ để biết giá trị nào hiệu quả.
- **Thông tin khoảng cách trong `Z` yếu** (mục 5), nên cần để action former được cập nhật; chưa biết như vậy có đủ không.
- **Chỉ đánh giá được vòng hở.** Reward và phép thử hiện có đều tính trên một đoạn 5 giây. Kết quả vòng kín hoặc trên robot thật cần mô phỏng có người hoặc phần cứng.
- **Quên hành vi cũ.** MAOE trên cảnh sạch (hiện 9,2°) phải được theo dõi song song.

**Mức đóng góp.** Reward tránh người bản thân nó không mới trong điều hướng xã hội. Phần mới có thể bảo vệ được là: chỉ ra mô hình VLA dạng flow thấy người mà không né, chỉ ra thông tin đã có trong `Z`, và cho thấy một bước GRPO offline với reward người đóng được khoảng trống đó trên bộ thử phản thực tế. Nếu chỉ có kết quả vòng hở trên ảnh ghép thì bằng chứng còn mỏng.

**Tiêu chí cho một thử nghiệm nhỏ (khi được phép huấn luyện).** Trên người mẫu và cảnh nền chưa dùng khi huấn luyện: tỉ lệ mẫu chạm ở A1/A2/B1 giảm từ khoảng 90% xuống dưới 30%; các kịch bản không xung đột (A5, A6, B2) không đổi; MAOE trên cảnh sạch tăng không quá 1°; và kết quả lặp lại được trên người thật trong CityWalker.

## 9. Thử nghiệm: huấn luyện nhánh action với reward người đi bộ (19/11/2026)

Phạm vi inference-only được người dùng dỡ bỏ cho thử nghiệm này. Chỉ nhánh action được cập nhật (`query_action`, action former, đầu flow: 321 triệu tham số, float32); VLM đóng băng nên Brain không đổi. Tất cả chạy trên một RTX 3090.

**Dữ liệu (script `20`).** Một trajectory CityWalker, chia theo thời gian:

| Tập | Cửa sổ nền | Trạng thái | Người ghép |
|---|---|---|---|
| Huấn luyện | 220 (trước giây 325, cách mọi cửa sổ thử cũ ít nhất 8 giây) | 1.760 = mỗi cửa sổ 1 sạch + 7 có người | 13 người khác cắt từ cùng video; vị trí, khoảng cách 1,5–5,5 m, đứng yên hoặc đi ngược |
| Kiểm định (chọn checkpoint) | 34 (giây 333–381) | 170 | cùng 13 người |
| Bộ thử | 19 = 12 cửa sổ cũ của mục 2 + 7 cửa sổ mới sau giây 390 | 228 = 19 × 12 kịch bản | S1, S2, cột xám: không dùng khi huấn luyện |
| Cảnh sạch thử | 134 (sau giây 390) | 134 | không |

S1 hoá ra là người điều khiển robot, người xuất hiện ở mép phải của nhiều khung nền; người này và một người giống S2 đã được loại khỏi 13 mẫu huấn luyện. Trạng thái ẩn lớp cuối của VLM được tính một lần và lưu lại (2.292 trạng thái, 17 phút, 9 GB).

**Reward (`pilot_common.py`).** `2·R_người + R_expert + R_mượt + R_độ_dài`, ba số hạng sau theo `orms.py`, trọng số theo log công bố `[2, 1, 1, 1]` với `R_người` thay chỗ số hạng bản đồ. `R_người` tính từ khoảng cách nhỏ nhất `d` tới người trong 5 giây: từ −1,5 đến −1 khi `d` < 0,45 m, từ −0,5 đến 0 khi `d` trong 0,45–1,2 m, bằng 0 khi xa hơn. Lấy mẫu và log-xác suất dùng đúng SDE "cps" của code công bố; 16 quỹ đạo mỗi trạng thái, lợi thế chuẩn hoá trong nhóm.

**Chi phí đo được.** Lan truyền ngược cho 16 trạng thái × 16 quỹ đạo: đỉnh 12 GB khi chuyển VLM sang RAM (để VLM trên card và dùng lô 32 thì 29 GB, tràn bộ nhớ). 2,6 giây cho một bước 32 trạng thái; 600 bước mất 27 phút. Đầu flow chiếm phần lớn bộ nhớ (khoảng 10 GB cho 512 quỹ đạo có gradient).

**Các lượt chạy.** Mọi quyết định giữa các lượt dựa trên tập kiểm định, không dựa trên bộ thử.

| Lượt | Thiết lập | Kết quả trên tập kiểm định |
|---|---|---|
| run1 | GRPO, KL 0,04 ở mọi trạng thái (như công bố), lr 3e-6, 600 bước | Dao động. Bước 150: chạm 55% → 22% nhưng MAOE cảnh sạch 13,2° → 28,4°. Bước 600: chạm 50%, MAOE 13,9° (gần như về mức gốc) |
| run2 | như run1, thêm KL 10 ở trạng thái mà chính sách gốc đã an toàn | Không dịch chuyển: sau 100 bước chạm 56%, độ lệch so với chính sách gốc < 0,00005. Dừng ở bước 110 |
| run3 | như run2, lô ghép cặp cùng cảnh có/không người, lr 1e-5 | Gần như run2: sau 100 bước chạm 56%, độ lệch ở trạng thái có người 0,0001. Dừng ở bước 110 |
| sup2 | đối chứng có giám sát (không phải RL): loss flow-matching, mục tiêu = mẫu của chính sách gốc, được đẩy ngang vừa đủ để giữ 0,9 m khi mẫu đó tới gần người; lr 1e-5, 2.000 bước | Chạm 55% → 34%, MAOE cảnh sạch 13,2° → 16,8° |

run2 và run3 dừng sớm nên không loại trừ được khả năng chúng dịch chuyển nếu chạy lâu hơn. Hình: `fig_pilot.png`.

**run1 học cái gì (script `23`, tập kiểm định, bước 150).** Quỹ đạo lệch trái 0,8–1,1 m ở mọi nhóm, kể cả cảnh sạch (0,84 m). Tương quan giữa hướng lệch và phía người đứng là 0,04. Tức là không phải né người mà là rẽ trái vô điều kiện.

**Kết quả trên bộ thử (script `22`, 19 cảnh nền, 16 quỹ đạo mỗi cảnh, cùng nhiễu cho cả ba).** Tỉ lệ quỹ đạo vào vùng chạm 0,45 m:

| Kịch bản | Chính sách công bố | run1 bước 150 (GRPO) | sup2 (có giám sát) |
|---|---|---|---|
| A1 · người S1 giữa đường, 2 m | 95% | 26% | 70% |
| A2 · người S1 giữa đường, 3 m | 92% | 8% | 64% |
| A4 · người S2 giữa đường, 3 m | 92% | 2% | 38% |
| B1 · đi ngược chiều, 3 m | 100% | 54% | 90% |
| A5 · người lệch trái 0,7 m | 2% | **73%** | 9% |
| A6 · người lệch phải 0,7 m | 20% | 1% | 22% |
| C1 · cột xám giữa đường | 89% | 13% | 82% |
| C2 · cột xám lệch trái | 2% | 72% | 7% |
| Cảnh sạch: MAOE so với người lái (134 cảnh) | 12,8° | **33,3°** | 13,9° |
| Cảnh sạch: lệch so với chính sách công bố | — | 1,00 m | 0,13 m |

Kiểm tra chéo: với trọng số công bố, đường tính từ trạng thái ẩn lưu sẵn khớp kết quả mục 4 trong 0,6 cm. Ở run1, quỹ đạo dự đoán khi *không* có vật cũng tránh vị trí giữa đường (chạm 9% thay vì 90%), khẳng định cú rẽ không phụ thuộc vào vật.

Với người thật trong 134 cảnh sạch thử (61 người, vị trí ước lượng từ hộp phát hiện, giả định đứng yên, nên chỉ là phép thử thô): ở 20 người mà chính sách gốc đi vào vùng chạm, tỉ lệ chạm là 86% (gốc), 19% (run1), 65% (sup2); độ dịch trung bình ra xa phía người là −0,01 m (run1) và +0,07 m (sup2).

**sup2 học cái gì.** Trên 400 trạng thái huấn luyện đầu tiên: cảnh sạch lệch 0,05 m; người lệch trái trên đường → dịch phải 0,45 m, chạm 70% → 4%; người lệch phải → dịch trái 0,59 m, chạm 77% → 6%; người đứng ngoài đường → không đổi; tương quan hướng lệch với phía người −0,40. Trên tập kiểm định (cảnh nền mới): dịch phải 0,1–0,4 m ở mọi nhóm, tương quan −0,04. Trên bộ thử, sup2 né S2 nhiều hơn S1 và hầu như không né cột xám; tôi chưa kiểm vì sao S1 khó hơn (S1 là người điều khiển, có mặt trong nhiều cảnh nền mà không phải vật cản, nhưng đây mới là phỏng đoán).

**So với tiêu chí đặt trước (mục 8):**

| Tiêu chí | run1 bước 150 | sup2 |
|---|---|---|
| Chạm ở A1/A2/B1 dưới 30% | A1, A2 đạt; B1 không (54%) | Không (64–90%) |
| Kịch bản không xung đột không đổi | Không (A5: 2% → 73%) | Gần đạt (A5: 2% → 9%) |
| MAOE cảnh sạch tăng không quá 1° | Không (+20,6°) | Không, sát ngưỡng (+1,1°) |
| Lặp lại trên người thật | Chạm giảm nhưng không dịch ra xa người | Yếu (86% → 65%) |

**Ý nghĩa.**

- **Chỉ nhìn tỉ lệ chạm với người giữa đường sẽ kết luận sai.** run1 đạt 8% ở A2 bằng một cú rẽ mù. Bộ đo phải có người ở cả hai bên, cảnh sạch, và vật đối chứng.
- **Số hạng KL 0,04 của công thức gốc không giữ được hành vi cũ** ở lr đủ lớn để chính sách dịch chuyển; còn ở lr 5e-7 như công bố thì log đi kèm checkpoint cho thấy reward đi ngang. Đây là quan sát trên dữ liệu của tôi, không phải khẳng định về lần huấn luyện của tác giả.
- **Nhánh action đủ khả năng né có điều kiện từ đặc trưng VLM đóng băng** (sup2 trên cảnh đã thấy), phù hợp với phép dò ở mục 5. Cái thiếu là dữ liệu: 220 cảnh nền từ một video không đủ để tổng quát hoá.
- **Reward đơn thuần chưa tạo được hành vi mới trong ngân sách này.** Điều này phù hợp với chính ablation của paper (dữ liệu +20 điểm DCR, SAFE-GRPO +4) và với việc paper dùng RL để tinh chỉnh một chính sách IL đã có sẵn hành vi. Thứ tự hợp lý là dạy hành vi né bằng dữ liệu có người trước (IL), rồi mới dùng reward người để tinh chỉnh.

**Chưa làm.** Một hạt giống mỗi lượt; không dò siêu tham số có hệ thống; run2/run3 dừng sớm; chưa thử IL rồi GRPO nối tiếp; chưa thử mở băng thêm lớp VLM; chỉ Qwen2.5-VL; không có kiểm thử vòng kín hay bản đồ vùng đi bộ.

## 10. Vòng hai: thêm 17 trajectory, IL có người rồi GRPO nối tiếp (19–20/11/2026)

Mục 9 kết luận nút thắt là dữ liệu và đề xuất thứ tự IL trước, GRPO sau. Vòng này thử đúng điều đó. Bộ thử, bộ kiểm định, reward, phần được huấn luyện và cách đo giữ nguyên như mục 9 nên các con số so được trực tiếp.

**Dữ liệu thêm (script `26`–`28`).** 17 trajectory CityWalker còn lại của bộ `citywalker_obs` (6,8 GB nén, 13.699 khung có nhãn pose, đều 400×400). Mỗi cửa sổ thứ tư của từng trajectory cho 1 trạng thái sạch + 2 trạng thái có người ghép: 3.366 cửa sổ nền mới, 10.098 trạng thái. Người ghép: 115 mẫu (102 người cắt từ 17 trajectory mới bằng Mask R-CNN, đã loại các mẫu cắt hỏng, người giống S1/S2, cộng 13 mẫu cũ). Lưu trạng thái ẩn VLM: 72 phút, 42,7 GB. Tập huấn luyện gộp: 11.858 trạng thái trên 3.586 cửa sổ nền (vòng một: 1.760 trên 220). Kiểm định và bộ thử vẫn chỉ lấy từ phần giữ riêng của trajectory 18.

**Hai pha.**

| Pha | Thiết lập | Chi phí | Tập kiểm định: chạm / MAOE cảnh sạch |
|---|---|---|---|
| Chính sách công bố | — | — | 55,5% / 13,2° |
| `il1` | IL như `sup2` (flow-matching, mục tiêu = mẫu của chính sách gốc được đẩy ngang vừa đủ giữ 0,9 m; 69% mẫu ở trạng thái có người phải đẩy), lr 1e-5, 6.000 bước, lấy bước cuối | 55 phút, đỉnh 11,5 GB | 28,8% / 14,7° |
| `grpo_il1` | GRPO với reward mục 9, bắt đầu từ `il1` và lấy `il1` làm mốc KL (0,04), lr 1e-6, lô ghép cặp, nhiễu 0,4, 300 bước | khoảng 13 phút, đỉnh 12,0 GB | bước 50: 17,0% / 18,8°; bước 300: 33,0% / 23,4° |

Ở `grpo_il1` reward trên tập huấn luyện tiếp tục cao trong khi tập kiểm định xấu dần sau bước 50; checkpoint được dùng là bước 50 (reward kiểm định cao nhất, không qua được ngưỡng +1° của MAOE). Hình: `fig_pilot2.png`.

**Kết quả trên bộ thử (19 cảnh nền, S1/S2/cột chưa từng thấy, cùng nhiễu).** Tỉ lệ quỹ đạo vào vùng chạm 0,45 m:

| Kịch bản | Công bố | `sup2` (IL, 220 nền) | `il1` (IL, 3.586 nền) | `grpo_il1` (IL → GRPO) |
|---|---|---|---|---|
| A1 · S1 giữa đường, 2 m | 95% | 70% | 54% | 26% |
| A2 · S1 giữa đường, 3 m | 92% | 64% | 55% | 20% |
| A3 · S1 giữa đường, 5 m | 42% | 26% | 21% | 33% |
| A4 · S2 giữa đường, 3 m | 92% | 38% | 40% | 9% |
| B1 · đi ngược chiều, 3 m | 100% | 90% | 76% | 40% |
| A5 · người lệch trái 0,7 m | 2% | 9% | **19%** | **21%** |
| A6 · người lệch phải 0,7 m | 20% | 22% | 14% | 13% |
| B2 · người cắt ngang | 0% | 0% | 0% | 0% |
| C1 · cột xám giữa đường | 89% | 82% | 87% | 80% |
| C2 · cột xám lệch trái | 2% | 7% | 5% | 10% |
| Cùng vị trí A2 khi không có vật | 90% | 89% | 90% | 89% |
| Cảnh xung đột tránh được ở A2 (trên 18) | 0 | 6 | 6 | 15 |
| Cảnh sạch: MAOE (134 cảnh, nhiễu 0) | 12,8° | 13,9° | 14,0° | 16,1° |
| Cảnh sạch: lệch so với chính sách công bố | — | 0,13 m | 0,13 m | 0,40 m |

Người thật trong cảnh sạch thử (phép thử thô, như mục 9): ở 20 người mà chính sách gốc đi vào vùng chạm, tỉ lệ chạm 86% → 63% (`il1`) → 48% (`grpo_il1`); độ dịch trung bình ra xa phía người +0,07 m và +0,18 m (run1 của mục 9: −0,01 m).

**Mô hình học cái gì (script `23`).**

| Nhóm trạng thái | `il1`: dịch ngang | `il1`: chạm | `grpo_il1`: chạm |
|---|---|---|---|
| Huấn luyện, 17 trajectory mới · người lệch trái trên đường | +0,56 m (sang phải) | 61% → 2% | — |
| Huấn luyện, 17 trajectory mới · người lệch phải trên đường | −0,46 m (sang trái) | 66% → 4% | — |
| Huấn luyện, 17 trajectory mới · cảnh sạch | 0,00 m (lệch tuyệt đối 0,09 m) | — | — |
| Kiểm định (cảnh nền mới) · người lệch trái trên đường | +0,27 m | 82% → 44% | 82% → 22% |
| Kiểm định · người lệch phải trên đường | −0,05 m (lệch tuyệt đối 0,58 m) | 88% → 41% | 88% → 26% |
| Kiểm định · cảnh sạch | −0,01 m (lệch tuyệt đối 0,17 m) | — | lệch tuyệt đối 0,70 m |

Tương quan giữa hướng dịch và phía người đứng trên tập kiểm định: −0,21 (`il1`), −0,22 (`grpo_il1`); `sup2` là −0,04, run1 là +0,04.

**So với tiêu chí đặt trước (mục 8):**

| Tiêu chí | `il1` | `grpo_il1` bước 50 |
|---|---|---|
| Chạm ở A1/A2/B1 dưới 30% | Không (54%, 55%, 76%) | A1, A2 đạt (26%, 20%); B1 không (40%) |
| Kịch bản không xung đột không đổi | Không (A5: 2% → 19%) | Không (A5: 2% → 21%) |
| MAOE cảnh sạch tăng không quá 1° | Không, sát ngưỡng (+1,2°) | Không (+3,4°) |
| Lặp lại trên người thật | Yếu (86% → 63%) | Một phần (86% → 48%, có dịch ra xa người) |

Chưa cấu hình nào đạt cả bốn tiêu chí.

**Đánh giá.**

- **Hướng IL trước, GRPO sau cho hành vi khác hẳn GRPO trực tiếp.** `grpo_il1` giảm chạm với người chưa từng thấy trên cảnh nền chưa từng thấy (A2: 92% → 20%, A4: 92% → 9%) mà vẫn đi giữa đường khi không có ai (89%, gốc 90%) và gần như không né cột (80%). run1 của mục 9 đạt con số A2 tương tự bằng cách bỏ hẳn vị trí giữa đường (9%) và né cả cột (13%). Tức là lần này mô hình phản ứng với *người*, không phải rẽ mù.
- **Thêm dữ liệu giúp ít hơn dự đoán ở mục 9.** Số cảnh nền tăng 16 lần nhưng trên bộ thử `il1` chỉ hơn `sup2` 9–16 điểm ở A1/A2/B1 và ngang nhau ở A4. `il1` khớp gần hoàn hảo trên trạng thái đã thấy (chạm 2–4%) nhưng còn 41–44% trên cảnh nền mới: khoảng cách tổng quát hoá vẫn lớn. Tính có điều kiện thì cải thiện rõ (tương quan −0,04 → −0,21). Lưu ý `il1` chỉ gặp mỗi trạng thái khoảng 16 lần (`sup2`: 36 lần) và mỗi cảnh nền mới chỉ có 2 vị trí người.
- **Điểm yếu là chọn phía né.** Trên cảnh mới, khi người lệch phải, quỹ đạo toả ra (lệch tuyệt đối 0,58 m) nhưng dịch trung bình gần 0: mô hình biết có người mà né sang cả hai bên. Đây cũng là lý do A5 xấu đi (2% → 19–21%): một phần quỹ đạo né *về phía* người đứng lệch trái. Trong trang phát lại, quỹ đạo của `grpo_il1` tách thành hai nhánh quanh người, nên đường trung bình (nét đậm) vẫn đi xuyên qua người dù phần lớn mẫu đi vòng (ví dụ A2, cảnh 3: lúc tới gần người nhất chỉ 2/16 mẫu nằm trong vùng chạm); phải đọc theo từng mẫu.
- **GRPO vẫn đánh đổi lấy cảnh sạch và không ổn định.** Lợi ích đạt đỉnh ngay điểm kiểm định đầu tiên (50 bước) rồi mất dần; MAOE cảnh sạch tăng 3,4°; bước đi dài thêm 13% (0,85 → 0,96 m, do số hạng độ dài); quỹ đạo ở cảnh sạch toả rộng hơn (0,70 m). KL 0,04 lại không giữ được hành vi cũ, như ở mục 9.
- **Kết luận cho câu hỏi "thêm dữ liệu rồi tune RL có cho đóng góp thật không".** Có tín hiệu thật nhưng chưa đủ để công bố: hành vi né người có điều kiện xuất hiện và chuyển được sang người/cảnh chưa thấy, nhưng hai tiêu chí về tác dụng phụ (kịch bản không xung đột, cảnh sạch) chưa đạt ở bất kỳ cấu hình nào, và mọi thứ mới là một hạt giống, vòng hở, người ghép.

**Việc tiếp theo có cơ sở nhất** (chưa làm): (1) sửa việc chọn phía: nhiều vị trí người hơn trên mỗi cảnh nền và mục tiêu né nhất quán về một phía khi hai phía gần tương đương, vì quy tắc "phía cần dịch ít hơn" đổi phía theo từng mẫu nhiễu; (2) GRPO với ràng buộc cảnh sạch tường minh (KL về chính sách công bố trên trạng thái sạch, bỏ số hạng độ dài) và dừng sớm theo tập kiểm định; (3) tập kiểm định lấy từ một trajectory không dùng khi huấn luyện, và ít nhất ba hạt giống; (4) IL lâu hơn, vì loss của `il1` còn giảm (0,047 ở bước 3.000, 0,028 ở bước 6.000).

**Chưa làm trong vòng này.** Một hạt giống; một cấu hình cho mỗi pha; tập kiểm định nhỏ (34 cửa sổ, cùng trajectory với bộ thử); không thử mở băng VLM; chỉ Qwen2.5-VL; vẫn vòng hở, không bản đồ vùng đi bộ.

## 11. Vòng ba: né một phía, neo cảnh sạch, tập kiểm định riêng, ba hạt giống (20/11/2026)

Vòng này làm bốn việc đã nêu ở cuối mục 10, không tải thêm dữ liệu. Bộ thử (trajectory 18: 19 cảnh nền × 12 kịch bản, 134 cảnh sạch, người thật) và cách đo giữ nguyên nên so được trực tiếp với mục 9 và 10. Mọi quy tắc chọn checkpoint dưới đây được đặt trước khi có kết quả huấn luyện nào của vòng ba.

**Thay đổi so với vòng hai.**

| Hạng mục | Vòng hai | Vòng ba |
|---|---|---|
| Mục tiêu né của IL | Phía cần dịch ít hơn, tính riêng cho từng mẫu nhiễu | Một phía cho mỗi trạng thái: đi bên phải người, trừ khi người đứng lệch phải hơn 0,15 m so với đường tham chiếu thì đi bên trái (`SN_SIDE=convention`) |
| Vị trí người trên mỗi cảnh nền mới | 2 | 4 (thêm 6.566 trạng thái, `H_more2.npy` 27,7 GB, 0,43 s/trạng thái) |
| Tập kiểm định | 34 cửa sổ của trajectory 18 (cùng trajectory với bộ thử) | Trajectory 8, loại hẳn khỏi huấn luyện: 83 cảnh sạch + 249 trạng thái có người, chỉ dùng 8 người cắt từ chính trajectory 8; các trạng thái huấn luyện có 8 người này cũng bị loại |
| MAOE cảnh sạch khi kiểm định | Mọi cửa sổ | Chỉ cửa sổ người lái đi ≥ 2 m trong 5 s (15/83 cửa sổ của trajectory 8 người lái đứng yên, góc khi đó là nhiễu: 29,1° trên tất cả, 6,8° trên phần có di chuyển) |
| GRPO: mốc KL ở trạng thái sạch | Điểm xuất phát (`il1`), trọng số 0,04 | Chính sách **công bố**, trọng số riêng (`SN_BETA_CLEAN`) |
| GRPO: số hạng độ dài | Như bản công bố (thưởng đi xa hơn người lái) | Chỉ giữ phần phạt khi đi dưới nửa quãng đường của người lái |
| Chọn checkpoint | IL: bước cuối; GRPO: reward kiểm định cao nhất (không qua được ngưỡng) | IL: bước cuối; GRPO: reward kiểm định cao nhất **trong** ngưỡng MAOE cảnh sạch +1° |
| Số hạt giống | 1 | 3 |

Tập huấn luyện: 17.677 trạng thái trên 3.503 cảnh nền (3.503 sạch + 14.174 có người) từ 17 trajectory. 69% mẫu ở trạng thái có người phải đẩy ngang, trong đó 61% sang phải.

**Các lần chạy.**

| Lần chạy | Thiết lập | Chi phí | Tập kiểm định mới: chạm / MAOE cảnh sạch |
|---|---|---|---|
| Chính sách công bố | — | — | 51,2% / 6,78° |
| `il3_s0`, `il3_s1`, `il3_s2` | IL như `il1` với mục tiêu né một phía, lr 1e-5, 6.000 bước, bước cuối | 54–57 phút, đỉnh 11,5 GB | 29,8% / 7,42° · 23,2% / 7,83° · 28,0% / 7,39° |
| `il2` (đối chứng) | Như `il3_s0` nhưng chỉ dùng 2 vị trí người của vòng hai | 54 phút | 30,3% / 7,22° |
| `grpo3_s0_b1`, `grpo3_s1_b1`, `grpo3_s2_b1` | GRPO từ `il3_s*`: KL 0,04 về điểm xuất phát ở trạng thái có người, KL trọng số 1 về chính sách công bố ở trạng thái sạch; lr 1e-6, nhiễu 0,4, lô ghép cặp, 150 bước, kiểm định mỗi 25 bước | 10 phút, 4,0 s/bước, đỉnh 13,2 GB | bước 150: 19,5% / 7,15° · bước 150: 19,1% / 7,01° · bước 125: 20,9% / 7,40° |
| `grpo3_s0_b10` | Như `grpo3_s0_b1` với trọng số neo cảnh sạch 10 | 10 phút | bước 150: 21,8% / 6,91° |

Trọng số neo cảnh sạch được chọn giữa 1 và 10 trên hạt giống 0 theo reward kiểm định (1,041 so với 0,970), rồi dùng giá trị 1 cho hai hạt giống còn lại. Cả ba lần GRPO đều có checkpoint nằm trong ngưỡng, nên không phải dùng tới phương án dự phòng. Khác vòng hai, tập kiểm định không xấu dần: tỉ lệ chạm giữ ở 18–22% từ bước 50 tới 150 và MAOE cảnh sạch quay về bằng hoặc dưới mức của IL.

**Kết quả trên bộ thử** (tỉ lệ quỹ đạo vào vùng chạm 0,45 m; vòng ba ghi trung bình ba hạt giống và khoảng nhỏ nhất–lớn nhất). Hình: `fig_pilot3.png`.

| Kịch bản | Công bố | `il1` (vòng hai) | `grpo_il1` (vòng hai) | IL vòng ba | IL → GRPO vòng ba | `il2` |
|---|---|---|---|---|---|---|
| A1 · S1 giữa đường, 2 m | 95% | 54% | 26% | 41% (36–50) | **26% (23–29)** | 55% |
| A2 · S1 giữa đường, 3 m | 92% | 55% | 20% | 40% (32–53) | **25% (21–28)** | 51% |
| A3 · S1 giữa đường, 5 m | 42% | 21% | 33% | 24% (15–35) | 17% (13–25) | 23% |
| A4 · S2 giữa đường, 3 m | 92% | 40% | 9% | 41% (27–60) | 21% (12–28) | 42% |
| B1 · đi ngược chiều, 3 m | 100% | 76% | 40% | 60% (51–78) | 40% (33–46) | 74% |
| A5 · người lệch trái 0,7 m | 2% | 19% | 21% | 7% (7–8) | 14% (13–14) | 5% |
| A6 · người lệch phải 0,7 m | 20% | 14% | 13% | 19% (14–21) | 15% (12–17) | 26% |
| B2 · người cắt ngang | 0% | 0% | 0% | 0% | 0% | 0% |
| C1 · cột xám giữa đường | 89% | 87% | 80% | 85% (82–91) | 82% (72–87) | 91% |
| C2 · cột xám lệch trái | 2% | 5% | 10% | 3% (3–4) | 2% (1–6) | 3% |
| Cùng vị trí A2 khi không có vật | 90% | 90% | 89% | 90% (89–92) | 92% (90–94) | 91% |
| Cảnh xung đột tránh được ở A2 (trên 18) | 0 | 6 | 15 | 7, 11, 12 | 13, 14, 13 | 8 |
| Cảnh sạch: MAOE (134 cảnh, nhiễu 0) | 12,76° | 13,96° | 16,13° | 14,11° (13,42–14,53) | 15,77° (15,18–16,40) | 14,11° |
| Cảnh sạch: lệch so với chính sách công bố | — | 0,13 m | 0,40 m | 0,16 m | 0,31 m | 0,11 m |
| Người thật, 20 ca chính sách gốc đi vào vùng chạm | 86% | 63% | 48% | 58% (52–64) | 50% (47–54) | 65% |
| Người thật, 41 ca không xung đột | 4% | 10% | 15% | 9% (8–9) | 9% (8–10) | — |

`grpo3_s0_b10` (trọng số neo 10, không được chọn) trên bộ thử: A1 29%, A2 36%, B1 48%, A5 11%, MAOE cảnh sạch 14,81°.

**Mô hình học cái gì (script `23`, tập kiểm định trajectory 8; cột cuối là trạng thái huấn luyện, hạt giống 0).** Tỉ lệ chạm, chính sách gốc → sau huấn luyện:

| Nhóm trạng thái | IL vòng ba (3 hạt giống) | IL → GRPO vòng ba | `il2` | Trên dữ liệu huấn luyện: IL / IL → GRPO |
|---|---|---|---|---|
| Người trên đường, lệch trái | 79% → 27% (20–31), dịch +0,62 m sang phải | 79% → 20% (18–22), +0,74 m | 79% → 46% | 76% → 8% / 2% |
| Người trên đường, lệch phải | 68% → 45% (43–47) | 68% → 31% (28–33) | 68% → 42% | 68% → 20% / 6% |
| Người đứng ngoài đường (lệch > 0,8 m) | 1% → 4% | 1% → 6% | 1% → 0% | 0% → 3% / 6% |
| Cảnh sạch: lệch tuyệt đối so với gốc | 0,18 m | 0,34 m | 0,12 m | 0,15 m / 0,24 m |

Nhóm "lệch phải" trộn hai hướng né theo đúng quy tắc (lệch phải dưới 0,15 m thì vẫn đi bên phải người, lệch nhiều hơn thì đi bên trái), nên độ dịch ngang trung bình của nhóm này gần 0 là điều phải có và không dùng để kết luận; con số có nghĩa là tỉ lệ chạm. Nhóm này khó hơn nhóm lệch trái ở cả mức khớp dữ liệu huấn luyện (20% so với 8%) lẫn tổng quát hoá (45% so với 27%).

**MAOE cảnh sạch tăng ở đâu (script `30`, phân tích làm sau khi thấy kết quả, không thay cho tiêu chí đặt trước).** 134 "cảnh sạch" của bộ thử là cảnh không ghép vật, nhưng 45 cảnh có người thật trong vòng 5 m quanh đường đi. MAOE nhiễu 0, độ:

| Nhóm cửa sổ thử | Công bố | `il1` | `grpo_il1` | IL vòng ba | IL → GRPO vòng ba |
|---|---|---|---|---|---|
| Tất cả (134) — tiêu chí đặt trước | 12,76 | 13,96 | 16,13 | 14,11 | 15,77 |
| Không có người thật gần đường (89) | 14,62 | 14,78 | 15,83 | 14,97 | 15,09 |
| Có người thật gần đường (45) | 9,09 | 12,33 | 16,72 | 12,41 | 17,11 |
| Người lái đi ≥ 2 m, không có người gần (70) | 7,75 | 7,81 | 8,77 | 8,06 | 8,20 |
| Người lái đi ≥ 2 m, có người gần (34) | 5,83 | 10,44 | 14,95 | 10,55 | 15,51 |

Ở cảnh thật sự không có người, GRPO vòng ba chỉ lệch thêm 0,5° (vòng hai: 1,2°); gần như toàn bộ phần tăng nằm ở 45 cảnh có người thật (+8,0°).

**So với tiêu chí đặt trước (mục 8), ba hạt giống:**

| Tiêu chí | IL vòng ba | IL → GRPO vòng ba |
|---|---|---|
| Chạm ở A1/A2/B1 dưới 30% | Không (41%, 40%, 60%; không hạt giống nào đạt) | A1, A2 đạt ở cả ba hạt giống (23–29%, 21–28%); B1 không (33–46%) |
| Kịch bản không xung đột không đổi | Gần đạt (A5: 2% → 7–8%; cột và A6 không đổi) | Không (A5: 2% → 13–14%) |
| MAOE cảnh sạch tăng không quá 1° | Không (+1,35° trung bình; chỉ hạt giống 0 đạt, +0,66°) | Không (+3,0°; từ +2,4° tới +3,6°) |
| Lặp lại trên người thật | Một phần (86% → 58%) | Một phần (86% → 50%; 8–11 trên 20 ca hết xung đột) |

Chưa cấu hình nào đạt cả bốn tiêu chí.

**Đánh giá.**

- **Kết quả né người giữa đường nay lặp lại được.** Ba hạt giống của IL → GRPO cho 23–29% ở A1 và 21–28% ở A2 trên người và cảnh nền chưa thấy (gốc 95% và 92%), ổn định suốt 150 bước thay vì mất sau bước 50. GRPO còn thu hẹp độ lệch giữa các hạt giống: IL dao động 32–53% ở A2, sau GRPO còn 21–28%. Mô hình vẫn đi giữa đường khi không có ai (92%) và không né cột (82%), tức vẫn là phản ứng với người.
- **Quy tắc né một phía sửa được lỗi của IL ở người lệch trái.** A5 giảm từ 19% (vòng hai) xuống 5–8% ở cả bốn lần chạy IL, kể cả `il2` không có dữ liệu thêm, nên hiệu ứng này thuộc về quy tắc chọn phía. Nhưng GRPO làm nó xấu lại một nửa (13–14%): reward người không phân biệt phía, nên GRPO không có lý do giữ quy ước.
- **Không thấy lợi ích của việc thêm vị trí người.** Cùng hạt giống 0, có thêm vị trí (50% / 53% ở A1/A2) gần như bằng không thêm (55% / 51%); hai hạt giống kia tốt hơn hẳn (32–37%). Độ lệch giữa các hạt giống lớn hơn hiệu ứng cần đo, và chỉ có một lần chạy đối chứng, nên chưa kết luận được. IL cũng chưa hội tụ: loss cuối 0,036 (`il2`: 0,027), mỗi trạng thái chỉ được gặp khoảng 11 lần, và tỉ lệ chạm trên tập kiểm định dao động 27–38% giữa các điểm kiểm định của cùng một lần chạy.
- **Neo cảnh sạch có tác dụng ở nơi nó được áp, nhưng tiêu chí cảnh sạch vẫn không đạt.** Ở 89 cảnh thử không có người, MAOE chỉ tăng 0,5°. Phần tăng còn lại đến từ 45 cảnh có người thật. Một phần là điều mong muốn: ở 20 ca chính sách gốc đi vào người, tỉ lệ chạm giảm từ 86% xuống 50%. Một phần là phản ứng thừa: ở 41 người không cản đường, tỉ lệ chạm tăng từ 4% lên 9% (3–4 người từ an toàn thành chạm), khớp với A5 ở trên. Tiêu chí "MAOE cảnh sạch" như đã đặt trộn hai thứ này; nó cần được định nghĩa lại **trước** vòng sau, không phải sửa sau khi thấy số.
- **Ngưỡng bảo vệ trên tập kiểm định không báo trước được bộ thử.** Cả ba hạt giống qua ngưỡng +1° trên trajectory 8 (+0,2° tới +0,6°) nhưng trượt trên bộ thử. Số đo trên tập kiểm định cũ (trajectory 18) thì có báo: MAOE tăng 2,5–2,9° trong lúc chạy GRPO. Tôi chưa đếm số người thật trong các cửa sổ sạch của trajectory 8, nên chưa khẳng định được đó là nguyên nhân.
- **Người đi ngược chiều vẫn là kịch bản yếu nhất** (40%, không đổi so với vòng hai); chỉ 30% vị trí người trong dữ liệu là đi ngược.

**Trả lời câu hỏi "thêm dữ liệu rồi tune RL có cho đóng góp thật không".** Bằng chứng mạnh hơn vòng hai: né người có điều kiện, chuyển được sang người và cảnh chưa thấy, lặp lại qua ba hạt giống, với tác dụng phụ nhỏ hơn (cảnh không người +0,5° thay vì +1,2°; người không cản đường 9% thay vì 15%). Nhưng vẫn chưa đạt bộ tiêu chí đã đặt, và vẫn là vòng hở, người ghép, một trajectory thử.

**Việc tiếp theo có cơ sở nhất** (chưa làm): (1) sửa phép đo trước: tách tiêu chí cảnh sạch thành "cảnh không có người" và "người thật không cản đường", và đưa cả hai vào tập kiểm định; (2) đưa quy ước phía né vào GRPO (số hạng reward cho phía đi, hoặc giữ loss IL làm điều chuẩn trong lúc chạy GRPO) để không mất phần IL đã sửa; (3) IL lâu hơn với lr giảm dần và lấy trung bình trọng số các bước cuối, vì bước cuối đang nhiễu; (4) tăng tỉ lệ người đi ngược chiều; (5) kiểm thử vòng kín.

**Giới hạn của vòng này.** Nhiều yếu tố đổi cùng lúc; chỉ `il2` (một hạt giống) tách quy tắc né khỏi dữ liệu thêm. Trọng số neo cảnh sạch chọn giữa hai giá trị trên một hạt giống. Tập kiểm định mới chỉ có 8 người. Bộ thử vẫn là phần giữ riêng của trajectory 18, cùng con phố với một phần dữ liệu huấn luyện. Phép tách MAOE theo người thật là phân tích sau, dùng bộ phát hiện người và ước lượng vị trí thô. Chỉ Qwen2.5-VL, VLM đóng băng.

## 12. Vòng bốn: tách tiêu chí cảnh sạch, đưa quy ước phía né vào reward GRPO (20/11/2026)

**Tiêu chí và quy tắc, viết trước khi chạy lần huấn luyện nào của vòng bốn.** Các ngưỡng số ở tiêu chí 2 và 3b được chọn sau khi đã thấy số của vòng ba, nên chỉ dùng để chấm vòng bốn; số của vòng ba ghi lại để so, không chấm lại.

| Tiêu chí trên bộ thử (trajectory 18) | Ngưỡng |
|---|---|
| 1. Chạm ở A1, A2, B1 | dưới 30% (như cũ) |
| 2. Kịch bản không xung đột (A5, A6, B2, C2, C3) | tỉ lệ chạm ở mỗi kịch bản tăng không quá 5 điểm phần trăm so với chính sách công bố |
| 3a. Cảnh không có người thật gần đường (89 trên 134 cảnh sạch) | MAOE nhiễu 0 tăng không quá 1° |
| 3b. Người thật mà chính sách công bố không chạm (41 người) | tỉ lệ chạm tăng không quá 5 điểm (4,3% → tối đa 9,3%) |
| 4. Người thật mà chính sách công bố đi vào vùng chạm (20 người) | đạt nếu dưới 30%; "một phần" nếu giảm nhưng chưa dưới 30% |

MAOE trên cả 134 cảnh (tiêu chí 3 cũ) vẫn được ghi để so với các vòng trước nhưng không còn dùng để chấm, vì nó trộn cảnh không người với cảnh có người thật (mục 11). Kết quả tính trên trung bình ba hạt giống, kèm số hạt giống đạt.

Thiết kế: chỉ đổi pha GRPO, xuất phát từ ba checkpoint IL của vòng ba (`il3_s0`–`il3_s2`), nên khác biệt so với vòng ba quy được cho hai thay đổi sau.

- **Reward phía né (`SN_W_SIDE`).** Với mỗi trạng thái có người, phía quy ước được tính đúng như mục tiêu IL vòng ba (đi bên phải người, trừ khi người lệch phải hơn 0,15 m so với đường của chính sách công bố). Mẫu quỹ đạo nào tới gần người dưới 2 m mà đi sai phía thì bị trừ `W_SIDE`. Số hạng này chỉ có trong reward huấn luyện; reward kiểm định giữ định nghĩa vòng ba để so được giữa các cấu hình.
- **Ngưỡng bảo vệ khi chọn checkpoint.** Reward kiểm định cao nhất trong số các điểm kiểm định thoả cả hai: (i) MAOE trên cửa sổ sạch của trajectory 8 có người lái di chuyển và **không có người thật gần đường** tăng không quá 1°; (ii) tỉ lệ chạm với người ghép mà chính sách công bố không chạm tăng không quá 5 điểm. Người thật trên trajectory 8 được dò bằng Mask R-CNN (script `32`): 12 trên 83 cửa sổ sạch có người gần đường, so với 45 trên 134 ở bộ thử. Điều này giải thích vì sao ngưỡng bảo vệ vòng ba, đo trên trajectory 8, không báo trước được MAOE của bộ thử.
- **Đối chứng `g4c`:** `W_SIDE = 0` với ngưỡng bảo vệ mới, ba hạt giống, để tách tác dụng của reward phía né khỏi tác dụng của việc đổi cách chọn checkpoint.
- **Chọn `W_SIDE`:** thử 0,5 và 1,0 trên hạt giống 0, lấy giá trị có reward kiểm định cao hơn trong ngưỡng, rồi chạy giá trị đó cho hai hạt giống còn lại.

**Biến thể 4b, thêm giữa chừng** (sau khi thấy reward phía né không có tác dụng ở hạt giống 0, trước khi có kết quả nào của 4b; tiêu chí và quy tắc chọn checkpoint giữ nguyên). Công thức GRPO chia advantage cho độ lệch chuẩn reward của từng nhóm 16 mẫu. Đo trong lúc huấn luyện: nhóm cảnh sạch có độ lệch chuẩn 0,05–0,08, nhóm có người trung vị 0,10–0,18, trong khi nhóm có mẫu chạm người trên 0,5. Tức là ở trạng thái không có gì cần sửa, chênh lệch reward vài phần trăm vẫn được khuếch đại thành tín hiệu lớn ngang với nhóm xung đột. 4b đặt sàn 0,2 cho độ lệch chuẩn này (`SN_ADV_STD_MIN=0.2`, một giá trị duy nhất, không dò), `W_SIDE = 0`, ba hạt giống (`g4f_s0`–`g4f_s2`).

**Các lần chạy** (150 bước, 14–16 phút mỗi lần, đỉnh 13,2 GB; mọi lần đều có checkpoint nằm trong cả hai ngưỡng bảo vệ). Tập kiểm định trajectory 8, chính sách công bố: chạm 51,2%, MAOE cảnh sạch không người 7,24°, chạm người không xung đột 5,4%.

| Lần chạy | Thiết lập | Bước được chọn | Kiểm định: chạm / MAOE / chạm người không xung đột |
|---|---|---|---|
| `il3_s0`–`il3_s2` (điểm xuất phát) | IL vòng ba | — | 29,8% / 7,56° / 8,0% · 23,2% / 7,81° / 6,0% · 28,0% / 7,48° / 5,8% |
| `g4s05_s0`, `g4s05_s1`, `g4s05_s2` | 4a: reward phía né, `W_SIDE` = 0,5 | 125, 150, 125 | 19,2% / 7,15° / 4,8% · 19,2% / 6,67° / 5,9% · 20,9% / 7,03° / 4,2% |
| `g4s10_s0` | 4a với `W_SIDE` = 1,0 | 125 | 19,2% / 7,16° / 4,7% |
| `g4f_s0`, `g4f_s1`, `g4f_s2` | 4b: sàn chuẩn hoá advantage 0,2 | 150, 150, 150 | 18,6% / 7,20° / 6,0% · 18,6% / 6,59° / 6,1% · 20,1% / 6,86° / 4,2% |

`W_SIDE` được chọn là 0,5 theo quy tắc (reward kiểm định 1,0375 so với 1,0368), dù hai giá trị cho kết quả gần như trùng nhau.

**Lệch so với kế hoạch đã ghi:** đối chứng `g4c` không được chạy riêng. Ba lần GRPO của vòng ba (`grpo3_s*_b1`) có đúng cấu hình huấn luyện của đối chứng (`W_SIDE = 0`, không sàn), và ở hạt giống 0 đường kiểm định của chúng trùng với 4a trong khoảng 0,3 điểm phần trăm ở mọi điểm kiểm định, nên chúng được dùng làm đối chứng. Khác biệt còn lại là cách chọn checkpoint (ngưỡng cũ thay vì ngưỡng mới).

**Kết quả trên bộ thử** (trung bình ba hạt giống, trong ngoặc là khoảng nhỏ nhất–lớn nhất):

| Chỉ số | Công bố | IL vòng ba | GRPO vòng ba (đối chứng) | 4a: reward phía né | 4b: sàn advantage |
|---|---|---|---|---|---|
| A1 · người giữa đường, 2 m | 95% | 41% (36–50) | 26% (23–29) | 26% (23–29) | 25% (24–28) |
| A2 · người giữa đường, 3 m | 92% | 40% (32–53) | 25% (21–28) | 25% (20–29) | 24% (21–26) |
| B1 · đi ngược chiều, 3 m | 100% | 60% (51–78) | 40% (33–46) | 40% (33–46) | 40% (34–45) |
| A3 · người giữa đường, 5 m | 42% | 24% | 17% (13–25) | 16% (12–21) | 13% (10–18) |
| A4 · S2 giữa đường, 3 m | 92% | 41% | 21% (12–28) | 22% (12–31) | 20% (12–27) |
| A5 · người lệch trái | 2% | 7% (7–8) | 14% (13–14) | 12% (11–13) | 14% (13–15) |
| A6 · người lệch phải | 20% | 19% (14–21) | 15% (12–17) | 15% (12–17) | 14% (12–16) |
| B2 · người cắt ngang | 0% | 0% | 0% | 0% | 0% |
| C1 · cột giữa đường | 89% | 85% | 82% (72–87) | 82% (72–87) | 83% (76–87) |
| C2 · cột lệch trái | 2% | 3% | 2% (1–6) | 2% (1–6) | 3% (1–6) |
| C3 · cột lệch phải | 19% | 20% (14–24) | 23% (17–28) | 24% (19–29) | 20% (15–25) |
| 3a · MAOE, 89 cảnh không có người thật | 14,62° | 14,97° | 15,08° | 15,07° | 15,29° |
| 3b · chạm người thật không cản đường (41) | 4,3% | 8,5% (7,5–9,3) | 8,9% (8,1–9,9) | 8,8% (7,9–9,9) | 9,0% (8,1–10,2) |
| 4 · chạm người thật trong xung đột (20) | 86% | 58% (52–64) | 50% (47–54) | 49% (44–54) | 46% (42–54) |
| MAOE trên cả 134 cảnh (tiêu chí cũ, chỉ để so) | 12,76° | 14,11° | 15,77° | 15,69° | 15,93° |
| A5: quỹ đạo đi vòng sai phía người | 0% | 1% | 4% | 3% | 4% |

**Chấm theo tiêu chí đã đặt** (trung bình ba hạt giống; trong ngoặc là số hạt giống đạt):

| Tiêu chí | IL vòng ba | GRPO vòng ba (đối chứng) | 4a | 4b |
|---|---|---|---|---|
| 1. A1, A2, B1 dưới 30% | Không (0/3) | A1, A2 đạt (3/3); B1 không (0/3) | như đối chứng | như đối chứng |
| 2. Kịch bản không xung đột, tăng không quá 5 điểm | Không ở A5 (+5 tới +6; 1/3) | Không ở A5 (+12; 0/3) | Không ở A5 (+10; 0/3) | Không ở A5 (+12; 0/3) |
| 3a. MAOE cảnh không người, không quá +1° | Đạt (+0,35°; 3/3) | Đạt (+0,46°; 3/3) | Đạt (+0,45°; 3/3) | Đạt (+0,67°; 3/3) |
| 3b. Người thật không cản đường, không quá +5 điểm | Đạt (+4,2; 2/3) | Đạt sát ngưỡng (+4,6; 2/3) | Đạt sát ngưỡng (+4,5; 2/3) | Đạt sát ngưỡng (+4,7; 2/3) |
| 4. Người thật trong xung đột dưới 30% | Một phần (58%) | Một phần (50%) | Một phần (49%) | Một phần (46%) |

Chưa cấu hình nào đạt cả năm tiêu chí. Ở tiêu chí 2, hạt giống 2 của cả ba cấu hình GRPO còn vượt ngưỡng ở C3 (cột lệch phải, +6 tới +10 điểm).

**Hai thay đổi của vòng bốn đều không có tác dụng.** 4a và 4b cho số trùng với đối chứng ở mọi chỉ số, trong khoảng dao động giữa các hạt giống. Lý do đo được:

- *Reward phía né.* Trong lúc huấn luyện chỉ khoảng 5% mẫu đi sai phía, tức trên dữ liệu đã thấy mô hình vốn chọn đúng phía; số hạng phạt gần như không có gì để sửa.
- *Sàn advantage.* Giảm tín hiệu của nhóm cảnh sạch khoảng bốn lần không làm đổi độ lệch ở cảnh sạch (0,29 m so với 0,31 m). Độ lệch ấy vì thế không sinh ra từ các nhóm cảnh sạch, mà là phản ứng với người thật trong cảnh (mục 11).

**Lỗi ở A5 nằm ở vài cảnh nền cụ thể.** Tách 19 cảnh nền của bộ thử:

| Nhóm cảnh nền | Công bố | IL vòng ba | GRPO vòng ba | 4b |
|---|---|---|---|---|
| 12 cảnh cũ | 1% | 2–3% | 1–6% | 1–6% |
| 7 cảnh mới (phần cuối trajectory 18) | 4% | 15–17% | 28–36% | 29–37% |

Trong 7 cảnh mới, lỗi dồn vào 2–3 cảnh mỗi hạt giống (số của GRPO vòng ba): cảnh 452 chạm 100% ở cả ba hạt giống, các cảnh 409, 483, 504 chạm 44–100% ở hạt giống này và 0–6% ở hạt giống khác; ba cảnh còn lại 0%. Ở cảnh hỏng, cả chùm quỹ đạo dịch 0,2–0,7 m **về phía** người ghép so với khi không ghép người, và ở cảnh 452 có 50–88% mẫu đi vòng sai phía. Vậy đây là né sai hướng trong vài cảnh chưa thấy, không phải quỹ đạo toả rộng. Con số "4% đi sai phía" ở bảng trên là trung bình trên 19 cảnh nên che mất điều này.

**Kiểm giả thuyết "do người điều khiển robot" (21/11/2026, script `33`, `34`).** Ở ba cảnh hỏng nhiều nhất (452, 409, 504), người điều khiển robot đứng rất gần và chiếm nửa phải hoặc giữa khung hình; giả thuyết là mô hình né người thật này sang trái, đúng chỗ người ghép đứng. Ba phép thử, không huấn luyện thêm, dùng `il3_s*` và `grpo3_s*_b1`:

*(A) Đối chiếu trên 19 cảnh của bộ thử.* Người thật chiếm 5–20% diện tích khung ở 18/19 cảnh (tôi không kiểm từng cảnh xem đó có phải người điều khiển không). Diện tích này không tương quan với tỉ lệ chạm ở A5: Spearman +0,01 (p = 0,96) với GRPO, +0,21 (p = 0,38) với IL. Ba cảnh trong nhóm có nhiều diện tích người thật nhất (190, 242, 361) không hỏng ở hạt giống nào.

*(B) Cảnh nền không có người thật.* Trên trajectory 8 chỉ có 5 cửa sổ mà Mask R-CNN không thấy ai trong cả 5 khung và người lái đang đi. Ghép một người "giống người điều khiển" sát camera bên phải (0,35 m bên phải, 0,8 m phía trước, cùng vị trí ở cả 5 khung) làm quỹ đạo của GRPO dịch sang trái ở hai trên ba hạt giống (vị trí ngang lúc 3 s: −0,08 → −0,26 m, −0,04 → −0,06 m, −0,07 → −0,28 m; chính sách công bố không đổi). Vậy cơ chế "người sát bên phải đẩy quỹ đạo sang trái" có tồn tại. Nhưng khi có thêm người lệch trái thì tỉ lệ chạm người đó chỉ tăng ở một hạt giống (40% → 41%, 16% → 21%, 20% → 42%). Năm cửa sổ là quá ít, và ở chính các cửa sổ này quỹ đạo vốn đã đi qua vị trí bên trái khi chưa ghép ai (12–31%), nên phép thử này không kết luận được.

*(C) Xoá người thật khỏi 19 cảnh thử* (mặt nạ Mask R-CNN, inpainting OpenCV, cả 5 khung), kèm **đối chứng giả**: giữ nguyên người thật, xoá một vùng cùng kích thước ở phía đối diện. Tỉ lệ chạm ở A5, ba hạt giống:

| | Ảnh gốc | Xoá người thật | Đối chứng giả (người thật còn nguyên) |
|---|---|---|---|
| Chính sách công bố | 2% | 3% | 3% |
| IL vòng ba | 7% / 8% / 7% | 7% / 6% / 1% | 2% / 3% / 2% |
| IL → GRPO vòng ba | 14% / 13% / 14% | 4% / 5% / 5% | 2% / 6% / 11% |
| GRPO: số cặp (cảnh, hạt giống) chạm từ 40% trở lên, trên 57 | 9 | 3 | 3 |
| GRPO, cảnh 452 | 100% / 100% / 100% | 0% / 0% / 0% | 0% / 0% / 0% |

Xoá người thật làm lỗi giảm, nhưng đối chứng giả, nơi người điều khiển còn nguyên, làm lỗi giảm ngang như vậy; cảnh 452 hết hỏng ở cả hai cách. Cả hai cách cũng tạo ra lỗi mới ở cảnh khác (xoá người: cảnh 504 ở một hạt giống từ 0% lên 69%; đối chứng giả: cảnh 393 từ 0% lên 100%, cảnh 160 từ 0% lên 69%).

**Kết luận của phép kiểm:** giả thuyết người điều khiển không được ủng hộ. Việc lỗi biến mất không phụ thuộc vào chuyện người điều khiển còn hay mất, mà xảy ra với bất kỳ thay đổi đáng kể nào trên ảnh. Lỗi ở A5 là hành vi dễ vỡ, gắn với từng ảnh và từng hạt giống: sửa ảnh thì cảnh này hết hỏng, cảnh khác lại hỏng. Phép kiểm cũng cho thấy giới hạn của phép đo: với 19 cảnh nền, mỗi cảnh nặng 5,3 điểm phần trăm, nên mức 12–14% ở A5 là 2–3 cảnh hỏng, và ngưỡng "+5 điểm" của tiêu chí 2 tương đương đúng một cảnh. Tôi chưa giải thích được vì sao cả hai cách sửa ảnh đều làm số cảnh hỏng giảm (9 → 3) thay vì chỉ xáo trộn chúng.

**Đánh giá.**

- **Việc tách tiêu chí cảnh sạch là kết quả chính của vòng này.** Đo đúng thì cảnh không có người đạt ở mọi hạt giống (+0,5° tới +0,7°), và người thật không cản đường đạt sát ngưỡng (2/3 hạt giống). Con số +3° của tiêu chí cũ đến từ phản ứng với người thật trong cảnh. Ngưỡng bảo vệ vòng ba không báo trước được bộ thử vì trajectory kiểm định ít người thật hơn hẳn (12/83 so với 45/134 cửa sổ).
- **Chỉnh reward trong pha GRPO không sửa được phần còn lại.** Ba cấu hình GRPO khác nhau cho cùng một kết quả. Trên dữ liệu huấn luyện mô hình đã né đúng (chạm 2–6%); các lỗi còn lại xuất hiện ở cảnh chưa thấy, tức là lỗi tổng quát hoá của việc xác định người ở đâu, và reward tính trên trạng thái đã khớp không chạm tới được.
- **Hai lỗi còn lại:** người đi ngược chiều (40%, không đổi qua ba vòng) và A5 trên vài cảnh nền (2% → 12–14%).

**Việc tiếp theo có cơ sở nhất** (chưa làm; đã cập nhật sau phép kiểm ngày 21/11): (1) mở rộng bộ thử trước khi sửa mô hình: nhiều cảnh nền hơn và từ nhiều trajectory hơn, để một cảnh hỏng không quyết định tiêu chí, kèm khoảng tin cậy theo cảnh nền; (2) tăng độ bền với thay đổi ảnh trong pha IL (nhiều cảnh nền hơn, biến đổi ảnh), vì lỗi còn lại là hành vi dễ vỡ theo từng ảnh; (3) tăng tỉ lệ người đi ngược chiều trong dữ liệu; (4) kiểm thử vòng kín. Cả (1)–(3) cần lưu thêm trạng thái ẩn, trong khi ổ C chỉ còn khoảng 12 GB.

**Giới hạn của vòng này.** Ngưỡng số của tiêu chí 2 và 3b đặt sau khi đã thấy vòng ba. 4b được thêm giữa chừng. Đối chứng dùng lại các lần chạy vòng ba với cách chọn checkpoint cũ. `W_SIDE` chỉ thử hai giá trị, sàn advantage một giá trị, cùng 150 bước và lr như vòng ba; reward phía né chỉ xét việc đi vòng qua người, không xét việc dịch về phía người. Phân tích theo cảnh nền làm sau khi thấy kết quả. Phép kiểm giả thuyết người điều khiển dùng ảnh inpainting (để lại vết nhoè lớn, 1–23% khung hình), 16 mẫu cùng nhiễu cho mỗi cảnh, và chỉ 5 cửa sổ không có người thật ở phần (B).

## 13. Kiểm tra cách thêm người và chi tiết phần VLM (24/11/2026)

Làm theo yêu cầu của co-author: xác nhận từ mã cách thêm người vào ảnh, và viết chi tiết hơn phần VLM. Bản đầy đủ nằm ở tài liệu trực tuyến (mục 2.3 và 2.4); mục này ghi phần mới so với mục 1–12.

**Xác nhận từ mã** (`stimuli.py`, `04_detect_persons.py`, `07_run_pedestrian.py`, `20`, `26`, `27`, `28`):

- Người ghép là hình cắt người thật (Mask R-CNN, alpha làm mờ Gaussian 0,8 px), không có mô phỏng. Thư viện 115 hình: 13 từ trajectory 18, 102 từ 17 trajectory mới; hình S1, S2 của bộ thử không dùng khi huấn luyện.
- Đặt vị trí bằng mô hình mặt phẳng đất danh định: F = 185 px, camera cao 0,60 m, chân trời hàng 200, người cao 1,70 m. Hình cắt được co bằng LANCZOS với hệ số 0,16–0,54.
- Cùng một hình cắt ở cả 5 khung (không có dáng bước); robot trong 5 khung lịch sử được giả định đi thẳng với bước `step_scale`.
- Vị trí ngẫu nhiên khi huấn luyện: 70% đứng yên và 30% đi ngược chiều; lệch ngang ±0,5 m (60%) hoặc ±1,2 m (40%); khoảng cách 1,5–5,5 m; vận tốc người 0,6–1,4 m/s; 7, 2 và 4 vị trí người mỗi cửa sổ nền ở vòng 1, 2 và 3.

**Số đo mới** (`35_insertion_check.py` → `results_35_insertion_check.json`):

| Phép kiểm | Kết quả |
|---|---|
| Bộ dò độc lập thấy người ghép (IoU > 0,3) | 12/12 cửa sổ ở cả 8 điều kiện có người; 0/12 ở 3 điều kiện cột xám |
| Số người bộ dò thấy mỗi khung | 1,67 (sạch) so với 2,50–2,83 (có người ghép) |
| Mô hình camera danh định so với 215 người thật toàn thân trong 539 khung: chiều cao ảnh / (hàng chân − 200), mô hình cho 2,83 | Trung vị 2,47 (chân tương ứng 0–1,5 m), 2,32 (1,5–3 m), 2,01 (3–6 m), 2,02 (trên 6 m): người ghép cao hơn người thật ở cùng hàng chân 15%, 22%, 41%, 40% |
| Khoảng cách suy từ chiều cao so với từ hàng chân, người ở 1,5–6 m | Trung vị 1,28 lần (10–90%: 0,74–2,54) |

Hệ quả: ước lượng "sai số khoảng cách khoảng ±30%" (mục 2) cùng bậc độ lớn với số đo, nhưng ở 3–6 m số đo lên tới khoảng 40%. Chưa biết tín hiệu nào đúng (nguyên nhân có thể: chân trời không ở hàng 200, camera nghiêng, méo ống kính góc rộng, chiều cao người). Phép so cặp có/không người và trái/phải không phụ thuộc hiệu chuẩn; các mốc 2 m, 3 m, 5 m và ngưỡng 0,45 m thì có.

**Phần VLM, thêm so với mục 3 và 5** (số từ `results_08`, `results_09`, `results_16`; không có phép đo VLM mới):

- Khung có người thật (30 khung) và khung sạch (16 khung): mô tả tự do nhắc tới người ở 28/30 và 0/16; VQA có định hướng trả lời "yes" ở 16/30 và 0/16.
- Quyết định trong CoT: không phân tích được ở 7/12 cửa sổ nền sạch và 3–4/12 ở điều kiện khác; "Move Forward-Left" nhiều nhất kể cả với cột xám (5/12), nên không riêng cho người.
- Polygon vùng đi được: 12/12 đúng định dạng, chân người ghép nằm trong polygon ở 11/12, IoU giữa polygon có và không có người là 0,79–0,81.
- Các câu hỏi VQA, mô tả và polygon chỉ dùng khung hiện tại, còn Action Expert nhận 5 khung: phép dò trên Z mới đo đúng đường đi của Action Expert.

## Giới hạn

Một trajectory, một miền ảnh (camera thấp, góc rộng, 400×400); người điều khiển robot xuất hiện ở mép phải nhiều khung nền và 3/12 cửa sổ nền đã có người thật ở xa. Người chèn là ảnh ghép, không có bóng đổ. Open-loop: không có phản hồi vật lý, không có `M_occ`. Phần Brain chỉ đo trên Qwen2.5-VL. Prompt nhánh action và prompt CoT đều là dựng lại. Phép dò ở mục 5 chỉ cho biết thông tin giải mã tuyến tính được, không chứng minh nó đủ chính xác để né; nó dùng hai người mẫu ghép vào ảnh và 30 cửa sổ nền.

## Tệp

Vòng bốn (mục 12): `32_detect_valx.py` và `results_32_valx_persons.json` (người thật trên trajectory 8), `pilot_common.py` (thêm `convention_side`, `wrong_side`, `near_people`), script `21` (thêm `SN_W_SIDE`, `SN_VAL_NOPERSON`, `SN_NC_GUARD`, `SN_ADV_STD_MIN`, `SN_SLIM`) và `22` (thêm MAOE cảnh không người, người thật không xung đột, tỉ lệ đi sai phía; các file `results_22_pilot_eval_*` của vòng ba đã được tính lại, số cũ không đổi), `run_round4_grpo.ps1`; kết quả `results_21_train_g4*.json`, `results_22_*_g4*.json`, `results_23_diagnose_g4*_valx.json`. Checkpoint ở `C:\Users\N4G\qcom_socialnav\pilot`: `g4s05_s0`–`g4s05_s2`, `g4s10_s0`, `g4f_s0`–`g4f_s2` (`best.pt`, 1,2 GB mỗi cái). Vòng này không thêm hình và không thêm checkpoint vào trang phát lại, vì hành vi trùng với vòng ba. Phép kiểm giả thuyết người điều khiển: `33_operator_correlation.py` → `results_33_operator_correlation.json`; `34_operator_causal.py` → `results_34_operator_causal.json`, `fig_34_stimuli.png` (ảnh kích thích: người ghép trên cảnh không người, cảnh 452 và 504 trước và sau khi xoá người).

Vòng ba (mục 11): script `21`, `24`, `27`, `pilot_common.py` (thêm các biến `SN_ROUND`, `SN_SIDE`, `SN_BETA_CLEAN`, `SN_LEN_MODE`, `SN_VAL_SPLIT`, `SN_MOVE_MIN`), `30_clean_breakdown.py`, `31_pilot3_figure.py`, hai script chạy chuỗi `run_round3_il.ps1` và `run_round3_grpo.ps1`; kết quả `results_24_supervised_il3_s*.json`, `results_24_supervised_il2.json`, `results_21_train_grpo3_*.json`, `results_22_*_il3_s*_last.json`, `results_22_*_il2_last.json`, `results_22_*_grpo3_*.json`, `results_23_diagnose_*_valx.json`, `results_30_clean_breakdown.json`, hình `fig_pilot3.png`. Ngoài OneDrive (`C:\Users\N4G\qcom_socialnav\pilot`, nay khoảng 98 GB): `H_more2.npy` (27,7 GB), `H_valx.npy` (1,1 GB), checkpoint `il3_s0`–`il3_s2\last.pt`, `il2\last.pt`, `grpo3_s*_b1\best.pt` và `best_any.pt`, `grpo3_s0_b10\best.pt` và `best_any.pt` (1,2 GB mỗi cái). Trang phát lại nay có tám checkpoint (vòng ba: hạt giống 0 của IL và của IL → GRPO).

Vòng hai (mục 10): script `26`–`29`, `pilot_sprites/` (115 mẫu `keep_*.png`, bảng `_kept.png`), kết quả `results_24_supervised_il1.json`, `results_21_train_grpo_il1.json`, `results_22_*_il1_last.json`, `results_22_*_grpo_il1_best_any.json`, `results_23_diagnose_il1_*`, `results_23_diagnose_grpo_il1_*`, hình `fig_pilot2.png`. Ngoài OneDrive: 18 file nén trajectory và khung đã giải nén ở `C:\Users\N4G\qcom_socialnav\data\citywalker_obs`, trạng thái ẩn `pilot\H_more.npy` (42,7 GB), checkpoint `pilot\il1\last.pt`, `pilot\il1\best_any.pt`, `pilot\grpo_il1\best_any.pt`, `pilot\grpo_il1\last.pt` (1,2 GB mỗi cái). Trang phát lại nay có sáu checkpoint.

Thử nghiệm mục 9: script `20`–`25`, `pilot_common.py`, `pilot_sprites.json`/`.png`, kết quả `results_21_*`, `results_22_*`, `results_23_*`, `results_24_*`, log `log_2*.txt`, hình `fig_pilot.png`; trạng thái ẩn lưu sẵn (9 GB) và hai checkpoint nhánh action (1,2 GB mỗi cái: `run1\best_step150.pt`, `sup2\last.pt`) nằm ở `C:\Users\N4G\qcom_socialnav\pilot`. Trang phát lại có thêm hai checkpoint này.

Script `00`–`19`, `stimuli.py`, `sn_common.py`, `case_viewer_template.html`, trang phát lại `mo_phong_cac_case.html` (đặc trưng của phép dò, khoảng 80 MB, nằm ở `C:\Users\N4G\qcom_socialnav\probe`); kết quả `results_*.json` (hậu tố `_fixed` = trọng số đã khôi phục, `_qwen2_vl` = backbone Qwen2-VL); `log_*.txt`; ảnh mẫu trong `stimuli_examples/`. Chọn backbone bằng biến môi trường `SN_BACKBONE`, tắt bước khôi phục bằng `SN_FIX_LOAD=0`. Trọng số, repo, môi trường Python và khung hình nằm ngoài OneDrive tại `C:\Users\N4G\qcom_socialnav` (khoảng 25 GB).
