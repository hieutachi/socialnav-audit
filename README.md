# SocialNav Audit: source snapshot and results for Co-Authors

## Tóm tắt điều hành

Repo này bàn giao phần audit SocialNav open-loop và một **source snapshot chọn lọc** của upstream SocialNav để mentor theo được planner: config → model/flow sampling → GRPO/rewards → inference/metric; phần reranking/evaluation của audit được lưu trong `code/` và dữ liệu số trong `results/`.

- Câu hỏi nghiên cứu hiện tại: risk-aware candidate ranking/guidance có giảm proxy contact với pedestrian tương lai tổng hợp trong open-loop không, và trade-off hình học là gì?
- Kết quả chính script 48: ở `il3_s0 / ped_headon_3m`, `guided_rr_exp` 4.71% so với `exp` 30.30%; contrast −25.58 pp, sprite-conditional CI [−27.30, −23.90]. Lateral deviation và góc ngoặt lớn hơn rõ; cần constraint/closed-loop trước khi nói “socially better”.
- Kết luận: **GO có điều kiện** cho phase feasibility có thêm pedestrian motion, evaluation closed-loop và constraint comfort/progress; chưa GO cho claim novelty/safety hoặc “beats SocNav paper”. Script 48 sửa baseline và bổ sung held-out cell, **không phải phương pháp mới**, compute không matched.

## 1. Tôi đang làm gì?

Kiểm tra lựa chọn quỹ đạo cho robot navigation có người đi bộ: so sánh các policy point/expected/CVaR, guided candidates và guided rerank dựa trên contact proxy qua nhiều future samples. Đây hiện là thí nghiệm **open-loop** với future pedestrian tổng hợp, không phải robot đóng vòng tương tác người thật.

Phạm vi của bộ kết quả: chỉ 1/5 condition có pedestrian motion (`ped_headon_3m`); script 48 dùng 380 head-on states, 306 sprites, 19 windows; cả run xử lý 608/1,900 state theo whole batches.

## 2. Làm như thế nào?

1. Lấy image/state và candidate pool từ SocialNav pipeline.
2. So sánh các selector: `pt`, `exp`, `cvar`; các baseline `ung`, `gui`; và `guided_rr_pt/exp/cvar`.
3. Chọn trajectory theo planning score; protocol 48 dùng 16 candidates, 8 planning futures, 32 held-out futures.
4. Đo contact proxy `hel` và geometry; 32 futures/state không phải 32 scene độc lập.
5. Tái tạo bảng từ result JSON bằng Python stdlib: `python code/reproduce_tables.py`.

### Code path liên quan

- `code/socialnav_core/src/params.py`: config (action chunk/dim, flow steps, policy toggles, freeze, sigma).
- `code/socialnav_core/src/train/train_grpo.py` + `src/trainer/grpo_trainer.py`: khởi tạo và training GRPO.
- `code/socialnav_core/src/model/modeling_cls.py`: model wrapper.
- `code/socialnav_core/src/train/sde_with_logprob.py`: custom flow/SDE sampling và log-prob; không nên gọi chung là vanilla DDPM.
- `code/socialnav_core/src/train/orms.py`: rewards (occupancy/smoothness/expert similarity etc.); khác với offline risk ranking của audit.
- `code/socialnav_core/src/dataset/`: GRPO input/preprocessing.
- `code/socialnav_core/utils/citywalker.py`: inference và waypoint/arrival/angle metric.
- `code/43_rerank_baseline.py` → `44_uncertainty_planning.py` → `46_heldout_v2.py` → `47b_diversity_audit.py` → `48_matched_guided_rerank.py`: audit evolution.

**Lưu ý snapshot:** upstream module được sao chép có chọn lọc để đọc hiểu, giữ license trong `code/socialnav_core/LICENSE`. Không bao gồm toàn bộ fork/upstream Transformers, checkpoint, cache, dataset hoặc config bí mật. README upstream và môi trường cần xem ở upstream SocialNav gốc; source snapshot ở đây không tự nhận là fork độc lập hoàn chỉnh.

## 3. Kết quả hiện tại

Held-out contact (%), head-on 3m:

| Arm | published | il3_s0 |
|---|---:|---:|
| ung | 74.1 | 53.4 |
| pt | 70.3 | 29.1 |
| exp | 66.1 | 30.3 |
| cvar | 67.5 | 31.7 |
| gui | 62.1 | 27.2 |
| guided_rr_pt | 45.8 | 4.3 |
| guided_rr_exp | 47.1 | 4.7 |
| guided_rr_cvar | 49.1 | 3.6 |

Primary `il3_s0`: `guided_rr_exp − exp = −25.58 pp [−27.30, −23.90]` (CI bootstrap sprite-conditional; fixed 19 windows). Đây là contact proxy; không phải safety rate.

Geometry trade-off `il3_s0`: `exp → guided_rr_exp`, lateral max vs pool-mean 0.66→1.36 m; origin-inclusive max turn 18.8→34.0°; endpoint norm/driver 0.919→0.945; clearance mean 0.742→1.275 m. Đây là lý do chưa kết luận policy guided tốt hơn về social behavior tổng thể.

## 4. Đánh giá và bước cần làm

**GO conditional:** chuyển sang một phase feasibility rõ giả thuyết, thêm nhiều cảnh pedestrian chuyển động (đặc biệt crossing/overtaking/crowd), giữ test set theo trajectory/scene tách biệt, so sánh với baseline cùng compute, rồi chạy closed-loop và chấm collision, comfort, path length, goal success, yielding/social compliance. Ràng buộc lateral deviation/turn/progress để tránh giảm contact bằng cách đi vòng quá xa.

**NO-GO cho các phát biểu:** “48 là proposed method mới”; “4.7% chứng minh an toàn”; “đã thắng SocNav paper”; “held-out scenes”; “compute matched”. Không so con số open-loop này trực tiếp với SR/DCR/SPL/MAOE paper.

## 5. Kiểm kê dữ liệu và quyền truy cập

Bản kiểm kê local: 18 CityWalker trajectories, 18,551 ảnh; 42,714 pose lines = 14,238 records. Cache metadata train 19,205 + Penn-Fudan 1,900 = 21,105 samples; tổng cache khoảng 83.08 GB. Những raw data/cache/checkpoint này không được commit; kết quả cần đọc qua JSON/NPZ đã lọc và HTML reports. Chi tiết: [dataset inventory](reports/DATASET_KIEM_KE_THEO_SOCNAV_2026-11-26.html).

## 6. Bằng chứng trong repo

| Nội dung | Đường dẫn |
|---|---|
| Bảng kết quả 41, 43, 44, 46, 47b, 48 | `results/*.json` |
| Raw held-out arrays script 48 | `results/raw_48_guided_rerank.npz` |
| Protocol manifest | `results/manifest_48_guided_rerank.json` |
| Bản đọc chi tiết | `reports/BAN_DOC_CHI_TIET_PHUONG_PHAP_VA_GO_NOGO_2026-11-26.html` |
| Dataset inventory | `reports/DATASET_KIEM_KE_THEO_SOCNAV_2026-11-26.html` |
| Co-Author slides | `reports/presentation_coauthor_meeting.html` |
| Council slides | `reports/report_presentation_council.html` |
| Full technical log | `docs/BAO_CAO_OPENLOOP.md` |

**Reproduce JSON tables:** `python code/reproduce_tables.py` — không cần model/GPU/network. Full rerun của original scripts cần môi trường/dataset/checkpoint riêng (paths hiện hardcode trong `code/sn_common.py`); repo không chứa những file đó.

## 7. Liên kết tài liệu

- [Project Knowledge Base](docs/PROJECT_KNOWLEDGE_BASE.md)
- [Planner Source Guide](docs/PLANNER_SOURCE_GUIDE.md)
- [Experiment 48 protocol](docs/EXPERIMENT_48.md)
- [Glossary](docs/GLOSSARY.md)
- [Audit changelog](docs/CHANGELOG_AUDIT.md)
- [Kết quả tái tạo](code/reproduce_tables.py)
