# Planner source guide (selected upstream snapshot)

## Mục tiêu và giới hạn

Tài liệu này chỉ dẫn Co-Author đọc phần SocialNav liên quan tới action/Flow Matching, training và reward, sau đó nối sang candidate ranking của audit. `code/socialnav_core/` là **snapshot có chọn lọc từ upstream SocialNav**, không phải toàn bộ repository upstream và không đảm bảo chạy độc lập. License upstream Apache-2.0 được giữ tại `code/socialnav_core/LICENSE`. Hãy tham chiếu upstream tương ứng cho provenance/version/complete environment; source snapshot không chứa toàn bộ dependency source, checkpoints hay dữ liệu.

## Read order

1. `src/params.py` — xem `TrainingArguments` và `GRPOArguments`: `action_dim=2`, `action_chunk=5`, `flow_matching_policy`, `num_flow_steps=3`, `action_former`, `query_action_layer`, `sigma`, freeze flags, learning rates, data paths. Default là default, không tự động chứng minh đúng với một checkpoint/run cụ thể.
2. `src/train/train_grpo.py` — xem `configure_flow_matching`, config/model init, truyền action/flow parameters và khởi tạo trainer. Đây là nối giữa CLI/config và model.
3. `src/model/modeling_cls.py` — model wrapper Qwen2-VL/Qwen2.5-VL, output pooling/head. Khi lần tới action expert, theo references được wiring từ training/model code upstream thay vì suy từ tên `modeling_cls`.
4. `src/train/sde_with_logprob.py` — `ConditionalFlowMatcherWithSigmaSchedule` và `_CPS`, timestep schedule, stochastic step, log-prob. Hai biến thể có phương trình cập nhật khác nhau; `noise_level` ảnh hưởng độ nhiễu.
5. `src/trainer/grpo_trainer.py` — rollout/completion, rewards và policy update.
6. `src/train/orms.py`, `reward_funcs.py` — reward/occupancy/smoothness/similarity, coordinate conversions. Đây là training reward, không đồng nhất với offline held-out contact `hel` hoặc rerank score.
7. `src/dataset/grpo_dataset.py`, `src/dataset/data_utils.py` — input record path/format/preprocess.
8. `utils/citywalker.py` — inference và waypoint metrics (L1, arrival, angle); đây không phải closed-loop benchmark score.
9. Trở lại root `code/43...48...` để xem candidate pools, selectors và audit metrics.

## Khái niệm config

| Trường | Diễn giải để đọc code |
|---|---|
| `action_dim` | số chiều vector hành động/waypoint (thường x,y). |
| `action_chunk` | số điểm trong một chunk. |
| `flow_matching_policy` | bật/tắt nhánh flow matching ở config/model. |
| `num_flow_steps` | số bước integration/sampling theo config. |
| `action_former` / `query_action_layer` | bật và cấu hình phần attention/query cho hành động. |
| `sigma`, `noise_level` | tham số nhiễu sampler; kiểm tra đường truyền trong run cụ thể. |
| `freeze_flow_matching`, `freeze_action_former`, `freeze_input_wp_encoder` | khóa gradient module tương ứng. |
| `min_value`, `max_value` | giới hạn giá trị waypoint/action khi nhánh áp dụng. |
| `flow_matching_lr`, `action_former_lr` | learning rates riêng nếu trainer tách param groups. |

## Ranking của audit: tách bạch với training

Trong scripts audit, candidate được sinh/chọn rồi đánh giá bằng energy selectors: `pt` (point/nominal), `exp` (expected energy qua planning futures), `cvar` (tail risk), `gui` (guided pool) và `guided_rr_*` (guided pool + selector). `hel_*` đo contact trên held-out futures. Đây là offline open-loop audit cụ thể, không phải phần GRPO model loss/reward.

- `43_rerank_baseline.py`: rerank candidates; `rerank16g` dùng true future velocity, nên oracle-advantaged upper bound, không deployment number.
- `44_uncertainty_planning.py`: so sánh selector in-sample/perturbation.
- `46_heldout_v2.py`: held-out future evaluation protocol.
- `47b_diversity_audit.py`: diversity; `div_pw` là pairwise diversity, không forecast accuracy.
- `48_matched_guided_rerank.py`: guided candidates và rerank trên held-out futures; baseline repair, không method mới; guidance compute không ngang rerank.

## Kết quả/đường dẫn kiểm chứng

- Main JSON: `results/results_48_guided_rerank.json` → `res.il3_s0.per_cond.ped_headon_3m`.
- Raw arrays: `results/raw_48_guided_rerank.npz`; held-out `hel` arrays có dạng `(32, 608)` cho policy, state batch run đầy đủ; 380 states được lọc cho head-on metric.
- Manifest: `results/manifest_48_guided_rerank.json`.
- Tái tạo mọi bảng JSON: `python code/reproduce_tables.py`.

Liên hệ kết quả không nên vượt quá thiết kế: 380 head-on states/306 sprites/19 windows; CI sprite-conditional, fixed windows; 32 futures không tương đương 32 scene độc lập. Contact proxy và hình học chưa phải social safety/closed-loop.

## Re-run nguyên trạng

Các helper audit tham chiếu `C:\Users\N4G\qcom_socialnav` qua `code/sn_common.py`, bao gồm repo, checkpoint và dữ liệu local; những file không có trong repo. Hãy dùng đường dẫn này như provenance nội bộ, không yêu cầu người đọc tự có quyền truy cập. Reproduce tables là đường kiểm tra không cần GPU/model; rerun đầy đủ phải có dữ liệu/checkpoint/quyền truy cập và dependency đúng.
