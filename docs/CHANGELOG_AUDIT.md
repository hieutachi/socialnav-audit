# Audit changelog (scripts 41–48)

| Script | Vai trò / kết luận cần đọc |
|---|---|
| 41 | Khởi đầu benchmark pedestrian/Penn-Fudan; in-sample, kiểm tra pipeline đầu tiên. |
| 43 | Candidate rerank baseline; biến thể dùng true future velocity là oracle upper bound, không deployment. |
| 44 | Point/expected/CVaR và perturbation audit; kết quả in-sample. |
| 45 | Held-out futures v1; giữ làm lịch sử, protocol được cập nhật ở 46. |
| 46 | Held-out v2; phát hiện đảo chiều theo policy giữa published và `il3_s0`, làm cơ sở protocol 48. |
| 47 | Moving-pedestrian counterfactual dùng oracle velocity; debug-only, không dùng làm evidence. |
| 47b | Schema-2 diversity audit; `div_pw` là candidate diversity, không forecast accuracy. |
| 48 | Guided candidate pool + rerank với held-out futures. Đây là baseline repair/matched held-out cell, chưa phải phương pháp mới; compute chưa matched. |

Các JSON/NPZ tương ứng ở `results/`; bảng tổng hợp tái tạo bằng `python code/reproduce_tables.py`. Không xem thứ tự script là bằng chứng nhân quả hoặc so sánh benchmark trực tiếp với SocialNav paper.
