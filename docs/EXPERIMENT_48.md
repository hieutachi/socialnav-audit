# Experiment 48 protocol and result guide

## Design

- Base protocol: script 46 held-out v2 (same state order/batch grid/seeds/sampler/energy reference).
- Candidate pool: 16 candidates; planning scoring uses 8 futures; primary contact evaluation uses 32 held-out futures.
- Run scope: 608/1,900 total states, 38 whole batches of 16; head-on condition subset 380 states / 306 sprites / 19 windows. Thus 608 is not the effective n for the primary cell.
- Main policies: `published`, `il3_s0`; primary contrast is on `il3_s0`, condition `ped_headon_3m`.
- Arms: `ung`, `pt`, `exp`, `cvar`, `gui`, `guided_rr_pt`, `guided_rr_exp`, `guided_rr_cvar`.
- Seeds in script protocol: noise seed 123; future seed rule `100 + global_batch_offset`.
- Energy lateral reference: pool mean (`ref_pool = wp_u.clone()` per code). The lateral diagnostic `lat` uses pool-mean reference, while `energy_single` uses each candidate's own unguided counterpart. Do not treat `lat` as direct audit of energy penalty.

## Claim hierarchy

- Primary `il3_s0 guided_rr_exp − exp`: −25.58 pp, sprite-conditional 95% CI [−27.30, −23.90].
- Contact means: 30.30% (`exp`) and 4.71% (`guided_rr_exp`) on head-on held-out proxy.
- Geometry: lateral mean 0.659→1.361 m; origin-inclusive turn 18.77→33.99 degrees; endpoint norm/driver 0.919→0.945; clearance 0.742→1.275 m.
- Contact is averaged over state × 32 futures; state-level zero-contact over all futures is a separate statistic. Neither represents closed-loop collision probability or success.
- CI is sprite-conditional on fixed windows; 32 futures/state are not independent scenes. Do not attach post-hoc p-values.

## Compute and novelty

Guidance generation incurs forward and backward model work (80 forward + 80 backward in the documented accounting), whereas the rerank baseline uses 80 forward. These are not compute-matched. Script 48 establishes a held-out guided-rerank baseline/repairs the evaluation cell; it does not establish a novel proposed method or beat the SocNav paper.

## Source-of-truth files and exact paths

- Result object: `results/results_48_guided_rerank.json`
- Primary value: `res.il3_s0.per_cond.ped_headon_3m.PRIMARY_d_hel_guided_rr_exp_minus_exp.mean`
- Primary CI: same key `.ci_sprite_conditional`
- Contact: `res.il3_s0.per_cond.ped_headon_3m.hel_exp.mean`, `.hel_guided_rr_exp.mean`
- Geometry: `.lat_exp`, `.lat_guided_rr_exp`, `.turn_origin_exp`, `.turn_origin_guided_rr_exp`, `.progress_vs_gt_exp`, `.progress_vs_gt_guided_rr_exp`, `.clr_hel_exp`, `.clr_hel_guided_rr_exp`
- Run metadata: top-level `n_states_total`, `n_states_run`, `batch_offsets`, `batch_size`, `n_cand`, `n_plan`, `n_held`, `primary_cond`, `ci_semantics`.
- Raw evidence: `results/raw_48_guided_rerank.npz`; arrays include `il3_s0__hel__<arm>` (32×608), `il3_s0__state_idx`, plus policy/action geometry arrays.
- Manifest: `results/manifest_48_guided_rerank.json`.

## Reproduction

1. JSON tables: `python code/reproduce_tables.py` (stdlib only).
2. Pipeline/geometry checks: `python code/test_48_pipeline.py` and `python code/test_48_geometry.py` in upstream-compatible PyTorch/numpy environment.
3. Full model rerun requires local SocialNav checkout, model checkpoint and data paths configured in `code/sn_common.py`; none are committed.
