"""47b - Descriptive candidate-pool diversity diagnostic, not a causal test.

Predeclared primary focus: ped_headon_3m, the moving condition in 46.
The four static conditions are descriptive controls; do not select a focus post hoc.
Existing checkpoints, fixed cached rows, no training, no new motion data. All policies
use identical source rows and candidate noise seed 123. First NSTATE available rows
per condition are used (not a random or population-representative subsample).

For each state, 16 candidate trajectories are in metres, shape (16,5,2):
  div_pw: mean over the 120 unordered candidate pairs of the mean over 5 waypoints
          of the 2D Euclidean distance. This is ADE between candidate paths, in metres.
  dup_frac: fraction of those pairs whose pairwise ADE is strictly below 0.05 m.
  div_end: radial endpoint population spread: sqrt(mean_candidates ||end-mean_end||²).
  spread_wp: mean over waypoints of radial population spread across candidates.
  maoe: candidate-0 MAOE against the existing driver path, degrees (context only).

Every metric is per-state, then averaged with equal state weights. Population spread
uses correction=0 and is not sample uncertainty. No across-state std enters diversity.
Raw NPZ stores source rows, condition/window/sprite IDs, all pools and per-state metrics;
JSON includes descriptive paired state differences against published (no p-values/CIs).
Same-state pairing does not make backgrounds/sprites independent or establish causality.
Lower diversity can be consistent with a restricted candidate pool, but cannot explain
or prove the cause of selection-gain changes; checkpoint changes are confounded and
this audit does not measure the energy landscape, calibration or counterfactual gains.
Any older div_end from the across-state implementation is invalid and not comparable.
"""
import copy
import json
import os

import numpy as np
import torch

from sn_common import SocialNav
from pilot_common import PILOT, prepare_head, maoe_deg

dev = torch.device("cuda:0")
POL = ["published", "il3_s0:last.pt", "il3_s1:last.pt", "il3_s2:last.pt"]
NCAND = 16
STEPS = 5
NSTATE = int(os.environ.get("DIV_NSTATE", "160"))   # per condition, capped by availability
DUPTOL = 0.05                                       # m

meta = json.load(open(os.path.join(PILOT, "states_pf.json")))
S = meta["states"][:meta["done"]]
H = np.load(os.path.join(PILOT, "H_pf.npy"), mmap_mode="r")

CONDS = ["ped_headon_3m", "ped_static_3m", "ped_left_3m", "ped_right_3m", "ped_static_2m"]
if NSTATE <= 0:
    raise ValueError("DIV_NSTATE must be positive")
rows = {c: [i for i, s in enumerate(S) if s["cond"] == c][:NSTATE] for c in CONDS}

sn = SocialNav()
head = prepare_head(sn.model)
sn.model.model.to("cpu")
sn.model.lm_head.to("cpu")
torch.cuda.empty_cache()
PUB = copy.deepcopy(head.state_dict())
noise = torch.randn(NCAND, 5, 2, generator=torch.Generator().manual_seed(123)).to(dev)


def sample(z, nz, sc):
    x = nz.clone()
    for k in range(STEPS):
        t = torch.tensor([k / STEPS], device=dev)
        with torch.no_grad():
            x = x + head.flow.predict_velocity(z, x, t).float() / STEPS
    with torch.no_grad():
        return head.model.recover_waypoints_from_pred(x) * sc[:, None, None]


def pool_stats(wp, duplicate_tolerance=0.05):
    """Pure tensor helper: (...state, G, T, 2) -> one scalar per state.

    No model, global device, candidate-count constant or CPU conversion is used.
    Expects (N,G,T,2), G>=2, T>=1, finite floating-point waypoints in metres.
    """
    if wp.ndim != 4 or wp.shape[1] < 2 or wp.shape[2] < 1 or wp.shape[3] != 2:
        raise ValueError("expected (N,G,T,2), G>=2, T>=1")
    if not wp.is_floating_point() or not bool(torch.isfinite(wp).all()):
        raise ValueError("waypoints must be finite floating-point metres")
    if duplicate_tolerance < 0:
        raise ValueError("duplicate tolerance must be nonnegative")
    iu = torch.triu_indices(wp.shape[1], wp.shape[1], offset=1, device=wp.device)
    pair_ade = (wp[:, iu[0]] - wp[:, iu[1]]).norm(dim=-1).mean(dim=-1)
    centred = wp - wp.mean(dim=1, keepdim=True)
    radial_spread = centred.square().sum(dim=-1).mean(dim=1).sqrt()
    return {"div_pw": pair_ade.mean(dim=1),
            "dup_frac": (pair_ade < duplicate_tolerance).to(wp.dtype).mean(dim=1),
            "div_end": radial_spread[:, -1],
            "spread_wp": radial_spread.mean(dim=-1)}


OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))
PRIMARY = "ped_headon_3m"
METRICS = ("div_pw", "dup_frac", "div_end", "spread_wp", "maoe")
out = {"schema_version": 2, "n_cand": NCAND, "dup_tol_m": DUPTOL,
       "n_state_per_cond_requested": NSTATE, "primary_condition": PRIMARY,
       "other_conditions": "descriptive static controls",
       "units": {"div_pw": "m, pairwise ADE", "dup_frac": "fraction, not percent",
                 "div_end": "m, radial endpoint population spread",
                 "spread_wp": "m, mean radial waypoint population spread", "maoe": "degrees"},
       "note": "Descriptive diagnostic, not causal evidence; fixed first source rows, noise seed 123",
       "limitations": ["Not a random subsample", "Shared backgrounds and repeated sprites are not independent",
                       "Diversity alone cannot explain selection gains", "Old div_end is invalid"],
       "res": {}, "raw_files": {}, "paired_vs_published": {}, "skipped": []}
reference = {}
for spec in POL:
    name, _, ck = spec.partition(":")
    path = os.path.join(PILOT, name, ck)
    if name != "published" and not os.path.isfile(path):
        out["skipped"].append(spec)
        print("SKIP", spec, flush=True)
        continue
    head.load_state_dict(PUB if name == "published" else torch.load(path, map_location="cpu"))
    out["res"][name] = {}
    policy_raw = {key: [] for key in METRICS}
    policy_rows, policy_pools = [], []
    for cd in CONDS:
        idx = rows[cd]
        if not idx:
            raise ValueError(f"no source rows for {cd}")
        parts = {key: [] for key in METRICS}
        for k in range(0, len(idx), 16):
            batch = idx[k:k + 16]
            n = len(batch)
            Hb = torch.from_numpy(np.asarray(H[batch])).to(dev).float()
            sc = torch.tensor([S[i]["step_scale"] for i in batch], device=dev).repeat_interleave(NCAND)
            gt = torch.tensor([S[i]["gt_m"] for i in batch], device=dev, dtype=torch.float32)
            with torch.no_grad():
                z = head.pool(Hb).repeat_interleave(NCAND, 0)
                wp = sample(z, noise.repeat(n, 1, 1), sc).view(n, NCAND, 5, 2)
                metrics = pool_stats(wp, DUPTOL)
                metrics["maoe"] = maoe_deg(wp[:, 0], gt)
            for key in METRICS:
                parts[key].append(metrics[key].cpu())
            policy_pools.append(wp.cpu().numpy())
        values = {key: torch.cat(parts[key]).numpy() for key in METRICS}
        for key in METRICS:
            if values[key].shape != (len(idx),) or not np.isfinite(values[key]).all():
                raise AssertionError((key, values[key].shape, "expected finite per-state metric"))
            if key == "dup_frac" and not ((values[key] >= 0) & (values[key] <= 1)).all():
                raise AssertionError("duplicate fractions outside [0,1]")
            policy_raw[key].append(values[key])
        policy_rows.extend(idx)
        r = {key: float(values[key].mean()) for key in METRICS}
        r["n"] = len(idx)
        out["res"][name][cd] = r
        if name == "published":
            reference[cd] = (np.asarray(idx), values)
        else:
            ref_idx, ref_values = reference[cd]
            if not np.array_equal(ref_idx, np.asarray(idx)):
                raise AssertionError("paired policy comparison requires identical source rows")
            differences = {key: values[key] - ref_values[key] for key in METRICS}
            out["paired_vs_published"].setdefault(name, {})[cd] = {
                "n": len(idx), "direction": "policy minus published", "descriptive_only": True,
                "metrics": {key: {"mean_difference": float(d.mean()),
                                  "median_difference": float(np.median(d))} for key, d in differences.items()}}
            for key in METRICS:
                policy_raw.setdefault("delta_vs_published_" + key, []).append(differences[key])
        print(f"{name:10s} {cd:16s} ADE {r['div_pw']:.3f} m | dup {100*r['dup_frac']:5.1f}% | "
              f"endpoint spread {r['div_end']:.3f} m | waypoint spread {r['spread_wp']:.3f} m | "
              f"MAOE {r['maoe']:.2f} deg", flush=True)
    raw = {key: np.concatenate(parts, axis=0) for key, parts in policy_raw.items()}
    raw.update(source_row=np.asarray(policy_rows),
               condition=np.asarray([S[i]["cond"] for i in policy_rows]),
               sprite=np.asarray([str(S[i]["sprite"]) for i in policy_rows]),
               source_window=np.asarray([S[i].get("ps", -1) for i in policy_rows]),
               trajectory=np.asarray([S[i].get("traj", -1) for i in policy_rows]),
               step_scale=np.asarray([S[i]["step_scale"] for i in policy_rows]),
               pool_wp=np.concatenate(policy_pools, axis=0), noise_seed=np.asarray(123))
    raw_name = f"results_47b_diversity_{name}_raw.npz"
    np.savez_compressed(os.path.join(OUTPUT_DIR, raw_name), **raw)
    out["raw_files"][name] = raw_name
    with open(os.path.join(OUTPUT_DIR, "results_47b_diversity.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1, allow_nan=False)
print("DONE")
