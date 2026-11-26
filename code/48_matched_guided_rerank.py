"""48 - Matched guided re-ranking on held-out futures (decisive fair contrast).

WHY THIS EXPERIMENT EXISTS
--------------------------
43 showed that "sample 16, keep the lowest-energy one" matches gradient
guidance on contact, and it ALREADY contained a guided re-rank: its `rerank16g`
arm picks argmin energy over the guided pool. 43's selector, however, scores
against the NOMINAL pedestrian only, and it scores on the same nominal state it
is evaluated on. 46 then moved to futures the selector never saw and found that
distributional selection over the UNGUIDED pool (exp / cvar) moves held-out
contact only slightly relative to the point forecast (46, head-on, il3_s0:
d(exp-pt) = +1.2 pts [+0.5, +1.9]; published: -4.2 pts), while `gui` -- guided
candidate 0, i.e. guidance with no selection -- is the only arm that clearly
moves it.

What neither 43 nor 46 provides is the matched cell: guided pool + the SAME
distributional selectors + the SAME held-out futures. 43's guided re-rank is
nominal and in-sample; 46's held-out protocol never re-ranks the guided pool.
This script fills exactly that cell and nothing else. It is a BASELINE REPAIR,
not a new method: every component already exists in 43/44/46, and the only
thing 48 contributes is scoring them under one protocol.

    guided_rr_pt    argmin over wp_g of E(wp_g, nominal future)
    guided_rr_exp   argmin over wp_g of mean_k E(wp_g, future_k), k in 0..7
    guided_rr_cvar  argmin over wp_g of mean of the worst half of those 8

The guided pool `wp_g` is the SAME pool 46 already draws (same z, same noise,
same reference), so the three new arms cost nothing but extra scoring, and they
are drawn from the SAME candidate set as `gui`.

PRIMARY POLICY
--------------
il3_s0 is the primary policy for the head-on contrast (it is the policy the
round-3 pilot actually ships); `published` is reported as a secondary, since
its behaviour under selection differs qualitatively (see 46). Both are run in
one pass so the protocol cannot drift between them.

PROTOCOL: COPIED FROM 46, NOT RE-DERIVED
----------------------------------------
Everything that could move a number is byte-identical to 46_heldout_v2.py and is
checked as such by test_48_geometry.py (AST source comparison):

  * state order           meta["states"][:done], unsliced, same file
  * batching              range(0, len(S), 16), contiguous, full batches only
  * pedestrian futures    sample_ped_futures(ped, 40, seed=100 + k)  <- k is the
                          GLOBAL state offset of the batch, so a batch that 48
                          computes gets exactly the futures 46 gave it
  * candidate noise       torch.randn(16, 5, 2), generator seed 123
  * sampler               sample() / energy_single() / path_points() verbatim
  * energy reference      each candidate's OWN unguided path (44/46's
                          `ref_pool = wp_pool.clone()`), NOT the pool mean, for
                          both the unguided and the guided pool. The guided
                          candidate j of state i is referenced against unguided
                          candidate j of state i -- same noise draw, so it is
                          the counterfactual "what this draw would have been".
  * planning / held-out   futures 0..7 select, futures 8..39 score. Never mixed.

SUBSET WITHOUT BREAKING REPRODUCIBILITY
---------------------------------------
The primary condition is ped_headon_3m (the only moving-pedestrian condition in
the suite; 46 found the static conditions saturated). Restricting the run to it
saves ~80 % of the GPU time, but the per-batch seed is a function of the batch
offset, so states may NOT be re-packed into new batches: that would change every
future. Instead this script keeps 46's batch grid exactly and runs only those
WHOLE batches that contain at least one head-on state. Such a batch also
contains states of the neighbouring conditions; those are computed honestly
(they are 46's own batch-mates) and are reported, but the head-on mask is what
the primary contrast uses. `--all` runs the full 1900 states.

WHAT IS REPORTED
----------------
  primary       guided_rr_exp vs exp, held-out contact, head-on, IL3_S0,
                paired by sprite
  secondary     the same contrast for `published`
  nominal       guided_rr_pt  vs pt,  held-out contact, head-on
  everything else is EXPLORATORY. No p-values are computed anywhere in this
  file, and the two uncertainty columns are labelled by what they are:
      ci_sprite_conditional : cluster bootstrap over sprites, CONDITIONAL on the
                              19 suite windows (windows are fixed, not resampled)
      ci_window_descriptive : the 2.5/97.5 PERCENTILES of the 19 per-window
                              means. This is a descriptive spread, not a
                              bootstrap and not a CI: 19 windows of one
                              trajectory are not a sample from a population.
                              Report it as "window spread", never as a CI.

Raw per-state material (chosen paths, per-future contact and clearance for every
arm, selected candidate indices, energies) goes to the NPZ uncompressed so that
any later re-analysis does not have to re-run the GPU.

OUTPUTS (all script-relative)
    results_48_guided_rerank.json
    raw_48_guided_rerank.npz
    manifest_48_guided_rerank.json

USAGE
    run_48.bat                     primary: head-on batches, both policies
    run_48.bat --smoke             2 batches, published only (runtime check)
    run_48.bat --all               all 1900 states, 46's full grid
    run_48.bat --max-batches 5
    python 48_matched_guided_rerank.py --self-test     no GPU, no model
"""
import argparse
import copy
import json
import os
import platform
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))

DELTA, R_GUIDE, STEPS = 0.3, 1.2, 5
NCAND = 16
N_PLAN = 8          # futures visible to the selector (44/46's budget)
N_HELD = 32         # futures never visible to the selector
N_ALL = N_PLAN + N_HELD
REALISED = 1        # 44/46's in-sample anchor index
BATCH = 16          # 46's batch size; the per-batch seed depends on it
PRIMARY_COND = "ped_headon_3m"
POL_DEFAULT = ["published", "il3_s0:last.pt"]

UNGUIDED_ARMS = ("ung", "pt", "exp", "cvar")
GUIDED_ARMS = ("gui", "guided_rr_pt", "guided_rr_exp", "guided_rr_cvar")
ARMS = UNGUIDED_ARMS + GUIDED_ARMS

# ----------------------------------------------------------------- geometry
# The five functions below are COPIED VERBATIM from 46_heldout_v2.py.
# test_48_geometry.py compares their source text against 46 and fails if they
# drift. Do not "clean them up".
_T = torch.linspace(0, 5, 51)


def path_points(wp):
    t = _T.to(wp.device)
    P = torch.cat([torch.zeros_like(wp[:, :1]), wp], 1)
    k = t.floor().clamp(max=4).long()
    a = (t - k)[None, :, None]
    return P[:, k] * (1 - a) + P[:, k + 1] * a


def energy_single(wp, peds, ref):
    """Byte-identical to 41/43/44/46: collision term + lateral corridor (+-1 m)."""
    pos = path_points(wp)
    t = _T.to(wp.device)
    pp = peds[:, None, :2] + peds[:, None, 2:] * t[None, :, None]
    D = (pos - pp).norm(dim=-1)
    act = (D < R_GUIDE).float()
    e = ((1 - D / R_GUIDE).clamp_min(0) ** 2 * act).sum(1) / (act.sum(1) + 1e-6)
    return e + ((wp[..., 0] - ref[..., 0]).abs() - 1.0).clamp_min(0).pow(2).mean(-1)


def turn_deg(wp):
    """46's turn: heading changes BETWEEN waypoint segments, origin excluded.

    Kept unchanged so 48's turn column reproduces 46's. It ignores the first
    segment (origin -> waypoint 0), which is where a swerve actually starts, so
    turn_deg_origin() below is reported alongside it.
    """
    d = wp[:, 1:] - wp[:, :-1]
    h = torch.atan2(d[..., 1], d[..., 0])
    dh = (h[:, 1:] - h[:, :-1]).abs()
    dh = torch.atan2(torch.sin(dh), torch.cos(dh)).abs()
    return torch.rad2deg(dh).max(-1).values


def turn_deg_origin(wp):
    """Same statistic with the origin segment prepended (robot starts at 0,0).

    This is NOT a correction of turn_deg -- both are reported. The origin
    segment is the one the pedestrian term acts on first, so a selector that
    buys safety with an immediate swerve is invisible to turn_deg and visible
    here.
    """
    P = torch.cat([torch.zeros_like(wp[:, :1]), wp], 1)
    d = P[:, 1:] - P[:, :-1]
    h = torch.atan2(d[..., 1], d[..., 0])
    dh = (h[:, 1:] - h[:, :-1]).abs()
    dh = torch.atan2(torch.sin(dh), torch.cos(dh)).abs()
    return torch.rad2deg(dh).max(-1).values


def final_dist(wp):
    """Euclidean distance of the last waypoint from the origin, metres."""
    return wp[..., -1, :].norm(dim=-1)


def cluster_boot(per_person_vals, per_person_n, rng, B=2000):
    """46's ratio-estimator bootstrap: resample people, weight by state counts."""
    v = np.asarray(per_person_vals, float)
    n = np.asarray(per_person_n, float)
    mean = v.sum() / n.sum()
    out = []
    idx_all = np.arange(len(v))
    for _ in range(B):
        b = rng.choice(idx_all, len(v))
        out.append(v[b].sum() / max(n[b].sum(), 1e-9))
    return mean, [float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))]


# ----------------------------------------------------------------- batching
def batch_offsets(n_states, conds, only_cond, max_batches):
    """46's batch grid, optionally filtered to whole batches touching one condition.

    Returns a list of global offsets k, each the start of a contiguous
    range(k, min(k + 16, n_states)). The offsets are a SUBSET of 46's offsets
    and are never re-packed, because sample_ped_futures is seeded with k.
    """
    all_k = list(range(0, n_states, BATCH))
    if only_cond is None:
        keep = all_k
    else:
        keep = [k for k in all_k
                if any(conds[i] == only_cond for i in range(k, min(k + BATCH, n_states)))]
    if max_batches is not None:
        keep = keep[:max_batches]
    return keep


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="48 - matched guided re-ranking, held-out futures")
    p.add_argument("--all", action="store_true",
                   help="run all 1900 states (46's full grid) instead of head-on batches only")
    p.add_argument("--max-batches", type=int, default=None,
                   help="stop after this many batches (runtime bound)")
    p.add_argument("--smoke", action="store_true",
                   help="= --max-batches 2 --policies published")
    p.add_argument("--policies", default=None,
                   help="comma list, e.g. 'published' or 'published,il3_s0:last.pt'")
    p.add_argument("--self-test", action="store_true",
                   help="geometry / protocol checks only; no model, no CUDA")
    p.add_argument("--tag", default="", help="suffix for the output file names")
    a = p.parse_args(argv)
    if a.smoke:
        if a.max_batches is None:
            a.max_batches = 2
        if a.policies is None:
            a.policies = "published"
    a.pol = POL_DEFAULT if a.policies is None else [s for s in a.policies.split(",") if s]
    return a


# ----------------------------------------------------------------- self-test
def self_test():
    """Checks that need neither the VLM nor a GPU. Exits non-zero on failure."""
    import test_48_geometry
    return test_48_geometry.main()


def main(argv=None):
    args = parse_args(argv)
    if args.self_test:
        return self_test()

    t_start = time.time()
    from sn_common import SocialNav
    from pilot_common import PILOT, CONTACT, prepare_head, min_ped_distance

    dev = torch.device(os.environ.get("SN48_DEV", "cuda:0"))
    meta = json.load(open(os.path.join(PILOT, "states_pf.json")))
    S = meta["states"][:meta["done"]]
    H = np.load(os.path.join(PILOT, "H_pf.npy"), mmap_mode="r")
    conds_per_state = [s["cond"] for s in S]
    conds = sorted(set(conds_per_state))
    people = np.array([s["sprite"] for s in S])
    windows = np.array([s["ps"] for s in S])

    only = None if args.all else PRIMARY_COND
    if only is not None and only not in conds:
        raise SystemExit(f"primary condition {only!r} not in states file; conds={conds}")
    offs = batch_offsets(len(S), conds_per_state, only, args.max_batches)
    sel_idx = np.array([i for k in offs for i in range(k, min(k + BATCH, len(S)))], dtype=np.int64)
    print(f"[48] {len(offs)} batches, {len(sel_idx)} states "
          f"({'all conds' if only is None else only + ' batches'}), policies={args.pol}", flush=True)

    # ---- model: identical bring-up to 43/46 ----
    sn = SocialNav()
    head = prepare_head(sn.model)
    sn.model.model.to("cpu")
    sn.model.lm_head.to("cpu")
    if dev.type == "cuda":
        torch.cuda.empty_cache()
    # deep copy: head.state_dict() hands back live tensors for `q`; without the
    # copy the second policy's load would be scored against mutated "published"
    # weights.
    PUB = copy.deepcopy(head.state_dict())
    noise = torch.randn(NCAND, 5, 2, generator=torch.Generator().manual_seed(123)).to(dev)

    def sample(z, nz, sc, peds=None, ref=None):
        x = nz.clone()
        for k in range(STEPS):
            t = torch.tensor([k / STEPS], device=dev)
            if peds is not None:
                xg = x.detach().requires_grad_(True)
                v = head.flow.predict_velocity(z, xg, t).float()
                wp = head.model.recover_waypoints_from_pred(xg + (1 - k / STEPS) * v) * sc[:, None, None]
                g, = torch.autograd.grad(energy_single(wp, peds, ref).sum(), xg)
                nrm = g.flatten(1).norm(dim=1)[:, None, None]
                x = (xg + v / STEPS - torch.where(nrm > 0, g * DELTA / nrm.clamp_min(1e-12),
                                                  torch.zeros_like(g))).detach()
            else:
                with torch.no_grad():
                    x = x + head.flow.predict_velocity(z, x, t).float() / STEPS
        if peds is not None:
            x = x.clamp(-1, 1)
        with torch.no_grad():
            return head.model.recover_waypoints_from_pred(x) * sc[:, None, None]

    def sample_ped_futures(ped_nom, n_futures, seed):
        """Verbatim 46: future 0 nominal, 1.. are heading/speed/hesitation draws."""
        B = ped_nom.shape[0]
        futures = [ped_nom.unsqueeze(1)]
        rng = torch.Generator(device=dev).manual_seed(seed)
        for _ in range(1, n_futures):
            p = ped_nom.clone()
            vx, vy = p[:, 2], p[:, 3]
            speed = (vx ** 2 + vy ** 2).sqrt()
            is_moving = speed > 0.1
            ang = torch.atan2(vx, vy)
            d_ang = torch.randn(B, generator=rng, device=dev) * (15.0 * np.pi / 180.0)
            speed_scale = 1.0 + torch.randn(B, generator=rng, device=dev) * 0.20
            stop_mask = (torch.rand(B, generator=rng, device=dev) < 0.15) & is_moving
            speed_scale = torch.where(stop_mask, torch.zeros_like(speed_scale),
                                      speed_scale.clamp(0.2, 1.8))
            new_speed = speed * speed_scale
            new_ang = ang + d_ang
            p[:, 2] = torch.where(is_moving, new_speed * torch.sin(new_ang), vx)
            p[:, 3] = torch.where(is_moving, new_speed * torch.cos(new_ang), vy)
            futures.append(p.unsqueeze(1))
        return torch.cat(futures, dim=1)

    def run():
        """One pass over the selected batches for the currently loaded policy."""
        store = {
            "state_idx": [],
            "paths": {a: [] for a in ARMS},           # (n, 5, 2) per arm
            "cand": {a: [] for a in ARMS},            # chosen candidate index
            "ins": {a: [] for a in ARMS},             # contact on future REALISED
            "hel": {a: [] for a in ARMS},             # (32, n) contact per held future
            "clr_hel": {a: [] for a in ARMS},         # (32, n) clearance per held future
            "clr_ins": {a: [] for a in ARMS},
            "turn": {a: [] for a in ARMS},
            "turn_o": {a: [] for a in ARMS},
            "lat": {a: [] for a in ARMS},
            "fin": {a: [] for a in ARMS},
            "E_sel": {a: [] for a in ARMS},           # planning energy of the pick
            "fin_ung0": [], "fin_gt": [],
            "flip_exp_pt": [], "flip_grr_exp_pt": [], "flip_grr_exp_vs_exp": [],
        }
        for bi, k in enumerate(offs):
            idx = list(range(k, min(k + BATCH, len(S))))
            n = len(idx)
            Hb = torch.from_numpy(np.asarray(H[idx])).to(dev).float()
            sc = torch.tensor([S[i]["step_scale"] for i in idx], device=dev).repeat_interleave(NCAND)
            ped = torch.tensor([S[i]["ped"] for i in idx], device=dev, dtype=torch.float32)
            gt = torch.tensor([S[i]["gt_m"] for i in idx], device=dev, dtype=torch.float32)
            with torch.no_grad():
                z = head.pool(Hb).repeat_interleave(NCAND, 0)
            nz = noise.repeat(n, 1, 1)

            wp_u = sample(z, nz, sc)
            p_rep = ped.repeat_interleave(NCAND, 0)
            wp_g = sample(z, nz, sc, p_rep, wp_u)     # ref = own unguided path, as in 44/46

            wp_u4 = wp_u.view(n, NCAND, 5, 2)
            wp_g4 = wp_g.view(n, NCAND, 5, 2)
            ref_pool = wp_u.clone()                   # exactly 44/46: ref_pool = wp_pool.clone()

            fut = sample_ped_futures(ped, N_ALL, seed=100 + k)      # (n, 40, 4)

            # ---- planning energies on the FIRST 8 futures, both pools ----
            # The guided pool is referenced against the SAME ref_pool, i.e. each
            # guided candidate against its OWN unguided draw, not the pool mean.
            Eu, Eg = [], []
            for f in range(N_PLAN):
                pf = fut[:, f].repeat_interleave(NCAND, 0)
                Eu.append(energy_single(wp_u, pf, ref_pool).view(n, NCAND))
                Eg.append(energy_single(wp_g, pf, ref_pool).view(n, NCAND))
            Eu = torch.stack(Eu, -1)                                # (n, 16, 8)
            Eg = torch.stack(Eg, -1)

            def picks(E):
                nom, ex = E[..., 0], E.mean(-1)
                cv = torch.topk(E, k=N_PLAN // 2, dim=-1).values.mean(-1)
                return nom, ex, cv, nom.argmin(1), ex.argmin(1), cv.argmin(1)

            Eu_nom, Eu_exp, Eu_cvar, i_pt, i_exp, i_cvar = picks(Eu)
            Eg_nom, Eg_exp, Eg_cvar, j_pt, j_exp, j_cvar = picks(Eg)

            ar = torch.arange(n, device=dev)
            zero_i = torch.zeros(n, dtype=torch.long, device=dev)
            sel = {"ung": wp_u4[:, 0], "pt": wp_u4[ar, i_pt], "exp": wp_u4[ar, i_exp],
                   "cvar": wp_u4[ar, i_cvar], "gui": wp_g4[:, 0],
                   "guided_rr_pt": wp_g4[ar, j_pt], "guided_rr_exp": wp_g4[ar, j_exp],
                   "guided_rr_cvar": wp_g4[ar, j_cvar]}
            cand = {"ung": zero_i, "pt": i_pt, "exp": i_exp, "cvar": i_cvar, "gui": zero_i,
                    "guided_rr_pt": j_pt, "guided_rr_exp": j_exp, "guided_rr_cvar": j_cvar}
            e_sel = {"ung": Eu_exp[:, 0], "pt": Eu_exp[ar, i_pt], "exp": Eu_exp[ar, i_exp],
                     "cvar": Eu_exp[ar, i_cvar], "gui": Eg_exp[:, 0],
                     "guided_rr_pt": Eg_exp[ar, j_pt], "guided_rr_exp": Eg_exp[ar, j_exp],
                     "guided_rr_cvar": Eg_exp[ar, j_cvar]}

            store["flip_exp_pt"].append((i_exp != i_pt).float().cpu())
            store["flip_grr_exp_pt"].append((j_exp != j_pt).float().cpu())
            store["flip_grr_exp_vs_exp"].append((j_exp != i_exp).float().cpu())

            # lateral deviation of EVERY arm against the SAME common reference,
            # the unguided pool mean (43's convention), so the arms are
            # comparable to each other and to 43.
            lat_ref = wp_u4.mean(1)[..., 0]
            for a in ARMS:
                w = sel[a]
                store["paths"][a].append(w.detach().cpu())
                store["cand"][a].append(cand[a].cpu())
                store["E_sel"][a].append(e_sel[a].detach().cpu())
                store["ins"][a].append((min_ped_distance(w, fut[:, REALISED]) < CONTACT).float().cpu())
                store["clr_ins"][a].append(min_ped_distance(w, fut[:, REALISED]).cpu())
                dm = torch.stack([min_ped_distance(w, fut[:, f]) for f in range(N_PLAN, N_ALL)], 0)
                store["hel"][a].append((dm < CONTACT).float().cpu())      # (32, n)
                store["clr_hel"][a].append(dm.cpu())                      # (32, n)
                store["turn"][a].append(turn_deg(w).cpu())
                store["turn_o"][a].append(turn_deg_origin(w).cpu())
                store["lat"][a].append((w[..., 0] - lat_ref).abs().max(-1).values.detach().cpu())
                store["fin"][a].append(final_dist(w).detach().cpu())
            store["fin_ung0"].append(final_dist(wp_u4[:, 0]).detach().cpu())
            store["fin_gt"].append(final_dist(gt).cpu())
            store["state_idx"].append(np.asarray(idx, dtype=np.int64))
            if (bi + 1) % 10 == 0 or bi + 1 == len(offs):
                print(f"[48]   batch {bi+1}/{len(offs)} (offset {k})  {time.time()-t_start:.0f}s",
                      flush=True)
        return store

    def flatten(store):
        out = {"state_idx": np.concatenate(store["state_idx"])}
        for grp in ("paths", "cand", "ins", "clr_ins", "turn", "turn_o", "lat", "fin", "E_sel"):
            for a in ARMS:
                out[f"{grp}__{a}"] = torch.cat(store[grp][a], 0).numpy()
        for grp in ("hel", "clr_hel"):
            for a in ARMS:
                out[f"{grp}__{a}"] = torch.cat(store[grp][a], 1).numpy()   # (32, n)
        for k2 in ("fin_ung0", "fin_gt", "flip_exp_pt", "flip_grr_exp_pt", "flip_grr_exp_vs_exp"):
            out[k2] = torch.cat(store[k2]).numpy()
        return out

    # ----------------------------------------------------------- statistics
    def summarise(R, mask, people_sub, windows_sub):
        """All CIs are labelled by what they resample. No p-values anywhere."""
        uniq = np.unique(people_sub[mask]) if mask.any() else np.array([])
        d = {"n_states": int(mask.sum()),
             "n_sprites": int(len(uniq)),
             "n_windows": int(len(np.unique(windows_sub[mask]))) if mask.any() else 0}
        if not mask.any():
            return d

        HEL = {a: R[f"hel__{a}"].mean(0) for a in ARMS}     # per state, over 32 futures
        INS = {a: R[f"ins__{a}"] for a in ARMS}

        def sprite_ci(arr, seed=0):
            vals = [arr[mask & (people_sub == u)].sum() for u in uniq]
            cnts = [int((mask & (people_sub == u)).sum()) for u in uniq]
            mu, ci = cluster_boot(vals, cnts, np.random.RandomState(seed))
            return {"mean": float(mu * 100), "ci_sprite_conditional": [c * 100 for c in ci]}

        def window_spread(arr):
            ws = np.unique(windows_sub[mask])
            per = np.array([arr[mask & (windows_sub == w)].mean() for w in ws], float)
            return {"per_window_mean": float(per.mean() * 100),
                    "ci_window_descriptive": [float(np.percentile(per, 2.5) * 100),
                                              float(np.percentile(per, 97.5) * 100)],
                    "n_windows": int(len(ws))}

        for a in ARMS:
            d[f"hel_{a}"] = sprite_ci(HEL[a])
            d[f"hel_{a}"].update(window=window_spread(HEL[a]))
            d[f"ins_{a}"] = sprite_ci(INS[a])
            d[f"turn_{a}"] = float(R[f"turn__{a}"][mask].mean())
            d[f"turn_origin_{a}"] = float(R[f"turn_o__{a}"][mask].mean())
            d[f"lat_{a}"] = float(R[f"lat__{a}"][mask].mean())
            d[f"clr_hel_{a}"] = float(R[f"clr_hel__{a}"].mean(0)[mask].mean())
            d[f"clr_ins_{a}"] = float(R[f"clr_ins__{a}"][mask].mean())
            d[f"E_sel_{a}"] = float(R[f"E_sel__{a}"][mask].mean())
            # progress ratios, both denominators stated explicitly
            d[f"progress_vs_ung0_{a}"] = float(
                (R[f"fin__{a}"][mask] / np.maximum(R["fin_ung0"][mask], 1e-6)).mean())
            d[f"progress_vs_gt_{a}"] = float(
                (R[f"fin__{a}"][mask] / np.maximum(R["fin_gt"][mask], 1e-6)).mean())

        def paired(a, b, seed=1):
            """Per-sprite paired difference on the held-out futures, a minus b."""
            dv = [HEL[a][mask & (people_sub == u)].sum() - HEL[b][mask & (people_sub == u)].sum()
                  for u in uniq]
            dn = [int((mask & (people_sub == u)).sum()) for u in uniq]
            mu, ci = cluster_boot(dv, dn, np.random.RandomState(seed))
            return {"mean": float(mu * 100), "ci_sprite_conditional": [c * 100 for c in ci]}

        d["PRIMARY_d_hel_guided_rr_exp_minus_exp"] = paired("guided_rr_exp", "exp")
        d["NOMINAL_d_hel_guided_rr_pt_minus_pt"] = paired("guided_rr_pt", "pt")
        for a, b in (("guided_rr_cvar", "cvar"), ("guided_rr_exp", "gui"),
                     ("guided_rr_exp", "guided_rr_pt"), ("exp", "pt"),
                     ("gui", "pt"), ("ung", "pt")):
            d[f"EXPLORATORY_d_hel_{a}_minus_{b}"] = paired(a, b)
        d["flip_exp_vs_pt"] = float(R["flip_exp_pt"][mask].mean() * 100)
        d["flip_guided_rr_exp_vs_guided_rr_pt"] = float(R["flip_grr_exp_pt"][mask].mean() * 100)
        d["flip_guided_rr_exp_vs_exp"] = float(R["flip_grr_exp_vs_exp"][mask].mean() * 100)
        return d

    tag = args.tag
    res_path = os.path.join(HERE, f"results_48_guided_rerank{tag}.json")
    npz_path = os.path.join(HERE, f"raw_48_guided_rerank{tag}.npz")
    man_path = os.path.join(HERE, f"manifest_48_guided_rerank{tag}.json")

    out = {
        "script": os.path.basename(__file__),
        "protocol_source": "46_heldout_v2.py (state order, batch grid, seeds, sampler, energy ref)",
        "n_states_total": len(S), "n_states_run": int(len(sel_idx)),
        "n_batches": len(offs), "batch_size": BATCH, "batch_offsets": offs,
        "n_cand": NCAND, "n_plan": N_PLAN, "n_held": N_HELD, "realised_index": REALISED,
        "subset": "all" if only is None else f"whole batches containing {only}",
        "primary_cond": PRIMARY_COND, "arms": list(ARMS), "conds": conds,
        "claims": {
            "primary": "guided_rr_exp vs exp, held-out contact, " + PRIMARY_COND
                       + ", policy il3_s0 (published is secondary)",
            "nominal": "guided_rr_pt vs pt, held-out contact, " + PRIMARY_COND,
            "everything_else": "exploratory; no p-values are computed in this script",
            "contribution": "baseline repair: 43 already re-ranked the guided pool but "
                            "only on the nominal pedestrian and in-sample; 46 used held-out "
                            "futures but never re-ranked the guided pool. 48 is the matched cell.",
        },
        "ci_semantics": {
            "ci_sprite_conditional": "cluster bootstrap over sprites, conditional on the fixed suite windows",
            "ci_window_descriptive": "2.5/97.5 percentiles of the 19 per-window means; a DESCRIPTIVE "
                                     "spread, not a bootstrap and not a population CI",
        },
        "res": {},
    }

    raw_all = {}
    for spec in args.pol:
        name, _, ck = spec.partition(":")
        if name == "published":
            sd = PUB
        else:
            ckpt = os.path.join(PILOT, name, ck)
            if not os.path.exists(ckpt):
                # do not silently skip: a missing checkpoint invalidates the contrast
                raise FileNotFoundError(f"checkpoint for policy {spec!r} not found: {ckpt}")
            sd = torch.load(ckpt)
        head.load_state_dict(sd)
        R = flatten(run())
        si = R["state_idx"]
        people_sub, windows_sub = people[si], windows[si]
        conds_sub = np.array([conds_per_state[i] for i in si])

        per_cond = {}
        for cd in sorted(set(conds_sub.tolist())):
            m = conds_sub == cd
            per_cond[cd] = summarise(R, m, people_sub, windows_sub)
            if cd == PRIMARY_COND:
                d = per_cond[cd]
                print(f"[48] {name:10s} {cd:16s} HELDOUT "
                      f"ung {d['hel_ung']['mean']:5.1f} pt {d['hel_pt']['mean']:5.1f} "
                      f"exp {d['hel_exp']['mean']:5.1f} cvar {d['hel_cvar']['mean']:5.1f} "
                      f"gui {d['hel_gui']['mean']:5.1f} "
                      f"grr_pt {d['hel_guided_rr_pt']['mean']:5.1f} "
                      f"grr_exp {d['hel_guided_rr_exp']['mean']:5.1f} "
                      f"grr_cvar {d['hel_guided_rr_cvar']['mean']:5.1f} || PRIMARY "
                      f"d(grr_exp-exp) {d['PRIMARY_d_hel_guided_rr_exp_minus_exp']['mean']:+5.1f} "
                      f"{d['PRIMARY_d_hel_guided_rr_exp_minus_exp']['ci_sprite_conditional']} || NOMINAL "
                      f"d(grr_pt-pt) {d['NOMINAL_d_hel_guided_rr_pt_minus_pt']['mean']:+5.1f} "
                      f"{d['NOMINAL_d_hel_guided_rr_pt_minus_pt']['ci_sprite_conditional']}", flush=True)
        out["res"][name] = {"per_cond": per_cond,
                            "all_run_states": summarise(R, np.ones(len(si), bool),
                                                        people_sub, windows_sub)}
        for k2, v in R.items():
            raw_all[f"{name}__{k2}"] = v
        json.dump(out, open(res_path, "w"), indent=1)
        np.savez(npz_path, **raw_all)

    json.dump({
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "script": os.path.basename(__file__),
        "argv": sys.argv[1:], "args": vars(args),
        "python": sys.version, "torch": torch.__version__, "numpy": np.__version__,
        "platform": platform.platform(),
        "cuda_device": torch.cuda.get_device_name(0) if (dev.type == "cuda" and torch.cuda.is_available()) else None,
        "states_file": os.path.join(PILOT, "states_pf.json"),
        "n_states_total": len(S), "n_states_run": int(len(sel_idx)),
        "batch_offsets": offs, "policies": args.pol, "arms": list(ARMS),
        "noise_seed": 123, "future_seed_rule": "100 + global_batch_offset",
        "energy_ref": "each candidate's own unguided path (ref_pool = wp_u.clone()), for both pools",
        "outputs": {"results": os.path.basename(res_path), "raw": os.path.basename(npz_path)},
        "runtime_s": round(time.time() - t_start, 1),
    }, open(man_path, "w"), indent=1)
    print(f"[48] DONE {time.time()-t_start:.0f}s -> {os.path.basename(res_path)}, "
          f"{os.path.basename(npz_path)}, {os.path.basename(man_path)}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
