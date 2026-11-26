"""46 - Corrected held-out audit of the distributional-selection claim in 44.

WHAT WAS WRONG WITH 44 (and why 45 v1 was discarded)
----------------------------------------------------
44 draws 8 pedestrian futures, scores all 16 robot candidates against those 8,
selects by expected risk / CVaR, and then evaluates contact on
`ped_realized = ped_futures[:, 1]` -- one of the same 8 scored models. The
selector is therefore allowed to see the future that materialises. That is an
in-sample evaluation and it flatters any distributional selector by construction.

45 v1 tried to fix this but changed the energy reference from each candidate's own
unguided path (44: `ref_pool = wp_pool.clone()`) to the pool mean, so its numbers
are not comparable to 43/44. Also, with N_FUT=8 and K_FIT=7 the "held7" arm
collapsed to a single held-out future. Both are fixed here:

  * energy reference = each candidate's own unguided path, byte-identical to 44;
  * 40 futures are drawn per state, planning uses futures 0..7 exactly as 44,
    and futures 8..39 (32 of them) are NEVER used for selection -- they are the
    held-out test set.

Arms (16 candidates each, same sampler, same noise seed as 41/43/44):
  ung   candidate 0 of the unguided pool            (prior draw, no selection)
  pt    argmin E(wp, pedestrian nominal)             (point forecast)
  exp   argmin mean_k E(wp, future_k), k in 0..7     (expected risk)
  cvar  argmin mean of worst half of E(wp, future_k) (CVaR, alpha = 0.5)
  gui   guided sample, candidate 0                   (gradient guidance)

Each arm is scored twice on the SAME selected trajectory:
  insample  contact on future 1        (44's protocol, sanity anchor)
  heldout   mean contact over 8..39    (the honest number)

Statistics: contact rate is a mean over states within a condition, so the
bootstrap resamples PEOPLE and re-weights each person by their number of states
(cluster bootstrap on the ratio of sums). The exp-vs-pt contrast is reported as a
paired per-person difference with its own CI, because paired is what the claim
needs: does the distribution help the SAME state that the point forecast fails?

Output: results_46_heldout.json
"""
import json
import os

import numpy as np
import torch

from sn_common import SocialNav
from pilot_common import PILOT, CONTACT, prepare_head, min_ped_distance

dev = torch.device("cuda:0")
POL = ["published", "il3_s0:last.pt"]
DELTA, R_GUIDE, STEPS = 0.3, 1.2, 5
NCAND = 16
N_PLAN = 8          # futures visible to the selector (44's budget)
N_HELD = 32         # futures never visible to the selector
N_ALL = N_PLAN + N_HELD
REALISED = 1        # 44 evaluates on this index; kept only for the anchor

meta = json.load(open(os.path.join(PILOT, "states_pf.json")))
S = meta["states"][:meta["done"]]
H = np.load(os.path.join(PILOT, "H_pf.npy"), mmap_mode="r")
sn = SocialNav()
head = prepare_head(sn.model)
sn.model.model.to("cpu")
sn.model.lm_head.to("cpu")
torch.cuda.empty_cache()
PUB = head.state_dict()
noise = torch.randn(NCAND, 5, 2, generator=torch.Generator().manual_seed(123)).to(dev)
_T = torch.linspace(0, 5, 51)


def path_points(wp):
    t = _T.to(wp.device)
    P = torch.cat([torch.zeros_like(wp[:, :1]), wp], 1)
    k = t.floor().clamp(max=4).long()
    a = (t - k)[None, :, None]
    return P[:, k] * (1 - a) + P[:, k + 1] * a


def energy_single(wp, peds, ref):
    """Byte-identical to 41/43/44: collision term + lateral corridor (+-1 m)."""
    pos = path_points(wp)
    t = _T.to(wp.device)
    pp = peds[:, None, :2] + peds[:, None, 2:] * t[None, :, None]
    D = (pos - pp).norm(dim=-1)
    act = (D < R_GUIDE).float()
    e = ((1 - D / R_GUIDE).clamp_min(0) ** 2 * act).sum(1) / (act.sum(1) + 1e-6)
    return e + ((wp[..., 0] - ref[..., 0]).abs() - 1.0).clamp_min(0).pow(2).mean(-1)


def sample_ped_futures(ped_nom, n_futures, seed):
    """Byte-identical to 44's generator, just asked for more draws.

    Future 0 is the nominal; 1..n-1 are heading (+-15 deg), speed (+-20 %) and
    hesitation (15 %) perturbations. Seeds are per batch as in 44, so futures
    0..7 are exactly 44's futures and 8..39 are fresh draws from the same model.
    """
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


def turn_deg(wp):
    d = wp[:, 1:] - wp[:, :-1]
    h = torch.atan2(d[..., 1], d[..., 0])
    dh = (h[:, 1:] - h[:, :-1]).abs()
    dh = torch.atan2(torch.sin(dh), torch.cos(dh)).abs()
    return torch.rad2deg(dh).max(-1).values


ARMS = ("ung", "pt", "exp", "cvar", "gui")
KEYS = ([f"ins_{a}" for a in ARMS] + [f"hel_{a}" for a in ARMS] +
        [f"turn_{a}" for a in ARMS] + ["lat_pt", "lat_exp", "clr_ins_pt", "clr_ins_exp",
                                       "clr_hel_pt", "clr_hel_exp", "flip"])


def run():
    acc = {k: [] for k in KEYS}
    for k in range(0, len(S), 16):
        idx = list(range(k, min(k + 16, len(S))))
        n = len(idx)
        Hb = torch.from_numpy(np.asarray(H[idx])).to(dev).float()
        sc = torch.tensor([S[i]["step_scale"] for i in idx], device=dev).repeat_interleave(NCAND)
        ped = torch.tensor([S[i]["ped"] for i in idx], device=dev, dtype=torch.float32)
        with torch.no_grad():
            z = head.pool(Hb).repeat_interleave(NCAND, 0)
        nz = noise.repeat(n, 1, 1)

        wp_u = sample(z, nz, sc)
        p_rep = ped.repeat_interleave(NCAND, 0)
        wp_g = sample(z, nz, sc, p_rep, wp_u)      # ref = own unguided path, as in 44

        wp_u4 = wp_u.view(n, NCAND, 5, 2)
        wp_g4 = wp_g.view(n, NCAND, 5, 2)
        ref_pool = wp_u.clone()                    # exactly 44: ref_pool = wp_pool.clone()

        fut = sample_ped_futures(ped, N_ALL, seed=100 + k)      # (n, 40, 4)

        # ---- score/select using the FIRST 8 futures only ----
        E = []
        for f in range(N_PLAN):
            pf = fut[:, f].repeat_interleave(NCAND, 0)
            E.append(energy_single(wp_u, pf, ref_pool).view(n, NCAND))
        E = torch.stack(E, -1)                                  # (n, 16, 8)
        E_nom, E_exp = E[..., 0], E.mean(-1)
        E_cvar = torch.topk(E, k=N_PLAN // 2, dim=-1).values.mean(-1)

        ar = torch.arange(n, device=dev)
        i_pt, i_exp = E_nom.argmin(1), E_exp.argmin(1)
        sel = {"ung": wp_u4[:, 0],
               "pt": wp_u4[ar, i_pt],
               "exp": wp_u4[ar, i_exp],
               "cvar": wp_u4[ar, E_cvar.argmin(1)],
               "gui": wp_g4[:, 0]}
        acc["flip"].append((i_exp != i_pt).float().cpu())

        # ---- score on the realised future 1 (44's protocol) ----
        for a in ARMS:
            acc[f"ins_{a}"].append((min_ped_distance(sel[a], fut[:, REALISED]) < CONTACT).float().cpu())

        # ---- score on the 32 held-out futures, averaged ----
        # hs[a] collects one (n,) contact vector per held-out future, then is
        # stacked into a single (N_HELD, n) tensor per batch. appending that once
        # keeps acc["hel_*"] a flat list of per-batch tensors, so torch.cat over
        # dim=1 later gives (N_HELD, n_states) exactly once.
        hs = {a: [] for a in ARMS}
        for f in range(N_PLAN, N_ALL):
            for a in ARMS:
                hs[a].append((min_ped_distance(sel[a], fut[:, f]) < CONTACT).float().cpu())
        for a in ARMS:
            acc[f"hel_{a}"].append(torch.stack(hs[a], 0))   # (32, n)
        # turn / lateral / clearance
        for a in ARMS:
            acc[f"turn_{a}"].append(turn_deg(sel[a]).cpu())
        acc["lat_pt"].append((sel["pt"][..., 0] - ref_pool.view(n, NCAND, 5, 2).mean(1)[..., 0]).abs().max(-1).values.cpu())
        acc["lat_exp"].append((sel["exp"][..., 0] - ref_pool.view(n, NCAND, 5, 2).mean(1)[..., 0]).abs().max(-1).values.cpu())
        for tag, a in (("ins_pt", "pt"), ("ins_exp", "exp")):
            acc["clr_" + tag].append(min_ped_distance(sel[a], fut[:, REALISED]).cpu())
        for a in ("pt", "exp"):
            dm = [min_ped_distance(sel[a], fut[:, f]) for f in range(N_PLAN, N_ALL)]
            acc[f"clr_hel_{a}"].append(torch.stack(dm, 0).mean(0).cpu())
    return acc


def cluster_boot(per_person_vals, per_person_n, rng, B=2000):
    """Ratio estimator bootstrap: resample people, weight by state counts."""
    v = np.asarray(per_person_vals, float)
    n = np.asarray(per_person_n, float)
    mean = v.sum() / n.sum()
    out = []
    idx_all = np.arange(len(v))
    for _ in range(B):
        b = rng.choice(idx_all, len(v))
        out.append(v[b].sum() / max(n[b].sum(), 1e-9))
    return mean, [float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))]


conds = sorted({s["cond"] for s in S})
people = np.array([s["sprite"] for s in S])
uniq = np.unique(people)
out = {"n_states": len(S), "n_people": int(len(uniq)), "n_cand": NCAND,
       "n_plan": N_PLAN, "n_held": N_HELD, "conds": conds,
       "note": "plan futures 0..7 (=44); test futures 8..39 never seen by selector; "
               "ins_* = 44 protocol on future 1 (anchor only)",
       "res": {}}

for spec in POL:
    name, _, ck = spec.partition(":")
    head.load_state_dict(PUB if name == "published" else torch.load(os.path.join(PILOT, name, ck)))
    acc = run()
    res = {}
    for cd in conds:
        m = np.array([s["cond"] == cd for s in S])
        per_people = people
        d = {}
        IN, HEL = {}, {}
        for a in ARMS:
            IN[a] = torch.cat(acc[f"ins_{a}"]).numpy()
            HEL[a] = torch.cat([t.mean(0) for t in acc[f"hel_{a}"]]).numpy()
            for tag, arr in (("ins", IN[a]), ("hel", HEL[a])):
                vals = [arr[m & (per_people == u)].sum() for u in uniq if (m & (per_people == u)).any()]
                cnts = [int((m & (per_people == u)).sum()) for u in uniq if (m & (per_people == u)).any()]
                mean, ci = cluster_boot(vals, cnts, np.random.RandomState(0))
                d[f"{tag}_{a}"] = {"mean": float(mean * 100), "ci": [c * 100 for c in ci]}
            d[f"turn_{a}"] = float(torch.cat(acc[f"turn_{a}"]).numpy()[m].mean())

        # ---- paired per-person contrasts on the HELD-OUT futures ----
        # The claim we care about is "the distribution helps the same state that the
        # point forecast fails", so the contrast must be paired, not two marginals.
        # Bootstrapping people with the difference taken *within* person removes the
        # state difficulty term that dominates the marginal CIs.
        def paired(a, b):
            idx = [u for u in uniq if (m & (per_people == u)).any()]
            dv = [HEL[a][m & (per_people == u)].sum() - HEL[b][m & (per_people == u)].sum() for u in idx]
            dn = [int((m & (per_people == u)).sum()) for u in idx]
            mu, ci = cluster_boot(dv, dn, np.random.RandomState(1))
            return {"mean": float(mu * 100), "ci": [c * 100 for c in ci]}

        d["d_hel_exp_pt"] = paired("exp", "pt")
        d["d_hel_cvar_pt"] = paired("cvar", "pt")
        d["d_hel_gui_pt"] = paired("gui", "pt")
        d["d_hel_ung_pt"] = paired("ung", "pt")
        for k2 in ("lat_pt", "lat_exp", "clr_ins_pt", "clr_ins_exp", "clr_hel_pt", "clr_hel_exp"):
            d[k2] = float(torch.cat(acc[k2]).numpy()[m].mean())
        d["flip_exp_vs_pt"] = float(torch.cat(acc["flip"]).numpy()[m].mean() * 100)
        d["n_states"] = int(m.sum())
        res[cd] = d
        print(f"{name:10s} {cd:16s} "
              f"INSAMPLE ung {d['ins_ung']['mean']:5.1f} pt {d['ins_pt']['mean']:5.1f} "
              f"exp {d['ins_exp']['mean']:5.1f} cvar {d['ins_cvar']['mean']:5.1f} "
              f"gui {d['ins_gui']['mean']:5.1f} || HELDOUT "
              f"ung {d['hel_ung']['mean']:5.1f} pt {d['hel_pt']['mean']:5.1f} "
              f"exp {d['hel_exp']['mean']:5.1f} cvar {d['hel_cvar']['mean']:5.1f} "
              f"gui {d['hel_gui']['mean']:5.1f} || PAIRED d(exp-pt) "
              f"{d['d_hel_exp_pt']['mean']:+5.1f} [{d['d_hel_exp_pt']['ci'][0]:+.1f},"
              f"{d['d_hel_exp_pt']['ci'][1]:+.1f}] "
              f"d(cvar-pt) {d['d_hel_cvar_pt']['mean']:+5.1f} [{d['d_hel_cvar_pt']['ci'][0]:+.1f},"
              f"{d['d_hel_cvar_pt']['ci'][1]:+.1f}] "
              f"d(gui-pt) {d['d_hel_gui_pt']['mean']:+5.1f} [{d['d_hel_gui_pt']['ci'][0]:+.1f},"
              f"{d['d_hel_gui_pt']['ci'][1]:+.1f}]", flush=True)
    out["res"][name] = res
    json.dump(out, open("results_46_heldout.json", "w"), indent=1)

print("DONE")
