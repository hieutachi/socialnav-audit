"""43 - Matched-budget selection baseline for the energy guidance of 38/41.

Question this answers, and why it must be answered before writing anything else:
  41 samples 16 noise draws per state, and reports the FRACTION of those 16 candidates whose predicted path
  enters the 0.45 m pedestrian radius.  That is a distributional number P(contact | noise).  Any selection
  baseline that hands back one trajectory per state is a point estimate, so the two are not directly
  comparable.  The honest comparison under an equal sampling budget of 16 forward passes is:

      A. guided     : run guidance on all 16 draws, report mean contact probability
      B. unguided   : run the prior on all 16 draws, report mean contact probability
      C. rerank16   : run the prior on all 16 draws, pick argmin energy, report contact of the pick
      D. rerank16g  : run guidance on all 16 draws, pick argmin energy, report contact of the pick

  A vs C is the decisive contrast: does the gradient buy anything beyond "sample 16, keep the safest"?
  C is best-of-16 and therefore has an advantage in raw contact; that is exactly why the plausibility
  columns (lateral deviation, max turn) are reported too.  If C matches A on contact while keeping the
  prior's smoothness, gradient guidance contributes nothing measurable at this budget.

Energy is the SAME function 41 optimises (collision term + lateral corridor term), so the selection
baseline cannot be accused of using a different criterion.

Also reported: E_unguided_mean, E_unguided_best, E_guided_mean per condition.  If E_guided_mean is not
below E_unguided_best, the gradient is not finding lower-energy trajectories than selection does and the
"guidance optimises the energy" story fails on its own terms.

Unrun in this session: C:\\Users\\N4G\\qcom_socialnav is access-denied and the available interpreter has
no torch.  Same sampler, same noise seed, same policy list and same statistics as 41 so the numbers drop
straight into the existing table.  Output: results_43_rerank.json + stdout log.
"""
import json
import os
import numpy as np
import torch
from sn_common import SocialNav
from pilot_common import PILOT, CONTACT, prepare_head, min_ped_distance

dev = torch.device("cuda:0")
POL = ["published", "il3_s0:last.pt", "il3_s1:last.pt", "il3_s2:last.pt"]
DELTA, R_GUIDE, STEPS = 0.3, 1.2, 5
NCAND = 16

meta = json.load(open(os.path.join(PILOT, "states_pf.json")))
S = meta["states"][:meta["done"]]
H = np.load(os.path.join(PILOT, "H_pf.npy"), mmap_mode="r")
sn = SocialNav()
head = prepare_head(sn.model)
sn.model.model.to("cpu")
sn.model.lm_head.to("cpu")
torch.cuda.empty_cache()
PUB = head.state_dict()
noise = torch.randn(NCAND, 5, 2, generator=torch.Generator().manual_seed(123)).to(dev)  # identical to 41
_T = torch.linspace(0, 5, 51)


def path_points(wp):
    t = _T.to(wp.device)
    P = torch.cat([torch.zeros_like(wp[:, :1]), wp], 1)
    k = t.floor().clamp(max=4).long()
    a = (t - k)[None, :, None]
    return P[:, k] * (1 - a) + P[:, k + 1] * a


def energy(wp, peds, ref):
    """Byte-identical to 41 so the selection criterion is the criterion guidance optimises."""
    pos = path_points(wp)
    t = _T.to(wp.device)
    pp = peds[:, None, :2] + peds[:, None, 2:] * t[None, :, None]
    D = (pos - pp).norm(dim=-1)
    act = (D < R_GUIDE).float()
    e = ((1 - D / R_GUIDE).clamp_min(0) ** 2 * act).sum(1) / (act.sum(1) + 1e-6)
    return e + ((wp[..., 0] - ref[..., 0]).abs() - 1.0).clamp_min(0).pow(2).mean(-1)


def sample(z, nz, sc, peds=None, ref=None):
    """Identical to 41. With peds=None this is the unguided prior; with peds not None it is guidance."""
    x = nz.clone()
    for k in range(STEPS):
        t = torch.tensor([k / STEPS], device=dev)
        if peds is not None:
            xg = x.detach().requires_grad_(True)
            v = head.flow.predict_velocity(z, xg, t).float()
            wp = head.model.recover_waypoints_from_pred(xg + (1 - k / STEPS) * v) * sc[:, None, None]
            g, = torch.autograd.grad(energy(wp, peds, ref).sum(), xg)
            n = g.flatten(1).norm(dim=1)[:, None, None]
            x = (xg + v / STEPS - torch.where(n > 0, g * DELTA / n.clamp_min(1e-12), torch.zeros_like(g))).detach()
        else:
            with torch.no_grad():
                x = x + head.flow.predict_velocity(z, x, t).float() / STEPS
    if peds is not None:
        x = x.clamp(-1, 1)
    with torch.no_grad():
        return head.model.recover_waypoints_from_pred(x) * sc[:, None, None]


def turn_deg(wp):
    """Max absolute heading change between consecutive segments, degrees. Same spirit as 39 max_turn."""
    d = wp[:, 1:] - wp[:, :-1]
    h = torch.atan2(d[..., 1], d[..., 0])
    dh = (h[:, 1:] - h[:, :-1]).abs()
    dh = torch.atan2(torch.sin(dh), torch.cos(dh)).abs()
    return torch.rad2deg(dh).max(-1).values


def run():
    """One pass over the suite for one policy; returns per-state dict of arrays."""
    acc = {k: [] for k in ("p_cont_ung", "p_cont_gui", "cont_rerank", "cont_rerank_g", "lat_rerank",
                           "lat_gui", "turn_ung", "turn_rerank", "turn_gui", "E_ung_mean", "E_ung_best",
                           "E_gui_mean", "E_rerank")}
    for k in range(0, len(S), 16):
        idx = list(range(k, min(k + 16, len(S))))
        n = len(idx)
        Hb = torch.from_numpy(np.asarray(H[idx])).to(dev).float()
        sc = torch.tensor([S[i]["step_scale"] for i in idx], device=dev).repeat_interleave(NCAND)
        ped = torch.tensor([S[i]["ped"] for i in idx], device=dev, dtype=torch.float32)
        with torch.no_grad():
            z = head.pool(Hb).repeat_interleave(NCAND, 0)
        nz = noise.repeat(n, 1, 1)

        ref = sample(z, nz, sc)                                   # unguided pool, 16 per state
        p = ped.repeat_interleave(NCAND, 0)
        wp_g = sample(z, nz, sc, p, ref)                          # guided pool, 16 per state

        E_u = energy(ref, p, ref)                                 # ref vs itself -> corridor term is 0
        E_g = energy(wp_g, p, ref)
        au = E_u.view(n, NCAND).argmin(1)
        ag = E_g.view(n, NCAND).argmin(1)
        ar = torch.arange(n, device=dev)

        wp_u4 = ref.view(n, NCAND, 5, 2)
        wp_g4 = wp_g.view(n, NCAND, 5, 2)
        ped4 = ped[:, None]                                       # (n,1,4)

        acc["p_cont_ung"].append((min_ped_distance(ref, p) < CONTACT).float().view(n, NCAND).mean(1).cpu())
        acc["p_cont_gui"].append((min_ped_distance(wp_g, p) < CONTACT).float().view(n, NCAND).mean(1).cpu())
        sel_u = wp_u4[ar, au]
        sel_g = wp_g4[ar, ag]
        acc["cont_rerank"].append((min_ped_distance(sel_u[:, None], ped4) < CONTACT).float().squeeze(-1).cpu())
        acc["cont_rerank_g"].append((min_ped_distance(sel_g[:, None], ped4) < CONTACT).float().squeeze(-1).cpu())
        # Lateral deviation of each arm is measured against the SAME reference
        # (the unguided pool mean), so rerank and guidance are directly comparable.
        acc["lat_rerank"].append((sel_u[..., 0] - wp_u4.mean(1)[..., 0]).abs().max(-1).values.cpu())
        acc["lat_gui"].append((sel_g[..., 0] - wp_u4.mean(1)[..., 0]).abs().max(-1).values.cpu())
        acc["turn_ung"].append(turn_deg(wp_u4.mean(1)).cpu())
        acc["turn_rerank"].append(turn_deg(sel_u).cpu())
        acc["turn_gui"].append(turn_deg(sel_g).cpu())
        acc["E_ung_mean"].append(E_u.view(n, NCAND).mean(1).cpu())
        acc["E_ung_best"].append(E_u.view(n, NCAND).min(1).values.cpu())
        acc["E_gui_mean"].append(E_g.view(n, NCAND).mean(1).cpu())
        acc["E_rerank"].append(E_u.view(n, NCAND).min(1).values.cpu())
        for key in acc:
            acc[key][-1] = acc[key][-1].reshape(-1)
    return {k: torch.cat(v).numpy() for k, v in acc.items()}


def summarise(v, mask, people, uniq, rng):
    """Cluster bootstrap over people, exactly as 41 summarise()."""
    per = {u: v[mask & (people == u)].mean() for u in uniq if (mask & (people == u)).any()}
    vals = np.array([per[u] for u in sorted(per)])
    bs = [vals[rng.choice(len(vals), len(vals))].mean() for _ in range(1000)]
    return dict(mean=float(v[mask].mean()), ci=[float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))],
                n_people=len(vals))


conds = sorted({s["cond"] for s in S})
people = np.array([s["sprite"] for s in S])
uniq = np.unique(people)
rng = np.random.RandomState(0)
res = {}

for spec in POL:
    name, _, ck = spec.partition(":")
    head.load_state_dict(PUB if name == "published" else torch.load(os.path.join(PILOT, name, ck)))
    r = run()
    res[name] = {}
    for cd in conds:
        m = np.array([s["cond"] == cd for s in S])
        d = {}
        for key, scale in (("p_cont_ung", 100), ("p_cont_gui", 100), ("cont_rerank", 100), ("cont_rerank_g", 100)):
            d[key] = summarise(r[key] * scale, m, people, uniq, rng)
        for key in ("lat_rerank", "lat_gui", "turn_ung", "turn_rerank", "turn_gui",
                    "E_ung_mean", "E_ung_best", "E_gui_mean"):
            d[key] = float(r[key][m].mean())
        d["n_states"] = int(m.sum())
        res[name][cd] = d
        print(f"{name:12s} {cd:18s} "
              f"ung {d['p_cont_ung']['mean']:5.1f} | gui {d['p_cont_gui']['mean']:5.1f} | "
              f"rerank16 {d['cont_rerank']['mean']:5.1f} | rerank16g {d['cont_rerank_g']['mean']:5.1f} | "
              f"E ung {d['E_ung_mean']:.4f} best {d['E_ung_best']:.4f} gui {d['E_gui_mean']:.4f} | "
              f"turn ung {d['turn_ung']:.1f} rr {d['turn_rerank']:.1f} gui {d['turn_gui']:.1f}", flush=True)
    json.dump(dict(n_people=int(len(uniq)), n_states=len(S), n_cand=NCAND, conds=conds, res=res),
              open("results_43_rerank.json", "w"), indent=1)

print("DONE")
