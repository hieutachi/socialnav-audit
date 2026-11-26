"""41 - Evaluate on the Penn-Fudan held-out pedestrian suite (40): 19 backgrounds x 5 conditions x 20 people each
(338 different people, never seen in training).  Same sampler and selection as 38 v3 (trust radius 0.3 chosen on valx),
no new tuning.  Policies: published, il3_s0..s2, grpo3_s0..s2_b1; each unguided and with oracle / static guidance.
Contact = path within 0.45 m of the person (16 noises per state).  95 % CI: bootstrap over people (cluster resampling).
Output: results_41_pf.json, log on stdout.
"""
import json
import os
import numpy as np
import torch
from sn_common import SocialNav
from pilot_common import PILOT, CONTACT, prepare_head, min_ped_distance

dev = torch.device("cuda:0")
POL = ["published", "il3_s0:last.pt", "il3_s1:last.pt", "il3_s2:last.pt", "grpo3_s0_b1:best.pt", "grpo3_s1_b1:best.pt",
       "grpo3_s2_b1:best.pt"]
DELTA, R_GUIDE, STEPS = 0.3, 1.2, 5
meta = json.load(open(os.path.join(PILOT, "states_pf.json")))
S = meta["states"][:meta["done"]]
H = np.load(os.path.join(PILOT, "H_pf.npy"), mmap_mode="r")
sn = SocialNav()
head = prepare_head(sn.model)
sn.model.model.to("cpu"); sn.model.lm_head.to("cpu"); torch.cuda.empty_cache()
PUB = head.state_dict()
noise = torch.randn(16, 5, 2, generator=torch.Generator().manual_seed(123)).to(dev)
_T = torch.linspace(0, 5, 51)


def path_points(wp):
    t = _T.to(wp.device); P = torch.cat([torch.zeros_like(wp[:, :1]), wp], 1)
    k = t.floor().clamp(max=4).long(); a = (t - k)[None, :, None]
    return P[:, k] * (1 - a) + P[:, k + 1] * a


def energy(wp, peds, ref):
    pos = path_points(wp); t = _T.to(wp.device)
    pp = peds[:, None, :2] + peds[:, None, 2:] * t[None, :, None]
    D = (pos - pp).norm(dim=-1); act = (D < R_GUIDE).float()
    e = ((1 - D / R_GUIDE).clamp_min(0) ** 2 * act).sum(1) / (act.sum(1) + 1e-6)
    return e + ((wp[..., 0] - ref[..., 0]).abs() - 1.0).clamp_min(0).pow(2).mean(-1)


def sample(z, nz, sc, peds=None, ref=None):
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


def run(mode):
    """mode: unguided | oracle | static. Returns per-state contact fraction and mean lateral shift vs unguided."""
    cont, lat = [], []
    for k in range(0, len(S), 16):
        idx = list(range(k, min(k + 16, len(S))))
        Hb = torch.from_numpy(np.asarray(H[idx])).to(dev).float()
        sc = torch.tensor([S[i]["step_scale"] for i in idx], device=dev).repeat_interleave(16)
        ped = torch.tensor([S[i]["ped"] for i in idx], device=dev, dtype=torch.float32)
        with torch.no_grad():
            z = head.pool(Hb).repeat_interleave(16, 0)
        nz = noise.repeat(len(idx), 1, 1)
        ref = sample(z, nz, sc)
        if mode == "unguided":
            wp = ref
        else:
            p = ped.repeat_interleave(16, 0).clone()
            if mode == "static":
                p[:, 2:] = 0
            wp = sample(z, nz, sc, p, ref)
        wp, ref = wp.view(len(idx), 16, 5, 2), ref.view(len(idx), 16, 5, 2)
        cont.append((min_ped_distance(wp, ped[:, None]) < CONTACT).float().mean(1).cpu())
        lat.append((wp - ref)[..., 0].abs().max(-1).values.mean(1).cpu())
    return torch.cat(cont).numpy(), torch.cat(lat).numpy()


conds = sorted({s["cond"] for s in S})
people = np.array([s["sprite"] for s in S])
uniq = np.unique(people)
rng = np.random.RandomState(0)
boot = [rng.choice(len(uniq), len(uniq)) for _ in range(1000)]


def summarise(c):
    out = {}
    for cd in conds:
        m = np.array([s["cond"] == cd for s in S])
        per = {u: c[m & (people == u)].mean() for u in uniq if (m & (people == u)).any()}
        keys = list(per)
        vals = np.array([per[u] for u in keys])
        bs = [vals[rng.choice(len(vals), len(vals))].mean() for _ in range(1000)]
        out[cd] = dict(mean=float(c[m].mean() * 100), ci=[float(np.percentile(bs, 2.5) * 100), float(np.percentile(bs, 97.5) * 100)],
                       n_states=int(m.sum()), n_people=len(keys))
    return out


res = {}
for spec in POL:
    name, _, ck = spec.partition(":")
    head.load_state_dict(PUB if name == "published" else torch.load(os.path.join(PILOT, name, ck)))
    res[name] = {}
    for mode in ("unguided", "oracle", "static"):
        c, l = run(mode)
        res[name][mode] = summarise(c)
        res[name][mode]["latdev_mean_m"] = float(l.mean())
        print(name, mode, " | ".join(f"{cd}: {v['mean']:.1f} [{v['ci'][0]:.1f},{v['ci'][1]:.1f}]" for cd, v in res[name][mode].items()
                                     if isinstance(v, dict)), f"| lat {l.mean():.2f} m", flush=True)
    json.dump(dict(n_people=int(len(uniq)), n_states=len(S), conds=conds, res=res), open("results_41_pf.json", "w"), indent=1)
print("DONE")
