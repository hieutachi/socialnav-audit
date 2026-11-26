"""44 - Evaluate Pedestrian Motion Uncertainty in Robot Planning.

Compares:
1. Unguided Prior (16 samples, mean / single pick)
2. Point-Prediction Planning (Rerank-16 and Guidance against deterministic CV model)
3. Distributional Planning (Expected Risk and Risk-Averse CVaR over multimodal / uncertain pedestrian futures)
4. Distributional Guidance (Gradient guidance with expected energy over pedestrian motion distribution)

Evaluated under:
- Nominal motion (exact constant velocity)
- Realized motion perturbations (heading change +/- 15 deg, speed variation +/- 25%, hesitation/stop)

Outputs:
- results_44_uncertainty.json
- Direct comparison metrics: contact rate (%), min clearance (m), lateral deviation (m), max turn (deg), action flip rate (%)
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
N_FUTURES = 8  # Number of sampled pedestrian motion futures per state

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
_T = torch.linspace(0, 5, 51).to(dev)

def path_points(wp):
    P = torch.cat([torch.zeros_like(wp[:, :1]), wp], 1)
    k = _T.floor().clamp(max=4).long()
    a = (_T - k)[None, :, None]
    return P[:, k] * (1 - a) + P[:, k + 1] * a

def energy_single(wp, peds, ref):
    pos = path_points(wp)
    pp = peds[:, None, :2] + peds[:, None, 2:] * _T[None, :, None]
    D = (pos - pp).norm(dim=-1)
    act = (D < R_GUIDE).float()
    e = ((1 - D / R_GUIDE).clamp_min(0) ** 2 * act).sum(1) / (act.sum(1) + 1e-6)
    return e + ((wp[..., 0] - ref[..., 0]).abs() - 1.0).clamp_min(0).pow(2).mean(-1)

def sample_ped_futures(ped_nom, n_futures=N_FUTURES, seed=42):
    """
    Given nominal pedestrian [x, y, vx, vy], generate n_futures plausible future motion trajectories.
    Models uncertainty:
    - Heading perturbation: +/- 15 deg std
    - Speed perturbation: +/- 20% std
    - Stop / hesitation probability: 15%
    """
    B = ped_nom.shape[0]
    futures = []
    rng = torch.Generator(device=dev).manual_seed(seed)
    
    # Nominal is future 0
    futures.append(ped_nom.unsqueeze(1))
    
    for f in range(1, n_futures):
        p = ped_nom.clone()
        vx, vy = p[:, 2], p[:, 3]
        speed = (vx**2 + vy**2).sqrt()
        is_moving = speed > 0.1
        
        # Heading angle
        ang = torch.atan2(vx, vy) # ego frame: y forward, x lateral
        d_ang = torch.randn(B, generator=rng, device=dev) * (15.0 * np.pi / 180.0)
        new_ang = ang + d_ang
        
        # Speed scaling
        speed_scale = 1.0 + torch.randn(B, generator=rng, device=dev) * 0.20
        # Hesitation / stop
        stop_mask = (torch.rand(B, generator=rng, device=dev) < 0.15) & is_moving
        speed_scale = torch.where(stop_mask, torch.zeros_like(speed_scale), speed_scale.clamp(0.2, 1.8))
        
        new_speed = speed * speed_scale
        p[:, 2] = torch.where(is_moving, new_speed * torch.sin(new_ang), vx)
        p[:, 3] = torch.where(is_moving, new_speed * torch.cos(new_ang), vy)
        futures.append(p.unsqueeze(1))
        
    return torch.cat(futures, dim=1) # (B, n_futures, 4)

def turn_deg(wp):
    d = wp[:, 1:] - wp[:, :-1]
    h = torch.atan2(d[..., 1], d[..., 0])
    dh = (h[:, 1:] - h[:, :-1]).abs()
    dh = torch.atan2(torch.sin(dh), torch.cos(dh)).abs()
    return torch.rad2deg(dh).max(-1).values

def run_eval():
    acc = {k: [] for k in (
        "cont_ung", "cont_pt_rr", "cont_exp_rr", "cont_cvar_rr", "cont_pt_gui",
        "cont_pert_ung", "cont_pert_pt_rr", "cont_pert_exp_rr", "cont_pert_cvar_rr", "cont_pert_pt_gui",
        "dmin_nom_pt_rr", "dmin_nom_exp_rr", "dmin_pert_pt_rr", "dmin_pert_exp_rr",
        "turn_ung", "turn_pt_rr", "turn_exp_rr", "turn_pt_gui",
        "lat_pt_rr", "lat_exp_rr", "lat_pt_gui",
        "flip_rate_exp_vs_pt"
    )}
    
    for k in range(0, len(S), 16):
        idx = list(range(k, min(k + 16, len(S))))
        n = len(idx)
        Hb = torch.from_numpy(np.asarray(H[idx])).to(dev).float()
        sc = torch.tensor([S[i]["step_scale"] for i in idx], device=dev).repeat_interleave(NCAND)
        ped_nom = torch.tensor([S[i]["ped"] for i in idx], device=dev, dtype=torch.float32)
        
        with torch.no_grad():
            z = head.pool(Hb).repeat_interleave(NCAND, 0)
        nz = noise.repeat(n, 1, 1)
        
        # 1. Generate unguided pool (16 per state)
        x = nz.clone()
        for step_i in range(STEPS):
            t = torch.tensor([step_i / STEPS], device=dev)
            with torch.no_grad():
                x = x + head.flow.predict_velocity(z, x, t).float() / STEPS
        with torch.no_grad():
            wp_pool = head.model.recover_waypoints_from_pred(x) * sc[:, None, None] # (n*16, 5, 2)
            
        wp_u4 = wp_pool.view(n, NCAND, 5, 2)
        
        # 2. Generate point-guided pool
        x_g = nz.clone()
        p_rep = ped_nom.repeat_interleave(NCAND, 0)
        ref_pool = wp_pool.clone()
        for step_i in range(STEPS):
            t = torch.tensor([step_i / STEPS], device=dev)
            xg = x_g.detach().requires_grad_(True)
            v = head.flow.predict_velocity(z, xg, t).float()
            wp_cur = head.model.recover_waypoints_from_pred(xg + (1 - step_i / STEPS) * v) * sc[:, None, None]
            g, = torch.autograd.grad(energy_single(wp_cur, p_rep, ref_pool).sum(), xg)
            norm = g.flatten(1).norm(dim=1)[:, None, None]
            x_g = (xg + v / STEPS - torch.where(norm > 0, g * DELTA / norm.clamp_min(1e-12), torch.zeros_like(g))).detach()
        x_g = x_g.clamp(-1, 1)
        with torch.no_grad():
            wp_gui = (head.model.recover_waypoints_from_pred(x_g) * sc[:, None, None]).view(n, NCAND, 5, 2)
            
        # 3. Pedestrian motion futures
        ped_futures = sample_ped_futures(ped_nom, n_futures=N_FUTURES, seed=100 + k) # (n, N_FUTURES, 4)
        
        # Realized perturbed human (Scenario: human took future #1 which had deviation)
        ped_realized = ped_futures[:, 1] # (n, 4)
        
        # Evaluate energies for each of 16 robot candidates across all pedestrian futures
        # wp_u4 is (n, 16, 5, 2) -> flatten to (n*16, 5, 2)
        # Compute energy against nominal ped
        E_nom = energy_single(wp_pool, p_rep, ref_pool).view(n, NCAND) # (n, 16)
        
        # Compute energy against all futures:
        # shape (n, 16, N_FUTURES)
        E_all_futures = []
        for f in range(N_FUTURES):
            pf = ped_futures[:, f].repeat_interleave(NCAND, 0)
            ef = energy_single(wp_pool, pf, ref_pool).view(n, NCAND)
            E_all_futures.append(ef.unsqueeze(-1))
        E_stack = torch.cat(E_all_futures, dim=-1) # (n, 16, N_FUTURES)
        
        E_exp = E_stack.mean(dim=-1) # (n, 16) - expected risk
        E_cvar = torch.topk(E_stack, k=max(1, N_FUTURES // 2), dim=-1).values.mean(dim=-1) # worst half
        
        # Selection indices
        idx_pt = E_nom.argmin(dim=1)
        idx_exp = E_exp.argmin(dim=1)
        idx_cvar = E_cvar.argmin(dim=1)
        
        ar = torch.arange(n, device=dev)
        sel_ung = wp_u4[:, 0] # first sample / unguided baseline
        sel_pt = wp_u4[ar, idx_pt]
        sel_exp = wp_u4[ar, idx_exp]
        sel_cvar = wp_u4[ar, idx_cvar]
        sel_gui = wp_gui[:, 0] # guided pick
        
        # Flip rate: does distribution planning pick a different trajectory than point planning?
        flipped = (idx_exp != idx_pt).float()
        acc["flip_rate_exp_vs_pt"].append(flipped.cpu())
        
        # Contact and clearance under NOMINAL pedestrian
        d_nom_pt = min_ped_distance(sel_pt, ped_nom)
        d_nom_exp = min_ped_distance(sel_exp, ped_nom)
        acc["dmin_nom_pt_rr"].append(d_nom_pt.cpu())
        acc["dmin_nom_exp_rr"].append(d_nom_exp.cpu())
        
        acc["cont_ung"].append((min_ped_distance(sel_ung, ped_nom) < CONTACT).float().cpu())
        acc["cont_pt_rr"].append((d_nom_pt < CONTACT).float().cpu())
        acc["cont_exp_rr"].append((d_nom_exp < CONTACT).float().cpu())
        acc["cont_cvar_rr"].append((min_ped_distance(sel_cvar, ped_nom) < CONTACT).float().cpu())
        acc["cont_pt_gui"].append((min_ped_distance(sel_gui, ped_nom) < CONTACT).float().cpu())
        
        # Contact and clearance under REALIZED PERTURBED pedestrian
        d_pert_pt = min_ped_distance(sel_pt, ped_realized)
        d_pert_exp = min_ped_distance(sel_exp, ped_realized)
        acc["dmin_pert_pt_rr"].append(d_pert_pt.cpu())
        acc["dmin_pert_exp_rr"].append(d_pert_exp.cpu())
        
        acc["cont_pert_ung"].append((min_ped_distance(sel_ung, ped_realized) < CONTACT).float().cpu())
        acc["cont_pert_pt_rr"].append((d_pert_pt < CONTACT).float().cpu())
        acc["cont_pert_exp_rr"].append((d_pert_exp < CONTACT).float().cpu())
        acc["cont_pert_cvar_rr"].append((min_ped_distance(sel_cvar, ped_realized) < CONTACT).float().cpu())
        acc["cont_pert_pt_gui"].append((min_ped_distance(sel_gui, ped_realized) < CONTACT).float().cpu())
        
        # Smoothness & distortion metrics
        mean_ung = wp_u4.mean(dim=1)
        acc["turn_ung"].append(turn_deg(mean_ung).cpu())
        acc["turn_pt_rr"].append(turn_deg(sel_pt).cpu())
        acc["turn_exp_rr"].append(turn_deg(sel_exp).cpu())
        acc["turn_pt_gui"].append(turn_deg(sel_gui).cpu())
        
        acc["lat_pt_rr"].append((sel_pt[..., 0] - mean_ung[..., 0]).abs().max(-1).values.cpu())
        acc["lat_exp_rr"].append((sel_exp[..., 0] - mean_ung[..., 0]).abs().max(-1).values.cpu())
        acc["lat_pt_gui"].append((sel_gui[..., 0] - mean_ung[..., 0]).abs().max(-1).values.cpu())
        
    return {k: torch.cat(v).numpy() for k, v in acc.items()}

conds = sorted({s["cond"] for s in S})
people = np.array([s["sprite"] for s in S])
uniq = np.unique(people)
rng = np.random.RandomState(0)

results = {}

for spec in POL:
    name, _, ck = spec.partition(":")
    head.load_state_dict(PUB if name == "published" else torch.load(os.path.join(PILOT, name, ck)))
    print(f"\n================ Running evaluation for: {name} ================", flush=True)
    r = run_eval()
    results[name] = {}
    
    for cd in conds:
        m = np.array([s["cond"] == cd for s in S])
        d = {}
        # Contact rates (mean %)
        d["cont_nom_ung"] = float(r["cont_ung"][m].mean() * 100)
        d["cont_nom_pt_rr"] = float(r["cont_pt_rr"][m].mean() * 100)
        d["cont_nom_exp_rr"] = float(r["cont_exp_rr"][m].mean() * 100)
        d["cont_nom_cvar_rr"] = float(r["cont_cvar_rr"][m].mean() * 100)
        d["cont_nom_pt_gui"] = float(r["cont_pt_gui"][m].mean() * 100)
        
        d["cont_pert_ung"] = float(r["cont_pert_ung"][m].mean() * 100)
        d["cont_pert_pt_rr"] = float(r["cont_pert_pt_rr"][m].mean() * 100)
        d["cont_pert_exp_rr"] = float(r["cont_pert_exp_rr"][m].mean() * 100)
        d["cont_pert_cvar_rr"] = float(r["cont_pert_cvar_rr"][m].mean() * 100)
        d["cont_pert_pt_gui"] = float(r["cont_pert_pt_gui"][m].mean() * 100)
        
        # Clearances
        d["dmin_nom_pt"] = float(r["dmin_nom_pt_rr"][m].mean())
        d["dmin_nom_exp"] = float(r["dmin_nom_exp_rr"][m].mean())
        d["dmin_pert_pt"] = float(r["dmin_pert_pt_rr"][m].mean())
        d["dmin_pert_exp"] = float(r["dmin_pert_exp_rr"][m].mean())
        
        # Flip rate
        d["flip_rate_exp_vs_pt"] = float(r["flip_rate_exp_vs_pt"][m].mean() * 100)
        
        # Smoothness
        d["turn_ung"] = float(r["turn_ung"][m].mean())
        d["turn_pt_rr"] = float(r["turn_pt_rr"][m].mean())
        d["turn_exp_rr"] = float(r["turn_exp_rr"][m].mean())
        d["turn_pt_gui"] = float(r["turn_pt_gui"][m].mean())
        
        d["lat_pt_rr"] = float(r["lat_pt_rr"][m].mean())
        d["lat_exp_rr"] = float(r["lat_exp_rr"][m].mean())
        d["lat_pt_gui"] = float(r["lat_pt_gui"][m].mean())
        
        d["n_states"] = int(m.sum())
        results[name][cd] = d
        
        print(f"[{cd}] (n={d['n_states']}):", flush=True)
        print(f"  Nominal Motion Contact:  Unguided={d['cont_nom_ung']:4.1f}% | Point-RR={d['cont_nom_pt_rr']:4.1f}% | Exp-RR={d['cont_nom_exp_rr']:4.1f}% | CVaR-RR={d['cont_nom_cvar_rr']:4.1f}% | Point-GUI={d['cont_nom_pt_gui']:4.1f}%", flush=True)
        print(f"  Perturbed Motion Contact: Unguided={d['cont_pert_ung']:4.1f}% | Point-RR={d['cont_pert_pt_rr']:4.1f}% | Exp-RR={d['cont_pert_exp_rr']:4.1f}% | CVaR-RR={d['cont_pert_cvar_rr']:4.1f}% | Point-GUI={d['cont_pert_pt_gui']:4.1f}%", flush=True)
        print(f"  Action Flip Rate (Exp vs Point): {d['flip_rate_exp_vs_pt']:4.1f}% | Min Clearance Perturbed: Point={d['dmin_pert_pt']:.2f}m vs Exp={d['dmin_pert_exp']:.2f}m", flush=True)
        print(f"  Turn (deg): Ung={d['turn_ung']:.1f} | Pt-RR={d['turn_pt_rr']:.1f} | Exp-RR={d['turn_exp_rr']:.1f} | Pt-GUI={d['turn_pt_gui']:.1f}", flush=True)

out_path = os.path.join(os.path.dirname(__file__), "results_44_uncertainty.json")
json.dump(dict(results=results, n_states=len(S), conds=conds), open(out_path, "w"), indent=2)
print(f"\nALL COMPLETED. Results saved to {out_path}", flush=True)
