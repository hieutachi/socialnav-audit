"""Shared pieces of the pedestrian-reward pilot (scripts 21-23).

Only the action branch is touched: `query_action` + the action former + the flow head
(321 M parameters, cast to float32).  The VLM is frozen and is not even run here: its
last-layer hidden states were cached by 20_build_pilot_states.py.

Sampling for the policy gradient uses the SDE of the released code
(`src/train/sde_with_logprob.py`, mode "cps"), exactly as `_grpo_sample_actions` does.
"""
import copy
import json
import math
import os
import numpy as np
import torch
from sn_common import ROOT

PILOT = os.path.join(ROOT, "pilot")
CONTACT = 0.45      # m: pedestrian radius 0.25 + robot half width 0.20 (same proxy as scripts 14 / 18)
COMFORT = 1.20      # m: beyond this the pedestrian term is zero
W_PED, W_SIM, W_SMOOTH, W_LEN = 2.0, 1.0, 1.0, 1.0     # the released run used [2, 1, 1, 1] with the map term first
VAL_TRAJ = 8        # round 3: held-out validation trajectory
# "full" = LengthReward as released (bonus for going further than the driver + penalty for less than half);
# "penalty" = only the penalty (round 2 showed the bonus lengthens every path by 13 %)
LEN_MODE = os.environ.get("SN_LEN_MODE", "full")


class HStore:
    """Several hidden-state files seen as one array; indexed with a sorted list of state numbers."""

    def __init__(self, parts):
        self.parts = parts
        self.offs = np.cumsum([0] + [p.shape[0] for p in parts])
        self.shape = (int(self.offs[-1]),) + parts[0].shape[1:]

    def __getitem__(self, idx):
        out = []
        for j, p in enumerate(self.parts):
            loc = [i - self.offs[j] for i in idx if self.offs[j] <= i < self.offs[j + 1]]
            if loc:
                out.append(np.asarray(p[loc]))
        return np.concatenate(out)


def load_states(more=None):
    """Trajectory 18 (train / val / suite / test_clean) and, if present and SN_MORE != 0, the training states of
    trajectories 1..17 appended after them (so the indices of the trajectory-18 states never change)."""
    meta = json.load(open(os.path.join(PILOT, "states.json")))
    H = np.load(os.path.join(PILOT, "H.npy"), mmap_mode="r")
    assert meta["done"] == len(meta["states"]) == H.shape[0]
    states = meta["states"]
    for s in states:
        s.setdefault("traj", 18)
    if more is None:
        more = os.environ.get("SN_MORE", "1") == "1"
    round3 = os.environ.get("SN_ROUND") == "3"
    parts = [H]
    tags = os.environ.get("SN_PARTS", "more,valx,more2" if round3 else "more").split(",") if more else []
    for tag in tags:
        mp = os.path.join(PILOT, f"states_{tag}.json")
        if not os.path.exists(mp):
            continue
        m2 = json.load(open(mp))
        n = m2["done"]
        for s in m2["states"][:n]:
            s["tag"] = tag
        states = states + m2["states"][:n]
        parts.append(np.load(os.path.join(PILOT, f"H_{tag}.npy"), mmap_mode="r")[:n])
    if round3:
        # Round 3: trajectory VAL_TRAJ and the people cut from it are a validation set of their own ("valx": clean
        # windows + placements of those people only); training states that show one of those people are dropped.
        pre = "t%02d_" % VAL_TRAJ
        for s in states:
            if s["traj"] == VAL_TRAJ:
                s["split"] = "valx" if (s.get("tag") == "valx" or s["cond"] == "none") else "drop"
            elif s["split"] == "train" and str(s.get("sprite", "")).startswith(pre):
                s["split"] = "drop"
    return states, HStore(parts)


class Head:
    """The part that is trained (or a frozen copy of it)."""

    def __init__(self, model, frozen_copy=False):
        self.model = model
        if frozen_copy:
            self.q = copy.deepcopy(model.query_action).requires_grad_(False)
            self.former = copy.deepcopy(model.query_multihead_multi_attn).requires_grad_(False)
            self.flow = copy.deepcopy(model.flow_matching_model).requires_grad_(False)
        else:
            self.q, self.former, self.flow = model.query_action, model.query_multihead_multi_attn, model.flow_matching_model

    def parameters(self):
        return [self.q] + list(self.former.parameters()) + list(self.flow.parameters())

    def state_dict(self):
        return dict(q=self.q.detach().cpu(), former={k: v.cpu() for k, v in self.former.state_dict().items()},
                    flow={k: v.cpu() for k, v in self.flow.state_dict().items()})

    def load_state_dict(self, sd):
        with torch.no_grad():
            self.q.copy_(sd["q"].to(self.q.device, self.q.dtype))
        self.former.load_state_dict(sd["former"])
        self.flow.load_state_dict(sd["flow"])

    def pool(self, H):
        return self.former(self.q.expand(H.shape[0], -1, -1), H, H).squeeze(1)

    def ode(self, z, noise, steps=5):
        """Deployed sampler: Euler integration, identical to forward(train=False)."""
        x = noise
        for k in range(steps):
            t = torch.tensor([k / steps], device=x.device)
            x = x + self.flow.predict_velocity(z, x, t).float() / steps
        return self.model.recover_waypoints_from_pred(x)

    def sde(self, z, sched, xs=None, noise=None, upto=None):
        """SDE chain with per-step log-probabilities.  xs=None: sample a new chain from `noise`.
        xs given: re-score that chain under the current weights (gradients flow into `z` and the flow head).
        upto: number of steps to evaluate (the last step is deterministic and carries no gradient)."""
        x = noise if xs is None else xs[0]
        chain, logps, means = [x], [], []
        for i, ts in enumerate(sched.timesteps[:-1][:upto]):
            ts = ts.unsqueeze(0)
            v = self.flow.predict_velocity(z, x, 1 - ts)
            x_new, logp, mean, _ = sched.sde_step_with_logprob(v_t=v, timestep=ts, x_t=x, x_t_new=None if xs is None else xs[i + 1])
            x = x_new.float().detach()
            chain.append(x)
            logps.append(logp.float())
            means.append(mean.float())
        return chain, logps, means


def make_sched(noise_level, device, steps=5):
    from src.train.sde_with_logprob import ConditionalFlowMatcherWithSigmaSchedule_CPS
    return ConditionalFlowMatcherWithSigmaSchedule_CPS(num_inference_steps=steps, noise_level=noise_level, device=device)


def prepare_head(model):
    """float32 for the trained part (bf16 cannot hold updates of this size); everything else frozen."""
    for p in model.parameters():
        p.requires_grad_(False)
    model.query_action.data = model.query_action.data.float()
    model.query_multihead_multi_attn.float()
    model.flow_matching_model.float()
    head = Head(model)
    for p in head.parameters():
        p.requires_grad_(True)
    return head


# ------------------------------------------------------------------ geometry and rewards (torch, metres)
_T = torch.linspace(0, 5, 51)


def min_ped_distance(wp_m, ped):
    """wp_m (..., 5, 2) metres in the ego frame at t=0, 1 Hz; ped (..., 4) = x, y, vx, vy.  Returns (...,)."""
    t = _T.to(wp_m.device)
    P = torch.cat([torch.zeros_like(wp_m[..., :1, :]), wp_m], dim=-2)
    k = t.floor().clamp(max=4).long()
    a = (t - k).unsqueeze(-1)
    pos = P[..., k, :] * (1 - a) + P[..., k + 1, :] * a
    pp = ped[..., None, :2] + ped[..., None, 2:] * t.unsqueeze(-1)
    return (pos - pp).norm(dim=-1).min(dim=-1).values


def ped_reward(d):
    """-1.5 .. -1 inside the contact zone (deeper is worse), -0.5 .. 0 between contact and the comfort radius, else 0.
    Same two-level shape as the released DtMapReward (-1.0 / -0.5), plus a slope so that a group whose samples
    are all in contact still has a direction to improve in."""
    inside = -1.0 - 0.5 * (CONTACT - d) / CONTACT
    near = -0.5 * (COMFORT - d) / (COMFORT - CONTACT)
    return torch.where(d < CONTACT, inside, torch.where(d < COMFORT, near, torch.zeros_like(d)))


def rewards(wp_s, step_scale, gt_m, ped, has_ped):
    """wp_s (B, G, 5, 2) in model units; step_scale (B,); gt_m (B, 5, 2); ped (B, 4); has_ped (B,) bool.
    The three non-pedestrian terms follow src/train/orms.py (SimilarityToExpertReward, SmoothnessReward, LengthReward)."""
    wp_m = wp_s * step_scale[:, None, None, None]
    gt = gt_m[:, None]
    d = min_ped_distance(wp_m, ped[:, None])
    r_ped = torch.where(has_ped[:, None], ped_reward(d), torch.zeros_like(d))
    # similarity to the human driver
    avg = (wp_m - gt).norm(dim=-1).mean(-1)
    dp, dg = wp_m[..., 1:, :] - wp_m[..., :-1, :], gt[..., 1:, :] - gt[..., :-1, :]
    unit = lambda v: torch.where(v.norm(dim=-1, keepdim=True) > 0, v / v.norm(dim=-1, keepdim=True).clamp_min(1e-12), torch.zeros_like(v))
    ang = torch.arccos((unit(dp) * unit(dg)).sum(-1).clamp(-1, 1)).mean(-1)
    r_sim = 0.7 * torch.exp(-avg / 1.0).clamp(0, 1) + 0.3 * ((torch.cos(ang) + 1) / 2).clamp(0, 1)
    # smoothness (model units, as released)
    r_smooth = torch.exp(-(wp_s[..., 1:, :] - wp_s[..., :-1, :]).norm(dim=-1).std(dim=-1, correction=0) / 0.8)
    # progress
    pl, gl = wp_m[..., -1, :].norm(dim=-1), gt[..., -1, :].norm(dim=-1)
    r_len = - 0.5 * (pl < 0.5 * gl).float()
    if LEN_MODE == "full":
        r_len = r_len + (2.0 * torch.sigmoid((pl - gl) / 5.0)).clamp(0, 1.2)
    total = W_PED * r_ped + W_SIM * r_sim + W_SMOOTH * r_smooth + W_LEN * r_len
    return total, dict(ped=r_ped, sim=r_sim, smooth=r_smooth, len=r_len, dmin=d)


# ------------------------------------------------------------------ round 4: passing side and real people
SIDE_THR = 0.15     # m; same rule as the IL targets of round 3 (24_supervised_bound.py, SN_SIDE=convention)
SIDE_RADIUS = 2.0   # m; the passing side is only judged for paths that come this close to the person


def closest_offset(wp_m, ped):
    """wp_m (N, 5, 2) m, ped (N, 4).  Pedestrian's sideways offset from the path at the closest approach
    (> 0: the person is to the right of the path, i.e. the robot passes on the person's left)."""
    t = _T.to(wp_m.device)
    P = torch.cat([torch.zeros_like(wp_m[:, :1]), wp_m], 1)
    k = t.floor().clamp(max=4).long()
    a = (t - k).unsqueeze(-1)
    pos = P[:, k] * (1 - a) + P[:, k + 1] * a
    pp = ped[:, None, :2] + ped[:, None, 2:] * t[None, :, None]
    i = (pos - pp).norm(dim=-1).argmin(1)
    r = torch.arange(len(i), device=i.device)
    return pp[r, i, 0] - pos[r, i, 0]


def convention_side(ref_wp_m, ped):
    """ref_wp_m (B, G, 5, 2): paths of the PUBLISHED policy for the same states; ped (B, 4).
    +1 = pass on the person's right (default), -1 = pass on the left (person more than SIDE_THR right of the mean path)."""
    B, G = ref_wp_m.shape[:2]
    off = closest_offset(ref_wp_m.reshape(B * G, 5, 2), ped.repeat_interleave(G, 0)).view(B, G).mean(1)
    return torch.where(off > SIDE_THR, -1.0, 1.0)


def wrong_side(wp_m, ped, side):
    """wp_m (B, G, 5, 2), ped (B, 4), side (B,).  Bool (B, G): the path comes within SIDE_RADIUS of the person and
    passes on the side the convention does not prescribe."""
    B, G = wp_m.shape[:2]
    off = closest_offset(wp_m.reshape(B * G, 5, 2), ped.repeat_interleave(G, 0)).view(B, G)
    passes = torch.where(off < 0, 1.0, -1.0)                 # robot to the right of the person: +1
    return (passes != side[:, None]) & (min_ped_distance(wp_m, ped[:, None]) < SIDE_RADIUS)


def near_people(dets, score=0.7):
    """Person boxes of one 400x400 frame (04_detect_persons.py / 32_detect_valx.py) -> rough ego positions (lat, d) of
    people within 1..5 m ahead and 1.5 m of the centre line.  Nominal pinhole, feet on the ground: exactly the filter of
    the real-pedestrian check in 22_eval_pilot.py."""
    import stimuli as st
    out = []
    for p in dets:
        x0, y0, x1, y1 = p["box"]
        if p["score"] < score or x1 > 380 or y1 <= st.HORIZON + 8 or y1 > 396:
            continue
        d = st.F_PX * st.CAM_H / (y1 - st.HORIZON)
        lat = ((x0 + x1) / 2 - st.CX) * d / st.F_PX
        if 1.0 <= d <= 5.0 and abs(lat) <= 1.5:
            out.append((float(lat), float(d)))
    return out


def batch_tensors(states, H, idx, device):
    idx = sorted(int(i) for i in idx)
    Hb = torch.from_numpy(H[idx]).to(device).float()
    st = [states[i] for i in idx]
    scale = torch.tensor([s["step_scale"] for s in st], device=device)
    gt = torch.tensor([s["gt_m"] for s in st], device=device)
    ped = torch.tensor([s["ped"] or [0, 0, 0, 0] for s in st], device=device, dtype=torch.float32)
    has = torch.tensor([bool(s["ped"]) and s["is_person"] for s in st], device=device)
    return idx, Hb, scale, gt, ped, has


def maoe_deg(pred, gt):
    """CityWalker MAOE per sample (pred (..., 5, 2), gt broadcastable), degrees."""
    cos = (pred * gt).sum(-1) / (pred.norm(dim=-1) * gt.norm(dim=-1) + 1e-12)
    return torch.rad2deg(torch.arccos(cos.clamp(-1, 1))).max(dim=-1).values
