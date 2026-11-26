"""Shared helpers for the SocialNav open-loop (inference-only) pedestrian probe.

Nothing here trains or changes weights: the model is loaded in eval mode and
only `forward(train=False)` (action expert) and `generate` (Brain) are called.

Paths point outside OneDrive on purpose (weights, repo, frames are large).
"""
import os
import sys
import random

import numpy as np
import torch
from PIL import Image
from scipy.spatial.transform import Rotation as R

ROOT = r"C:\Users\N4G\qcom_socialnav"
REPO = os.path.join(ROOT, "repo", "SocialNav")
# Backbone is chosen with the environment variable SN_BACKBONE (qwen2_5_vl | qwen2_vl).
BACKBONE = os.environ.get("SN_BACKBONE", "qwen2_5_vl")
_BACKBONES = {
    "qwen2_5_vl": ("SocialNav-Qwen2.5-VL-SAFE-GRPO", "Qwen2_5_VLForConditionalGeneration",
                   ["replace_qwen2_5_with_mixed_modality_forward", "replace_qwen2_5_vision"]),
    "qwen2_vl": ("SocialNav-Qwen2-VL-SAFE-GRPO", "Qwen2VLForConditionalGeneration",
                 ["replace_qwen_2_with_mixed_modality_forward"]),
}
CKPT = os.path.join(ROOT, "ckpt", _BACKBONES[BACKBONE][0])
# SN_FIX_LOAD=1 (default): after from_pretrained, copy back every checkpoint tensor that the
# released loader re-initialised (see restore_checkpoint_tensors).  SN_FIX_LOAD=0 reproduces
# the behaviour of the released code as-is.
FIX_LOAD = os.environ.get("SN_FIX_LOAD", "1") == "1"
TAG = ("" if BACKBONE == "qwen2_5_vl" else "_" + BACKBONE) + ("_fixed" if FIX_LOAD else "")   # result-file suffix
POSE_DIR = os.path.join(ROOT, "data", "citywalker_pose", "pose_label")
OBS_DIR = os.path.join(ROOT, "data", "citywalker_obs")

if REPO not in sys.path:
    sys.path.insert(0, REPO)

CATEGORIES = ["crowd", "person_close_by", "turn", "action_target_mismatch", "crossing", "other"]
WAYPOINT_TOKENS = [f"<input_pos{i}>" for i in range(1, 6)] + ["<input_target>"]
EXTRA_SPECIAL_TOKENS = ["<flow_matching_policy>", "<time>"]

# Action-branch prompt.  The training prompt is not published; this wording is the one that
# agreed best with the human driver on CityWalker (see 01/13).  Override with SN_ACT_PROMPT.
_POS = "".join(f"<input_pos{i}>" for i in range(1, 6))
ACT_PROMPT = os.environ.get(
    "SN_ACT_PROMPT", f"Historical positions: {_POS}. Target position: <input_target>. Predict the future trajectory.")

CONTEXT = 5      # history frames (t-4..t)
WP_LEN = 5       # predicted waypoints
SEARCH_WINDOW = 50
ARRIVED_THRESHOLD = 5
ARRIVED_PROB = 0.3


# --------------------------------------------------------------------------
# CityWalker teleop sample construction (mirrors ai4ce/CityWalker
# data/teleop_dataset.py, mode='test', cord_embedding='input_target')
# --------------------------------------------------------------------------
def _latlon_to_local(lat, lon, lat0, lon0):
    r_earth = 6378137
    lat_r, lon_r, lat0_r, lon0_r = map(np.radians, (lat, lon, lat0, lon0))
    x = (lon_r - lon0_r) * np.cos((lat_r + lat0_r) / 2) * r_earth
    y = (lat_r - lat0_r) * r_earth
    return x, y


def _pose_to_matrix(pose):
    m = np.eye(4)
    m[:3, :3] = R.from_rotvec(pose[3:6]).as_matrix()
    m[:3, 3] = pose[:3]
    return m


def _transform_poses(poses, current_pose):
    inv = np.linalg.inv(_pose_to_matrix(current_pose))
    out = []
    for p in poses:
        pos = (inv @ _pose_to_matrix(p))[:3, 3].copy()
        pos[[0, 1]] = pos[[1, 0]]   # lidar extrinsic, as in CityWalker
        pos[1] *= -1
        out.append(pos)
    return np.array(out)


def _transform_input(gps_xy):
    tr = gps_xy - gps_xy[-1]
    second_last = tr[-2]
    ang = -np.pi / 2 - np.arctan2(second_last[1], second_last[0])
    rot = np.array([[np.cos(ang), -np.sin(ang)], [np.sin(ang), np.cos(ang)]])
    return tr[:, :2] @ rot.T


def load_traj(traj_id):
    """Parse one pose_traj_XX.txt -> dict of arrays."""
    path = os.path.join(POSE_DIR, f"pose_traj_{traj_id:02d}.txt")
    lines = open(path).read().strip().split("\n")
    gps, poses, names, cats = [], [], [], []
    ref = None
    for i in range(0, len(lines), 3):
        g = lines[i].split(",")
        p = lines[i + 1].split(",")
        lat, lon, alt = float(g[1]), float(g[2]), float(g[4])
        if ref is None:
            ref = (lat, lon, alt)
        x, y = _latlon_to_local(lat, lon, ref[0], ref[1])
        gps.append([x, y, alt - ref[2]])
        poses.append([float(v) for v in p[1:7]])
        names.append(f"forward_{int(p[7]):04d}.jpg")
        cats.append([int(v) for v in lines[i + 2].split(",")])
    gps, poses, cats = np.array(gps), np.array(poses), np.array(cats)
    step_scale = float(np.linalg.norm(np.diff(poses[:, [0, 1]], axis=0), axis=1).mean())
    return dict(traj_id=traj_id, gps=gps, poses=poses, names=names, cats=cats,
                step_scale=max(step_scale, 1e-2),
                image_dir=os.path.join(OBS_DIR, f"traj_nav_{traj_id:02d}"))


def build_sample(traj, pose_start, seed=0):
    """One open-loop sample. Units: everything divided by step_scale (as the model expects)."""
    poses, gps = traj["poses"], traj["gps"]
    rng = random.Random(seed * 1_000_003 + pose_start)
    future = poses[pose_start + CONTEXT: pose_start + CONTEXT + SEARCH_WINDOW]
    max_idx = future.shape[0] - 1
    arrived = rng.random() < ARRIVED_PROB
    if arrived:
        target_idx = rng.randint(WP_LEN, min(WP_LEN + ARRIVED_THRESHOLD, max_idx))
    else:
        target_idx = rng.randint(WP_LEN + ARRIVED_THRESHOLD, max_idx)

    cur = poses[pose_start + CONTEXT - 1]
    gt = _transform_poses(poses[pose_start + CONTEXT: pose_start + CONTEXT + WP_LEN], cur)[:, :2]
    target = _transform_poses(poses[pose_start + CONTEXT + target_idx][None], cur)[0, :2]
    hist = _transform_input(gps[pose_start: pose_start + CONTEXT])
    s = traj["step_scale"]
    input_wp = np.concatenate([hist, target[None]], axis=0) / s
    idx_now = pose_start + CONTEXT - 1
    return dict(
        traj_id=traj["traj_id"], pose_start=pose_start, idx_now=idx_now,
        images=[os.path.join(traj["image_dir"], n) for n in traj["names"][pose_start: pose_start + CONTEXT]],
        input_waypoints=input_wp.astype(np.float32),     # (6,2) scaled
        gt_waypoints=(gt / s).astype(np.float32),        # (5,2) scaled
        step_scale=s, arrived=bool(arrived),
        categories=traj["cats"][idx_now].tolist(),
    )


def usable_starts(traj):
    n = traj["poses"].shape[0] - CONTEXT - max(ARRIVED_THRESHOLD * 2, WP_LEN)
    return list(range(0, max(n, 0)))


# --------------------------------------------------------------------------
# Model
# --------------------------------------------------------------------------
def restore_checkpoint_tensors(model, ckpt_dir, apply=True):
    """Make the in-memory model equal to the checkpoint file.

    With the transformers tree shipped in the SocialNav repo, `from_pretrained` leaves 43
    tensors of the action branch (`query_action`, the whole `query_multihead_multi_attn`
    action former, and the flow head's `pos_emb` / `cond_pos_emb`) at fresh random values
    although they are stored in the checkpoint.  This copies the stored values back.
    It does not train or alter anything: the result is exactly the published weights.
    Returns {group: number of tensors that differed}.
    """
    import glob
    import collections
    from safetensors import safe_open
    sd = model.state_dict()
    restored = collections.Counter()
    unmatched = []
    with torch.no_grad():
        for f in sorted(glob.glob(os.path.join(ckpt_dir, "*.safetensors"))):
            with safe_open(f, "pt") as h:
                for k in h.keys():
                    cands = [k]
                    if k.startswith("model."):
                        cands.append("model.language_model." + k[len("model."):])
                    if k.startswith("visual."):
                        cands.append("model." + k)
                    tgt = next((c for c in cands if c in sd), None)
                    if tgt is None:
                        unmatched.append(k)
                        continue
                    src = h.get_tensor(k)
                    cur = sd[tgt]
                    if src.shape != cur.shape:
                        unmatched.append(k + " (shape)")
                        continue
                    if not torch.equal(cur.detach().cpu().to(src.dtype), src):
                        restored[k.split(".")[0] if not k.startswith("flow_matching_model") else ".".join(k.split(".")[:3])] += 1
                        if apply:
                            cur.copy_(src.to(cur.device, cur.dtype))
    return dict(restored), unmatched


class SocialNav:
    def __init__(self, device="cuda:0", flow_steps=5):
        import transformers
        from transformers import AutoProcessor
        from src.train import monkey_patch_forward, monkey_patch_vision

        _, class_name, patches = _BACKBONES[BACKBONE]
        for name in patches:
            fn = getattr(monkey_patch_forward, name, None) or getattr(monkey_patch_vision, name)
            fn()

        kw = dict(action_dim=2, action_chunk=5, flow_matching_policy=True, num_flow_steps=flow_steps,
                  action_former=True, query_action_layer=4, sigma=0.0, ar_lambda_loss=1.0,
                  min_value=-1.25, max_value=1.25, sde_mode="cps")
        self.device = torch.device(device)
        self.model = getattr(transformers, class_name).from_pretrained(
            CKPT, dtype=torch.bfloat16, device_map=str(device), **kw)
        self.load_diff, self.load_unmatched = restore_checkpoint_tensors(self.model, CKPT, apply=FIX_LOAD)
        print(f"[load] backbone={BACKBONE} | tensors that differed from the checkpoint file after from_pretrained: "
              f"{self.load_diff or 'none'} | {'restored from file' if FIX_LOAD else 'LEFT AS LOADED (SN_FIX_LOAD=0)'}"
              f" | unmatched file keys: {len(self.load_unmatched)}", flush=True)
        self.model.eval()
        self.processor = AutoProcessor.from_pretrained(CKPT)
        tok = self.processor.tokenizer
        self.special_token2id = {t: tok.convert_tokens_to_ids(t) for t in WAYPOINT_TOKENS + EXTRA_SPECIAL_TOKENS}
        self.model.special_token2id = self.special_token2id

    def _inputs(self, images, text, generation):
        content = [{"type": "image", "image": im} for im in images] + [{"type": "text", "text": text}]
        chat = self.processor.apply_chat_template(
            [{"role": "user", "content": content}], tokenize=False, add_generation_prompt=True)
        if not generation:
            chat = chat + "<|im_end|>"   # exactly as utils/citywalker.py does for the action branch
        return self.processor(text=chat, images=images, padding=True, return_tensors="pt").to(self.device)

    @torch.no_grad()
    def act(self, images, input_waypoints, prompt, num_samples=1, noise=None):
        """Action expert. Returns (num_samples, 5, 2) waypoints in step_scale units, ego frame (+Y fwd, +X right)."""
        inputs = self._inputs(images, prompt, generation=False)
        inputs["input_waypoints"] = torch.as_tensor(input_waypoints, dtype=torch.float32)[None].to(self.device)
        if noise is not None:
            noise = torch.as_tensor(noise)
            num_samples = noise.shape[0]
        out = self.model(**inputs, train=False, train_branch="fm", num_samples=num_samples,
                         select_noise=noise, special_token2id=self.special_token2id)
        return out[0].detach().float().cpu().numpy()

    @torch.no_grad()
    def gen(self, images, prompt, input_waypoints=None, max_new_tokens=512):
        """Brain (autoregressive head). Greedy decoding. Returns generated text."""
        inputs = self._inputs(images, prompt, generation=True)
        extra = {}
        if input_waypoints is not None:
            extra["input_waypoints"] = torch.as_tensor(input_waypoints, dtype=torch.float32)[None].to(self.device)
        ids = self.model.generate(**inputs, **extra, max_new_tokens=max_new_tokens, do_sample=False,
                                  use_cache=True)
        new = ids[0, inputs["input_ids"].shape[1]:]
        return self.processor.tokenizer.decode(new, skip_special_tokens=True).strip()


def load_images(paths):
    return [Image.open(p).convert("RGB") for p in paths]


def max_angle_deg(pred, gt):
    """CityWalker MAOE per sample: max over 5 steps of angle between pred and gt waypoint vectors."""
    p, g = np.asarray(pred, dtype=np.float64), np.asarray(gt, dtype=np.float64)
    cos = (p * g).sum(1) / (np.linalg.norm(p, axis=1) * np.linalg.norm(g, axis=1) + 1e-12)
    ang = np.degrees(np.arccos(np.clip(cos, -1, 1)))
    return float(ang.max()), ang
