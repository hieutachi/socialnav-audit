"""Pedestrian insertion for the open-loop counterfactual (no simulator).

A pedestrian cut out of the same camera stream (Mask R-CNN mask) is pasted into the five
history frames with a ground-plane pinhole model.  The camera of the CityWalker teleop
robot is not calibrated, so F_PX / CAM_H / HORIZON are nominal values read off the frames:
all distances below are *nominal* (expect roughly +-30%).  Left/right placement and the
with/without contrast do not depend on that calibration.
"""
import os
import json
import numpy as np
from PIL import Image, ImageFilter

F_PX = 185.0      # nominal focal length in pixels for the 400x400 frame
CAM_H = 0.60      # nominal camera height above ground (m)
HORIZON = 200.0   # nominal horizon row
CX = 200.0
PERSON_H = 1.70   # m

HERE = os.path.dirname(os.path.abspath(__file__))
FRAME_DIR = r"C:\Users\N4G\qcom_socialnav\data\citywalker_obs\traj_nav_18"
SPRITES = {"S1": ("forward_0850.jpg", 0), "S2": ("forward_3120.jpg", 0)}
_cache = {}


def load_sprite(key):
    """RGBA cut-out of a real pedestrian from the same camera."""
    if key in _cache:
        return _cache[key]
    name, det_idx = SPRITES[key]
    det = json.load(open(os.path.join(HERE, "results_04_person_detections.json")))[name]
    z = np.load(os.path.join(HERE, "masks", name.replace(".jpg", ".npz")))
    mask = z["masks"][list(z["idx"]).index(det_idx)]
    rgb = np.array(Image.open(os.path.join(FRAME_DIR, name)).convert("RGB"))
    ys, xs = np.where(mask)
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    a = Image.fromarray((mask[y0:y1, x0:x1] * 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(0.8))
    rgba = Image.fromarray(rgb[y0:y1, x0:x1]).convert("RGBA")
    rgba.putalpha(a)
    _cache[key] = rgba
    return rgba


def pillar_like(sprite):
    """Control object: grey pillar with exactly the same bounding box as the pedestrian."""
    w, h = sprite.size
    rng = np.random.RandomState(0)
    tex = np.clip(95 + rng.randn(h, w, 1) * 6 + np.linspace(-12, 12, w)[None, :, None], 0, 255)
    arr = np.concatenate([np.repeat(tex, 3, axis=2), np.full((h, w, 1), 255.0)], axis=2).astype(np.uint8)
    return Image.fromarray(arr, "RGBA")


def paste(frame, sprite, dist, lat):
    """Paste `sprite` standing on the ground at nominal (lat m right, dist m ahead)."""
    if dist <= 0.6:
        return frame
    h_px = F_PX * PERSON_H / dist
    if h_px < 10:
        return frame
    sc = h_px / sprite.size[1]
    sp = sprite.resize((max(2, int(round(sprite.size[0] * sc))), max(2, int(round(h_px)))), Image.LANCZOS)
    foot_y = HORIZON + F_PX * CAM_H / dist
    foot_x = CX + F_PX * lat / dist
    out = frame.copy()
    out.paste(sp, (int(round(foot_x - sp.size[0] / 2)), int(round(foot_y - sp.size[1]))), sp)
    return out


def bbox_px(sprite, dist, lat):
    h_px = F_PX * PERSON_H / dist
    w_px = sprite.size[0] * h_px / sprite.size[1]
    foot_y = HORIZON + F_PX * CAM_H / dist
    foot_x = CX + F_PX * lat / dist
    return [foot_x - w_px / 2, foot_y - h_px, foot_x + w_px / 2, foot_y]


# Each scenario returns, for history index k = 4..0 (k frames before now), the pedestrian's
# position in the robot frame at that time: (lateral m, distance m).  v_robot in m/frame (1 Hz).
def scenario_positions(kind, d_now, lat_now, v_robot, v_ped=1.0):
    pos = []
    for k in range(4, -1, -1):
        if kind == "static":            # standing still: only the robot's own advance changes the range
            pos.append((lat_now, d_now + k * v_robot))
        elif kind == "headon":          # walking towards the robot at v_ped
            pos.append((lat_now, d_now + k * (v_robot + v_ped)))
        elif kind == "crossing":        # walking left -> right across the path at v_ped, reaches lat_now now
            pos.append((lat_now - k * v_ped, d_now + k * v_robot))
        else:
            raise ValueError(kind)
    return pos


def render(frames, sprite, kind, d_now, lat_now, v_robot):
    pos = scenario_positions(kind, d_now, lat_now, v_robot)
    return [paste(f, sprite, d, x) for f, (x, d) in zip(frames, pos)], pos


CONDITIONS = {
    # name: (object, scenario, nominal distance now [m], lateral offset now [m])
    "none":            None,
    "ped_static_2m":   ("S1", "static", 2.0, 0.0),
    "ped_static_3m":   ("S1", "static", 3.0, 0.0),
    "ped_static_5m":   ("S1", "static", 5.0, 0.0),
    "ped2_static_3m":  ("S2", "static", 3.0, 0.0),
    "ped_left_3m":     ("S1", "static", 3.0, -0.7),
    "ped_right_3m":    ("S1", "static", 3.0, +0.7),
    "ped_headon_3m":   ("S1", "headon", 3.0, 0.0),
    "ped_crossing_3m": ("S1", "crossing", 3.0, 0.0),
    "pillar_static_3m": ("PILLAR", "static", 3.0, 0.0),
    "pillar_left_3m":  ("PILLAR", "static", 3.0, -0.7),
    "pillar_right_3m": ("PILLAR", "static", 3.0, +0.7),
}


def make(frames, cond, v_robot):
    spec = CONDITIONS[cond]
    if spec is None:
        return list(frames), None
    obj, kind, d, lat = spec
    sprite = pillar_like(load_sprite("S1")) if obj == "PILLAR" else load_sprite(obj)
    return render(frames, sprite, kind, d, lat, v_robot)
