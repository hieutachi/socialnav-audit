"""End-to-end pipeline test for 48_matched_guided_rerank.py -- CPU, stub model.

The geometry tests (test_48_geometry.py) prove the protocol functions match 46.
They do not prove that main() runs: the shapes, the NPZ layout, the masking of
the non-head-on batch-mates, the per-arm bookkeeping and the manifest only
break when the whole thing is executed. Running the real thing needs the VLM
and a GPU, so this test substitutes:

  * a fake `sn_common.SocialNav` + `pilot_common` whose flow head is a tiny
    deterministic differentiable function (guidance still needs real gradients
    through `predict_velocity`, so the stub is a genuine autograd graph);
  * a temporary PILOT directory with a miniature states_pf.json / H_pf.npy that
    reproduces the real suite's layout (windows x 5 conditions x K sprites).

Then it asserts the properties that a reviewer would check in the output:
  - only whole 16-state batches containing head-on states are computed
  - every head-on state is present exactly once, in ascending order
  - per-arm raw arrays have the right shapes, held-out arrays are (32, n)
  - contact is a 0/1 flag, consistent with clearance < CONTACT
  - guided_rr_* really come from the guided pool and ung/pt/exp/cvar from the
    unguided pool (checked by looking the chosen candidate index back up)
  - the selectors pick the argmin of the score they claim to use
  - a selector is never worse than its own arm's candidate 0 on the planning
    objective it optimises
  - results JSON carries the primary / nominal contrasts, labelled CIs and no
    p-values; the manifest records seeds, offsets and the energy reference
  - a missing checkpoint raises instead of being silently skipped

Run:  python test_48_pipeline.py
"""
import importlib.util
import json
import os
import shutil
import sys
import tempfile
import types

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
SRC_48 = os.path.join(HERE, "48_matched_guided_rerank.py")
FAILURES = []
NCAND, N_PLAN, N_HELD, N_ALL = 16, 8, 32, 40
CONTACT = 0.45
HDIM = 8


def check(cond, label, detail=""):
    if cond:
        print(f"  PASS  {label}")
    else:
        print(f"  FAIL  {label}  {detail}")
        FAILURES.append(label)
    return bool(cond)


# ----------------------------------------------------------------- stub model
class _Flow(torch.nn.Module):
    """Tiny velocity field: differentiable in x and conditioned on z."""

    def __init__(self):
        super().__init__()
        g = torch.Generator().manual_seed(7)
        self.w = torch.nn.Parameter(torch.randn(HDIM, 10, generator=g) * 0.3)
        self.b = torch.nn.Parameter(torch.zeros(10))

    def predict_velocity(self, z, x, t):
        v = (z @ self.w + self.b).view(-1, 5, 2)
        return (v + 0.25 * torch.tanh(x) + float(t) * 0.1).float()


class _Inner(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = torch.nn.Linear(2, 2)          # stands in for the VLM trunk
        self.lm_head = torch.nn.Linear(2, 2)
        self.query_action = torch.nn.Parameter(torch.zeros(1, 1, HDIM))
        self.query_multihead_multi_attn = torch.nn.Linear(HDIM, HDIM)
        self.flow_matching_model = _Flow()

    def recover_waypoints_from_pred(self, x):
        return torch.cumsum(x, dim=-2)              # monotone, like the real decoder


class _Head:
    def __init__(self, inner):
        self.model = inner
        self.flow = inner.flow_matching_model
        self.former = inner.query_multihead_multi_attn
        self.q = inner.query_action

    def pool(self, H):
        return self.former(H.mean(1)[:, :HDIM])

    def state_dict(self):
        return {"q": self.q.detach().clone(),
                "flow": {k: v.detach().clone() for k, v in self.flow.state_dict().items()}}

    def load_state_dict(self, sd):
        with torch.no_grad():
            self.q.copy_(sd["q"])
        self.flow.load_state_dict(sd["flow"])


def install_stubs(pilot_dir):
    torch.manual_seed(1234)
    inner = _Inner()
    sn_mod = types.ModuleType("sn_common")
    sn_mod.ROOT = pilot_dir

    class SocialNav:
        def __init__(self):
            self.model = inner

    sn_mod.SocialNav = SocialNav
    pc = types.ModuleType("pilot_common")
    pc.PILOT = pilot_dir
    pc.CONTACT = CONTACT
    pc.prepare_head = lambda model: _Head(model)
    _T = torch.linspace(0, 5, 51)

    def min_ped_distance(wp_m, ped):
        t = _T.to(wp_m.device)
        P = torch.cat([torch.zeros_like(wp_m[..., :1, :]), wp_m], dim=-2)
        k = t.floor().clamp(max=4).long()
        a = (t - k).unsqueeze(-1)
        pos = P[..., k, :] * (1 - a) + P[..., k + 1, :] * a
        pp = ped[..., None, :2] + ped[..., None, 2:] * t.unsqueeze(-1)
        return (pos - pp).norm(dim=-1).min(dim=-1).values

    pc.min_ped_distance = min_ped_distance
    sys.modules["sn_common"] = sn_mod
    sys.modules["pilot_common"] = pc
    return inner


def build_pilot(tmp, n_windows=4, K=4):
    """Miniature suite with the real layout: window -> 5 conditions -> K sprites."""
    conds = ("ped_static_2m", "ped_static_3m", "ped_headon_3m", "ped_left_3m", "ped_right_3m")
    states, r = [], 0
    keys = [f"pf_{i:03d}" for i in range(7)]
    for w in range(n_windows):
        for cd in conds:
            kind = "headon" if cd == "ped_headon_3m" else "static"
            lat = -0.7 if cd.endswith("left_3m") else (0.7 if cd.endswith("right_3m") else 0.0)
            d = 2.0 if cd.endswith("2m") else 3.0
            for _ in range(K):
                ped = [lat, d, 0.0, 0.0] if kind == "static" else [lat, d, 0.0, -1.0]
                states.append(dict(split="pf", traj=18, ps=100 + w, cond=cd, kind=kind, d=d,
                                   lat=lat, v_ped=1.0, sprite=keys[r % len(keys)],
                                   ped=ped, is_person=True, step_scale=1.0 + 0.01 * (r % 5),
                                   gt_m=[[0., 1.], [0., 2.], [0., 3.], [0., 4.], [0., 5.]]))
                r += 1
    json.dump(dict(states=states, done=len(states)),
              open(os.path.join(tmp, "states_pf.json"), "w"))
    g = np.random.RandomState(0)
    np.save(os.path.join(tmp, "H_pf.npy"),
            g.randn(len(states), 6, HDIM).astype(np.float16))
    return states


def load_48():
    spec = importlib.util.spec_from_file_location("m48_pipe", SRC_48)
    m = importlib.util.module_from_spec(spec)
    sys.modules["m48_pipe"] = m
    spec.loader.exec_module(m)
    return m


def main():
    tmp = tempfile.mkdtemp(prefix="t48_")
    cwd = os.getcwd()
    try:
        states = build_pilot(tmp)
        install_stubs(tmp)
        os.environ["SN48_DEV"] = "cpu"
        m = load_48()
        # keep every output inside the temp dir
        m.HERE = tmp
        os.chdir(tmp)

        print("H. end-to-end run on the stub model (head-on subset, published only)")
        rc = m.main(["--policies", "published", "--tag", "_pipe"])
        check(rc == 0, "main() returns 0")

        res = json.load(open(os.path.join(tmp, "results_48_guided_rerank_pipe.json")))
        man = json.load(open(os.path.join(tmp, "manifest_48_guided_rerank_pipe.json")))
        _R = np.load(os.path.join(tmp, "raw_48_guided_rerank_pipe.npz"))
        # raw arrays are namespaced by policy so several policies share one NPZ
        check(all(k.startswith("published__") for k in _R.files),
              "every raw array is namespaced by its policy")
        R = {k.split("__", 1)[1]: _R[k] for k in _R.files}

        conds = [s["cond"] for s in states]
        si = R["state_idx"]
        n = len(si)
        check(list(si) == sorted(si), "states come out in ascending order")
        check(len(set(si.tolist())) == n, "no state computed twice")
        headon = {i for i, c in enumerate(conds) if c == "ped_headon_3m"}
        check(headon.issubset(set(si.tolist())), "every head-on state was computed",
              f"missing {len(headon - set(si.tolist()))}")
        offs = res["batch_offsets"]
        expect = [i for k in offs for i in range(k, min(k + 16, len(states)))]
        check(list(si) == expect, "computed states are exactly the whole chosen batches")
        check(all(k % 16 == 0 for k in offs), "offsets lie on 46's grid")
        extra = set(si.tolist()) - headon
        check(len(extra) > 0, "batch-mates of other conditions were computed and kept",
              f"{len(extra)} extra states")
        check("ped_headon_3m" in res["res"]["published"]["per_cond"],
              "head-on results present")
        other = [c for c in res["res"]["published"]["per_cond"] if c != "ped_headon_3m"]
        check(len(other) > 0, "batch-mate conditions are reported separately, not merged",
              str(other))
        nh = res["res"]["published"]["per_cond"]["ped_headon_3m"]["n_states"]
        check(nh == len(headon), "head-on mask selects exactly the head-on states",
              f"{nh} vs {len(headon)}")
        print()

        print("I. raw arrays")
        for a in m.ARMS:
            check(R[f"paths__{a}"].shape == (n, 5, 2), f"paths__{a} is (n,5,2)",
                  str(R[f"paths__{a}"].shape))
            check(R[f"hel__{a}"].shape == (N_HELD, n), f"hel__{a} is (32,n)",
                  str(R[f"hel__{a}"].shape))
            check(R[f"clr_hel__{a}"].shape == (N_HELD, n), f"clr_hel__{a} is (32,n)")
            check(set(np.unique(R[f"hel__{a}"]).tolist()) <= {0.0, 1.0},
                  f"hel__{a} is a 0/1 flag")
            agree = ((R[f"clr_hel__{a}"] < CONTACT).astype(np.float32) == R[f"hel__{a}"]).all()
            check(bool(agree), f"hel__{a} == (clr_hel__{a} < CONTACT)")
            check(R[f"cand__{a}"].shape == (n,) and R[f"cand__{a}"].max() < NCAND,
                  f"cand__{a} is a valid candidate index")
            check(np.isfinite(R[f"E_sel__{a}"]).all(), f"E_sel__{a} finite")
            check((R[f"turn__{a}"] >= 0).all() and (R[f"turn_o__{a}"] >= 0).all(),
                  f"turn columns non-negative for {a}")
        check((R["cand__ung"] == 0).all() and (R["cand__gui"] == 0).all(),
              "ung and gui are candidate 0 by definition")
        check(R["fin_ung0"].shape == (n,) and (R["fin_ung0"] > 0).all(),
              "progress denominator fin_ung0 is positive")
        check(R["fin_gt"].shape == (n,) and (R["fin_gt"] > 0).all(),
              "progress denominator fin_gt is positive")
        check(np.allclose(R["fin__ung"], R["fin_ung0"]),
              "fin__ung equals the ung0 denominator (same path)")
        print()

        print("J. the arms really come from the pools they claim")
        # re-run one batch by hand and look the chosen indices back up
        head = _Head(sys.modules["sn_common"].SocialNav().model)
        meta = json.load(open(os.path.join(tmp, "states_pf.json")))
        S = meta["states"][:meta["done"]]
        H = np.load(os.path.join(tmp, "H_pf.npy"), mmap_mode="r")
        k0 = offs[0]
        idx = list(range(k0, min(k0 + 16, len(S))))
        nb = len(idx)
        noise = torch.randn(NCAND, 5, 2, generator=torch.Generator().manual_seed(123))
        Hb = torch.from_numpy(np.asarray(H[idx])).float()
        sc = torch.tensor([S[i]["step_scale"] for i in idx]).repeat_interleave(NCAND)
        ped = torch.tensor([S[i]["ped"] for i in idx], dtype=torch.float32)
        with torch.no_grad():
            z = head.pool(Hb).repeat_interleave(NCAND, 0)
        nz = noise.repeat(nb, 1, 1)

        def sample(peds=None, ref=None):
            x = nz.clone()
            for s in range(m.STEPS):
                t = torch.tensor([s / m.STEPS])
                if peds is not None:
                    xg = x.detach().requires_grad_(True)
                    v = head.flow.predict_velocity(z, xg, t).float()
                    wp = head.model.recover_waypoints_from_pred(
                        xg + (1 - s / m.STEPS) * v) * sc[:, None, None]
                    g, = torch.autograd.grad(m.energy_single(wp, peds, ref).sum(), xg)
                    nrm = g.flatten(1).norm(dim=1)[:, None, None]
                    x = (xg + v / m.STEPS - torch.where(
                        nrm > 0, g * m.DELTA / nrm.clamp_min(1e-12), torch.zeros_like(g))).detach()
                else:
                    with torch.no_grad():
                        x = x + head.flow.predict_velocity(z, x, t).float() / m.STEPS
            if peds is not None:
                x = x.clamp(-1, 1)
            with torch.no_grad():
                return head.model.recover_waypoints_from_pred(x) * sc[:, None, None]

        wp_u = sample()
        p_rep = ped.repeat_interleave(NCAND, 0)
        wp_g = sample(p_rep, wp_u)
        check(not torch.allclose(wp_u, wp_g), "the guided pool differs from the unguided pool")
        wp_u4, wp_g4 = wp_u.view(nb, NCAND, 5, 2), wp_g.view(nb, NCAND, 5, 2)
        sl = slice(0, nb)
        for a, pool in (("ung", wp_u4), ("pt", wp_u4), ("exp", wp_u4), ("cvar", wp_u4),
                        ("gui", wp_g4), ("guided_rr_pt", wp_g4),
                        ("guided_rr_exp", wp_g4), ("guided_rr_cvar", wp_g4)):
            c = torch.from_numpy(R[f"cand__{a}"][sl]).long()
            want = pool[torch.arange(nb), c].numpy()
            got = R[f"paths__{a}"][sl]
            check(np.allclose(want, got, atol=1e-4),
                  f"{a} is candidate cand__{a} of the {'guided' if pool is wp_g4 else 'unguided'} pool",
                  f"max |d| = {np.abs(want - got).max():.2e}")

        # selectors pick the argmin of the score they advertise
        fut = []
        rng = torch.Generator().manual_seed(100 + k0)
        futs = [ped.unsqueeze(1)]
        for _ in range(1, N_ALL):
            p = ped.clone()
            vx, vy = p[:, 2], p[:, 3]
            speed = (vx ** 2 + vy ** 2).sqrt()
            is_moving = speed > 0.1
            ang = torch.atan2(vx, vy)
            d_ang = torch.randn(nb, generator=rng) * (15.0 * np.pi / 180.0)
            ss = 1.0 + torch.randn(nb, generator=rng) * 0.20
            stop = (torch.rand(nb, generator=rng) < 0.15) & is_moving
            ss = torch.where(stop, torch.zeros_like(ss), ss.clamp(0.2, 1.8))
            p[:, 2] = torch.where(is_moving, speed * ss * torch.sin(ang + d_ang), vx)
            p[:, 3] = torch.where(is_moving, speed * ss * torch.cos(ang + d_ang), vy)
            futs.append(p.unsqueeze(1))
        fut = torch.cat(futs, 1)
        for pool, pre in ((wp_u, ""), (wp_g, "guided_rr_")):
            E = torch.stack([m.energy_single(pool, fut[:, f].repeat_interleave(NCAND, 0),
                                             wp_u).view(nb, NCAND) for f in range(N_PLAN)], -1)
            want_pt = E[..., 0].argmin(1).numpy()
            want_exp = E.mean(-1).argmin(1).numpy()
            want_cv = torch.topk(E, k=N_PLAN // 2, dim=-1).values.mean(-1).argmin(1).numpy()
            nm = {"": ("pt", "exp", "cvar"), "guided_rr_": ("guided_rr_pt", "guided_rr_exp",
                                                            "guided_rr_cvar")}[pre]
            for arm, want in zip(nm, (want_pt, want_exp, want_cv)):
                check((R[f"cand__{arm}"][sl] == want).all(),
                      f"{arm} is the argmin of its own score")
            # a selector cannot do worse than candidate 0 on its own objective
            e_exp = E.mean(-1).numpy()
            c = R[f"cand__{nm[1]}"][sl]
            check((e_exp[np.arange(nb), c] <= e_exp[:, 0] + 1e-6).all(),
                  f"{nm[1]} never worse than candidate 0 on the expected energy")
        print()

        print("K. results JSON and manifest")
        d = res["res"]["published"]["per_cond"]["ped_headon_3m"]
        check("PRIMARY_d_hel_guided_rr_exp_minus_exp" in d, "primary contrast present")
        check("NOMINAL_d_hel_guided_rr_pt_minus_pt" in d, "nominal contrast present")
        expl = [k for k in d if k.startswith("EXPLORATORY_")]
        check(len(expl) >= 6, "exploratory contrasts present and labelled as such", str(len(expl)))
        blob = json.dumps(res)
        check('"p_value"' not in blob and '"pval"' not in blob and '"p"' not in blob,
              "no p-values anywhere in the results")
        check("ci_sprite_conditional" in d["hel_exp"], "sprite CI is labelled conditional")
        check("ci_window_descriptive" in d["hel_exp"]["window"],
              "window spread is labelled descriptive")
        check(d["hel_exp"]["window"]["n_windows"] == len({s["ps"] for s in states}),
              "window spread covers every suite window")
        for a in m.ARMS:
            check(f"progress_vs_ung0_{a}" in d and f"progress_vs_gt_{a}" in d,
                  f"both progress denominators reported for {a}")
            check(f"turn_{a}" in d and f"turn_origin_{a}" in d,
                  f"legacy turn and origin-inclusive turn both reported for {a}")
            check(f"lat_{a}" in d, f"lateral deviation reported for {a}")
        check(abs(d["progress_vs_ung0_ung"] - 1.0) < 1e-6,
              "ung's progress ratio against itself is exactly 1")
        check(res["res"]["published"]["all_run_states"]["n_states"] == n,
              "the all-run-states block covers every computed state")
        check(man["noise_seed"] == 123 and man["future_seed_rule"] == "100 + global_batch_offset",
              "manifest records the seeds")
        check(man["batch_offsets"] == offs, "manifest records the batch offsets")
        check("own unguided path" in man["energy_ref"], "manifest records the energy reference")
        check(man["n_states_run"] == n and man["n_states_total"] == len(states),
              "manifest records run vs total state counts")
        check(set(d.keys()) >= {f"hel_{a}" for a in m.ARMS},
              "held-out contact reported for all 8 arms")
        print()

        print("L. failure modes")
        try:
            m.main(["--policies", "il3_s0:nope.pt", "--max-batches", "1", "--tag", "_fail"])
            check(False, "a missing checkpoint raises")
        except FileNotFoundError:
            check(True, "a missing checkpoint raises FileNotFoundError, not a silent skip")
        except Exception as e:                                   # noqa: BLE001
            check(False, "a missing checkpoint raises FileNotFoundError", repr(e))
        rc2 = m.main(["--smoke", "--tag", "_smoke"])
        smoke = json.load(open(os.path.join(tmp, "results_48_guided_rerank_smoke.json")))
        check(rc2 == 0 and smoke["n_batches"] == 2, "--smoke runs and stops after 2 batches",
              str(smoke["n_batches"]))
        check(smoke["batch_offsets"] == offs[:2], "--smoke keeps the leading real offsets")
        check(os.path.exists(os.path.join(tmp, "raw_48_guided_rerank_smoke.npz")),
              "--tag keeps the smoke outputs off the real file names")
        rc3 = m.main(["--all", "--max-batches", "2", "--policies", "published", "--tag", "_all"])
        allr = json.load(open(os.path.join(tmp, "results_48_guided_rerank_all.json")))
        check(rc3 == 0 and allr["batch_offsets"] == [0, 16], "--all starts from offset 0")
        check(allr["subset"] == "all", "--all is recorded in the results")
        print()
    finally:
        os.chdir(cwd)
        shutil.rmtree(tmp, ignore_errors=True)

    if FAILURES:
        print(f"FAILED {len(FAILURES)}: " + ", ".join(FAILURES))
        return 1
    print("ALL PIPELINE TESTS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
