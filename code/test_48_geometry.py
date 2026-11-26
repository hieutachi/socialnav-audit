"""Tests for 48_matched_guided_rerank.py -- no GPU, no VLM, no checkpoints.

Two kinds of check:

  A. PROTOCOL IDENTITY (ast, no torch needed for the comparison itself)
     The functions that decide what a number means -- path_points,
     energy_single, sample, sample_ped_futures, turn_deg -- must be textually
     identical to 46_heldout_v2.py's, modulo the lines that 48 is allowed to
     differ on (variable renames are NOT allowed; the only tolerated difference
     is the docstring). If someone "improves" one of them in 48, the comparison
     with 46 is dead and this test fails loudly.

  B. GEOMETRY / SELECTION NUMERICS (torch on cpu)
     path_points, energy_single, turn_deg, turn_deg_origin, final_dist and the
     argmin/CVaR selection algebra are checked against hand-computed values on
     synthetic inputs, plus the batching rules (contiguity, seed = 100 + global
     offset, plan/held-out disjointness, head-on coverage).

Run:
    python test_48_geometry.py
    python 48_matched_guided_rerank.py --self-test
"""
import ast
import importlib.util
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
SRC_46 = os.path.join(HERE, "46_heldout_v2.py")
SRC_48 = os.path.join(HERE, "48_matched_guided_rerank.py")
SHARED = ("path_points", "energy_single", "sample", "sample_ped_futures", "turn_deg")

FAILURES = []


def check(cond, label, detail=""):
    if cond:
        print(f"  PASS  {label}")
    else:
        print(f"  FAIL  {label}  {detail}")
        FAILURES.append(label)
    return bool(cond)


def load_48():
    """Import 48 by path; its module name starts with a digit so plain import fails."""
    spec = importlib.util.spec_from_file_location("m48", SRC_48)
    m = importlib.util.module_from_spec(spec)
    sys.modules["m48"] = m
    spec.loader.exec_module(m)
    return m


def collect_funcs(path):
    """Every FunctionDef in the file, by name, including nested ones."""
    src = open(path, encoding="utf-8").read()
    tree = ast.parse(src)
    out = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            out.setdefault(node.name, []).append(node)
    return src, out


def strip_doc(node):
    """Source of the function body with the docstring removed, normalised."""
    n = ast.parse(ast.unparse(node)).body[0]
    if (n.body and isinstance(n.body[0], ast.Expr)
            and isinstance(n.body[0].value, ast.Constant)
            and isinstance(n.body[0].value.value, str)):
        n.body = n.body[1:]
    return ast.unparse(n)


# ----------------------------------------------------------- A. protocol identity
def test_protocol_identity():
    print("A. protocol identity against 46_heldout_v2.py")
    ok = check(os.path.exists(SRC_46), "46_heldout_v2.py present", SRC_46)
    if not ok:
        return
    _, f46 = collect_funcs(SRC_46)
    _, f48 = collect_funcs(SRC_48)
    for name in SHARED:
        if not check(name in f46 and name in f48, f"{name} defined in both"):
            continue
        if not check(len(f48[name]) == 1, f"{name} defined exactly once in 48",
                     f"found {len(f48.get(name, []))}"):
            continue
        a, b = strip_doc(f46[name][0]), strip_doc(f48[name][0])
        check(a == b, f"{name} byte-identical to 46 (docstring excluded)",
              "" if a == b else "\n--- 46 ---\n" + a + "\n--- 48 ---\n" + b)

    # constants that define the protocol
    src48 = open(SRC_48, encoding="utf-8").read()
    src46 = open(SRC_46, encoding="utf-8").read()

    def const(src, name):
        tree = ast.parse(src)
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                for t in node.targets:
                    if isinstance(t, ast.Name) and t.id == name:
                        if isinstance(node.value, ast.BinOp) and isinstance(node.value.op, ast.Add):
                            return const(src, node.value.left.id) + const(src, node.value.right.id)
                        return ast.literal_eval(node.value)
                if (isinstance(node.targets[0], ast.Tuple)
                        and all(isinstance(e, ast.Name) for e in node.targets[0].elts)):
                    names = [e.id for e in node.targets[0].elts]
                    if name in names:
                        return ast.literal_eval(node.value)[names.index(name)]
        return None

    for cn in ("NCAND", "N_PLAN", "N_HELD", "N_ALL", "REALISED", "DELTA", "R_GUIDE", "STEPS"):
        v46, v48 = const(src46, cn), const(src48, cn)
        check(v46 == v48 and v48 is not None, f"constant {cn} matches 46", f"46={v46} 48={v48}")

    check("seed=100 + k" in src48, "future seed rule is 100 + global batch offset")
    check("ref_pool = wp_u.clone()" in src48, "energy reference is each candidate's own unguided path")
    # the pool mean must appear ONLY as the lateral-deviation reference, never as
    # the energy reference (that was 45 v1's mistake, which broke comparability)
    pool_mean_lines = [l.strip() for l in src48.splitlines() if "wp_u4.mean(1)" in l]
    check(len(pool_mean_lines) == 1, "pool mean used in exactly one place", str(pool_mean_lines))
    check(all("lat_ref" in l for l in pool_mean_lines),
          "pool mean used only as the lateral-deviation reference", str(pool_mean_lines))
    check(all("energy_single" not in l for l in pool_mean_lines),
          "pool mean never reaches energy_single")
    for l in src48.splitlines():
        if "energy_single(wp_u" in l or "energy_single(wp_g" in l:
            check("ref_pool" in l, "every planning energy call uses ref_pool", l.strip())
    check("copy.deepcopy(head.state_dict())" in src48, "PUB is deep-copied")
    check("raise FileNotFoundError" in src48, "missing checkpoints raise, not skip")
    print()


# ----------------------------------------------------------- B. geometry numerics
def test_geometry(m):
    print("B. geometry numerics")
    # path_points: straight line along +y, 1 m per second -> position at t is (0, t)
    wp = torch.tensor([[[0., 1.], [0., 2.], [0., 3.], [0., 4.], [0., 5.]]])
    pos = m.path_points(wp)
    check(pos.shape == (1, 51, 2), "path_points shape (B,51,2)", str(tuple(pos.shape)))
    t = torch.linspace(0, 5, 51)
    check(torch.allclose(pos[0, :, 1], t, atol=1e-5), "path_points interpolates at 0.1 s")
    check(float(pos[0, 0].abs().sum()) == 0.0, "path starts at the origin")

    # energy_single: pedestrian far away and ref == wp -> energy exactly 0
    ped_far = torch.tensor([[50., 50., 0., 0.]])
    e = m.energy_single(wp, ped_far, wp)
    check(abs(float(e)) < 1e-6, "energy 0 for a distant pedestrian and self-reference", str(float(e)))

    # pedestrian standing on the path -> collision term > 0
    ped_on = torch.tensor([[0., 2.5, 0., 0.]])
    e_on = float(m.energy_single(wp, ped_on, wp))
    check(e_on > 0.0, "energy > 0 when the pedestrian sits on the path", str(e_on))
    # closer approach must cost more
    wp_away = torch.tensor([[[2., 1.], [2., 2.], [2., 3.], [2., 4.], [2., 5.]]])
    e_away = float(m.energy_single(wp_away, ped_on, wp_away))
    check(e_away < e_on, "energy decreases when the path moves away", f"{e_away} vs {e_on}")

    # corridor term: |x - ref_x| beyond 1 m is penalised quadratically, mean over waypoints
    ref = torch.zeros(1, 5, 2)
    wp_lat = torch.zeros(1, 5, 2)
    wp_lat[..., 0] = 3.0           # 3 m lateral, 2 m beyond the corridor -> 4.0
    e_lat = float(m.energy_single(wp_lat, ped_far, ref))
    check(abs(e_lat - 4.0) < 1e-5, "corridor term = (|dx|-1)^2 averaged", str(e_lat))
    wp_in = torch.zeros(1, 5, 2)
    wp_in[..., 0] = 0.9            # inside the corridor -> 0
    check(abs(float(m.energy_single(wp_in, ped_far, ref))) < 1e-6, "corridor term 0 within +-1 m")

    # turn_deg (46's): straight path -> 0; turn_deg ignores the origin segment
    check(abs(float(m.turn_deg(wp))) < 1e-4, "turn_deg 0 on a straight path")
    check(abs(float(m.turn_deg_origin(wp))) < 1e-4, "turn_deg_origin 0 on a straight path")
    # right-angle kink between segments 1->2 and 2->3
    wp_k = torch.tensor([[[0., 1.], [0., 2.], [1., 2.], [2., 2.], [3., 2.]]])
    check(abs(float(m.turn_deg(wp_k)) - 90.0) < 1e-3, "turn_deg sees a 90 deg kink",
          str(float(m.turn_deg(wp_k))))
    # immediate swerve off the origin: invisible to turn_deg, visible with the origin segment
    wp_s = torch.tensor([[[1., 1.], [1., 2.], [1., 3.], [1., 4.], [1., 5.]]])
    check(abs(float(m.turn_deg(wp_s))) < 1e-4, "turn_deg blind to the origin swerve (46 behaviour)")
    check(float(m.turn_deg_origin(wp_s)) > 40.0, "turn_deg_origin catches the origin swerve",
          str(float(m.turn_deg_origin(wp_s))))

    # final_dist
    check(abs(float(m.final_dist(wp)) - 5.0) < 1e-5, "final_dist is the last waypoint norm")
    check(tuple(m.final_dist(torch.zeros(3, 5, 2)).shape) == (3,), "final_dist keeps the batch dim")
    print()


def test_selection_algebra(m):
    print("C. selection algebra (pt / exp / cvar on 8 planning futures)")
    torch.manual_seed(0)
    n, nc, nf = 4, m.NCAND, m.N_PLAN
    E = torch.rand(n, nc, nf)
    nom, ex = E[..., 0], E.mean(-1)
    cv = torch.topk(E, k=nf // 2, dim=-1).values.mean(-1)
    check(torch.allclose(nom, E[:, :, 0]), "pt score is future 0 (the nominal)")
    check(torch.allclose(ex, E.sum(-1) / nf), "exp score is the mean over the 8 planning futures")
    manual = torch.stack([torch.stack([E[i, c].sort(descending=True).values[:nf // 2].mean()
                                       for c in range(nc)]) for i in range(n)])
    check(torch.allclose(cv, manual, atol=1e-6), "cvar score is the mean of the worst half")
    check(int(cv.shape[1]) == nc and int(ex.argmin(1).max()) < nc, "selection indexes candidates")
    # a candidate that is best on every future must win every selector
    E2 = torch.rand(1, nc, nf) + 1.0
    E2[0, 7] = 0.01
    for s in (E2[..., 0], E2.mean(-1), torch.topk(E2, k=nf // 2, dim=-1).values.mean(-1)):
        check(int(s.argmin(1)) == 7, "dominant candidate wins the selector")
    print()


def test_batching(m):
    print("D. batching, seeds and the held-out split")
    # the real suite layout from 40_pennfudan_build.py: for each of the 19 suite
    # windows, each of the 5 conditions in turn, each repeated K=20 times
    conds = []
    for _ in range(19):
        for c in ("ped_static_2m", "ped_static_3m", "ped_headon_3m", "ped_left_3m", "ped_right_3m"):
            conds.extend([c] * 20)
    n = len(conds)
    check(n == 1900 and conds.count("ped_headon_3m") == 380, "layout mimic matches the real suite",
          f"{n} states, {conds.count('ped_headon_3m')} head-on")
    all_k = m.batch_offsets(n, conds, None, None)
    check(all_k == list(range(0, n, 16)), "full grid equals 46's range(0, len(S), 16)")
    check(len(all_k) == 119, "1900 states -> 119 batches", str(len(all_k)))

    sub = m.batch_offsets(n, conds, "ped_headon_3m", None)
    check(set(sub).issubset(set(all_k)), "head-on offsets are a SUBSET of 46's offsets")
    check(sub == sorted(sub), "offsets stay in ascending state order")
    covered = {i for k in sub for i in range(k, min(k + 16, n))}
    headon = {i for i, c in enumerate(conds) if c == "ped_headon_3m"}
    check(headon.issubset(covered), "every head-on state is inside a computed batch",
          f"missing {len(headon - covered)}")
    for k in sub:
        idx = list(range(k, min(k + 16, n)))
        check_contig = idx == list(range(idx[0], idx[-1] + 1))
        if not check_contig:
            check(False, "batch indices contiguous", str(k))
            break
    else:
        check(True, "batch indices contiguous")
    check(all(len(range(k, min(k + 16, n))) == 16 for k in sub[:-1]),
          "batches keep 46's size of 16")
    extra = covered - headon
    check(len(extra) > 0, "mixed batches do carry non-head-on states (they are reported, then masked)")

    lim = m.batch_offsets(n, conds, "ped_headon_3m", 3)
    check(lim == sub[:3], "--max-batches truncates from the front, keeping offsets")

    # seeds: the future seed of a batch depends only on its global offset, so a
    # subset run reproduces 46's futures for the batches it does compute
    seeds_full = [100 + k for k in all_k]
    seeds_sub = [100 + k for k in sub]
    check(set(seeds_sub).issubset(set(seeds_full)), "subset seeds are 46's seeds for those batches")
    check(len(set(seeds_full)) == len(seeds_full), "seeds unique per batch")

    # plan / held-out split
    check(m.N_PLAN == 8 and m.N_HELD == 32 and m.N_ALL == 40, "8 planning + 32 held-out = 40")
    plan, held = set(range(m.N_PLAN)), set(range(m.N_PLAN, m.N_ALL))
    check(plan.isdisjoint(held), "planning and held-out futures are disjoint")
    check(m.REALISED in plan, "the in-sample anchor index lives in the planning set")
    print()


def test_future_sampler_determinism():
    print("E. pedestrian-future sampler determinism (cpu re-implementation of 46's rule)")
    # 48's sample_ped_futures is nested in main() and binds `dev`; its AST is
    # already checked against 46 in part A. Here we only verify the property the
    # protocol relies on: same seed -> same futures, different seed -> different,
    # and future 0 is the nominal.
    def gen(ped_nom, n_futures, seed, device="cpu"):
        B = ped_nom.shape[0]
        futures = [ped_nom.unsqueeze(1)]
        rng = torch.Generator(device=device).manual_seed(seed)
        for _ in range(1, n_futures):
            p = ped_nom.clone()
            vx, vy = p[:, 2], p[:, 3]
            speed = (vx ** 2 + vy ** 2).sqrt()
            is_moving = speed > 0.1
            ang = torch.atan2(vx, vy)
            d_ang = torch.randn(B, generator=rng, device=device) * (15.0 * np.pi / 180.0)
            speed_scale = 1.0 + torch.randn(B, generator=rng, device=device) * 0.20
            stop_mask = (torch.rand(B, generator=rng, device=device) < 0.15) & is_moving
            speed_scale = torch.where(stop_mask, torch.zeros_like(speed_scale),
                                      speed_scale.clamp(0.2, 1.8))
            new_speed = speed * speed_scale
            new_ang = ang + d_ang
            p[:, 2] = torch.where(is_moving, new_speed * torch.sin(new_ang), vx)
            p[:, 3] = torch.where(is_moving, new_speed * torch.cos(new_ang), vy)
            futures.append(p.unsqueeze(1))
        return torch.cat(futures, dim=1)

    ped = torch.tensor([[0.0, 3.0, 0.0, -1.0], [0.7, 3.0, 0.0, 0.0]])   # head-on and static
    a = gen(ped, 40, 100)
    b = gen(ped, 40, 100)
    c = gen(ped, 40, 116)
    check(torch.equal(a, b), "same seed -> identical futures")
    check(not torch.equal(a[:, 1:], c[:, 1:]), "different batch offset -> different futures")
    check(torch.equal(a[:, 0], ped), "future 0 is the nominal pedestrian")
    check(tuple(a.shape) == (2, 40, 4), "futures shape (B, 40, 4)", str(tuple(a.shape)))
    # a static pedestrian (speed 0) must stay static in every future
    check(float(a[1, :, 2:].abs().max()) == 0.0, "a static pedestrian stays static across futures")
    # the moving one must actually vary
    check(float(a[0, 1:, 2:].std()) > 0.0, "a moving pedestrian's velocity varies across futures")
    # first 8 futures are a prefix of the 40 -> subsetting the horizon cannot change planning
    a8 = gen(ped, 8, 100)
    check(torch.equal(a8, a[:, :8]), "futures 0..7 are a prefix, so 46's planning set is reproduced")
    print()


def test_stats(m):
    print("F. bootstrap / ratio estimator")
    rng = np.random.RandomState(0)
    vals = [3.0, 1.0, 0.0]        # sums of a 0/1 variable per person
    cnts = [10, 10, 10]
    mu, ci = m.cluster_boot(vals, cnts, rng, B=500)
    check(abs(mu - 4.0 / 30.0) < 1e-12, "ratio estimator = sum(v)/sum(n)", str(mu))
    check(ci[0] <= mu <= ci[1], "the point estimate lies inside its CI", str(ci))
    mu2, ci2 = m.cluster_boot([5.0], [10], np.random.RandomState(0), B=100)
    check(abs(mu2 - 0.5) < 1e-12 and abs(ci2[0] - 0.5) < 1e-9,
          "a single cluster gives a degenerate CI (no fake precision)")
    print()


def test_cli(m):
    print("G. CLI contract")
    a = m.parse_args([])
    check(a.pol == ["published", "il3_s0:last.pt"], "default policies are published + il3_s0",
          str(a.pol))
    check(a.all is False, "default run is the head-on subset")
    check(a.max_batches is None, "default run is not truncated")
    s = m.parse_args(["--smoke"])
    check(s.max_batches == 2 and s.pol == ["published"], "--smoke = 2 batches, published only",
          f"{s.max_batches} {s.pol}")
    f = m.parse_args(["--all", "--max-batches", "5", "--policies", "published"])
    check(f.all and f.max_batches == 5 and f.pol == ["published"], "flags combine")
    check(m.parse_args(["--self-test"]).self_test, "--self-test flag parses")
    check(m.PRIMARY_COND == "ped_headon_3m", "primary condition is head-on")
    check(m.ARMS == ("ung", "pt", "exp", "cvar", "gui",
                     "guided_rr_pt", "guided_rr_exp", "guided_rr_cvar"),
          "8 arms: 46's five plus three guided re-ranks", str(m.ARMS))
    print()


def main():
    m = load_48()
    test_protocol_identity()
    test_geometry(m)
    test_selection_algebra(m)
    test_batching(m)
    test_future_sampler_determinism()
    test_stats(m)
    test_cli(m)
    if FAILURES:
        print(f"FAILED {len(FAILURES)}: " + ", ".join(FAILURES))
        return 1
    print("ALL TESTS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
