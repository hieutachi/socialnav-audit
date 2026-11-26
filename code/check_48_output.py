"""Independent audit of a 48 run's saved outputs. Read-only; touches no GPU.

Usage:  python check_48_output.py [tag]     (default tag: _smoke)

Checks the things that would silently poison the conclusion:
  * the NPZ and the JSON agree on which states were computed
  * held-out arrays are (32, n), contact is 0/1 and equals clearance < 0.45
  * each arm's path really is the candidate its cand__ index names, in the pool
    it claims (verified indirectly: guided and unguided arms must not be drawn
    from the same path set)
  * selection is a true argmin: a selector's own planning energy is <= the
    energy of candidate 0 of its own pool
  * contact rates are ordered the way the protocol forces them to be
    (a selector cannot be worse than its pool's candidate 0 on the energy it
    optimises; it CAN be worse on held-out contact -- that is the finding)
  * no NaN/Inf anywhere, no degenerate all-equal columns
  * conditional vs descriptive CIs are present and the point estimate is inside
    the sprite CI
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CONTACT = 0.45
PROBLEMS = []
NOTES = []


def bad(msg):
    PROBLEMS.append(msg)
    print("  ANOMALY  " + msg)


def ok(msg):
    print("  ok       " + msg)


def note(msg):
    NOTES.append(msg)
    print("  note     " + msg)


def main(tag="_smoke"):
    res = json.load(open(os.path.join(HERE, f"results_48_guided_rerank{tag}.json")))
    man = json.load(open(os.path.join(HERE, f"manifest_48_guided_rerank{tag}.json")))
    Z = np.load(os.path.join(HERE, f"raw_48_guided_rerank{tag}.npz"))
    arms = res["arms"]
    pols = man["policies"]
    print(f"48 output audit  tag={tag!r}  policies={pols}  "
          f"{res['n_batches']} batches, {res['n_states_run']} states, "
          f"offsets {res['batch_offsets']}\n")

    for pol in pols:
        name = pol.partition(":")[0]
        print(f"--- {name} ---")
        R = {k.split("__", 1)[1]: Z[k] for k in Z.files if k.startswith(name + "__")}
        if not R:
            bad(f"{name}: no raw arrays in the NPZ")
            continue
        si = R["state_idx"]
        n = len(si)
        if n != res["n_states_run"]:
            bad(f"{name}: NPZ has {n} states, JSON claims {res['n_states_run']}")
        else:
            ok(f"{n} states, matching the JSON")
        expect = [i for k in res["batch_offsets"]
                  for i in range(k, min(k + res["batch_size"], res["n_states_total"]))]
        if list(si) != expect:
            bad(f"{name}: state_idx is not the concatenation of the whole batches")
        else:
            ok("state_idx == the whole declared batches, contiguous and ascending")

        for a in arms:
            hel, clr = R[f"hel__{a}"], R[f"clr_hel__{a}"]
            if hel.shape != (res["n_held"], n):
                bad(f"{name}/{a}: hel is {hel.shape}, expected {(res['n_held'], n)}")
            if set(np.unique(hel).tolist()) - {0.0, 1.0}:
                bad(f"{name}/{a}: hel is not a 0/1 flag")
            if not ((clr < CONTACT).astype(np.float32) == hel).all():
                bad(f"{name}/{a}: hel disagrees with clr_hel < {CONTACT}")
            for key in (f"paths__{a}", f"clr_hel__{a}", f"E_sel__{a}", f"turn__{a}",
                        f"turn_o__{a}", f"lat__{a}", f"fin__{a}"):
                v = R[key]
                if not np.isfinite(v).all():
                    bad(f"{name}: {key} has NaN or Inf")
            if R[f"cand__{a}"].max() >= res["n_cand"] or R[f"cand__{a}"].min() < 0:
                bad(f"{name}/{a}: cand index out of range")
        ok("held-out arrays well-formed, contact consistent with clearance, all finite")

        # pools: guided arms must not coincide with unguided arms
        pu, pg = R["paths__ung"], R["paths__gui"]
        if np.allclose(pu, pg):
            bad(f"{name}: the guided pool equals the unguided pool -- guidance did nothing")
        else:
            ok(f"guided candidate 0 differs from unguided candidate 0 "
               f"(mean |d| = {np.abs(pu - pg).mean():.4f} m)")
        for g, u in (("guided_rr_pt", "pt"), ("guided_rr_exp", "exp"),
                     ("guided_rr_cvar", "cvar")):
            same = np.isclose(R[f"paths__{g}"], R[f"paths__{u}"]).all(axis=(1, 2)).mean()
            if same > 0.5:
                bad(f"{name}: {g} equals {u} on {same:.0%} of states -- pools may be crossed")
        ok("guided re-ranks are distinct from their unguided counterparts")

        # selection sanity: the chosen candidate's planning energy beats candidate 0
        for a, base in (("pt", "ung"), ("exp", "ung"), ("cvar", "ung"),
                        ("guided_rr_pt", "gui"), ("guided_rr_exp", "gui"),
                        ("guided_rr_cvar", "gui")):
            worse = (R[f"E_sel__{a}"] > R[f"E_sel__{base}"] + 1e-5).mean()
            if a.endswith("exp") and worse > 0:
                bad(f"{name}: {a} has a higher expected planning energy than {base} "
                    f"on {worse:.1%} of states -- the argmin is not an argmin")
            elif worse > 0:
                note(f"{a} exceeds {base}'s EXPECTED energy on {worse:.1%} of states "
                     f"(allowed: {a} optimises a different score)")
        ok("expected-risk selectors are true argmins of the expected energy")

        # degenerate columns
        for a in arms:
            if np.unique(R[f"cand__{a}"]).size == 1 and a not in ("ung", "gui"):
                note(f"{a} always picks candidate {int(R[f'cand__{a}'][0])} "
                     f"-- check the pool is not collapsed")
            if R[f"hel__{a}"].mean() in (0.0, 1.0):
                note(f"{a} held-out contact is exactly {R[f'hel__{a}'].mean():.0%} "
                     f"(floor/ceiling, CI will be degenerate)")

        # per-condition numbers
        for cd, d in res["res"][name]["per_cond"].items():
            star = " <== PRIMARY" if cd == res["primary_cond"] else ""
            print(f"  {cd:16s} n={d['n_states']:4d} sprites={d['n_sprites']:3d} "
                  f"windows={d['n_windows']:2d}{star}")
            row = "    hel%: " + "  ".join(f"{a}={d[f'hel_{a}']['mean']:5.1f}" for a in arms)
            print(row)
            for key in ("PRIMARY_d_hel_guided_rr_exp_minus_exp",
                        "NOMINAL_d_hel_guided_rr_pt_minus_pt"):
                if key in d:
                    c = d[key]["ci_sprite_conditional"]
                    print(f"    {key}: {d[key]['mean']:+6.2f}  CI_sprite [{c[0]:+.2f}, {c[1]:+.2f}]"
                          + ("  (excludes 0)" if c[0] > 0 or c[1] < 0 else "  (includes 0)"))
            for a in arms:
                h = d[f"hel_{a}"]
                c = h["ci_sprite_conditional"]
                if not (c[0] - 1e-9 <= h["mean"] <= c[1] + 1e-9):
                    bad(f"{name}/{cd}/{a}: mean {h['mean']} outside its sprite CI {c}")
                if "ci_window_descriptive" not in h["window"]:
                    bad(f"{name}/{cd}/{a}: window CI missing its descriptive label")
            if d["n_windows"] < 3:
                note(f"{cd}: only {d['n_windows']} window(s) in this run -- the window "
                     f"spread is meaningless until the full head-on set is run")
            # progress / plausibility columns
            print("    turn46: " + "  ".join(f"{a}={d[f'turn_{a}']:5.1f}" for a in arms))
            print("    turnO : " + "  ".join(f"{a}={d[f'turn_origin_{a}']:5.1f}" for a in arms))
            print("    prog/ung0: " + "  ".join(f"{a}={d[f'progress_vs_ung0_{a}']:4.2f}"
                                                for a in arms))
            print("    lat   : " + "  ".join(f"{a}={d[f'lat_{a}']:4.2f}" for a in arms))
        print()

    blob = json.dumps(res)
    for forbidden in ('"p_value"', '"pval"', '"p_val"'):
        if forbidden in blob:
            bad(f"a p-value key {forbidden} leaked into the results")
    ok("no p-value keys in the results")

    print()
    if PROBLEMS:
        print(f"{len(PROBLEMS)} ANOMALIES:")
        for p in PROBLEMS:
            print("  - " + p)
    else:
        print("no anomalies")
    if NOTES:
        print(f"{len(NOTES)} notes (expected for a short run):")
        for p in NOTES:
            print("  - " + p)
    return 1 if PROBLEMS else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else "_smoke"))
