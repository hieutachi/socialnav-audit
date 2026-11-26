#!/usr/bin/env python3
"""Regenerate every headline table in this repository from the raw result JSONs.

No GPU, no model, no torch: this reads `results/*.json` and prints the tables
that appear in the reports. Its purpose is to let a reader check a number in a
slide against the file it came from without trusting the author.

Run:
    python code/reproduce_tables.py            # print all tables
    python code/reproduce_tables.py > out.txt  # save

Exit code is 0 only if every invariant check passes. Checks are listed at the
bottom of the output.
"""
from __future__ import annotations

import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RES = os.path.join(ROOT, "results")

POLICY_LABEL = {"published": "published", "il3_s0": "il3_s0"}
COND = "ped_headon_3m"
ARMS = ["ung", "pt", "exp", "cvar", "gui", "guided_rr_pt", "guided_rr_exp", "guided_rr_cvar"]

FAILURES: list[str] = []


# ---------------------------------------------------------------- helpers
def load(name: str) -> dict:
    path = os.path.join(RES, name)
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def mean_of(v):
    """Accept both a bare number and the {'mean':...,'ci':[...]} wrapper."""
    if isinstance(v, dict):
        return float(v["mean"])
    return float(v)


def ci_of(v):
    """CI field name differs per script: 'ci' (41/43/46) or 'ci_sprite_conditional' (48)."""
    if isinstance(v, dict):
        for key in ("ci", "ci_sprite_conditional"):
            if key in v:
                return v[key]
    return None


def fmt(x: float, nd: int = 1) -> str:
    return f"{x:.{nd}f}"


def table(title: str, header: list[str], rows: list[list[str]]) -> None:
    print()
    print(title)
    widths = [max(len(header[i]), *(len(r[i]) for r in rows)) for i in range(len(header))]
    print("| " + " | ".join(h.ljust(w) for h, w in zip(header, widths)) + " |")
    print("|" + "|".join("-" * (w + 2) for w in widths) + "|")
    for r in rows:
        print("| " + " | ".join(c.ljust(w) for c, w in zip(r, widths)) + " |")


def check(cond: bool, label: str, detail: str = "") -> None:
    if not cond:
        FAILURES.append(label)
        print(f"  FAIL  {label}  {detail}")


# ---------------------------------------------------------------- 48
def table_48() -> None:
    d = load("results_48_guided_rerank.json")
    res = d["res"]
    rows = []
    for arm in ARMS:
        cells = [arm]
        for pol in ("published", "il3_s0"):
            cells.append(fmt(mean_of(res[pol]["per_cond"][COND]["hel_" + arm])))
        rows.append(cells)
    table(
        "TABLE 1  Script 48 - held-out contact (%), mean over state x 32 futures\n"
        f"         {COND}, 380 states / 306 sprites / 19 windows",
        ["arm", "published", "il3_s0"],
        rows,
    )

    # contrast table
    rows = []
    for pol in ("published", "il3_s0"):
        pc = res[pol]["per_cond"][COND]
        for key in sorted(pc):
            if not key.startswith(("PRIMARY", "NOMINAL", "EXPLORATORY")):
                continue
            if "guided_rr" not in key:
                continue
            v = pc[key]
            ci = ci_of(v)
            rows.append([
                pol,
                key.split("d_hel_")[-1].replace("_minus_", " - "),
                f"{mean_of(v):+.2f}",
                f"[{ci[0]:+.2f}, {ci[1]:+.2f}]" if ci else "-",
            ])
    table("TABLE 2  Script 48 - contrasts (percentage points)", ["policy", "contrast", "delta", "CI 95%"], rows)

    # geometry
    geo = [
        ("lateral max vs pool-mean ref, mean (m)", "lat", 2),
        ("origin-inclusive max turn (deg)", "turn_origin", 1),
        ("endpoint norm / driver", "progress_vs_gt", 3),
        ("mean clearance to held-out futures (m)", "clr_hel", 3),
    ]
    rows = []
    for label, key, nd in geo:
        cells = [label]
        for pol in ("published", "il3_s0"):
            pc = res[pol]["per_cond"][COND]
            for arm in ("ung", "exp", "gui", "guided_rr_exp"):
                cells.append(fmt(mean_of(pc[f"{key}_{arm}"]), nd))
        rows.append(cells)
    table(
        "TABLE 3  Script 48 - geometry per arm (published: 4 cols, then il3_s0: 4 cols)\n"
        "         order within each policy: ung | exp | gui | guided_rr_exp",
        ["metric", "pub ung", "pub exp", "pub gui", "pub grr_exp", "il3 ung", "il3 exp", "il3 gui", "il3 grr_exp"],
        rows,
    )

    # The NPZ stores all 608 states but not the condition mask. Do not present
    # a zero-contact fraction here: exact primary-cell subset is not in this NPZ.
    # The report's 49.7% is a separate statistic and is not reconstructed here.

    # invariants
    prim = pc["PRIMARY_d_hel_guided_rr_exp_minus_exp"]
    ci = ci_of(prim)
    check(ci is not None and ci[1] < 0, "48 primary CI excludes zero on the negative side", str(ci))
    check(abs(mean_of(prim) + 25.58) < 0.05, "48 primary delta is -25.58 pp", str(mean_of(prim)))
    pub = res["published"]["per_cond"][COND]["PRIMARY_d_hel_guided_rr_exp_minus_exp"]
    check(abs(mean_of(pub) + 18.98) < 0.05, "48 published secondary delta is -18.98 pp", str(mean_of(pub)))
    check(
        mean_of(pc["hel_guided_rr_exp"]) < mean_of(pc["hel_exp"]),
        "48 guided_rr_exp has lower contact than exp on il3_s0",
    )
    check(
        mean_of(pc["lat_guided_rr_exp"]) > mean_of(pc["lat_exp"]),
        "48 guided_rr_exp has larger lateral deviation than exp (the trade-off)",
    )
    check("p_value" not in json.dumps(d).lower(), "48 JSON contains no p-values")


# ---------------------------------------------------------------- 46
def table_46() -> None:
    d = load("results_46_heldout.json")
    rows = []
    for pol in ("published", "il3_s0"):
        pc = d["res"][pol][COND]
        rows.append([
            policy_label(pol),
            fmt(mean_of(pc["ins_pt"])), fmt(mean_of(pc["hel_pt"])),
            fmt(mean_of(pc["ins_exp"])), fmt(mean_of(pc["hel_exp"])),
            f"{mean_of(pc['d_hel_exp_pt']):+.1f}",
            f"[{ci_of(pc['d_hel_exp_pt'])[0]:+.1f}, {ci_of(pc['d_hel_exp_pt'])[1]:+.1f}]",
            fmt(mean_of(pc["flip_exp_vs_pt"])),
        ])
    table(
        "TABLE 5  Script 46 - in-sample vs held-out contact (%), head-on 3 m, N=380\n"
        "         ins_* scores on future 1 (44 protocol); hel_* on futures 8..39",
        ["policy", "ins pt", "hel pt", "ins exp", "hel exp", "delta exp-pt", "CI 95%", "flip %"],
        rows,
    )

    s0 = d["res"]["il3_s0"][COND]
    pub = d["res"]["published"][COND]
    check(mean_of(s0["hel_exp"]) > mean_of(s0["hel_pt"]), "46 il3_s0 exp is worse than pt on held-out")
    check(mean_of(pub["hel_exp"]) < mean_of(pub["hel_pt"]), "46 published exp is better than pt on held-out")
    check(abs(mean_of(s0["d_hel_exp_pt"]) - 1.2) < 0.1, "46 il3_s0 delta is +1.2 pp")
    check(d["n_plan"] == 8 and d["n_held"] == 32, "46 uses 8 planning futures and 32 held-out futures")


def policy_label(p: str) -> str:
    return {"published": "published", "il3_s0": "il3_s0"}.get(p, p)


# ---------------------------------------------------------------- 43
def table_43() -> None:
    d = load("results_43_rerank.json")
    rows = []
    for pol in ("published", "il3_s0", "il3_s1", "il3_s2"):
        pc = d["res"][pol][COND]
        rows.append([
            pol,
            fmt(mean_of(pc["p_cont_ung"])),
            fmt(mean_of(pc["p_cont_gui"])),
            fmt(mean_of(pc["cont_rerank"])),
            fmt(mean_of(pc["cont_rerank_g"])),
            fmt(pc["turn_ung"]), fmt(pc["turn_rerank"]), fmt(pc["turn_gui"]),
        ])
    table(
        "TABLE 6  Script 43 - 16-candidate rerank, in-sample (%), head-on 3 m\n"
        "         rerank_g uses TRUE future velocity for guidance: oracle-advantaged upper bound",
        ["policy", "unguided", "guidance", "rerank16", "rerank16+guide", "turn ung", "turn rr", "turn gui"],
        rows,
    )
    s0 = d["res"]["il3_s0"][COND]
    check(mean_of(s0["cont_rerank_g"]) == 0.0, "43 rr16g is 0.0% on il3_s0 (oracle upper bound)")
    check(mean_of(s0["cont_rerank"]) > mean_of(s0["cont_rerank_g"]), "43 plain rerank16 is worse than oracle variant")


# ---------------------------------------------------------------- 44
def table_44() -> None:
    d = load("results_44_uncertainty.json")
    rows = []
    for pol in ("published", "il3_s0"):
        pc = d["results"][pol][COND]
        rows.append([
            pol,
            fmt(mean_of(pc["cont_nom_ung"])),
            fmt(mean_of(pc["cont_nom_pt_rr"])),
            fmt(mean_of(pc["cont_nom_exp_rr"])),
            fmt(mean_of(pc["cont_nom_cvar_rr"])),
            fmt(mean_of(pc["cont_pert_exp_rr"])),
            fmt(mean_of(pc["flip_rate_exp_vs_pt"])),
        ])
    table(
        "TABLE 7  Script 44 - nominal vs perturbed contact (%), head-on 3 m - IN-SAMPLE\n"
        "         The selector is scored on future 1, which is one of the 8 futures it used.",
        ["policy", "nom ung", "nom pt", "nom exp", "nom cvar", "pert exp", "flip %"],
        rows,
    )
    s0 = d["results"]["il3_s0"][COND]
    check(
        mean_of(s0["cont_nom_exp_rr"]) > mean_of(s0["cont_nom_pt_rr"]),
        "44 in-sample exp is worse than pt on il3_s0 (46 confirms reversal on held-out)",
    )


# ---------------------------------------------------------------- 47b
def table_47b() -> None:
    d = load("results_47b_diversity.json")
    rows = []
    for pol in ("published", "il3_s0", "il3_s1", "il3_s2"):
        pc = d["res"][pol][COND]
        rows.append([
            pol,
            fmt(pc["div_pw"], 3),
            f"{100 * pc['dup_frac']:.1f}",
            fmt(pc["spread_wp"], 3),
            fmt(pc["maoe"], 2),
            str(pc["n"]),
        ])
    table(
        "TABLE 8  Script 47b schema-2 - candidate diversity, head-on 3 m\n"
        "         div_pw = pairwise ADE between candidates (m) - a DIVERSITY metric, not forecast accuracy",
        ["policy", "div_pw (m)", "dup %", "wp spread (m)", "MAOE (deg)", "n"],
        rows,
    )
    check(d.get("schema_version") == 2, "47b is the schema-2 output (supersedes the buggy first run)")
    check(all(d["res"][p][COND]["dup_frac"] < 0.05 for p in d["res"]), "no candidate collapse (dup < 5%)")


# ---------------------------------------------------------------- 41
def table_41() -> None:
    d = load("results_41_pf.json")
    rows = []
    for pol in d["res"]:
        pc = d["res"][pol]["unguided"][COND]
        row = [pol, fmt(mean_of(pc))]
        rows.append(row)
    table(
        "TABLE 9  Script 41 - historical unguided baseline (%), head-on 3 m\n"
        "         in-sample by construction; later scripts are the rerank audit",
        ["policy", "unguided"],
        rows,
    )


# ---------------------------------------------------------------- main
def main() -> int:
    print("=" * 78)
    print("SocialNav open-loop audit - table reproduction")
    print("Reads results/*.json only. No model, no GPU, no network.")
    print("=" * 78)

    table_48()
    table_46()
    table_43()
    table_44()
    table_47b()
    table_41()

    print()
    print("=" * 78)
    if FAILURES:
        print(f"INVARIANT CHECKS: {len(FAILURES)} FAILURE(S)")
        for f in FAILURES:
            print("  -", f)
        return 1
    print("INVARIANT CHECKS: ALL PASS")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    sys.exit(main())
