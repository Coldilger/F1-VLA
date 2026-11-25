"""Experiments 6-7 (F1-VLA): graded and horizon-shifted forecast replacement.

Two readings of the same closed-loop runs (README.md, "What is replaced"):

1. Closed loop. Each condition's 288 episodes (216 for bank:other_task, which
   has no Eggplant donor) are paired by (task, seed, episode) with `control`
   (own tokens fed back through the same hook, same RNG use) via
   ../paired_stats.py. p-values are also Holm-adjusted over the 13
   conditions. `control` vs Experiment 1's `exp1_baseline` is a check.
2. Single decision. Per decision, d(donor, own) = mean over the 4-step chunk
   of the L2 distance between the action decoded with the donor forecast and
   with the own forecast, same action start noise (normalised action space);
   the noise floor d(own2, own) does the same with a second own forecast
   sampled with another seed. Averaged per episode, then over episodes, with
   a bootstrap CI (episodes resampled within task) on the paired difference
   donor - floor. Also: share of the 14 action-relevant tokens that differ
   from the own forecast, and the relative change of the 29 forecast
   positions' keys / values.

Decisions on the states a condition visits differ between conditions (the
closed loop diverges), so the single-decision numbers compare each donor
with the floor measured at the same decisions, not across conditions.

    python analyze_exp67.py     # prints markdown, writes exp67_summary.json
"""
from __future__ import annotations

import json
import pathlib
import re
import sys
from collections import defaultdict

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "experiment11_nogenloss"))
sys.path.insert(0, str(HERE.parent))
from analyze_exp11 import load  # noqa: E402
from paired_stats import paired_report  # noqa: E402

RECORDS = HERE / "records"
# tag -> (label, point of the advisor's list)
CONDS = {
    "own_m2": ("own forecast, 2 decisions earlier", "6"),
    "own_m1": ("own forecast, 1 decision earlier", "6"),
    "own_p1": ("own forecast, 1 decision later (lookahead)", "6"),
    "own_p2": ("own forecast, 2 decisions later (lookahead)", "6"),
    "bank_same_task": ("same task, another episode", "1"),
    "bank_other_task": ("same table, another task", "1"),
    "bank_other_scene": ("other scene", "1"),
    "real_m4": ("real frame t-4", "6"),
    "real_0": ("real frame t (now)", "6"),
    "real_p1": ("real frame t+1", "6"),
    "real_p3": ("real frame t+3 (trained horizon, oracle)", "6"),
    "real_p8": ("real frame t+8", "6"),
    "real_p16": ("real frame t+16", "6"),
}


LOOKAHEAD = ("own_p1", "own_p2", "real_p1", "real_p3", "real_p8", "real_p16")


def sfx(tag: str) -> str:
    """Lookahead conditions were rerun with a freshly built replay env (tag suffix
    _fresh, 2026-11-25); the first runs reused one replay env (README, caveat).
    The rerun is used once its records exist."""
    return "_fresh" if tag in LOOKAHEAD and (RECORDS / f"exp67_{tag}_seed0_fresh").exists() else ""


def outcomes(tag: str):
    return load(re.compile(rf"^(\S+?)_exp67_{tag}_seed(\d+){sfx(tag)}$"))


def holm(ps: dict[str, float]) -> dict[str, float]:
    order = sorted(ps, key=ps.get)
    out, run = {}, 0.0
    for i, k in enumerate(order):
        run = max(run, min(1.0, (len(order) - i) * ps[k]))
        out[k] = run
    return out


def decisions(tag: str):
    """Per-episode means of the single-decision quantities, keyed (task, seed, episode)."""
    eps = {}
    for seed in (0, 1, 2):
        root = RECORDS / f"exp67_{tag}_seed{seed}{sfx(tag)}"
        for f in sorted(root.glob("*/episode_*.npz")):
            recs = np.load(f, allow_pickle=True)["records"]
            d = lambda a, b: float(np.linalg.norm(a - b, axis=-1).mean())
            row = np.array([[d(r["action_donor"], r["action_own"]), d(r["action_own2"], r["action_own"]),
                             r["tok_diff_donor"], r["tok_diff_seed"], r["kv_change_donor"]["k"],
                             r["kv_change_donor"]["v"], r.get("replay_max_diff", 0.0)] for r in recs])
            eps[(f.parent.name, seed, int(f.stem.split("_")[1]))] = row.mean(0).tolist() + [row[:, 6].max(), len(recs)]
    return eps


def replay_share(tag: str) -> float | None:
    """Share of lookahead decisions whose replay env did not match the main env (> 1e-6)."""
    v = [r["replay_max_diff"] for f in RECORDS.glob(f"exp67_{tag}_seed[012]{sfx(tag)}/*/episode_*.npz")
         for r in np.load(f, allow_pickle=True)["records"] if "replay_max_diff" in r]
    return float(np.mean(np.array(v) > 1e-6)) if v else None


def boot_diff(eps: dict, n_boot: int = 10000, seed: int = 0):
    keys = sorted(eps)
    x = np.array([eps[k][0] - eps[k][1] for k in keys])
    tasks = np.array([k[0] for k in keys])
    groups = [np.flatnonzero(tasks == t) for t in np.unique(tasks)]
    rng = np.random.default_rng(seed)
    bs = [x[np.concatenate([rng.choice(g, len(g)) for g in groups])].mean() for _ in range(n_boot)]
    return float(x.mean()), (float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5)))


def main() -> None:
    control, _ = outcomes("control")
    base, _ = load(re.compile(r"^(\S+?)_exp1_baseline_seed(\d+)$"))
    print("lookahead runs: " + ", ".join(f"{t}{sfx(t) or ' (reused replay env)'}" for t in LOOKAHEAD) + "\n")
    summary = {"lookahead_suffix": {t: sfx(t) for t in LOOKAHEAD}, "control_vs_exp1_baseline": None, "closed_loop": {}, "single_decision": {}}

    r = paired_report(base, control)
    summary["control_vs_exp1_baseline"] = {k: r[k] for k in ("n", "base_successes", "cond_successes", "diff", "diff_ci95",
                                                              "mcnemar_p", "only_in_one")}
    print(f"Check: exp1_baseline {r['base_successes']}/{r['n']} vs control {r['cond_successes']}/{r['n']}, "
          f"{100 * r['diff']:+.1f} pp [{100 * r['diff_ci95'][0]:+.1f}, {100 * r['diff_ci95'][1]:+.1f}], "
          f"p = {r['mcnemar_p']:.3g} (only_in_one {r['only_in_one']})\n")

    reps = {}
    for tag in CONDS:
        cond, flags = outcomes(tag)
        reps[tag] = paired_report(control, cond)
        reps[tag]["flags"] = {t: dict(v) for t, v in flags.items()}
    adj = holm({t: reps[t]["mcnemar_p"] for t in CONDS})

    print("## Closed loop, paired with `control`\n")
    print("| condition | point | n | control | condition | diff [95% CI] | McNemar p | Holm p | lost / gained |")
    print("|---|---|---|---|---|---|---|---|---|")
    for tag, (lab, pt) in CONDS.items():
        r = reps[tag]
        lo, hi = r["diff_ci95"]
        print(f"| `{tag}` {lab} | {pt} | {r['n']} | {r['base_successes']} | {r['cond_successes']} | "
              f"{100 * r['diff']:+.1f} pp [{100 * lo:+.1f}, {100 * hi:+.1f}] | {r['mcnemar_p']:.3g} | {adj[tag]:.3g} | "
              f"{r['discordant_base_only']} / {r['discordant_cond_only']} |")
        summary["closed_loop"][tag] = {k: r[k] for k in ("n", "base_successes", "cond_successes", "diff", "diff_ci95",
                                                         "mcnemar_p", "discordant_base_only", "discordant_cond_only",
                                                         "only_in_one", "per_task", "flags")} | {"holm_p": adj[tag]}

    tasks = sorted({k[0] for k in control})
    print("\n| condition | " + " | ".join(t.split("-")[0] for t in tasks) + " |\n|---|" + "---|" * len(tasks))
    print("| `control` | " + " | ".join(f"{sum(v for k, v in control.items() if k[0] == t)}/72" for t in tasks) + " |")
    for tag in CONDS:
        pt = reps[tag]["per_task"]
        print(f"| `{tag}` | " + " | ".join(f"{pt[t]['cond']}/{pt[t]['n']}" if t in pt else "—" for t in tasks) + " |")

    # replay divergence (diag_replay.py): the lookahead future is exact at most Carrot and Stack
    # decisions but not at Eggplant and Spoon ones, so the lookahead conditions are also given for these two
    keep = ("PutCarrotOnPlateInScene-v0", "StackGreenCubeOnYellowCubeBakedTexInScene-v0")
    sub = lambda d: {k: v for k, v in d.items() if k[0] in keep}
    print("\n### Lookahead conditions, Carrot + Stack only (replay exact at most decisions)\n")
    print("| condition | n | control | condition | diff [95% CI] | McNemar p |\n|---|---|---|---|---|---|")
    summary["closed_loop_carrot_stack"] = {}
    for tag in LOOKAHEAD:
        r = paired_report(sub(control), sub(outcomes(tag)[0]))
        lo, hi = r["diff_ci95"]
        summary["closed_loop_carrot_stack"][tag] = {k: r[k] for k in ("n", "base_successes", "cond_successes", "diff",
                                                                      "diff_ci95", "mcnemar_p")}
        print(f"| `{tag}` | {r['n']} | {r['base_successes']} | {r['cond_successes']} | {100 * r['diff']:+.1f} pp "
              f"[{100 * lo:+.1f}, {100 * hi:+.1f}] | {r['mcnemar_p']:.3g} |")

    print("\n## Single decision (same states, same action noise)\n")
    print("| condition | decisions | d(donor, own) | floor d(own2, own) | donor − floor [95% CI] | tokens changed: donor / reseed | key / value change | replay inexact |")
    print("|---|---|---|---|---|---|---|---|")
    for tag in ("control",) + tuple(CONDS):
        eps = decisions(tag)
        a = np.array(list(eps.values()))
        m, (lo, hi) = boot_diff(eps)
        s = {"episodes": len(eps), "decisions": int(a[:, 8].sum()), "d_donor": float(a[:, 0].mean()),
             "d_floor": float(a[:, 1].mean()), "diff": m, "diff_ci95": (lo, hi), "tok_diff_donor": float(a[:, 2].mean()),
             "tok_diff_seed": float(a[:, 3].mean()), "kv_k": float(a[:, 4].mean()), "kv_v": float(a[:, 5].mean()),
             "replay_max_diff": float(a[:, 7].max()), "replay_inexact_share": replay_share(tag)}
        summary["single_decision"][tag] = s
        rep = "—" if s["replay_inexact_share"] is None else f"{100 * s['replay_inexact_share']:.0f}%"
        print(f"| `{tag}` | {s['decisions']} | {s['d_donor']:.4f} | {s['d_floor']:.4f} | {m:+.4f} [{lo:+.4f}, {hi:+.4f}] | "
              f"{s['tok_diff_donor']:.2f} / {s['tok_diff_seed']:.2f} | {s['kv_k']:.3f} / {s['kv_v']:.3f} | "
              f"{rep} |")

    (HERE / "exp67_summary.json").write_text(json.dumps(summary, indent=1, default=str))


if __name__ == "__main__":
    main()
