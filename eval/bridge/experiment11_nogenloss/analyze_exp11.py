"""Experiment 11 (F1-VLA): deployed Bridge finetune vs the same finetune with
forecasting-loss weight 0 (gen_out_loss_ratio 0), both 100k steps.
288 episodes (4 tasks x 24 episodes x policy seeds 0-2) paired by
(task, seed, episode) with ../paired_stats.py. Result dirs are matched by
exact name, so '..._seed0_smoke' etc. are not picked up.

    python analyze_exp11.py     # prints markdown, writes exp11_summary.json
"""
from __future__ import annotations

import json
import pathlib
import re
import sys
from collections import defaultdict

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from paired_stats import _FLAG, _NAME, paired_report  # noqa: E402

RESULTS = HERE.parent / "SimplerEnv" / "results"
RUNS = {"deployed (forecast loss 0.1)": re.compile(r"^(\S+?)_exp1_baseline_seed(\d+)$"),
        "no forecast loss (0)": re.compile(r"^(\S+?)_exp1_baseline_seed(\d+)_nogenloss$")}


def load(pat: re.Pattern):
    out, flags = {}, defaultdict(lambda: defaultdict(int))
    for mp4 in RESULTS.rglob("*.mp4"):
        m = _NAME.match(mp4.name)
        if not m:
            continue
        d = next((p for p in mp4.parents if pat.match(p.name)), None)
        if d is None:
            continue
        task, seed = pat.match(d.name).groups()
        out[(task, int(seed), int(m.group(2)))] = m.group(1) == "success"
        for name, val in _FLAG.findall(m.group(3) or ""):
            flags[task][name] += val == "True"
    return out, flags


def main() -> None:
    (base, bflags), (cond, cflags) = (load(p) for p in RUNS.values())
    r = paired_report(base, cond)
    lo, hi = r["diff_ci95"]
    tasks = sorted({t for t, _, _ in base})
    per = lambda d, t: sum(v for (tt, _, _), v in d.items() if tt == t)
    summary = {"n": r["n"], "deployed": r["base_successes"], "nogenloss": r["cond_successes"],
               "nogenloss_wilson": r["cond_wilson"], "diff": r["diff"], "diff_ci95": r["diff_ci95"],
               "mcnemar_p": r["mcnemar_p"], "lost_gained": [r["discordant_base_only"], r["discordant_cond_only"]],
               "only_in_one": r["only_in_one"],
               "per_task": {t: {"deployed": per(base, t), "nogenloss": per(cond, t),
                                "n": sum(1 for (tt, _, _) in cond if tt == t),
                                "flags_deployed": dict(bflags[t]), "flags_nogenloss": dict(cflags[t])} for t in tasks}}
    (HERE / "exp11_summary.json").write_text(json.dumps(summary, indent=1))
    print("| run | success | diff vs deployed [95% CI] | McNemar p | lost / gained |\n|---|---|---|---|---|")
    print(f"| deployed | {r['base_successes']}/{r['n']} | — | — | — |")
    print(f"| no forecast loss | {r['cond_successes']}/{r['n']} | {100 * r['diff']:+.1f} pp [{100 * lo:+.1f}, {100 * hi:+.1f}] "
          f"| {r['mcnemar_p']:.4g} | {r['discordant_base_only']} / {r['discordant_cond_only']} |")
    print(f"only_in_one: {r['only_in_one']}")
    print("\n| task | deployed | no forecast loss | flags deployed | flags no forecast loss |\n|---|---|---|---|---|")
    for t, v in summary["per_task"].items():
        print(f"| {t} | {v['deployed']}/{v['n']} | {v['nogenloss']}/{v['n']} | {v['flags_deployed']} | {v['flags_nogenloss']} |")


if __name__ == "__main__":
    main()
