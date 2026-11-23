"""Experiment 12 follow-up (F1-VLA): the deployed 100k checkpoint evaluated with
(a) the exact rotation state (F1_STATE_FRAME=bridge, tag suffix _e12statebridge)
and (b) the training history spacing (F1_HISTORY_SPACING=3, _e12hist3), each
paired with exp1_baseline_seed{0,1,2} by (task, seed, episode).
Audit: LDA-1B/eval/bridge/experiment12_audit/README.md.

    python analyze_exp12.py   # prints markdown, writes exp12_summary.json
"""
from __future__ import annotations

import json
import pathlib
import re
import sys

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "experiment11_nogenloss"))
sys.path.insert(0, str(HERE.parent))
from analyze_exp11 import load  # noqa: E402
from paired_stats import paired_report  # noqa: E402

BASE = re.compile(r"^(\S+?)_exp1_baseline_seed(\d+)$")
CONDS = {"exact rotation state": "_e12statebridge", "history spacing 0.6 s": "_e12hist3"}


def main() -> None:
    base, bflags = load(BASE)
    summary = {}
    print("| condition | n | baseline | condition | diff [95% CI] | McNemar p | lost / gained |\n|---|---|---|---|---|---|---|")
    per_task = {}
    for name, sfx in CONDS.items():
        cond, cflags = load(re.compile(rf"^(\S+?)_exp1_baseline_seed(\d+){sfx}$"))
        r = paired_report(base, cond)
        lo, hi = r["diff_ci95"]
        summary[name] = {k: r[k] for k in ("n", "base_successes", "cond_successes", "diff", "diff_ci95", "mcnemar_p",
                                           "discordant_base_only", "discordant_cond_only", "only_in_one")}
        print(f"| {name} | {r['n']} | {r['base_successes']} | {r['cond_successes']} | {100 * r['diff']:+.1f} pp "
              f"[{100 * lo:+.1f}, {100 * hi:+.1f}] | {r['mcnemar_p']:.3g} | {r['discordant_base_only']} / {r['discordant_cond_only']} |")
        for t in sorted({k[0] for k in cond}):
            per_task.setdefault(t, {"baseline": sum(v for k, v in base.items() if k[0] == t)})[name] = \
                sum(v for k, v in cond.items() if k[0] == t)
            per_task[t][name + " grasp"] = cflags[t].get("consecutive_grasp", 0)
        summary[name]["per_task"] = {t: v[name] for t, v in per_task.items()}
    print("\n| task | baseline | " + " | ".join(CONDS) + " |\n|---|---|" + "---|" * len(CONDS))
    for t, v in per_task.items():
        print(f"| {t} | {v['baseline']}/72 | " + " | ".join(f"{v.get(n, '—')}/72" for n in CONDS) + " |")
    (HERE / "exp12_summary.json").write_text(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
