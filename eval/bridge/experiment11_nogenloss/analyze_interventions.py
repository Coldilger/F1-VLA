"""Experiment 1 interventions (zero / shuffle / mean forecast) on the model
finetuned without the forecasting loss, next to the same interventions on the
deployed model. Each intervention is paired with its own model's baseline by
(task, seed, episode) with ../paired_stats.py. Conditions with fewer than
288 episodes are reported as incomplete.

    python analyze_interventions.py   # prints markdown, writes interventions_summary.json
"""
from __future__ import annotations

import json
import re

from analyze_exp11 import HERE, load
from paired_stats import paired_report

FULL = 288
MODELS = {"no forecast loss": "_nogenloss", "deployed": ""}


def pat(cond: str, suffix: str) -> re.Pattern:
    return re.compile(rf"^(\S+?)_exp1_{cond}_seed(\d+){suffix}$")


def main() -> None:
    summary = {}
    print("| model | intervention | n | baseline | intervened | diff [95% CI] | McNemar p | lost / gained |")
    print("|---|---|---|---|---|---|---|---|")
    for model, sfx in MODELS.items():
        base, _ = load(pat("baseline", sfx))
        for cond in ("zero", "shuffle", "mean"):
            res, _ = load(pat(cond, sfx))
            if not res:
                continue
            if len(res) < FULL:
                print(f"| {model} | {cond} | {len(res)}/{FULL} | incomplete | | | | |")
                continue
            r = paired_report(base, res)
            lo, hi = r["diff_ci95"]
            summary[f"{model}/{cond}"] = {k: r[k] for k in ("n", "base_successes", "cond_successes", "diff",
                                                            "diff_ci95", "mcnemar_p", "discordant_base_only",
                                                            "discordant_cond_only", "only_in_one")}
            print(f"| {model} | {cond} | {r['n']} | {r['base_successes']} | {r['cond_successes']} | "
                  f"{100 * r['diff']:+.1f} pp [{100 * lo:+.1f}, {100 * hi:+.1f}] | {r['mcnemar_p']:.3g} | "
                  f"{r['discordant_base_only']} / {r['discordant_cond_only']} |")
    (HERE / "interventions_summary.json").write_text(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
