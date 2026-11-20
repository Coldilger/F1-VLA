"""Experiment 9 (F1-VLA): success and latency against the number of forecast
scales, paired episode by episode with the Experiment 1 baseline
(exp1_baseline_seed{0,1,2}; same tasks, episodes and policy seeds;
../paired_stats.py: Wilson CIs, stratified paired bootstrap, exact McNemar).

Latency per decision comes from timing/*.json (one file per job, every
_predict_new_chunk call, H100). Only the full runs (no tag suffix) are
counted; smoke runs are listed separately.

    python analyze_scales.py         # prints markdown, writes scales_summary.json
"""

from __future__ import annotations

import json
import pathlib
import re
import sys
from collections import defaultdict

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from paired_stats import _NAME, paired_report, wilson  # noqa: E402

RESULTS = HERE.parent / "SimplerEnv" / "results"
CONDITIONS = [(1, 0), (1, 1), (2, 0), (2, 1), (4, 0), (6, 0), (6, 1)]
PATCH_NUMS = (1, 2, 3, 4, 5, 6, 8, 10, 13, 16)


def n_forecast(s: int) -> int:
    return sum(p * p for p in PATCH_NUMS[:s]) - 1


def load_exact(stem: str) -> dict[tuple[str, int, int], bool]:
    """(task, seed, episode) -> success, from result dirs named exactly
    <task>_<stem><seed> (so '..._seed0_smoke' is not picked up)."""
    pat = re.compile(rf"^(\S+?)_{re.escape(stem)}(\d+)$")
    out = {}
    for mp4 in RESULTS.rglob("*.mp4"):
        m = _NAME.match(mp4.name)
        if not m:
            continue
        d = next((p for p in mp4.parents if pat.match(p.name)), None)
        if d is None:
            continue
        task, seed = pat.match(d.name).groups()
        out[(task, int(seed), int(m.group(2)))] = m.group(1) == "success"
    return out


def latencies(cond: str) -> tuple[list[float], set[str]]:
    lat, gpus = [], set()
    for f in sorted((HERE / "timing").glob(f"{cond}_seed*.json")):
        d = json.loads(f.read_text())
        lat += d["latency_s"]
        gpus.add(d["gpu"])
    return lat, gpus


def main() -> None:
    base = load_exact("exp1_baseline_seed")
    rows = []
    for s, p in CONDITIONS:
        cond = f"scales{s}_pos{p}"
        res = load_exact(f"exp9_{cond}_seed")
        lat, gpus = latencies(cond)
        row = {"scales": s, "pos_match": p, "forecast_positions": n_forecast(s),
               "action_position_shift": 0 if p else n_forecast(s) - n_forecast(4),
               "n_decisions": len(lat), "gpus": sorted(gpus)}
        if lat:
            row["latency_ms_median"] = 1000 * float(np.median(lat))
            row["latency_ms_p95"] = 1000 * float(np.percentile(lat, 95))
        if res:
            r = paired_report(base, res)
            row.update({k: r[k] for k in ("n", "base_successes", "cond_successes", "cond_wilson", "diff", "diff_ci95",
                                          "discordant_base_only", "discordant_cond_only", "mcnemar_p", "only_in_one")})
            row["per_task"] = {t: v["cond"] for t, v in r["per_task"].items()}
            row["per_task_n"] = {t: v["n"] for t, v in r["per_task"].items()}
        rows.append(row)
    (HERE / "scales_summary.json").write_text(json.dumps(rows, indent=1))

    print("| scales | forecast positions | action position shift | episodes | success | Wilson 95% | diff vs deployed [95% CI] | McNemar p | lost / gained | latency median (p95), ms | decisions timed |")
    print("|---|---|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        shift = "0 (matched)" if r["pos_match"] else f"{r['action_position_shift']:+d}"
        lat = f"{r['latency_ms_median']:.1f} ({r['latency_ms_p95']:.1f})" if "latency_ms_median" in r else "—"
        if "n" in r:
            lo, hi = r["cond_wilson"]
            dlo, dhi = r["diff_ci95"]
            print(f"| {r['scales']} | {r['forecast_positions']} | {shift} | {r['n']} | {r['cond_successes']}/{r['n']} = {100 * r['cond_successes'] / r['n']:.1f}% "
                  f"| [{100 * lo:.1f}, {100 * hi:.1f}] | {100 * r['diff']:+.1f} pp [{100 * dlo:+.1f}, {100 * dhi:+.1f}] | {r['mcnemar_p']:.4f} "
                  f"| {r['discordant_base_only']} / {r['discordant_cond_only']} | {lat} | {r['n_decisions']} |")
        else:
            print(f"| {r['scales']} | {r['forecast_positions']} | {shift} | 0 | — | — | — | — | — | {lat} | {r['n_decisions']} |")
    tasks = sorted({t for r in rows for t in r.get("per_task", {})})
    if tasks:
        print("\n| scales | position | " + " | ".join(t.replace("InScene-v0", "").replace("Scene-v0", "") for t in tasks) + " |")
        print("|---|---|" + "---|" * len(tasks))
        bt = defaultdict(int)
        for (t, _, _), ok in base.items():
            bt[t] += ok
        print("| 4 | deployed (Exp 1 baseline, seeds 0-2) | " + " | ".join(f"{bt[t]}/72" for t in tasks) + " |")
        for r in rows:
            if "per_task" in r:
                label = "deployed (rerun, seed 0)" if r["scales"] == 4 else ("matched" if r["pos_match"] else "shifted")
                print(f"| {r['scales']} | {label} | " + " | ".join(f"{r['per_task'].get(t, '—')}/{r['per_task_n'].get(t, '—')}" for t in tasks) + " |")


if __name__ == "__main__":
    main()
