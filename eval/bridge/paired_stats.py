"""Paired statistics for closed-loop SimplerEnv comparisons (thesis protocol,
Section 4.4: same task instances across conditions, successes out of
trials, paired differences, confidence intervals -- Agarwal et al. 2021).

SimplerEnv's evaluator has no seed: with --obj-variation-mode episode the
object layout is a deterministic function of the episode index, so two
conditions run over the same --obj-episode-range are already paired episode
by episode. Per-episode outcomes are read from the evaluator's video file
names: {success|failure}_obj_episode_{id}.mp4.

    python paired_stats.py BASELINE_DIR CONDITION_DIR [--boot 10000]

Each DIR is searched recursively; the task is the env directory name
(e.g. PutCarrotOnPlateInScene-v0_...) and the seed a `seedN` save tag, so
one DIR may hold several tasks and seeds. Pairs are (task, seed, episode).
"""

from __future__ import annotations

import argparse
import math
import pathlib
import re
from collections import defaultdict

import numpy as np

_NAME = re.compile(r"^(success|failure)_obj_episode_(\d+)\.mp4$")
_SEED = re.compile(r"seed(\d+)")


def load_outcomes(root: pathlib.Path) -> dict[tuple[str, int, int], bool]:
    """(task, seed, episode) -> success. The seed comes from a `seedN` save
    tag in the path (F1-VLA's runs are genuinely seeded); 0 if absent
    (mimic-video's pipeline is deterministic, see its Exp1 method note)."""
    out = {}
    for mp4 in root.rglob("*.mp4"):
        m = _NAME.match(mp4.name)
        if not m:
            continue
        task = next((p.name.split("_")[0] for p in mp4.parents if "InScene" in p.name or "Scene-v" in p.name), "?")
        seed = _SEED.search(str(mp4.relative_to(root)))
        key = (task, int(seed.group(1)) if seed else 0, int(m.group(2)))
        if key in out and out[key] != (m.group(1) == "success"):
            raise ValueError(f"conflicting outcomes for {key} under {root}")
        out[key] = m.group(1) == "success"
    return out


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact McNemar p-value from the discordant counts."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2**n
    return min(1.0, 2 * tail)


def paired_report(base: dict, cond: dict, n_boot: int = 10000, seed: int = 0) -> dict:
    keys = sorted(set(base) & set(cond))
    if not keys:
        raise ValueError("no episodes in common")
    x = np.array([base[k] for k in keys], dtype=float)
    y = np.array([cond[k] for k in keys], dtype=float)
    tasks = np.array([k[0] for k in keys])
    b = int(((x == 1) & (y == 0)).sum())  # baseline succeeded, condition failed
    c = int(((x == 0) & (y == 1)).sum())
    # stratified paired bootstrap: resample episodes within each task
    rng = np.random.default_rng(seed)
    groups = [np.flatnonzero(tasks == t) for t in np.unique(tasks)]
    diffs = np.empty(n_boot)
    for i in range(n_boot):
        idx = np.concatenate([rng.choice(g, size=len(g), replace=True) for g in groups])
        diffs[i] = y[idx].mean() - x[idx].mean()
    per_task = defaultdict(dict)
    for t in np.unique(tasks):
        m = tasks == t
        per_task[t] = {"n": int(m.sum()), "base": int(x[m].sum()), "cond": int(y[m].sum())}
    return {
        "n": len(keys),
        "base_successes": int(x.sum()),
        "cond_successes": int(y.sum()),
        "base_wilson": wilson(int(x.sum()), len(keys)),
        "cond_wilson": wilson(int(y.sum()), len(keys)),
        "diff": float(y.mean() - x.mean()),
        "diff_ci95": (float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))),
        "discordant_base_only": b,
        "discordant_cond_only": c,
        "mcnemar_p": mcnemar_exact(b, c),
        "per_task": dict(per_task),
        "only_in_one": len(set(base) ^ set(cond)),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("baseline_dir", type=pathlib.Path)
    ap.add_argument("condition_dir", type=pathlib.Path)
    ap.add_argument("--boot", type=int, default=10000)
    a = ap.parse_args()
    r = paired_report(load_outcomes(a.baseline_dir), load_outcomes(a.condition_dir), a.boot)
    pct = lambda v: f"{100 * v:.1f}%"
    print(f"paired episodes: {r['n']}  (episodes present in only one run: {r['only_in_one']})")
    print("| task | n | baseline | condition |\n|---|---|---|---|")
    for t, d in r["per_task"].items():
        print(f"| {t} | {d['n']} | {d['base']}/{d['n']} | {d['cond']}/{d['n']} |")
    lo, hi = r["base_wilson"]
    print(f"baseline:  {r['base_successes']}/{r['n']} = {pct(r['base_successes'] / r['n'])} [Wilson 95%: {pct(lo)}, {pct(hi)}]")
    lo, hi = r["cond_wilson"]
    print(f"condition: {r['cond_successes']}/{r['n']} = {pct(r['cond_successes'] / r['n'])} [Wilson 95%: {pct(lo)}, {pct(hi)}]")
    lo, hi = r["diff_ci95"]
    print(f"paired difference: {100 * r['diff']:+.1f} pp [stratified bootstrap 95%: {100 * lo:+.1f}, {100 * hi:+.1f}]")
    print(f"discordant pairs: baseline-only {r['discordant_base_only']}, condition-only "
          f"{r['discordant_cond_only']}; exact McNemar p = {r['mcnemar_p']:.4f}")


if __name__ == "__main__":
    main()
