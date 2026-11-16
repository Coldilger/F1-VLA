"""Experiment 8, step 2: closed-loop outcomes of the four cells
(understanding block from A or B) x (generation block from A or B), read from
the SimplerEnv video names (success flag and the simulator's per-episode
flags), paired by episode (../paired_stats.py).

All flags are relative to the scene's own task A: `moved_correct_obj` = the
object A asks to move was moved, `moved_wrong_obj` = the other object was.

    python analyze_conflict_cl.py [alt]     # alt defaults to "mirror"
"""

from __future__ import annotations

import collections
import json
import pathlib
import re
import sys

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from paired_stats import load_outcomes, paired_report  # noqa: E402

RESULTS = HERE.parent / "SimplerEnv" / "results"
CELLS = [("A", "A"), ("A", "B"), ("B", "A"), ("B", "B")]
FLAGS = ["moved_correct_obj", "moved_wrong_obj", "is_src_obj_grasped", "src_on_target"]
SHORT = {"PutCarrotOnPlateInScene-v0": "Carrot", "PutSpoonOnTableClothInScene-v0": "Spoon",
         "StackGreenCubeOnYellowCubeBakedTexInScene-v0": "Stack", "PutEggplantInBasketScene-v0": "Eggplant"}
_NAME = re.compile(r"^(success|failure)_obj_episode_(\d+)(_.*)?\.mp4$")


def tag(und: str, gen: str, alt: str) -> str:
    return f"exp8_cl_und{und}_gen{gen}_{alt}_seed0"


def per_task(und: str, gen: str, alt: str) -> dict[str, dict[str, int]]:
    """task -> {n, success, <flag counts>}"""
    out: dict[str, dict[str, int]] = collections.defaultdict(lambda: collections.Counter())
    for mp4 in RESULTS.rglob("*.mp4"):
        m = _NAME.match(mp4.name)
        if not m or tag(und, gen, alt) not in str(mp4):
            continue
        task = next(SHORT[t] for t in SHORT if t in str(mp4))
        c = out[task]
        c["n"] += 1
        c["success"] += m.group(1) == "success"
        for f in FLAGS:
            c[f] += f"{f}_True" in mp4.name
    return out


def main() -> None:
    alt = sys.argv[1] if len(sys.argv) > 1 else "mirror"
    cells = {c: per_task(*c, alt) for c in CELLS}
    tasks = [t for t in ("Carrot", "Eggplant", "Spoon", "Stack") if any(t in cells[c] for c in CELLS)]
    summary = {"alt": alt, "cells": {f"und{u}_gen{g}": {t: dict(v) for t, v in cells[(u, g)].items()} for u, g in CELLS}, "paired": {}}

    print(f"### Per cell, all tasks (alternative instruction: {alt})\n")
    print("| understanding block | generation block | episodes | success (task A) | moved A's object | moved the other object | grasped A's object |")
    print("|---|---|---|---|---|---|---|")
    for u, g in CELLS:
        tot = collections.Counter()
        for t in tasks:
            tot.update(cells[(u, g)].get(t, {}))
        n = tot["n"]
        print(f"| {u} | {g} | {n} | {tot['success']}/{n} | {tot['moved_correct_obj']}/{n} | {tot['moved_wrong_obj']}/{n} | {tot['is_src_obj_grasped']}/{n} |")

    print("\n### Per task: success / moved A's object / moved the other object\n")
    print("| understanding | generation | " + " | ".join(tasks) + " |")
    print("|---|---|" + "---|" * len(tasks))
    for u, g in CELLS:
        row = []
        for t in tasks:
            c = cells[(u, g)].get(t)
            row.append("—" if not c else f"{c['success']} / {c['moved_correct_obj']} / {c['moved_wrong_obj']} of {c['n']}")
        print(f"| {u} | {g} | " + " | ".join(row) + " |")

    print("\n### Paired comparisons of success on task A\n")
    print("| comparison | what it tests | base | cond | diff [95% CI] | McNemar p | base only / cond only |")
    print("|---|---|---|---|---|---|---|")
    for (bu, bg), (cu, cg), what in [
        (("A", "A"), ("A", "B"), "forecast from B under instruction A"),
        (("B", "B"), ("B", "A"), "forecast from A under instruction B"),
        (("A", "A"), ("B", "B"), "instruction changed everywhere"),
    ]:
        base = load_outcomes(RESULTS, tag(bu, bg, alt) + "/")
        cond = load_outcomes(RESULTS, tag(cu, cg, alt) + "/")
        if not base or not cond:
            continue
        r = paired_report(base, cond)
        summary["paired"][f"und{bu}_gen{bg}->und{cu}_gen{cg}"] = {k: r[k] for k in ("n", "base_successes", "cond_successes", "diff", "diff_ci95", "mcnemar_p", "discordant_base_only", "discordant_cond_only") if k in r}
        lo, hi = r["diff_ci95"]
        print(f"| und {bu}/gen {bg} → und {cu}/gen {cg} | {what} | {r.get('base_successes', '?')}/{r['n']} | {r['cond_successes']}/{r['n']} "
              f"| {100 * r['diff']:+.1f} pp [{100 * lo:+.1f}, {100 * hi:+.1f}] | {r['mcnemar_p']:.4f} | {r['discordant_base_only']} / {r['discordant_cond_only']} |")
    (HERE / f"conflict_cl_summary_{alt}.json").write_text(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
