"""Experiment 8, steps 0-1: summarise records/<task>/episode_*.npz written by
f1_conflict_inference.py.

For every replanning decision the probe stored the action chunk decoded from
the same start noise under six combinations of "whose instruction is in the
understanding block" x "whose forecast is in the generation block":

  AA, AA2 (other forecast sampling seed), BB, AB, BA, ABf (only the 29
  forecast positions swapped).

Reported per task and alternative instruction B (mean over episodes of the
per-episode mean, 95% bootstrap over episodes):

  tok(seed)   share of the 30 sampled forecast tokens that differ between two
              sampling seeds under the same instruction A
  tok(B)      share that differ between instruction A and B, same seed
  kv_fut/hist relative change of the gen block's values, forecast positions /
              history positions, A -> B
  d(X,Y)      mean over the chunk's steps of the L2 distance between two
              action chunks (normalised action units)
  follow      d(AB,AA) / (d(AB,AA) + d(AB,BB)); 0 = the action follows the
              instruction in the understanding block, 1 = it follows the
              forecast in the generation block

    python analyze_conflict.py        # prints markdown, writes conflict_summary.json
"""

from __future__ import annotations

import collections
import json
import pathlib

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
SHORT = {"PutCarrotOnPlateInScene-v0": "Carrot", "PutSpoonOnTableClothInScene-v0": "Spoon",
         "StackGreenCubeOnYellowCubeBakedTexInScene-v0": "Stack", "PutEggplantInBasketScene-v0": "Eggplant"}


def dist(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b, axis=-1).mean())


def tok_diff(a, b) -> float:
    return float((np.asarray(a).ravel() != np.asarray(b).ravel()).mean())


def boot(per_episode: list[float], n: int = 4000) -> tuple[float, float, float]:
    x = np.asarray(per_episode, dtype=float)
    rng = np.random.default_rng(0)
    means = x[rng.integers(0, len(x), size=(n, len(x)))].mean(1)
    return float(x.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def episode_metrics(records) -> tuple[dict, dict]:
    """Per-episode means: seed-level metrics and per-alternative metrics."""
    seed = collections.defaultdict(list)
    alts: dict[str, dict[str, list]] = collections.defaultdict(lambda: collections.defaultdict(list))
    meta: dict[str, dict] = {}
    for r in records:
        aa = r["action_AA"]
        seed["tok_seed"].append(tok_diff(r["tokens_A"], r["tokens_A2"]))
        seed["d_AA2_AA"].append(dist(r["action_AA2"], aa))
        seed["seed_kv_fut"].append(r["seed_kv_diff_future"]["v"])
        seed["seed_kv_hist"].append(r["seed_kv_diff_hist"]["v"])
        seed["seed_kv_und"].append(r["seed_kv_diff_und"]["v"])
        # layout check: number of prefix positions that differ between seeds
        differs = r["seed_pos_diff_v"] > 1e-6
        seed["n_seed_pos"].append(float(differs.sum()))
        # earliest position that differs (prefix length if none does: the
        # tokens of scales 1-3 happened to coincide between the two seeds)
        seed["first_seed_pos"].append(float(np.argmax(differs)) if differs.any() else float(len(differs)))
        seed["no_seed_pos"].append(float(not differs.any()))
        seed["forecast_start"].append(float(r["und_len"] + r["hist_len"]))
        seed["prefix_len"].append(float(len(r["seed_pos_diff_v"])))
        for name, a in r["alts"].items():
            m = alts[name]
            meta[name] = {"instruction": a["instruction"], "same_len": bool(a["same_len"])}
            bb, ab, ba, abf = a["action_BB"], a["action_AB"], a["action_BA"], a["action_ABf"]
            m["tok_B"].append(tok_diff(r["tokens_A"], a["tokens_B"]))
            m["kv_fut"].append(a["kv_diff_future"]["v"])
            m["kv_hist"].append(a["kv_diff_hist"]["v"])
            m["d_BB_AA"].append(dist(bb, aa))
            m["d_AB_AA"].append(dist(ab, aa))
            m["d_ABf_AA"].append(dist(abf, aa))
            m["d_BA_BB"].append(dist(ba, bb))
            m["follow_AB"].append(dist(ab, aa) / (dist(ab, aa) + dist(ab, bb) + 1e-12))
            m["follow_BA"].append(dist(ba, bb) / (dist(ba, bb) + dist(ba, aa) + 1e-12))
    agg = {"first_seed_pos": np.min, "no_seed_pos": np.sum}
    return ({k: float(agg.get(k, np.mean)(v)) for k, v in seed.items()},
            {n: {**{k: float(np.mean(v)) for k, v in m.items()}, **meta[n]} for n, m in alts.items()})


def main() -> None:
    summary = {}
    for task_dir in sorted((HERE / "records").iterdir()):
        files = sorted(task_dir.glob("episode_*.npz"))
        seeds, alts, n_dec = [], collections.defaultdict(list), 0
        for f in files:
            recs = np.load(f, allow_pickle=True)["records"]
            n_dec += len(recs)
            s, a = episode_metrics(recs)
            seeds.append(s)
            for name, m in a.items():
                alts[name].append(m)
        task = {"episodes": len(files), "decisions": n_dec,
                "seed": {k: boot([s[k] for s in seeds]) for k in seeds[0]}, "alts": {}}
        task["seed"]["first_seed_pos_min"] = min(s["first_seed_pos"] for s in seeds)
        task["seed"]["no_seed_pos_sum"] = sum(s["no_seed_pos"] for s in seeds)
        for name, eps in alts.items():
            task["alts"][name] = {"instruction": eps[0]["instruction"], "same_len": eps[0]["same_len"],
                                  **{k: boot([e[k] for e in eps]) for k in eps[0] if k not in ("instruction", "same_len")}}
        summary[SHORT.get(task_dir.name, task_dir.name)] = task
    (HERE / "conflict_summary.json").write_text(json.dumps(summary, indent=1))

    def ci(t, nd=3):
        return f"{t[0]:.{nd}f} [{t[1]:.{nd}f}, {t[2]:.{nd}f}]"

    print("### Layout check and noise floor (same instruction, two forecast sampling seeds)\n")
    print("| task | episodes | decisions | prefix positions | forecast block starts at | earliest position that differs | positions that differ (mean) | decisions with none | tok(seed) | value change und / history / forecast | d(AA2, AA) |")
    print("|---|---|---|---|---|---|---|---|---|---|---|")
    for t, d in summary.items():
        s = d["seed"]
        print(f"| {t} | {d['episodes']} | {d['decisions']} | {s['prefix_len'][0]:.0f} | {s['forecast_start'][0]:.0f} | {s['first_seed_pos_min']:.0f} | {s['n_seed_pos'][0]:.1f} "
              f"| {s['no_seed_pos_sum']:.0f} | {ci(s['tok_seed'], 2)} | {s['seed_kv_und'][0]:.3f} / {s['seed_kv_hist'][0]:.3f} / {s['seed_kv_fut'][0]:.3f} | {ci(s['d_AA2_AA'], 4)} |")

    print("\n### Instruction A vs. instruction B\n")
    print("| task | B | same token count | tok(B) | kv forecast | kv history | d(BB, AA) | d(AB, AA) | d(ABf, AA) | d(BA, BB) | follow(AB) | follow(BA) |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for t, d in summary.items():
        for name, a in d["alts"].items():
            print(f"| {t} | {name}: \"{a['instruction']}\" | {'yes' if a['same_len'] else 'no'} | {ci(a['tok_B'], 2)} | {a['kv_fut'][0]:.3f} | {a['kv_hist'][0]:.3f} "
                  f"| {ci(a['d_BB_AA'])} | {ci(a['d_AB_AA'], 4)} | {ci(a['d_ABf_AA'], 4)} | {ci(a['d_BA_BB'], 4)} | {ci(a['follow_AB'])} | {ci(a['follow_BA'])} |")


if __name__ == "__main__":
    main()
