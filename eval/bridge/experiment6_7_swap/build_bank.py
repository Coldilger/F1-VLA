"""Donor bank for Experiment 6 (graded foresight replacement), F1-VLA.

The donors are F1-VLA's own forecasts on states of normal (deployed-policy)
rollouts: the 30 forecast token ids sampled at every replanning decision of
Experiment 8's probe (`../experiment8_conflict/records/<task>/episode_NNN.npz`,
`tokens_A`, instruction = the scene's own, seed 0, 4 tasks x 24 episodes;
Stack episode 0 is missing there). Writes `bank_tokens.npz` with, per task,
`tokens` [24, K, 30] (int64, -1 where the episode is missing or shorter)
and `n` [24] (decisions per episode).

    python build_bank.py
"""
from __future__ import annotations

import pathlib

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
SRC = HERE.parent / "experiment8_conflict" / "records"
TASKS = ["PutCarrotOnPlateInScene-v0", "PutSpoonOnTableClothInScene-v0",
         "StackGreenCubeOnYellowCubeBakedTexInScene-v0", "PutEggplantInBasketScene-v0"]


def main() -> None:
    out = {}
    for task in TASKS:
        eps = {}
        for f in sorted((SRC / task).glob("episode_*.npz")):
            recs = np.load(f, allow_pickle=True)["records"]
            steps = [int(r["global_step"]) for r in recs]
            assert steps == sorted(steps) and len(set(np.diff(steps))) <= 1, (f, steps)
            eps[int(f.stem.split("_")[1])] = np.stack([np.asarray(r["tokens_A"], dtype=np.int64) for r in recs])
        # Experiment 8 numbered episodes in the order it ran them; Stack's
        # first episode (0) was skipped, so its files 0..22 are episodes 1..23
        if len(eps) == 23:
            eps = {k + 1: v for k, v in eps.items()}
        k_max = max(v.shape[0] for v in eps.values())
        tok = -np.ones((24, k_max, 30), dtype=np.int64)
        n = np.zeros(24, dtype=np.int64)
        for e, v in eps.items():
            tok[e, :len(v)] = v
            n[e] = len(v)
        out[f"{task}/tokens"], out[f"{task}/n"] = tok, n
        print(task, "episodes", int((n > 0).sum()), "decisions per episode", sorted(set(n.tolist())))
    np.savez_compressed(HERE / "bank_tokens.npz", **out)


if __name__ == "__main__":
    main()
