# Experiment 2 — Oracle injection (F1-VLA)

**Status: done.** See [`ORACLE_EXPERIMENT.md`](ORACLE_EXPERIMENT.md) for the
full write-up (mechanism, metric, results, caveats).

Short version: replace F1's self-sampled VAR foresight tokens with the real
next frame's ground-truth VQ tokens, and compare predicted-action L1 error
against the unmodified (self-sampled) baseline on real logged Bridge
moments. Result: oracle is ~6x more accurate than the trivial "don't move"
baseline — F1 does use precise future information when given it, not just
carry the foresight computation as a ritual.

Files: `oracle_offline_probe.py` (offline probe, no simulator),
`oracle_offline_probe.slurm` / `oracle_smoke.slurm` (launchers). The oracle
hook itself lives in the model, not here — see
`f1_vla/src/models/modeling_f1.py`'s `oracle_indices` parameter on
`sample_actions_with_world_model`.
