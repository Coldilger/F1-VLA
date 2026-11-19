# Experiment 9 — Compute vs. performance (F1-VLA): number of forecast scales

**Question (advisor's point 4).** Does F1-VLA get better actions when it
spends more inference compute on its forecast? Plot task success against
the full latency of one decision.

**Prior work this extends.** The F1 paper (Lv et al. 2025, Table 4) varies
the number of forecast scales at inference (2 / 4 / 6; Bridge 52.1 / 59.4 /
57.3%, 24 episodes per task, one run). mimic-video's own denoising-depth
sweep (Pai et al. 2025, Fig. 6) is the same question for a video model;
our mimic-video version is in that repo's `eval/bridge/experiment9_depth`.

## Design

F1 generates its forecast coarse-to-fine over the VAR scales
`patch_nums = (1, 2, 3, 4, 5, 6, 8, 10, 13, 16)` and stops after
`num_resolutions` (= 4 in training and deployment). The inputs of scales
2..S are appended to the generation block:

| scales S | forecast positions in the cache | action tokens' position shift (unmatched) |
|---|---|---|
| 1 | 0 | −29 |
| 2 | 4 | −25 |
| 4 (deployed) | 29 | 0 |
| 6 | 90 | +61 |

Changing S also moves the action tokens' RoPE position ids (they start at
the number of valid prefix positions). Experiment 1 showed that a position
shift alone costs ~14 pp (removal shifts by ~709), so each S is run twice:
as is, and with **POS_MATCH=1**, where the action tokens get exactly the
position ids of the deployed layout and only the amount of forecast in the
cache changes (`f1_scales_inference.py`, a copy of `denoise_step` with an
offset; `modeling_f1.py` is not modified). S = 1 is the cheapest point: no
forecast positions at all, the 680 encoded history positions kept.

Protocol: closed loop, 4 tasks × 24 episodes × policy seeds 0–2 = 288
episodes, paired episode by episode with the Experiment 1 baseline
(`exp1_baseline_seed{0,1,2}`, same seeds and episodes); `../paired_stats.py`.
S = 4 is rerun on seed 0 only (reproducibility check and latency). Every
decision (`_predict_new_chunk`, with `cuda.synchronize`) is timed; all jobs
on H100 NVL.

## Files

| file | role |
|---|---|
| `f1_scales_inference.py` | `F1VLAScales`: sets `num_resolutions`, optional position matching, per-decision timing |
| `main_inference_scales.py` | entry point (env `SCALES`, `POS_MATCH`, `TIMING_OUT`) |
| `scales_closed_loop.slurm` | one task × seed × condition |
| `submit_feeder.sh`, `jobs.txt` | the grid, submitted under the 30-job queue limit |
| `analyze_scales.py` | success + latency table, `scales_summary.json` |
| `timing/*.json` | per-decision latencies |

## Checks (2026-11-19, Carrot, episodes 0–2, seed 0; `jobs_smoke.txt`)

- S = 4 through the wrapper reproduces the Experiment 1 baseline episodes
  exactly (same outcomes and progress flags).
- The copied `denoise_step` with offset 0 (`FORCE_COPY=1`) reproduces them
  as well, so the position-matched runs differ from the original only by
  the offset.
- Prefix length per condition as expected: 984 / 988 / 1013 / 1074 for
  S = 1 / 2 / 4 / 6.
- The fine-tuned checkpoint runs at S = 6 (trained at S = 4; the scale-5/6
  embeddings are those of the pre-trained model).
- Median latency per decision (45 decisions each, H100 NVL): S = 1 173.6 ms,
  S = 2 186.8 ms, S = 4 212.4 ms, S = 6 242.0 ms.

## Results

Running (submitted 2026-11-19).
