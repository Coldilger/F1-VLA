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

## Results (2026-11-20)

76 jobs (`jobs.txt`), all `EVAL_EXIT=0`. 288 paired episodes per condition
(S = 4 rerun: seed 0, 96 episodes). Table from `analyze_scales.py`;
latency per decision on H100 NVL, every decision of every episode timed.

| scales | forecast positions | action tokens' positions | success | paired diff vs deployed [95% CI] | McNemar p | lost / gained | latency median (p95), ms |
|---|---|---|---|---|---|---|---|
| 1 | 0 | shifted −29 | 128/288 = 44.4% | −4.2 pp [−10.1, +1.7] | 0.21 | 45 / 33 | 173.0 (175.6) |
| 1 | 0 | matched | 143/288 = 49.7% | +1.0 pp [−3.8, +5.9] | 0.78 | 25 / 28 | 172.6 (174.4) |
| 2 | 4 | shifted −25 | 137/288 = 47.6% | −1.0 pp [−6.6, +4.5] | 0.81 | 36 / 33 | 185.7 (188.3) |
| 2 | 4 | matched | 136/288 = 47.2% | −1.4 pp [−5.9, +3.1] | 0.65 | 24 / 20 | 185.5 (188.4) |
| **4 (deployed)** | 29 | — | 140/288 = 48.6% (Exp 1); rerun seed 0: 49/96, identical episode by episode | — | — | 0 / 0 | 213.2 (216.5) |
| 6 | 90 | shifted +61 | 133/288 = 46.2% | −2.4 pp [−8.3, +3.5] | 0.50 | 43 / 36 | 237.8 (241.5) |
| 6 | 90 | matched | 143/288 = 49.7% | +1.0 pp [−3.5, +5.6] | 0.77 | 21 / 24 | 238.6 (244.5) |

Per task (Carrot / Eggplant / Spoon / Stack, of 72): deployed 25 / 50 / 36 / 29;
S = 1 matched 28 / 52 / 34 / 29; S = 2 matched 25 / 45 / 35 / 31; S = 6
matched 26 / 51 / 35 / 31; S = 1 shifted 25 / 45 / 33 / 25; S = 6 shifted
21 / 53 / 39 / 20.

**What this shows.**
1. **More forecast does not buy success, less does not cost it.** From no
   forecast positions at all (S = 1) to three times the deployed amount
   (S = 6), every position-matched condition is within ±1.4 pp of the
   deployed policy, all intervals contain 0. The forecast part of the
   decision costs ~66 ms (173 → 239 ms, +38%) for no measurable gain.
2. **The F1 paper's scale ablation is not reproduced as an effect.** Lv et
   al. report 52.1 / 59.4 / 57.3% for 2 / 4 / 6 scales (24 episodes per
   task, one run); on 288 paired episodes the corresponding differences
   here are −1.4 and +1.0 pp.
3. **Shifting the action tokens' positions costs a little, at most.** The
   shifted variants are 0.4–5.2 pp below their matched counterparts
   (largest at S = 1, −29 positions: −4.2 pp vs deployed, CI [−10.1, +1.7]);
   no shifted condition differs significantly from the deployed policy. The
   ~709-position shift of Experiment 1's removal cost −13.9 pp; small shifts
   are tolerated.
4. Together with Experiments 1 and 8: at inference F1-VLA's action expert
   does not need the forecast's content or amount — only the sequence
   layout it was trained on.

Limits: one checkpoint, three policy seeds; S = 6 uses scale-5/6
embeddings that were not finetuned on Bridge.
