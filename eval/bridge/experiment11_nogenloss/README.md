# Experiment 11 — Is the forecast useful during training? (F1-VLA, finetuning stage)

**Question (advisor's point 5, reduced to what fits the compute budget).**
Experiments 1, 8 and 9 ask whether F1-VLA *uses* its forecast when it acts.
This one asks the other half of the thesis question: does learning to
forecast make the policy better, even if the forecast is not read at
inference?

**Prior work.** The F1 paper's No-Gen variant (Lv et al. 2025, Table 4;
Bridge 27.1% vs 59.4%) is trained without the generation expert, but it
also skips the pretraining stages and changes the input layout, so it is not
a matched comparison. LDA-1B's authors compare policy-only and co-trained
models by held-out action error on Agibot World (Lyu et al. 2026, Sec. V-C,
Fig. 10), not by task success. A matched action-only baseline is the
standard control for auxiliary objectives in representation learning
(e.g. Jaderberg et al. 2017, UNREAL).

## Design

Two finetunes of the same pretrained checkpoint (`f1_vla_stage2`, before
any Bridge finetuning) on BridgeData V2, identical in data, data order
(seed 42), steps (100,000 at batch 16 = 0.85 epoch), optimiser and
architecture. The only difference is the weight of the forecasting loss in

    loss = action_loss + gen_out_loss_ratio * forecast_loss   (f1_policy.py L200)

| model | config | gen_out_loss_ratio |
|---|---|---|
| deployed (all earlier experiments) | `f1_vla/config/bridge_finetune.yaml` | 0.1 |
| this experiment | `f1_vla/config/bridge_finetune_nogenloss.yaml` | 0.0 |

With weight 0 the forecast branch stays in the model — same tokens, same
positions — so there is no layout change (the confound of Experiment 1's
removal). It still receives gradients through the action expert's
attention, so it is free to become "whatever helps the action" instead of a
forecaster. The forecasting loss is still computed and logged, so its drift
can be followed.

**What it covers and what it does not.** Only the finetuning stage: both
models start from a checkpoint that already learned to forecast in
pretraining. A difference would show that forecasting supervision *during
finetuning* matters; no difference would not rule out a benefit from
pretraining.

**Comparisons planned.** (1) Closed-loop success, 288 paired episodes (4
tasks × 24 × seeds 0–2), against the deployed model. (2) Forecast quality
of both models (token accuracy against the real future). (3) The
Experiment 1 interventions and internals (attention on forecast positions)
on the new model. If the difference is small, the deployed run is repeated
as well, since GPU training is not bit-for-bit reproducible.

## Run

`slurm/full_bridge_finetune_nogenloss_selfchain.slurm` (self-chaining 6 h
waves on H100, resumes from the latest checkpoint; output
`outputs/bridge_finetune_nogenloss`). Submitted 2026-11-19, job 776902.

## Results (2026-11-22)

Training finished at 100,000 steps (job 782037). Evaluation: 12 jobs
782523–782536 (`jobs_eval.txt`), `CONDITION=baseline` of
`../experiment1_ablation/closed_loop/closed_loop.slurm` with
`F1_CKPT=outputs/bridge_finetune_nogenloss`, save tag
`exp1_baseline_seed{0,1,2}_nogenloss`, all `EVAL_EXIT=0`; logs confirm the
weight-0 checkpoint was loaded. `python analyze_exp11.py` (writes
`exp11_summary.json`), paired by (task, seed, episode) with the Experiment 1
baseline `exp1_baseline_seed{0,1,2}`.

**Training metrics at the end (mean over steps 90k–100k, training data).**

| | action loss | forecast loss | forecast token accuracy (mean / tail) |
|---|---|---|---|
| deployed (weight 0.1) | 0.122 | 3.69 | 0.311 / 0.227 |
| weight 0 | 0.122 | 8.65 | 0.017 / 0.004 |

**Closed loop (288 paired episodes).**

| run | success | diff vs deployed [95% CI] | McNemar p | lost / gained |
|---|---|---|---|---|
| deployed (weight 0.1) | 140/288 (48.6%) | — | — | — |
| weight 0 | 139/288 (48.3%) | −0.3 pp [−6.6, +5.9] | 1.0 | 44 / 43 |

| task (of 72) | deployed | weight 0 | grasped: deployed / weight 0 |
|---|---|---|---|
| Carrot | 25 | 32 | 44 / 45 |
| Eggplant | 50 | 49 | 69 / 65 |
| Spoon | 36 | 38 | 49 / 51 |
| Stack | 29 | 20 | 60 / 54 |

**What this shows.**
1. Without its loss the forecast branch stops forecasting (token accuracy
   0.31 → 0.02, near chance), yet the policy is as good as the deployed one:
   same action loss on the training data and the same closed-loop success
   (−0.3 pp, CI ±6 pp; 87 discordant episodes split 44 / 43). The per-task
   differences go both ways (Carrot +7, Stack −9 of 72, i.e. about ±3 per
   24-episode run); the deployed model's own runs with different policy
   seeds differ by up to 7 of 24 on one task (Eggplant 14 vs 21, Stack
   7 vs 13).
2. So forecasting supervision **during finetuning** does not make F1-VLA's
   Bridge policy better. Together with Experiments 1 and 8 (the deployed
   policy barely reads the forecast at inference), the forecast branch is
   not what the Bridge policy relies on, either at training or at test
   time.
3. **What it does not show.** Both runs start from `f1_vla_stage2`, which
   learned to forecast in pretraining; a benefit of forecasting during
   pretraining (the representations the action expert inherits) is not
   tested. The deployed run was not repeated, so run-to-run training noise
   is not measured; it would only matter if a difference had appeared.
4. Comparisons (2) and (3) of the plan: forecast quality is given by the
   training metrics above (evaluated rollouts not scored); for (3) see
   below.

### Experiment 1 interventions on the weight-0 model (2026-11-22, partly done)

Does the weight-0 model still read its forecast positions, now that they
hold no forecast? Same conditions and script as Experiment 1
(`../experiment1_ablation/closed_loop/closed_loop.slurm`) with
`F1_CKPT=outputs/bridge_finetune_nogenloss`, `TAG_SUFFIX=_nogenloss`, H100:
`zero` (forecast removed) and `shuffle` (forecast from another decision),
4 tasks × seeds 0–2; `meancollect` (seed 0) then `mean` (per-task mean
forecast KV of this model, kept in `closed_loop/meankv_nogenloss/`, seeds
0–2); `internals` (seed 0, records in
`../experiment1_ablation/internals/records_nogenloss/`, so the deployed
model's records are not overwritten). Job ids: `jobs_interventions.txt`
(submitted by `~/slurm_util/feeder.sh`; the `mean` jobs are queued once the
mean KV exists).

**All three interventions, complete (2026-11-22).** `python analyze_interventions.py`
(writes `interventions_summary.json`); each model's intervention paired
with its own baseline, 288 episodes:

| model | baseline | `zero` | diff [95% CI] | McNemar p | lost / gained |
|---|---|---|---|---|---|
| weight 0 (no forecasting loss) | 139 | 101 | −13.2 pp [−19.1, −7.3] | 4e-5 | 61 / 23 |
| deployed (weight 0.1), Experiment 1 | 140 | 100 | −13.9 pp [−20.5, −7.6] | 7e-5 | 69 / 29 |
| weight 0, `shuffle` | 139 | 138 | −0.3 pp [−4.9, +4.5] | 1 | 24 / 23 |
| deployed, `shuffle` (Experiment 1) | 140 | 144 | +1.4 pp [−3.1, +5.9] | 0.65 | 20 / 24 |
| weight 0, `mean` | 139 | 141 | +0.7 pp [−3.8, +5.6] | 0.89 | 23 / 25 |
| deployed, `mean` (Experiment 1) | 140 | 139 | −0.3 pp [−4.9, +4.2] | 1 | 21 / 20 |

Removing the forecast positions hurts the weight-0 model as much as the
deployed one, although in the weight-0 model these positions hold no
forecast (token accuracy 0.02). So Experiment 1's `zero` drop does not
measure the use of forecast *content*: the action expert depends on these
positions being present and in their usual range, whatever they encode.
`shuffle` and `mean` keep the positions in range and have no effect on
either model.

**Internals (seed 0, 96 episodes, 1,800 decisions; same conditions applied
to identical inputs at each decision).** `../experiment1_ablation/internals/
analyze_internals.py records_nogenloss/` → `summary_nogenloss.md` there
(deployed model: `summary.md`). Shapes and masks are identical in both
models, and the mask test passes (a masked position has exactly zero
effect).

| | deployed | weight 0 |
|---|---|---|
| attention of the action queries on the 29 forecast positions | 0.016 | 0.009 |
| attention on the 680 encoded history positions | 0.285 | 0.196 |
| attention on the language/image (und) prefix | 0.487 | 0.575 |
| action change with `shuffle` (relative L2) | 0.003 | 0.003 |
| action change with `mean` | 0.009 | 0.088 |
| action change with the generation block `removed` | 0.411 | 0.367 |
| forecast-position values changed by `mean` (relative L2) | 0.27 | 0.88 |

Without the forecasting loss the action expert still reads the generation
block, though less (history 0.20 against 0.29 of its attention; forecast
positions 0.009 against 0.016) and more from the und prefix. In the
weight-0 model the per-task mean is far from each decision's own block
(values 0.88 away against 0.27), and it moves the single-decision action
by 9% (against 1%), yet closed-loop success does not change (+0.7 pp).


