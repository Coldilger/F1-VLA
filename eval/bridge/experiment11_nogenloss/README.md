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

## Results

Training in progress.
