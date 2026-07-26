# F1-VLA on SimplerEnv Bridge — finetuning + eval log

Record of what was run, what was broken, and how each fix moved the number.
Everything below is measured, not estimated.

## Training (stage 3 finetune on BridgeData V2)

- Dataset: `IPEC-COMMUNITY/bridge_orig_lerobot` (53,192 episodes / 1,893,026 frames, 21 GB).
- Start point: the released stage-2 checkpoint `InternRobotics/F1-VLA` (stages 1-2
  were done by the authors; we only run stage 3).
- Config: `f1_vla/config/bridge_finetune.yaml`, 100k steps, bf16, 1x H200.
- Job 585744: `COMPLETED`, wall time **1d 5h 29m**, final `train_loss` **0.5108**
  (from ~0.64 at the start). Checkpoints every 10k steps.
- Final weights backed up outside scratch: `/home/3295540/f1_vla_models/bridge_finetune/`.

## Eval harness

Closed-loop rollout in SimplerEnv/ManiSkill2 (`eval/bridge/`), modeled on
mimic-video-project's `VAMInference`. See `SETUP.md` for the environment build.

## Bugs found and their measured effect

All measured on `PutCarrotOnPlateInScene-v0`, 24 episodes, 0 skipped episodes.
Paper (Table 3) reports **70.8%** for F1 on this task.

| # | Fix | Success | consecutive_grasp | src_on_target |
|---|---|---|---|---|
| 0 | original wrapper | **0.0%** (0/24) | 0 | 0 |
| 1 | gripper `[0,1]` -> binarized `[-1,+1]`, rotation euler -> axis-angle | **12.5%** (3/24) | 70 | 24 |
| 2 | + image 480x640 (4:3) -> 256x256 square | **29.2%** (7/24) | 154 | 195 |
| 3 | + world-model history stride 0.2s -> 0.6s (**REVERTED**, made it worse) | **4.2%** (1/24) | 80 | 36 |
| 4 | + state rotation expressed relative to episode-reset pose | **37.5%** (9/24) | 197 | 225 |

Details:

1. **Gripper + rotation.** bridge_orig gripper values are exactly `{0.0, 1.0}`
   (0=close, 1=open), but the widowx controller takes `[-1,+1]`. Passing raw
   `[0,1]` put "close" at 0 = mid joint range = half-open, so the gripper could
   never clamp anything — matching the symptom exactly (`is_src_obj_grasped`
   flickered True, `consecutive_grasp` never True). Rotation: bridge stores euler,
   the controller wants axis-angle. Both fixed to mirror SimplerEnv's own
   reference (`octo_model.py`, widowx_bridge branch).
2. **Image aspect.** Training frames are 256x256 square; SimplerEnv renders
   480x640. The raw 4:3 frame got letterboxed with black bars by F1's internal
   `resize_with_pad`, and the history `CenterCrop(256)` threw away ~1/3 of the
   horizontal FOV. Neither was ever seen in training.
3. **History stride (negative result, kept for the record).** Training sampled the
   world-model history every 0.6s spanning 1.8s (`obs_img_stride=3` @ 5 fps), so
   matching that at eval *looks* faithful and the arithmetic lines up exactly.
   It measured **worse** (4.2% vs 29.2%). Likely because with stride 3 the history
   is still mostly repeat-padding until ~step 12 of a 60-step episode, degrading
   the early actions that set up the grasp. Dense (every-step) history kept;
   `history_stride_steps` left as a knob.
4. **State rotation frame.** Found with `diag_state.py`, which diffs the proprio we
   feed the policy against the training distribution. Position matched (x/y/z all
   inside training ranges) but orientation was far outside it:

   | dim | training range | our eval value |
   |---|---|---|
   | roll | -0.357 .. 0.466 | -3.06 (~ -pi) |
   | pitch | -0.577 .. 0.250 | 1.51 (~ pi/2) |
   | yaw | -0.023 .. 0.981 | -3.08 (~ -pi) |

   bridge_orig stores orientation as small angles centred on zero, i.e. relative
   to the gripper-down home pose; SimplerEnv's `ee_pose_at_base` is absolute and
   at rest is a ~93 deg rotation sitting right at the pitch=pi/2 gimbal
   singularity. All 12 euler orderings were checked — none maps the sim rest pose
   into the training range, so the mismatch is the reference frame, not the
   ordering. Now reported relative to the pose captured at episode reset
   (per-episode, so it also covers the eggplant task's different robot).

**Resolved analytically (no GPU):** the 4-vs-5 history-frame question. The gen
expert's `TemporalDownsampling` is `Conv1d(kernel=4, stride=4, padding=0)` whose
output is reshaped to a hardcoded single timestep, so T=4 is exactly right. T=5
would silently drop a frame — and since the conv consumes the first 4, it would
discard the *most recent* observation.

## Bugs fixed in F1-VLA itself

The upstream repo only ever runs the model through `train_hf.py`, so the pure
inference path (`from_pretrained` + `select_action_with_world_model`) had never
been exercised. Four distinct bugs, all inference-only (none affects training):

1. `f1_policy.py`: `F1Config.from_pretrained` called with the wrong kwarg name.
2. `configuration_f1.py`: a loaded `vae` config (with the real `vae_ckpt` path)
   was correctly parsed and then unconditionally overwritten with a default
   containing `vae_ckpt: None`.
3. `modeling_f1.py`: `set_requires_grad` re-read `training_args.x` instead of the
   `self.x` it had just computed, crashing when `training_args is None`.
4. `modeling_f1.py`: `denoise_step` passed `is_eval=True` to a `forward()` that
   has no such parameter.

## Method note

Success rate over 24 episodes was the only trustworthy signal here. Watching
rollout videos cannot distinguish "model is mediocre" from "format is subtly
wrong", and single-digit episode counts are noise. It also caught a fix that was
theoretically well-motivated but empirically harmful (#3), and caught a run whose
0.0% was actually a stale cached result — the evaluator silently skips episodes
whose output video already exists, so every experiment needs its own
`--additional-env-save-tags`, and every run's `already done` count must be 0.

## Final eval: 3 seeds x 24 episodes per task

The world-model head samples (top_k/top_p), so rollouts are stochastic. A single
24-episode run swings a lot — the same config scored 29.2% and 45.8% on the carrot
task on different seeds — so every number below is the mean of 3 seeds.

All 12 runs valid (0 skipped episodes, 0 tracebacks).

| task | seeds (0/1/2) | mean | sd | paper | delta |
|---|---|---|---|---|---|
| Put Carrot on Plate | 29.2 / 45.8 / 41.7 | **38.9%** | 8.6 | 70.8% | **-31.9** |
| Put Spoon on Towel | 50.0 / 45.8 / 45.8 | **47.2%** | 2.4 | 50.0% | -2.8 |
| Stack Green Cube | 33.3 / 45.8 / 33.3 | **37.5%** | 7.2 | 50.0% | -12.5 |
| Put Eggplant in Basket | 70.8 / 70.8 / 66.7 | **69.4%** | 2.4 | 66.7% | **+2.7** |
| **average** | | **48.2%** | | **59.4%** | **-11.1** |

Paper column = the Success values of the "F1 (Ours), Pretrained ✔" row of Table 3.
Note the paper's headline 72.9% is its "Overall Average" column, which averages
Grasp *and* Success; the comparable success-only average is 59.4%.

The gap is not uniform, which is the informative part: spoon and eggplant are at
parity (within seed noise), while carrot is 32 points down. Seed variance splits
the same way — sd ~2.4 on the two tasks we match, sd ~7-9 on the two we don't.

## Grasp vs Success breakdown (seed 0)

| task | grasp | paper grasp | success | paper success |
|---|---|---|---|---|
| Carrot | 58% | 87.5% | 29% | 70.8% |
| Spoon | 62% | 70.8% | 50% | 50.0% |
| Stack | 70% | 87.5% | 33% | 50.0% |
| Eggplant | 91% | 100% | 70% | 66.7% |

We lose far more between grasping and placing than the paper does: carrot
58->29 (half of all grasps dropped) vs the paper's 87.5->70.8, and stack 70->33
vs 87.5->50. Transport/placement, not reaching, is where our runs fail.

## Root cause hypothesis: action chunk size

| source | chunk_size |
|---|---|
| authors' released stage-2 checkpoint (`InternRobotics/F1-VLA`) | **30** |
| repo default `f1_vla/config/f1_config.json` | 50 |
| **our finetune** (inherited from `debug_test.yaml`) | **4** |

We finetuned a model that plans only 4 action steps (0.8s at 5 Hz) ahead, where
the authors trained with 30. This was copied unnoticed from the repo's *debug*
config when only the dataset section was swapped for bridge.

Three independent observations line up with this being the remaining gap:
1. The loss is concentrated in transport/placement (see table above), which is
   the phase needing sustained coherent motion.
2. The two tasks we match are the ones needing least sustained precision; the two
   we lose are precise placements (carrot onto plate, cube onto cube).
3. Eggplant — the one task where we beat the paper — is also the only one with a
   120-step budget instead of 60, i.e. twice the time to recover from a myopic
   plan.

Next step: refinetune with `chunk_size: 30` to match the checkpoint we started
from.

## Table 8 of the paper: what our recipe actually mismatches

The paper's appendix Table 8 gives the full training recipe per stage/task. Two
corrections to earlier assumptions in this file:

- **Action chunk size for Simpler is 8**, not 30. The 30 came from the released
  checkpoint's `config.json`, which describes **Stage II** (pretrain). Table 8's
  Stage III column for Simpler says 8. So the chunk30 refinetune matches neither.
- **Our 4 was LIBERO's value**, not an arbitrary debug number: Table 8 lists
  LIBERO = 4, and `debug_test.yaml` is a LIBERO config. We inherited the right
  value for the wrong task.

Full comparison for Stage III / Simpler:

| hyperparameter | paper (Simpler) | ours | |
|---|---|---|---|
| Learning Rate | 5e-5 | 5e-5 | match |
| LR Scheduler | Cosine | Cosine | match |
| Loss Weight (Gen:Act) | 0.1:1 | 0.1:1 | match |
| Und / Gen Resolution | 224 / 256 | 224 / 256 | match |
| Num Predicted Scales | 4 | 4 | match |
| Denoise Steps | 10 | 10 | match |
| Action Chunk Size | **8** | 4 | mismatch |
| Batch Size | **128** | 16 | mismatch |
| Training duration | **10 epochs** | **0.85 epoch** | mismatch |

On the duration row: Table 8 specifies Stage III downstream tasks in *epochs* and
Stage I/II/LIBERO in *steps* — they are alternative parameterisations, not a
missing value (epochs are the sensible unit when per-task dataset sizes differ,
as the table's caption says). Our config set both `num_train_epochs: 50` and
`max_steps: 100_000`; HF Trainer's `max_steps` silently overrides the epoch
count, so we ran 100k steps at batch 16 = 1.6M samples = **0.85 epoch**, while
the paper sees 18.9M samples (11.8x more). Specify epochs and drop `max_steps`
to avoid this.

## Does more training close the gap? No.

Success rate vs training steps, `PutCarrotOnPlateInScene-v0`, 3 seeds x 24
episodes per checkpoint (15 runs, 0 skipped). Uses the intermediate checkpoints
of the chunk4 run, so this cost eval time only, no training.

| checkpoint | seeds (0/1/2) | mean |
|---|---|---|
| 20,000 | 41.7 / 37.5 / 29.2 | **36.1%** |
| 40,000 | 29.2 / 25.0 / 25.0 | 26.4% |
| 60,000 | 37.5 / 25.0 / 37.5 | 33.3% |
| 80,000 | 29.2 / 25.0 / 37.5 | 30.6% |
| 100,000 | 37.5 / 41.7 / 29.2 | **36.1%** |

Flat: 36 -> 26 -> 33 -> 31 -> 36, no trend, and 20k equals 100k exactly. The
spread across seeds within one checkpoint (up to 12.5pp) is as large as the
spread across checkpoints. So 5x more training bought nothing measurable, and
"we only trained 0.85 epoch" does not explain the gap to the paper — the cause is
more likely structural (chunk size, batch size as an optimisation regime,
residual eval mismatch, or different Stage III data).

Practical consequence: ~20k steps is enough for this configuration, i.e. ~7h per
run instead of ~33h. Caveat: measured on one task, in our regime (batch 16,
chunk 4); it does not test batch 128, which changes the optimisation dynamics.

## Eval was not reproducible under `--f1-seed` (fixed)

Evaluating the *same weights* with the *same seed* twice gave 38.9% and 36.1%
(verified same weights: `model.safetensors` and `checkpoint-100000/model.safetensors`
have identical key sets, identical key order, and 9/9 sampled tensors bitwise
equal; only file metadata differs, hence differing md5).

Cause: `--f1-seed` created a `torch.Generator` that only reaches the world-model
token sampler. The flow-matching start noise comes from
`F1FlowMatching.sample_noise`, which calls `torch.normal()` with no generator,
i.e. the **global** torch RNG, which was never seeded. Fixed by also calling
`torch.manual_seed` / `torch.cuda.manual_seed_all` in the wrapper.

This matters for any model comparison: with ~±6pp run-to-run noise on 24
episodes, two models differing by less than ~10pp cannot be distinguished
without more episodes or more repeats.

## Seeded chunk4 baseline (re-measured after the RNG fix)

All earlier numbers were measured with the flow-matching noise unseeded, so they
were not reproducible. Re-measured the same final chunk4 weights with the fix, to
have an apples-to-apples baseline for later comparisons:

| task | seeds (0/1/2) | mean | paper | delta |
|---|---|---|---|---|
| Put Carrot on Plate | 41.7 / 25.0 / 37.5 | 34.7% | 70.8% | -36.1 |
| Put Spoon on Towel | 62.5 / 41.7 / 45.8 | **50.0%** | 50.0% | +0.0 |
| Stack Green Cube | 37.5 / 29.2 / 54.2 | 40.3% | 50.0% | -9.7 |
| Put Eggplant in Basket | 62.5 / 58.3 / 87.5 | **69.4%** | 66.7% | +2.7 |
| **average** | | **48.6%** | **59.4%** | **-10.8** |

Reassuringly close to the unseeded measurement (48.6% vs 48.2%), so the earlier
conclusions were noisy but not biased.

Note what seeding does and does not fix: each individual seed is now
reproducible, but the spread *across* seeds is still large (stack 29.2-54.2,
eggplant 58.3-87.5 — up to 29pp). That spread is real sensitivity to the sampled
noise, not measurement error. For a Bernoulli success rate around 0.4, a
24-episode estimate has sd ~10pp, so a 3-seed mean still has a standard error of
~5-6pp. **Two models differing by less than ~10pp cannot be separated at this
sample size.** Since eval is cheap (~5 min per 24 episodes) and training plateaus
early, the right budget split for a comparative study is more evaluation
episodes, not more training steps.

## Checkpoint interval was wasting GPU time

`save_steps: 10_000` while a 6h wave completes roughly the same number of steps
meant the save period coincided with the wave length, so every interrupted wave
threw away everything since its last checkpoint. Observed directly: wave 604636
reached step 88,173, the last checkpoint was 80,000, and the next wave resumed
from 80,000 — **8,173 steps (~3 GPU-hours) discarded**, and this had been
happening on every wave.

Fixed to `save_steps: 2_000` with `save_total_limit: 3`: worst-case loss per
interruption drops to ~2,000 steps (~40 min), and disk drops from 176 GB to
~66 GB. Any future multi-wave run should keep the checkpoint interval well below
what one wave completes.


## chunk_size=30: negative result (2026-07-26)

Trained a second model identical to the chunk4 baseline except `chunk_size: 30`
(100k steps, batch 16, self-chaining across 6h waves). Evaluated on the full
4-task grid, 24 episodes, 3 seeds, same seeding fix as the chunk4 baseline.

| task     | s0   | s1   | s2   | chunk30 | chunk4 baseline |
|----------|------|------|------|---------|-----------------|
| carrot   |  4.2 |  8.3 | 12.5 |   8.3%  | 34.7%           |
| spoon    | 16.7 |  4.2 | 12.5 |  11.1%  | 50.0%           |
| stack    |  0.0 |  0.0 |  0.0 |   0.0%  | 40.3%           |
| eggplant | 29.2 | 25.0 | 12.5 |  22.2%  | 69.4%           |
| **mean** |      |      |      | **10.4%** | **48.6%**     |

Worse on every task, 4.7x worse on average, and identically zero on stack across
all three seeds — well outside the cross-seed spread, so this is not noise.

Two independent causes, both measured rather than assumed:

1. **The horizon does not fit the data.** Bridge episodes average 35.6 frames
   (7.1s at 5fps), median 35. At chunk_size=30, 32.4% of episodes are *entirely*
   shorter than the prediction horizon, and for a uniformly sampled start inside
   a median-length episode roughly 40% of the chunk positions fall past the end
   and are masked out of the loss. At chunk_size=8 (the paper's Simpler value)
   that drops to ~10%, at 4 to ~4%. Note this also makes `train_loss` across
   chunk sizes incomparable: the masked positions contribute zeros to the mean,
   which is why chunk30 reports 0.045 vs chunk4's 0.511 without being better.
   (The comparison is doubly invalid — chunk4's figure averages all 100k steps
   including the high-loss start, chunk30's only the last 10k after a resume.)

2. **Executing the whole chunk open-loop.** The eval wrapper played every
   predicted action before re-observing, i.e. 2 replans per 60-step episode at
   chunk30 vs 15 at chunk4. Predicting 30 but executing only 8 (receding
   horizon, `--f1-execute-steps 8`) lifted carrot from 8.3% to 22.2%.

Conclusion: chunk_size=30 is wrong for Bridge; the paper's 8 is a deliberate fit
to episode length. **But the bigger finding is (2): replanning frequency moved
the score by 13.9pp on one task, whereas 20k vs 100k training steps moved it by
0.0pp.** Control frequency, not training length, is where the headroom is.

### Replanning frequency is NOT a general lever (2026-07-26)

The chunk30 finding above (executing 8 of 30 beat executing 30 of 30 by 13.9pp)
suggested replanning more often might be free headroom. Tested directly on the
chunk4 baseline, carrot, 3 seeds:

| executed steps per chunk | s0   | s1   | s2   | mean  |
|--------------------------|------|------|------|-------|
| 1 of 4                   | 29.2 | 16.7 | 25.0 | 23.6% |
| 2 of 4                   | 37.5 | 37.5 | 29.2 | 34.7% |
| 4 of 4 (baseline)        | 41.7 | 25.0 | 37.5 | 34.7% |

Replanning *more* often than the trained chunk does not help, and replanning
every step actively hurts (-11.1pp). So the earlier reading was too broad: the
chunk30 gain was not "replanning helps", it was partial repair of a horizon that
did not fit the data. The correct rule is **execute roughly the chunk the model
was trained to emit**.

Mechanism, consistent with the model: actions are generated from fresh Gaussian
noise via flow matching, and the policy is deliberately multimodal (going left
around an object and going right are both valid). Playing a chunk out keeps one
sampled intent; re-drawing noise every step re-rolls the intent every step, so
the arm dithers between modes and completes none.

Side benefit: the baseline reproduced at exactly 34.7% for a third independent
time, reconfirming the harness is deterministic under --f1-seed.

### Where the gap to the paper's 72.9% stands

Ruled out by measurement: training length (20k == 100k, 0.0pp), longer action
chunk (30 is -38.2pp), replanning frequency (not a lever). Still untested and
now the obvious candidate: **chunk_size=8**, the value Table 8 actually lists for
Simpler. Our 4 came from the LIBERO config, not the Simpler one, and chunk size
is empirically the highest-leverage knob we have found (4 vs 30 = 38pp). Launched
as `slurm/full_bridge_finetune_chunk8_selfchain.slurm`, identical to the chunk4
baseline in every other respect (batch 16, lr 5e-5, cosine over 100k, 0.1:1 loss).

## CORRECTION: the learning curve is NOT flat (2026-07-26)

Earlier this file recorded "20k steps == 100k steps (36.1% both), so undertraining
does not explain the gap". **That was measured on carrot only and is wrong.**
Carrot is the one task of four where the curve happens to be flat. Full control,
chunk4 checkpoint-20000 vs the final 100k model, 24 episodes, 3 seeds:

| task     | @20k  | @100k | delta  |
|----------|-------|-------|--------|
| carrot   | 36.1% | 34.7% |  +1.4  |
| spoon    | 34.7% | 50.0% | -15.3  |
| stack    | 15.3% | 40.3% | -25.0  |
| eggplant | 18.1% | 69.4% | -51.3  |
| **mean** | **26.0%** | **48.6%** | **-22.6** |

The score nearly doubles from 20k to 100k and shows no sign of saturating.

Consequences:

1. **Budget guidance reversed.** The earlier advice ("~20k steps suffice, ~7h per
   run") was wrong; a run needs the full 100k, ~29h on one H100. Plan comparative
   studies from that number.
2. **chunk8 vs chunk4 must be compared at matched steps.** At 20k: chunk8 14.6%
   vs chunk4 26.0%, i.e. -11.5pp, not the -34.0pp implied by comparing against
   chunk4@100k. Too early to judge chunk8; its 100k run is in flight.
3. **Undertraining is back as the leading explanation for the gap to 72.9%.**
   Paper: 10 epochs at batch 128 (~148k steps, ~18.9M samples seen).
   Ours: 100k steps at batch 16 = 1.6M samples = **0.845 epochs**, i.e. 11.8x
   less data through the model, while still on the steep part of the curve.

Also refuted this session (both were mine, both measured, both dead):

- *"Replanning frequency is a lever."* Executing 1 or 2 of 4 does not beat 4 of 4.
- *"The harness penalises long chunks by executing the whole chunk."* chunk8 with
  --f1-execute-steps 4 scores 15.6% vs 14.6% at 8, and 18.1% at the reference
  value 5 (carrot). Execution horizon is not the explanation; the ordering of
  chunk4 > chunk8 > chunk30 by replan count was a coincidence.
- *"train_loss tracks policy quality."* chunk4 and chunk8 have near-identical loss
  curves (0.535 vs 0.515 at 20k, same slope) and differ 2x in success rate. Loss
  is useless for model selection here; only rollouts count.
