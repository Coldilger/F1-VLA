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
