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
