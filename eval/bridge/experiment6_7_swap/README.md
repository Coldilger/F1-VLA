# Experiments 6–7 — Graded and horizon-shifted foresight replacement (F1-VLA)

**Questions (advisor's points 1 and 6).** (6) Replace the forecast with one
from the same task and scene, another task in the same scene, or an
unrelated scene: how wrong must the forecast be before the action changes?
(7) Feed a forecast made for an earlier or later moment: does the *timing*
of the predicted future matter? Method: interchange interventions (Geiger
et al., 2021), extending Experiment 1's shuffle (forecast from another
episode) into a graded scale, and Experiment 2's oracle (the real future
frame) into a scale of horizons. Same design for mimic-video and LDA-1B.

Everything runs **in the closed loop**: the robot executes the actions
decoded with the replaced forecast, 4 tasks × 24 episodes × policy seeds
0–2 (288 episodes, paired with `exp1_baseline_seed{0,1,2}`). The
single-decision effect is recorded in the same runs (below).

## What is replaced

F1-VLA's action expert reads the 29 forecast positions of the generation
block, which are built from the forecast tokens of VAR scales 1–3 (14
tokens; the 16 scale-4 tokens are never fed back, Experiment 8). A
replacement is a list of token ids passed as `oracle_indices` to
`sample_actions_with_world_model` (`f1_vla/src/models/modeling_f1.py`), the
same hook as Experiment 2: the model then builds the 29 positions from them
under the current observation and instruction, exactly as for its own
sampled tokens.

| `SWAP` | donor of the forecast tokens | point |
|---|---|---|
| `control` | the decision's own sampled tokens, fed back through the hook (must reproduce the normal action) | check |
| `own:-2`, `own:-1` | this rollout's own forecast from 2 / 1 decisions earlier (8 / 4 control steps) | 6 (earlier), 1 (same episode) |
| `own:+1`, `own:+2` | the forecast the normal policy would make 1 / 2 decisions later (lookahead) | 6 (later) |
| `bank:same_task` | own forecast at the same decision index of another episode of the task (episode e+1) | 1 |
| `bank:other_task` | same table, another task, same episode and decision index (Carrot → Spoon → Stack → Carrot; none for Eggplant, the only sink task) | 1 |
| `bank:other_scene` | the other scene: table tasks ← Eggplant (sink), Eggplant ← Carrot | 1 |
| `real:-4`, `real:0` | the real camera frame 4 steps ago / now, tokenised by F1's VAE | 6 |
| `real:+1` | the real frame 1 step ahead: the next frame of the evaluation history, which is fed at consecutive steps (0.2 s) although training spaced it 0.6 s apart (Experiment 12) | 6 |
| `real:+3` | the real frame 3 steps ahead: the horizon F1 was trained to forecast (+3 frames at 5 fps), i.e. a closed-loop oracle | 6 |
| `real:+8`, `real:+16` | real frames further ahead (lookahead) | 6 |

Bank donors (`bank_tokens.npz`, `build_bank.py`) are F1's own forecasts on
normal rollouts (Experiment 8's per-decision `tokens_A`, seed 0) and are
time matched (same decision index), so the graded scale is about scene and
task, not about the phase of the episode. In SimplerEnv Carrot, Spoon and
Stack share one table scene (`bridge_table_1_v1`) with different objects;
Eggplant is the sink scene, the only other scene in the set.

**Lookahead.** In the closed loop the future does not exist yet, so it is
simulated: a second environment of the same episode is reset with the same
options and replays every action executed so far, and its full simulator
state is compared with the main environment's (`replay_max_diff`). From
there the normal policy (own forecast; wrapper state and RNG states saved
and restored) runs h control steps; its frame at t+h, or the tokens it
samples at decision t+d, are the donor. Nothing of the lookahead reaches
the main environment.

## Recorded per decision

`records/<tag>/<task>/episode_NNN.npz`: the action decoded with the own
forecast and with the donor from the same action start noise (single-
decision effect `d(donor, own)`), a second own forecast from another
sampling seed (noise floor `d(own2, own)`), donor / own token ids and
their difference on the 14 tokens that reach the action expert, the
relative change of the 29 forecast positions' keys and values (strength of
the intervention), the replay check and timing.

## Files

| file | role |
|---|---|
| `f1_swap_inference.py` | `F1VLASwapInference`, `EnvTap` (environment capture and replay lookahead) |
| `main_inference_swap.py` | entry point (wraps the evaluator's environment builder) |
| `swap_closed_loop.slurm` | one task × seed × `SWAP`; tag `exp67_<swap>_seed<S>` |
| `build_bank.py`, `bank_tokens.npz` | donor bank |
| `jobs_smoke.txt` | checks below |

## Status

**Smoke runs (2026-11-22/23, Carrot, episodes 0–1, seed 0, tag suffix
`_smoke`, jobs in `jobs_smoke.txt`):** all five `EVAL_EXIT=0`.
- `control`: `d(donor, own) = 0.0000`, i.e. the hook reproduces the normal
  action exactly.
- Lookahead (`real:+3`, `own:+1`): `replay_max_diff = 0` — the replayed
  environment matches the main one exactly, so the simulated future is the
  one the robot would meet.
- Time per decision: 0.6–0.7 s without lookahead, 2.4–2.6 s with it.
- Single-decision effect already visible: `d(donor, own)` 0.004–0.006
  against the noise floor `d(own2, own)` 0.004–0.006 (another forecast
  sample).

**Full runs queued 2026-11-23:** 14 conditions (table above, with `real:+1`)
× 4 tasks × seeds 0–2, 24 episodes each; `bank:other_task` has no Eggplant
donor (165 jobs, labels `f1e67_*` in `~/slurm_util/queue_done.txt`, 3 h
limit, alternating H200 / H100). Eggplant episodes have 30 decisions and its
Carrot donor (`bank:other_scene`) only 15; later decisions use the donor's
last one.

**Comparison.** The executed action differs from Experiment 1's baseline
wrapper in RNG use (a second forecast is sampled each decision), so each
condition is paired with `control` of the same task, seed and episode;
`control` vs `exp1_baseline` is reported as a check.
