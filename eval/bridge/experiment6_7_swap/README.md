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

**Full runs (2026-11-23 → 25):** 14 conditions × 4 tasks × seeds 0–2,
24 episodes each, all 165 jobs `EVAL_EXIT=0`; `bank:other_task` has no
Eggplant donor (216 episodes). Eggplant episodes have 30 decisions and its
Carrot donor (`bank:other_scene`) only 15; later decisions use the donor's
last one. Labels `f1e67_*` in `~/slurm_util/queue_done.txt`.

## Results (2026-11-25)

`python analyze_exp67.py` → `exp67_summary.json`.

**Check.** `control` reproduces the normal action exactly at every
decision (`d(donor, own) = 0` over 5,400 decisions). Its closed-loop
success, 124/288, is 5.6 pp below Experiment 1's `exp1_baseline` (140/288;
[−12.2, +0.7], p = 0.11), which differs only in RNG use (a second forecast
is sampled each decision). Within noise, but it is why every condition is
paired with `control`, not with `exp1_baseline`.

### Closed loop, each condition paired with `control` (same task, seed, episode)

| condition | point | n | control | condition | diff [95% CI] | McNemar p (Holm) |
|---|---|---|---|---|---|---|
| own forecast, 2 decisions earlier | 6 | 288 | 124 | 121 | −1.0 pp [−5.2, +3.1] | 0.75 (1) |
| own forecast, 1 decision earlier | 6 | 288 | 124 | 128 | +1.4 pp [−2.8, +5.6] | 0.63 (1) |
| own forecast, 1 decision later ‡ | 6 | 288 | 124 | 125 | +0.3 pp [−4.2, +4.9] | 1 (1) |
| own forecast, 2 decisions later ‡ | 6 | 288 | 124 | 127 | +1.0 pp [−3.5, +5.6] | 0.77 (1) |
| same task, another episode | 1 | 288 | 124 | 131 | +2.4 pp [−2.1, +6.9] | 0.36 (1) |
| same table, another task | 1 | 216 | 73 | 78 | +2.3 pp [−1.9, +6.5] | 0.41 (1) |
| other scene | 1 | 288 | 124 | 125 | +0.3 pp [−4.2, +4.9] | 1 (1) |
| real frame t−4 | 6 | 288 | 124 | 121 | −1.0 pp [−5.6, +3.5] | 0.77 (1) |
| real frame t (now) | 6 | 288 | 124 | 129 | +1.7 pp [−2.1, +5.9] | 0.50 (1) |
| real frame t+1 ‡ | 6 | 288 | 124 | 118 | −2.1 pp [−6.2, +2.1] | 0.42 (1) |
| real frame t+3 (trained horizon, oracle) ‡ | 6 | 288 | 124 | 125 | +0.3 pp [−4.2, +4.9] | 1 (1) |
| real frame t+8 ‡ | 6 | 288 | 124 | 129 | +1.7 pp [−2.8, +6.2] | 0.55 (1) |
| real frame t+16 ‡ | 6 | 288 | 124 | 125 | +0.3 pp [−3.8, +4.5] | 1 (1) |

‡ lookahead condition; see the replay caveat below. Restricted to Carrot
and Stack, where the replay is exact at most decisions (144 episodes,
control 37): own +1 / +2 → 40 / 38, real +1 / +3 / +8 / +16 → 36 / 41 / 40 / 43,
all p ≥ 0.26.

### Single decision (same state, same action noise)

`d` = L2 distance between two decoded 4-step action chunks (normalised
action space; for scale, a chunk's norm is 1.8 and consecutive decisions'
chunks differ by 1.9). The floor is the change caused by sampling the own
forecast again with another seed, which already changes 62% of the 14
action-relevant tokens.

| condition | d(donor, own) | floor | donor − floor [95% CI] | tokens changed | key / value change of the 29 positions |
|---|---|---|---|---|---|
| own −2 / −1 | 0.0038 / 0.0035 | 0.0038 / 0.0037 | ≈0 / −0.0002 | 65% / 62% | 0.03 / 0.08 |
| own +1 / +2 ‡ | 0.0039 / 0.0040 | 0.0039 | ≈0 | 67% / 70% | 0.03–0.04 / 0.09 |
| same task, other episode | 0.0040 | 0.0039 | +0.0001 [−0.0000, +0.0003] | 81% | 0.04 / 0.10 |
| same table, other task | 0.0038 | 0.0038 | +0.0001 [−0.0001, +0.0002] | 85% | 0.04 / 0.10 |
| other scene | 0.0048 | 0.0042 | **+0.0006 [+0.0002, +0.0010]** | 100% | 0.07 / 0.18 |
| real −4 / 0 / +1 / +3 / +8 / +16 | 0.0037–0.0043 | 0.0039–0.0042 | −0.0002 … +0.0001 | 51–67% | 0.03 / 0.07–0.08 |

**What this shows.**
1. **Point 1 (how wrong must the forecast be).** No replacement changes
   F1-VLA's success: 13 conditions, all within ±2.4 pp of `control`, none
   significant even before correction. At a single decision a replaced
   forecast moves the action exactly as much as resampling F1's own
   forecast does, and both are ~0.2% of the action's size. Only a forecast
   from the other scene (every token different, twice the change in the
   action expert's keys and values) exceeds that floor, by 15%, still far
   too little to change an outcome. This extends Experiment 1 (shuffle,
   mean): the action expert reads the forecast positions, but not their
   content.
2. **Point 6 (does the timing matter).** Neither earlier forecasts nor real
   frames from 4 steps ago up to 16 steps ahead, including the trained
   horizon t+3 (an oracle of the true future frame), change success. Giving
   F1-VLA the real future does not help, as in Experiment 2's open-loop
   oracle.

**Caveat: lookahead replay is not exact everywhere.** `replay_max_diff`
(max |state difference| between the replay env and the main env) is 0 at
every first decision of an episode but nonzero at 59–64% of later
lookahead decisions: almost all Eggplant decisions and ~90% of Spoon ones,
10–30% of late Carrot decisions, few Stack ones. The smoke runs (Carrot,
episodes 0–1) did not show this. The likely cause is reusing one replay
env across resets (decision 0 uses a freshly built env and is always
exact); `diag_replay.py` / `diag_replay.slurm` test it (reused vs freshly
built replay env, no model). **Rerun (queued 2026-11-25):** `EnvTap.replay` now builds a
fresh replay env for every lookahead; the 6 lookahead conditions are rerun
(72 jobs, tag suffix `_fresh`, labels `f1e67f_*`), after a smoke run
(Eggplant, `real:+3`, episodes 0–1, suffix `_freshsmoke_*`) must show
`replay_max_diff = 0` at every decision (`~/slurm_util/gate_exp67fresh.sh`).
`analyze_exp67.py` uses the `_fresh` runs once they exist; the numbers above
are from the first runs. For the ‡ conditions the donor is then the
future of a nearby, not the identical, state. This does not affect the
non-lookahead conditions, and on Carrot + Stack, where the replay is
mostly exact, the lookahead conditions show no effect either.
