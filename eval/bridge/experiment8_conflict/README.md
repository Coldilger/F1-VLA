# Experiment 8 — Instruction–foresight conflict (F1-VLA)

**Question.** The instruction reaches F1-VLA's action expert by two routes:
directly (the language tokens sit in the understanding block of the shared
KV cache) and through the forecast (the generation block attends to the
language tokens while it is produced). When the two routes disagree, which
one does the action follow?

**Prior work this extends.** Cue-conflict stimuli (Geirhos et al., 2019:
give a network two cues that point to different answers and see which one
it follows) and interchange interventions (Geiger et al., 2021: run the
model on two inputs and swap one internal component between the runs).
Here the two cues are the instruction and the forecast generated under
another instruction. Advisor's list, point 7.

## Design

Two instructions, A (the scene's own) and B, on the same observation, the
same forecast sampling seed and the same action start noise:

| | generation block from the run under A | generation block from the run under B |
|---|---|---|
| understanding block from the run under A | AA (normal) | AB (conflict) |
| understanding block from the run under B | BA (conflict) | BB (normal) |

plus AA2 (generation block from a second sampling seed under A: the noise
floor) and ABf (only the 29 forecast positions taken from B; the 680
history positions stay from A).

`follow = d(AB, AA) / (d(AB, AA) + d(AB, BB))`: 0 = the action stays with
the instruction in the understanding block, 1 = it goes with the forecast.
`d` is the mean over the chunk's steps of the L2 distance between two
action chunks, in normalised action units.

Instructions B per task (`INSTRUCTIONS` in `f1_conflict_inference.py`):

- **mirror** — the scene's instruction with the two objects swapped (e.g.
  "stack the yellow block on the green block"): both objects are present,
  and the token count is identical, so every position id is unchanged;
- the instructions of the **other three tasks** (objects absent from the
  scene; token count differs, which shifts the generation block's position
  ids — a layout confound, marked "same token count: no").

## Files

| file | role |
|---|---|
| `f1_conflict_inference.py` | `F1VLAConflictInference` (steps 0–1: baseline rollout that re-decodes every chunk under all cells) and `F1VLAConflictClosedLoop` (step 2: the robot is driven by one cell) |
| `main_inference_conflict.py` | entry point (`CONFLICT_MODE=probe` or `cl`) |
| `conflict_probe.slurm`, `jobs.txt` | steps 0–1, 4 tasks × 24 episodes, seed 0 |
| `conflict_closed_loop.slurm`, `jobs_cl.txt` | step 2, 4 cells × 4 tasks × 24 episodes, mirror instruction |
| `analyze_conflict.py`, `conflict_summary.json` | tables below |
| `records/<task>/episode_NNN.npz` | raw per-decision records |

## Steps 0–1: results (2026-11-16)

1785 replanning decisions on states visited by a normal rollout under A
(Carrot, Spoon: 15 per episode; Eggplant: 30). Stack has 23 of 24 episodes
(the last episode's record was not written; cause not checked). Numbers are
means over episodes with a 95% bootstrap interval over episodes.

### Layout check and noise floor (same instruction, two forecast sampling seeds)

| task | episodes | decisions | prefix positions | forecast block starts at | earliest position that differs | positions that differ (mean) | decisions with none | tok(seed) | value change und / history / forecast | d(AA2, AA) |
|---|---|---|---|---|---|---|---|---|---|---|
| Carrot | 24 | 360 | 1013 | 984 | 984 | 24.7 | 1 | 0.74 [0.71, 0.77] | 0.000 / 0.000 / 0.075 | 0.0048 [0.0038, 0.0063] |
| Eggplant | 24 | 720 | 1013 | 984 | 984 | 24.2 | 2 | 0.67 [0.62, 0.72] | 0.000 / 0.000 / 0.074 | 0.0046 [0.0036, 0.0061] |
| Spoon | 24 | 360 | 1013 | 984 | 984 | 24.9 | 0 | 0.74 [0.71, 0.76] | 0.000 / 0.000 / 0.072 | 0.0034 [0.0031, 0.0037] |
| Stack | 23 | 345 | 1013 | 984 | 984 | 24.1 | 10 | 0.71 [0.68, 0.74] | 0.000 / 0.000 / 0.069 | 0.0029 [0.0027, 0.0031] |

`tok(seed)`: share of the 30 sampled forecast tokens that differ between
the two seeds. The prefix is 304 understanding positions + 680 history
positions + 29 forecast positions. In all 1785 decisions the two seeds
never differ before position 984, which confirms the layout from the data:
**the generation block is 680 history positions followed by 29 forecast
positions** (the 30 + 679 split used in Experiment 1's internal diagnostics
was wrong). About 25 of the 29 forecast positions differ; in 13 decisions
none does, because the tokens of scales 1–3 coincided and the tokens of
scale 4 are never fed back.

### Instruction A vs. instruction B (same seed)

`tok(B)`: share of the 30 forecast tokens that differ between A and B.
`kv forecast` / `kv history`: relative change of the generation block's
values at the forecast / history positions.

| task | B | same token count | tok(B) | kv forecast | kv history | d(BB, AA) | d(AB, AA) | d(ABf, AA) | d(BA, BB) | follow(AB) | follow(BA) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Carrot | mirror: "put plate on carrot" | yes | 0.19 [0.17, 0.21] | 0.065 | 0.045 | 1.971 [1.761, 2.178] | 0.0053 [0.0046, 0.0063] | 0.0064 [0.0039, 0.0099] | 0.0053 [0.0049, 0.0058] | 0.012 [0.006, 0.020] | 0.013 [0.006, 0.021] |
| Carrot | spoon: "put the spoon on the towel" | no | 0.11 [0.10, 0.13] | 0.052 | 0.042 | 1.105 [0.990, 1.214] | 0.0094 [0.0066, 0.0128] | 0.0047 [0.0039, 0.0057] | 0.0064 [0.0055, 0.0075] | 0.022 [0.014, 0.031] | 0.019 [0.011, 0.028] |
| Carrot | stack: "stack the green block on the yellow block" | no | 0.17 [0.15, 0.20] | 0.068 | 0.055 | 1.578 [1.459, 1.695] | 0.0102 [0.0077, 0.0132] | 0.0049 [0.0040, 0.0061] | 0.0081 [0.0075, 0.0088] | 0.012 [0.008, 0.016] | 0.011 [0.007, 0.016] |
| Carrot | eggplant: "put eggplant into yellow basket" | no | 0.13 [0.12, 0.14] | 0.055 | 0.043 | 1.287 [1.146, 1.430] | 0.0079 [0.0057, 0.0107] | 0.0062 [0.0039, 0.0092] | 0.0067 [0.0057, 0.0077] | 0.020 [0.011, 0.030] | 0.021 [0.011, 0.032] |
| Eggplant | mirror: "put yellow basket into eggplant" | yes | 0.17 [0.15, 0.19] | 0.064 | 0.036 | 1.370 [1.265, 1.477] | 0.0052 [0.0045, 0.0060] | 0.0046 [0.0036, 0.0060] | 0.0071 [0.0058, 0.0087] | 0.009 [0.006, 0.012] | 0.010 [0.007, 0.014] |
| Eggplant | carrot: "put carrot on plate" | no | 0.16 [0.15, 0.18] | 0.067 | 0.042 | 1.310 [1.166, 1.446] | 0.0053 [0.0046, 0.0059] | 0.0042 [0.0037, 0.0047] | 0.0058 [0.0054, 0.0064] | 0.010 [0.006, 0.015] | 0.011 [0.007, 0.016] |
| Eggplant | spoon: "put the spoon on the towel" | no | 0.17 [0.16, 0.18] | 0.068 | 0.041 | 1.297 [1.184, 1.399] | 0.0057 [0.0050, 0.0064] | 0.0043 [0.0037, 0.0052] | 0.0066 [0.0058, 0.0077] | 0.010 [0.007, 0.015] | 0.011 [0.007, 0.015] |
| Eggplant | stack: "stack the green block on the yellow block" | no | 0.17 [0.14, 0.19] | 0.070 | 0.045 | 1.337 [1.250, 1.424] | 0.0087 [0.0068, 0.0110] | 0.0047 [0.0039, 0.0058] | 0.0074 [0.0069, 0.0080] | 0.009 [0.007, 0.011] | 0.008 [0.007, 0.009] |
| Spoon | mirror: "put the towel on the spoon" | yes | 0.22 [0.20, 0.24] | 0.073 | 0.051 | 2.003 [1.805, 2.204] | 0.0043 [0.0039, 0.0047] | 0.0034 [0.0030, 0.0039] | 0.0050 [0.0046, 0.0054] | 0.009 [0.004, 0.014] | 0.009 [0.005, 0.015] |
| Spoon | carrot: "put carrot on plate" | no | 0.12 [0.11, 0.12] | 0.054 | 0.043 | 1.047 [0.908, 1.186] | 0.0061 [0.0051, 0.0072] | 0.0037 [0.0031, 0.0046] | 0.0083 [0.0052, 0.0126] | 0.021 [0.013, 0.029] | 0.021 [0.014, 0.030] |
| Spoon | stack: "stack the green block on the yellow block" | no | 0.18 [0.15, 0.21] | 0.068 | 0.052 | 1.361 [1.253, 1.477] | 0.0054 [0.0046, 0.0062] | 0.0035 [0.0031, 0.0040] | 0.0076 [0.0061, 0.0095] | 0.008 [0.006, 0.011] | 0.009 [0.007, 0.012] |
| Spoon | eggplant: "put eggplant into yellow basket" | no | 0.16 [0.15, 0.18] | 0.064 | 0.048 | 1.307 [1.165, 1.450] | 0.0055 [0.0043, 0.0069] | 0.0033 [0.0030, 0.0037] | 0.0075 [0.0053, 0.0113] | 0.012 [0.007, 0.019] | 0.012 [0.008, 0.018] |
| Stack | mirror: "stack the yellow block on the green block" | yes | 0.19 [0.18, 0.21] | 0.063 | 0.042 | 1.589 [1.381, 1.805] | 0.0035 [0.0033, 0.0037] | 0.0029 [0.0028, 0.0031] | 0.0041 [0.0038, 0.0045] | 0.010 [0.007, 0.015] | 0.011 [0.008, 0.015] |
| Stack | carrot: "put carrot on plate" | no | 0.19 [0.17, 0.20] | 0.074 | 0.056 | 1.617 [1.474, 1.767] | 0.0081 [0.0070, 0.0095] | 0.0031 [0.0030, 0.0033] | 0.0100 [0.0091, 0.0110] | 0.010 [0.007, 0.013] | 0.011 [0.008, 0.014] |
| Stack | spoon: "put the spoon on the towel" | no | 0.15 [0.14, 0.17] | 0.063 | 0.045 | 1.074 [0.957, 1.192] | 0.0054 [0.0048, 0.0059] | 0.0030 [0.0028, 0.0032] | 0.0062 [0.0056, 0.0068] | 0.014 [0.008, 0.020] | 0.015 [0.009, 0.022] |
| Stack | eggplant: "put eggplant into yellow basket" | no | 0.15 [0.14, 0.17] | 0.064 | 0.046 | 0.984 [0.879, 1.087] | 0.0073 [0.0062, 0.0085] | 0.0031 [0.0029, 0.0033] | 0.0088 [0.0067, 0.0118] | 0.015 [0.010, 0.021] | 0.016 [0.011, 0.022] |

### What this shows

1. **The instruction changes the forecast less than resampling it does.**
   With the seed fixed, a different instruction changes 11–22% of the
   forecast tokens; a different seed under the same instruction changes
   67–74%. At the forecast positions the instruction moves the values by
   5–7%, the same size as a change of seed (7%).
2. **At a single decision the action follows the instruction in the
   understanding block.** Changing the instruction everywhere moves the
   action by 1.0–2.0; putting the generation block of the other
   instruction under an unchanged understanding block moves it by
   0.004–0.010. `follow` is 0.008–0.022 in all 16 pairs (0.009–0.012 for
   the four mirror pairs, which have no layout confound), in both
   directions (AB and BA).
3. **Swapping only the forecast positions is at the noise floor.**
   d(ABf, AA) = 0.003–0.006 against d(AA2, AA) = 0.003–0.005: a forecast
   generated under another instruction moves the action as much as a
   re-drawn forecast under the same instruction.
4. **Which forecast is sampled hardly matters for a single action.** Two
   seeds disagree on about 70% of the forecast tokens and the action moves
   by 0.003–0.005 — a few thousandths of what the instruction does. This
   agrees with Experiment 1 (task-average and wrong-episode forecasts leave
   closed-loop success unchanged).

**Limits.** These are single decisions on states visited under instruction
A; small differences could add up over an episode (step 2 tests this). The
mirror instructions are probably rare or absent in the Bridge training
data, so "BB" is the policy under an unusual instruction, not a validated
"go for the other object" policy — step 2 also checks which object the
robot moves under BB. One forecast seed pair, one action start noise per
decision, policy seed 0.

## Step 2: closed loop (submitted 2026-11-16, results pending)

`F1VLAConflictClosedLoop` drives the robot with one cell at every
replanning decision (two full policy calls for the mixed cells, same
forecast seed and start noise). Cells: und=A/gen=A (control through the
same decode path), B/B, A/B, B/A; B = mirror; 4 tasks × 24 episodes, seed
0 (`jobs_cl.txt`). Outcome: SimplerEnv's `moved_correct_obj` /
`moved_wrong_obj` flags and success, relative to the scene's task A.
Sanity check built in: re-decoding the unmixed cache reproduces the
policy's own action exactly (max abs difference 0 on the first episodes of
job 767448).
