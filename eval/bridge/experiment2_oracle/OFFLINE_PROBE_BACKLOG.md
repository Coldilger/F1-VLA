# Offline oracle probe — backlog, not a cited result

Moved out of `ORACLE_EXPERIMENT.md` (2026-08-20). This probe replays
`bridge_orig_lerobot` — the exact dataset F1 was finetuned on, with no
held-out split — so any gap it shows is confounded with memorization and
cannot be trusted as evidence about the causal question this experiment
asks. Established as a general rule for this thesis, not specific to F1:
offline/replay-based L1 probes are not a decisive or even citable metric
anywhere, precisely because of this confound (same reasoning that
already applies to Experiment 1 — see `feedback_exp1_success_rate_metric`
memory). The live oracle probe in `ORACLE_EXPERIMENT.md` (randomized
SimplerEnv-Bridge rollouts, not the training dataset) is what replaced
this, specifically because it does not have this problem.

Kept here only for provenance/traceability — not linked from the main
doc's narrative, not to be cited in the thesis write-up or presentation.

## What it measured

Take a real, logged moment from an actual Bridge episode: the "now" frame
plus what the robot actually did a second later. Show the model "now", ask
"what action would you take" — once with F1 imagining the future itself
(policy/default), once with the real next frame substituted in via
`oracle_indices` on `sample_actions_with_world_model`. Compare each
predicted action's L1 distance to the real logged action.

## Results (120 samples: 24 episodes × 5 moments)

| | oracle | baseline (zero action) |
|---|---|---|
| full action, L1 | 0.0157 (sd 0.0195) | 0.0946 |
| position (x,y,z), L1 | 0.0060 | — |
| gripper, L1 | 0.0202 | — |

Oracle scored better than a trivial zero-action baseline here — but since
this replays the training set with no held-out split, that gap cannot be
attributed to genuine use of foresight versus memorized recall of this
specific trajectory. Do not read anything into this table beyond "the
probe ran and produced a number" — the live probe is the one that answers
the actual question.
