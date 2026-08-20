# Offline ablation probes — backlog, not a cited result

Moved out of `README.md` (2026-08-20). Both variants below compare a
predicted action against the logged action from `bridge_orig_lerobot` — the
exact dataset F1 was finetuned on, with no held-out split — so any gap is
confounded with memorization and isn't decisive evidence either way. This
experiment's real, citable result is the closed-loop success-rate work in
`README.md`, which doesn't have this problem (SimplerEnv randomizes object
placement per episode, so most states beyond step 1 were never logged
anywhere in training). Same rule as `experiment2_oracle/OFFLINE_PROBE_BACKLOG.md`
and `feedback_exp1_success_rate_metric` memory: offline/replay-based L1
probes are not a decisive or citable metric anywhere in this thesis.

Kept here only for provenance — these ran first, chronologically, and
shaped which closed-loop conditions got prioritized. Not linked from the
main doc's narrative, not to be cited in the thesis write-up or
presentation.

## Variant 1 (remove the module) — offline probe (120 samples: 24 episodes × 5 moments)

| | full action L1 |
|---|---|
| ablated (no world model) | 0.0272 (sd 0.0319) |
| baseline (default, world model present) | 0.0176 (sd 0.0263) |
| zero-action baseline | 0.0946 |

## Variant 2 (shuffle across episodes) — offline probe (same 120 paired samples)

| condition | full action L1 |
|---|---|
| baseline (self-imagined foresight, default) | 0.0176 |
| oracle (real future, own episode) | 0.0157 |
| shuffled (real future, wrong episode) | 0.0157 |
| ablated (no foresight at all) | 0.0272 |
| zero-action baseline | 0.0946 |

Oracle and shuffled landed identical to three significant figures here,
which is what first suggested F1 might not care which episode's real image
fills the slot — but this is exactly the kind of pattern a memorized
dataset could produce for reasons unrelated to the model's actual causal
use of the signal. The closed-loop version of the same comparison
(`README.md`'s variant 1/2 real SimplerEnv results) tests this cleanly
instead, and is what the thesis should cite.
