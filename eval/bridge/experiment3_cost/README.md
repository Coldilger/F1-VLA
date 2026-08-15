# Experiment 3 — Cost per decision (F1-VLA)

**Status: planned, not yet run.**

## What this tests

Context for interpreting Experiments 1 and 2: how much inference-time
compute does each model actually spend on its world-model computation, and
at what latency / success-rate trade-off? A computation that turns out not
to causally matter (per E1/E2) is a much stronger result if it's also
*expensive* — that's compute being spent for nothing, not just an unused
but free side-effect.

| Category | Model | World-model compute @ inference | Latency / control step (median · p95) | Success rate, SimplerEnv-Bridge (95% CI) |
|---|---|---|---|---|
| 1 | **F1-VLA** | VAR foresight loop, re-run **every control step** | TBD | TBD |
| 2 | mimic-video | One video-backbone forward pass per action chunk (amortised over the chunk) | TBD | TBD |
| 3 | LDA-1B | None beyond the shared MM-DiT — the visual-forecasting head is a training-time co-objective, unused at inference | TBD | TBD |

F1 is in the worst position here structurally: unlike mimic-video (one
forward pass amortized over a whole action chunk) or LDA-1B (no extra
inference-time cost at all), F1 re-runs its VAR foresight sampling loop on
*every single control step*. Reference point: F1's own paper reports ≈235ms
total inference on an RTX 4090 (foresight generation 76ms + 10× flow action
steps 95ms + I/O ~64ms) — already above a 200ms/step budget, before
accounting for our own hardware.

## Not yet done

- [ ] Instrument `f1_vla_policy.py`'s `step()` to record per-step wall-clock
      latency, split into foresight-generation vs action-flow-matching vs
      I/O, matching the paper's own breakdown.
- [ ] Run on real hardware (this project's GPUs, not the paper's RTX 4090)
      across a real SimplerEnv-Bridge eval sweep, report median/p95.
- [ ] Cross-reference against the already-measured success rate
      (`../../RESULTS.md`) for the same checkpoint/seeds.
