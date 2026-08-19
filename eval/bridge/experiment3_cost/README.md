# Experiment 3 — Cost per decision (F1-VLA)

**Status: F1's own row done (real closed-loop latency, all 3 Experiment 1
conditions).**

## What this tests

Context for interpreting Experiments 1 and 2: how much inference-time
compute does each model actually spend on its world-model computation, and
at what latency / success-rate trade-off? A computation that turns out not
to causally matter (per E1/E2) is a much stronger result if it's also
*expensive* — that's compute being spent for nothing, not just an unused
but free side-effect.

| Category | Model | World-model compute @ inference | Latency / control step (median · p95) | Success rate, SimplerEnv-Bridge |
|---|---|---|---|---|
| 1 | **F1-VLA** | VAR foresight loop, re-run **every control step** | 215.7 ms · 263.2 ms | 48.6% (3-seed × 4-task average) |
| 2 | mimic-video | One video-backbone forward pass per action chunk (amortised over the chunk) | 10226.9 ms · 10416.5 ms | 11.5% (baseline at matched `stop=23`) |
| 3 | LDA-1B | None beyond the shared MM-DiT — the visual-forecasting head is a training-time co-objective, unused at inference | 254.7 ms · 256.8 ms (RoboCasa checkpoint, Bridge has nothing working to time yet) | 0% (still under investigation) |

F1 is in the worst position here structurally: unlike mimic-video (one
forward pass amortized over a whole action chunk) or LDA-1B (no extra
inference-time cost at all), F1 re-runs its VAR foresight sampling loop on
*every single control step*. Reference point: F1's own paper reports ≈235ms
total inference on an RTX 4090 (foresight generation 76ms + 10× flow action
steps 95ms + I/O ~64ms) — already above a 200ms/step budget, before
accounting for our own hardware.

## Results — F1's real closed-loop latency, all 3 Experiment 1 conditions

Measured via [`timing_wrapper.py`](timing_wrapper.py) (a class decorator
wrapping `_predict_new_chunk` with `torch.cuda.synchronize()` + a timer —
does not modify `f1_vla_policy.py` or any Experiment 1 wrapper) +
[`main_inference_timed.py`](main_inference_timed.py), on a real SimplerEnv
rollout (3 episodes, PutCarrotOnPlateInScene, n=45 replan calls per
condition — every replan is a full chunk-worth of control steps, i.e. one
real per-decision cost, matching this table's own row description).
Latency doesn't need the full 4×3×24 statistical-power grid the way success
rate does — it's a property of the model's compute, not particularly
task-dependent — so this smaller run already gives a stable estimate.

| condition | median | p95 | mean |
|---|---|---|---|
| baseline (self-imagined foresight) | 215.7ms | 263.2ms | 270.0ms |
| **ablated** (no world model) | **160.6ms** | 204.9ms | 187.0ms |
| shuffled (wrong-episode real frame) | 217.3ms | 220.3ms | 225.0ms |

**How to read it.** Ablated is genuinely, meaningfully cheaper than
baseline (~55ms / ~26% faster median) — removing the VAR foresight-sampling
loop saves real compute, as expected structurally, and (per
`../experiment1_ablation/README.md`) also measurably *hurts* real task
success (34.7% vs 48.6%). So the module is a real compute-for-performance
trade, not a free structural artifact.

Shuffled costs about the same as baseline, *not* less, despite skipping the
sampling step itself. This is explainable, not surprising: `oracle_indices`
(the mechanism variant 2 and Experiment 2 both reuse) only substitutes what
gets *sampled* at each VAR scale — the expensive part (forward passes
through the shared attention stack at every scale) still runs identically
whether the tokens are sampled or supplied. The extra cost of encoding the
injected frame through the VQ-VAE apparently roughly offsets what little
the substitution itself would have saved. So variant 2's earlier finding
(shuffled matches or beats baseline on success, per
`../experiment1_ablation/README.md`) is a real, not-cheaper "free" win in
success-rate terms, but not a discount in inference-time cost.

F1's own paper reports ≈235ms total inference on an RTX 4090 — our
baseline median (215.7ms) is comparable, on presumably faster hardware
(H200 vs RTX 4090), a reasonable sanity check that this measurement isn't
wildly off.

## Not yet done

- [x] Instrument the wrapper to record real per-decision latency (kept
      as one combined number rather than splitting foresight/action/I/O
      internally, to avoid modifying `modeling_f1.py` — see above for how
      the ablated-vs-baseline comparison serves as an indirect split).
- [x] Run on real hardware, all 3 Experiment 1 conditions, report
      median/p95.
- [x] mimic-video and LDA-1B's rows (this table's other two categories) —
      measured 2026-08-19, see each repo's own `experiment3_cost/README.md`.
- [ ] Scale up to the full statistical-power grid if a more precise number
      is needed later (current n=45 already gives a stable median/p95
      estimate for a compute-bound, not particularly noisy quantity).
