# Experiment 1 — Ablation of the world-model signal (F1-VLA)

**Status: done — both variants, offline probe and real closed-loop.**

## What this tests

One hypothesis, two opposite interventions, applied per-model depending on
whether the model's world-model computation normally runs at inference:

- Where it **normally runs** (F1, mimic-video): turn it **off**, and see if
  action prediction gets worse. If it doesn't, the computation the model
  spends on foresight at inference isn't load-bearing for the actions it
  actually picks.
- Where it **normally doesn't run** (LDA-1B, see the LDA-1B fork's copy of
  this experiment): turn it **on**, and see if action prediction gets
  better.

This is the complement to Experiment 2 (oracle injection): E2 asks "does a
*perfect* future help", E1 asks "does the *actual, currently-computed*
future matter at all, in either direction."

## F1's mechanism: suppress the foresight signal reaching the action expert

**Plain-language summary of variant 1.** F1 makes decisions using three
sub-networks ("experts") that share one memory (a KV cache) built from
attention:

1. **"What I see"** — the current camera image + the language instruction.
2. **"What I imagine next"** — F1's own world model: it imagines what the
   scene will look like a moment later, generated as a sequence of discrete
   tokens (autoregressive VQ-VAE sampling, "VAR").
3. **"What I do"** — the action expert. It reads the shared memory (built
   from #1 and #2) and outputs the actual robot action via flow-matching
   denoising.

**The ablation does not touch the model's weights, retrain anything, or
edit `modeling_f1.py`.** It exploits an already-existing, already-correct
behavior of the shared-memory code
(`f1_vla/src/models/paligemma_with_expert.py`, function `forward`): when
building or reading that memory, any expert slot that is `None` is simply
skipped — nothing is computed or stored for it. This exists for an
unrelated, legitimate reason (reusing the cache across denoising steps
without recomputing it). The ablation exploits it on purpose: `#2`'s slot is
always passed as `None`, for every step, so "what I imagine next" never
enters the shared memory at all — not even a placeholder. The action expert
(`#3`) reads a memory that contains only `#1`, exactly as if the world-model
expert did not exist for that call. The rest of the model — its trained
weights, image processing, action-expert internals, every other eval script
— is completely unmodified; only this one call's inputs differ. See
[`sample_without_world_model.py`](sample_without_world_model.py) for the
exact code (a new, standalone function, not a change to the shared model
file) and its docstring for the underlying mechanics in full technical
detail.

**Why not the originally-planned `use_world_model=False`?** F1 has a config
flag by that name, but it controls whether the world-model sub-network
(with its own trained weights) gets *constructed* at all when the model is
built — flipping it would require reconstructing the whole model with a
different architecture and reloading weights into it, which is a much
larger, riskier change than needed here. The `None`-slot trick achieves the
same experimental condition (the action expert gets nothing from the
world-model expert) while the model itself stays byte-for-byte the one
already validated for every other eval/oracle run in this repo.

Two ways to run this ablation, each with a different confound — **how
"suppress" is implemented matters**:

| | Sequence shape | Content of the foresight slot | Answers | Confound |
|---|---|---|---|---|
| **1. Remove the module** (`None`-slot trick above) | changed: tokens absent | nothing there | is the module needed at all? what does it cost? | shape the model never saw in training |
| **2. Shuffle the KV entries across episodes** | unchanged | real foresight, wrong episode | is *input-specific* information used? | none |

Variant 2 reuses Experiment 2's oracle mechanism (`oracle_indices`,
already validated, also unmodified) — see its own section below — so it
needed no new model-level code either, only a different frame source.

## Results — variant 1, offline probe (120 samples: 24 episodes × 5 moments)

Implemented WITHOUT modifying `modeling_f1.py`: see
[`sample_without_world_model.py`](sample_without_world_model.py)'s docstring
for the mechanism (relies on `paligemma_with_expert.forward` already
skipping any `None` entry in `inputs_embeds`, both when filling and reading
the KV cache, so the gen/world-model expert simply never gets queried for
this call — no config change, no reconstructed model, no risk to the
existing eval/oracle/training paths). Offline probe only so far (real
logged Bridge moments, no simulator), same protocol as
`../experiment2_oracle/oracle_offline_probe.py`.

| | full action L1 |
|---|---|
| **ablated** (no world model) | **0.0272** (sd 0.0319) |
| **baseline** (default, world model present) | **0.0176** (sd 0.0263) |
| zero-action baseline | 0.0946 |

**How to read it:** both conditions are far better than doing nothing
(3.5-5x lower L1 than the zero-action baseline), but removing the
world-model tokens makes action prediction ~55% worse relative to the
unmodified default (0.0272 vs 0.0176). Read together with Experiment 2's
result (oracle is ~6x better than zero-action) this is a second, independent
signal that F1's foresight computation is causally load-bearing for action
selection, not an unused ritual — consistent in direction with (not
identical to) E2's finding.

**Caveat — the confound the slide table already flags is not yet ruled
out.** This result cannot yet distinguish "the module's *information*
matters" from "the model has just never seen this shorter, gen-token-absent
sequence shape during training and degrades on any unfamiliar input shape."
Variant 2 (shuffle real foresight tokens across episodes — same sequence
shape as training, wrong episode's content) is the clean version without
this confound and is not yet implemented.

## Results — variant 2, offline probe (120 samples, same seed/protocol)

Implemented by reusing Experiment 2's already-validated oracle mechanism
unchanged (`oracle_indices` hook, `F1VLAOracleInference`) — no new model
code at all. The only change from the oracle condition: the injected real
frame comes from a *different, randomly paired episode* instead of this
episode's own true next frame (paired by a fixed shift through the same
seed=0 sample list, so every condition below is evaluated on the identical
120 moments). See
[`kv_shuffle_offline_probe.py`](kv_shuffle_offline_probe.py).

**Combined table, all four conditions on the same 120 paired samples:**

| condition | full action L1 |
|---|---|
| baseline (self-imagined foresight, default) | 0.0176 |
| **oracle** (real future, own episode — Experiment 2) | **0.0157** |
| **shuffled** (real future, wrong episode — variant 2) | **0.0157** |
| ablated (no foresight at all — variant 1) | 0.0272 |
| zero-action baseline | 0.0946 |

**How to read it — this changes variant 1's interpretation.** Oracle and
shuffled are *identical* to three significant figures, despite shuffled's
content being from a completely unrelated episode. If the model were using
episode-specific future information, shuffled should sit clearly between
baseline and oracle, not match oracle exactly. It doesn't: **the model
appears indifferent to whether the injected frame is correct, only to
whether *some* real (non-imagined) image is present in that slot at all.**
Combined with ablated being the clear outlier (much worse than all three
conditions with *some* image present, real or wrong-episode), the most
consistent reading is that variant 1's earlier degradation is better
explained by its confound (unfamiliar, shorter sequence shape) than by loss
of genuine information — variant 2 was designed to distinguish exactly this,
and comes down against "input-specific information is used."

This is a meaningful data point for the central research question: it leans
toward the foresight slot acting as something closer to a **structural/
training-time crutch** (needs *a* real image there, in-distribution,
regardless of content) rather than a channel carrying causally-used,
episode-specific predictive information — closer to mimic-video's finding
(oracle ≈ no help there either) than to a clean "world modelling as
control" story, though via a different mechanism (F1: any real image helps
equally; mimic: even the real image doesn't help).

## Results — variant 1, REAL closed-loop SimplerEnv-Bridge (the result that matters)

**Why this run exists, and why the offline-probe numbers above aren't
enough on their own:** the checkpoint was finetuned on the *entire* Bridge
dataset with no held-out split, so a close match between a predicted action
and the real logged action (what the offline probes measure) could reflect
memorization of that exact (image, action) pair — for *either* condition
equally — rather than genuine use of the foresight computation. L1-against-
logged-action cannot tell those two apart. Real closed-loop rollout can:
SimplerEnv randomizes object placement every episode
(`obj_variation_mode: episode`), so beyond the first step the model faces
states that were never logged anywhere during training — they're the
consequence of its own actions in this specific simulated instantiation.
Comparing success rate here is the test that actually isolates causal
weight from memorization.

Implementation: [`f1_vla_policy_ablated.py`](f1_vla_policy_ablated.py)
(`F1VLAAblatedInference`, subclasses the real `F1VLAInference` unchanged,
only overrides `_predict_new_chunk`) +
[`main_inference_ablated.py`](main_inference_ablated.py) (mirrors
`../main_inference.py` exactly, swaps the model class). Same protocol as
the seeded baseline in `../../RESULTS.md`: 4 tasks × 3 seeds × 24 episodes,
same checkpoint (`outputs/bridge_finetune`).

| task | baseline (seeds 0/1/2, `RESULTS.md`) | baseline mean | **ablated** (seeds 0/1/2) | **ablated mean** | Δ |
|---|---|---|---|---|---|
| Put Carrot on Plate | 41.7 / 25.0 / 37.5 | 34.7% | 20.8 / 20.8 / 20.8 | **20.8%** | **−13.9pp** |
| Put Spoon on Towel | 62.5 / 41.7 / 45.8 | 50.0% | 33.3 / 41.7 / 50.0 | **41.7%** | **−8.3pp** |
| Stack Green Cube | 37.5 / 29.2 / 54.2 | 40.3% | 12.5 / 8.3 / 20.8 | **13.9%** | **−26.4pp** |
| Put Eggplant in Basket | 62.5 / 58.3 / 87.5 | 69.4% | 41.7 / 66.7 / 79.2 | **62.5%** | **−6.9pp** |
| **average** | | **48.6%** | | **34.7%** | **−13.9pp** |

**How to read it: ablated is worse than baseline on every single task, not
just on average.** Per `RESULTS.md`'s own documented sensitivity floor at
this sample size (3 seeds × 24 episodes → ~10pp needed to separate two
conditions), the overall drop (−13.9pp) and two of the four individual
tasks (Carrot −13.9pp, Stack −26.4pp) clear that bar; Spoon and Eggplant are
closer to the noise floor but both still land in the same direction, with
no task reversing it. Combined with the consistent direction across all
four tasks, this is a real, memorization-robust result: **removing F1's
world-model computation measurably hurts real task success.** This is the
strongest evidence yet (stronger than the offline probe, immune to its
memorization concern) that the foresight computation is causally
load-bearing for action selection, not a training-time-only effect.

Read alongside variant 2's offline finding (oracle ≈ shuffled — the model
doesn't care *which* episode's real image goes in the slot, only that a
real one is there): the combined picture is that the module's causal weight
comes from something like "conditioning on a real, in-distribution image
signal helps regardless of its specific predictive content" — this closed-
loop result confirms the module matters for real outcomes, while variant 2
narrows *why* (plausibly not "genuine future prediction," more like
"real-image conditioning/regularization"). Variant 2 has not yet been run
in real closed loop (see below) to confirm this reading holds under the
same memorization-robust test.

## Results — variant 2, REAL closed-loop SimplerEnv-Bridge

Same memorization-robustness motivation as variant 1's closed-loop run
above. Implementation: [`f1_vla_policy_shuffled.py`](f1_vla_policy_shuffled.py)
(`F1VLAShuffledInference`, subclasses `F1VLAInference` unchanged) +
[`main_inference_shuffled.py`](main_inference_shuffled.py). Reuses
Experiment 2's `oracle_indices` hook unchanged; the injected frame is drawn
each control step from a fixed pool of 64 real Bridge frames loaded once at
startup (`_load_frame_pool`), sampled with a seed-dependent RNG so each of
the 3 eval seeds is reproducible. Same 4-task × 3-seed × 24-episode protocol
as variant 1 and the baseline.

| task | baseline | ablated (variant 1) | **shuffled (variant 2)** |
|---|---|---|---|
| Put Carrot on Plate | 34.7% | 20.8% | **36.1%** (37.5/29.2/41.7) |
| Put Spoon on Towel | 50.0% | 41.7% | **50.0%** (50.0/41.7/58.3) |
| Stack Green Cube | 40.3% | 13.9% | **43.1%** (41.7/33.3/54.2) |
| Put Eggplant in Basket | 69.4% | 62.5% | **70.8%** (70.8/70.8/70.8) |
| **average** | **48.6%** | **34.7%** | **50.0%** |

**How to read it: this confirms the offline finding on the memorization-
robust metric.** Shuffled matches or slightly exceeds baseline on *every
single task*, never falling below it — the opposite pattern from ablated,
which fell below baseline on every task. Combined with variant 1's result,
the full picture across all three closed-loop conditions is consistent and
clean:

**no real image in the foresight slot (ablated) < self-imagined foresight
(baseline) ≤ any real image, right or wrong episode (shuffled)**

This closed-loop result rules out the memorization concern for the earlier
offline reading: it isn't that the model merely "recognizes" a memorized
(image, action) pair regardless of what's in the foresight slot — object
placement is randomized per SimplerEnv episode, so most of what the model
sees here was never in the training set. The module's causal weight for
real task success is real, and variant 2 narrows down *why*: what seems to
matter is that a real, in-distribution image occupies that slot at all —
not that it correctly predicts this specific episode's future. A plausible
mechanistic story: the world-model slot's real trained job may be closer to
"give the action expert a second, real look at plausible scene statistics
to condition on" than "tell the action expert what happens next" — closer
to a representation/regularization role than a genuine forecasting role,
even though removing it still measurably hurts.

## Not yet done

- [x] Implement variant 1 eval run (offline probe).
- [x] Implement variant 2 (KV-shuffle across episodes, offline probe).
- [x] Run variant 1 on real SimplerEnv-Bridge closed-loop.
- [x] Run variant 2 on real SimplerEnv-Bridge closed-loop.
