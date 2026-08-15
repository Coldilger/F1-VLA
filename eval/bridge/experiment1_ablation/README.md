# Experiment 1 — Ablation of the world-model signal (F1-VLA)

**Status: planned, not yet run.**

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

F1's VAR foresight loop feeds sampled future-frame tokens into the model as
part of `inputs_embeds` before the action expert. Two ways to suppress this,
each with a different confound — **how "suppress" is implemented matters**:

| | Sequence shape | Content of the foresight slot | Answers | Confound |
|---|---|---|---|---|
| **1. Remove the module** (`use_world_model=False`) | changed: tokens absent | nothing there | is the module needed at all? what does it cost? | shape the model never saw in training |
| **2. Shuffle the KV entries across episodes** | unchanged | real foresight, wrong episode | is *input-specific* information used? | none |

Variant 1 is already directly available: `use_world_model` is a real,
existing config flag (`f1_vla/src/models/configuration_f1.py`, threaded
through `modeling_f1.py:664` — `inputs_embeds = [None, None, act_embs] if
self.config.use_world_model else [None, act_embs]`). Running eval with it
flipped to `False` on the existing bridge-finetuned checkpoint (trained with
it `True`) gives variant 1 directly, with the noted confound: the model
never saw this shorter sequence shape in training, so a drop in performance
could be "the module matters" or just "unfamiliar input shape" — variant 2
is the clean version that doesn't have this confound, but needs new code
(shuffling the foresight KV cache entries across different episodes'
batches at inference) that doesn't exist yet.

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

## Not yet done

- [x] Implement variant 1 eval run (offline probe).
- [x] Implement variant 2 (KV-shuffle across episodes, offline probe).
- [x] Run variant 1 on real SimplerEnv-Bridge closed-loop — done above.
- [ ] Run variant 2 (KV-shuffle) on real SimplerEnv-Bridge closed-loop, to
      check whether "oracle ≈ shuffled" also holds for real task success,
      not just offline L1.
