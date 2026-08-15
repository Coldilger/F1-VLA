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

## Not yet done

- [x] Implement variant 1 eval run (offline probe).
- [ ] Implement variant 2 (KV-shuffle across episodes) — needs new code,
      rules out the shape confound above.
- [ ] Run variant 1 on real SimplerEnv-Bridge closed-loop (not just the
      offline probe) — compare success rate against the unmodified
      baseline (see `../../RESULTS.md`).
