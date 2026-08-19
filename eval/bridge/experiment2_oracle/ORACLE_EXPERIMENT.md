# Experiment 2: Oracle injection — F1-VLA

## What this is and why it's grounded, not invented

This experiment is not something we made up. It directly extends the
method from Section III / Fig. 2 of the mimic-video paper
(arXiv:2512.15692, "Case Study: How Does Video Generation Quality Affect
Robot Policy Performance?"). There, the authors do exactly one thing: give
the action decoder either a predicted future or the real ("oracle") one,
and measure how much that changes performance. We applied the same
method to all three models under comparison (F1-VLA, mimic-video,
LDA-1B), each time through that specific model's own natively-trained
future-prediction mechanism — not a bolted-on hack, the same mechanism
the model actually uses at inference.

The question the experiment answers: **does the "imagine the future"
computation carry causal weight for action selection, or is the benefit
already baked into the learned representations, making the inference-time
computation an expensive ritual with no real effect?**

## How exactly F1 "sees" the real future

By default F1 **imagines** the future frame itself: it samples it as a
sequence of discrete tokens through its own VQ-VAE (a finite "dictionary"
of image codes any image can be reassembled from). This process is called
VAR (Visual AutoRegressive) — the model predicts future-frame tokens one
at a time, across multiple resolution scales, the same way a language
model predicts words one at a time.

In the oracle condition we don't let the model guess. We take the
**real** next frame (from an actually-recorded Bridge episode, not the
simulator) and run it through the same VQ-VAE encoder — we get the same
kind of discrete tokens, but real ones, not sampled ones. Those tokens
are substituted exactly where the sampled ones would normally go — the
rest of the model has no way of telling whether the tokens were imagined
or came from reality.

Implementation: `f1_vla/src/models/modeling_f1.py`, `oracle_indices`
parameter on `sample_actions_with_world_model` — when given, it's used
instead of sampling at each VAR scale. Defaults to `None`, so normal
closed-loop rollout is unaffected.

## How the metric is computed

Take a real, logged moment from an actual Bridge episode: the "now"
frame plus what the robot **actually** did a second later (logged in the
dataset — a fact, not a guess). Show the model "now", ask "what action
would you take", compare its answer to reality.

We do this twice for the same moment:
- **policy / default** — the model answers while imagining the future
  itself;
- **oracle** — the model is additionally shown the real next frame before
  answering.

A robot action is a vector of several numbers (x/y/z translation,
rotation, gripper). **L1** is `|predicted − real|`, averaged across the
vector's numbers. Example: real action `[0.02, -0.01, 0.00]`, predicted
`[0.03, -0.01, 0.01]` → errors `[0.01, 0.00, 0.01]` → L1 = 0.0067. Lower
L1 means a closer guess.

We then compare L1 under oracle vs L1 under policy on the same moments.
If oracle is clearly more accurate, the mechanism genuinely uses precise
future information when it has it. If there's no difference, even the
real future doesn't help — meaning the inference-time computation carries
no causal weight for action selection.

## Current results (120 samples: 24 episodes × 5 moments)

| | oracle | baseline (zero action) |
|---|---|---|
| full action, L1 | **0.0157** (sd 0.0195) | 0.0946 |
| position (x,y,z), L1 | 0.0060 | — |
| gripper, L1 | 0.0202 | — |

**How to read it:** oracle is almost 6x more accurate than the trivial
baseline ("don't move"). Given a real future frame, F1 genuinely
recovers the correct action with good accuracy — the foresight
mechanism is being used, not just present as a ritual. This is a strong
candidate for the "world modelling as control" cell of the taxonomy
(Slide 16) — at least for F1.

## Not yet done

- [ ] **Closed-loop success-rate evaluation.** The precedent this experiment
  extends (mimic-video's own paper, Section III/Fig. 2) reports **closed-loop
  success rate**, not offline single-step L1 — the number above is a cheaper
  proxy for the causal question, not a replication of the paper's own
  reported metric. Getting a genuine closed-loop oracle number is harder than
  it looks: once the model's own action diverges from the logged trajectory,
  there is no pre-recorded "real future" left to inject at the next step. The
  source paper handled this via live human teleoperation (mimic-video's own
  `main_inference_hil.py` / `eval_hil.sh`, "human-in-the-loop evaluation
  (oracle study)") — expensive per episode, and not yet run for any of the
  three models. Until this exists, read the L1 numbers above as a cheap
  offline signal that the mechanism *can* use real future information, not
  as evidence about closed-loop success rate specifically.

## Caveats

1. **Memorization.** The model was trained on the whole Bridge dataset;
   there is no held-out split. The number may partly reflect "the model
   memorized this trajectory" rather than "the architecture can use
   foresight". Without a held-out split these are indistinguishable —
   see the general caveat that applies to all three models.
2. **One run, no seed repeats.** Given the seed-to-seed variance already
   documented on this benchmark (see `RESULTS.md`, up to 25-30pp spread
   between seeds on closed-loop success rate), a single number without
   repeats is a preliminary result, not final. Worth running 2-3 times
   with different seeds before citing the gap as robust.
3. **Units.** L1 here is in Bridge's physical units (meters for
   position, radians for rotation, 0-1 for gripper), because
   `F1VLAInference` already un-normalizes its prediction before
   comparison. This lets it be compared directly against the logged
   action with no extra conversion.
