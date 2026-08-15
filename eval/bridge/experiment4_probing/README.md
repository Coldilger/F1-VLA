# Experiment 4 — Representation probing (F1-VLA)

**Status: planned, not yet run.**

## What this tests

The direct test of the "training-time representational effect" side of the
central research question: does each model's frozen backbone already encode
useful future information in its representations, independent of what the
model's own inference-time computation (VAR foresight sampling, for F1)
does with it? If a small probe can recover future end-effector pose from a
single frozen hidden state, that's evidence the *training* process already
built the representation — separate from whether F1's own inference-time
foresight loop is what surfaces or uses it (that's E1/E2's question).

## Method

- Freeze the backbone, take one hidden state at the same relative position
  in all three models (needs a specific, comparable extraction point picked
  per architecture — not yet decided for F1).
- Train a small probe head (same size/architecture for all three models) on
  top of the frozen features.
- Probe target: **future end-effector pose** — deliberately *not* what the
  models are already trained to predict (F1 predicts action chunks and VQ
  tokens, not raw future pose directly), so the probe measures information
  present in the representation rather than just replaying the training
  objective.
- Use the same Bridge-finetuned checkpoint as Experiments 1–3
  (`../../RESULTS.md`).
- Cost: forward pass only, train just the small probe head — hours, no
  retraining of the backbone itself.

## Not yet done

- [ ] Decide the exact extraction point (which layer / token position) in
      F1's architecture, comparable to the equivalent choice in mimic-video
      and LDA-1B.
- [ ] Implement frozen-feature extraction + probe head training script.
- [ ] Run on real logged Bridge trajectories, report probe accuracy against
      future end-effector pose.
