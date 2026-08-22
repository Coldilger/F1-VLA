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

A robot action is a vector of several numbers (x/y/z translation, rotation,
gripper). **L1** is `|predicted − real|`, averaged across the vector's
numbers. Example: real action `[0.02, -0.01, 0.00]`, predicted
`[0.03, -0.01, 0.01]` → errors `[0.01, 0.00, 0.01]` → L1 = 0.0067. Lower L1
means a closer guess.

The comparison is between the oracle-conditioned prediction (the real next
frame substituted in via `oracle_indices` on `sample_actions_with_world_model`)
and a reference action, on the same real decision points. If oracle is
clearly closer to the reference than the model's normal, self-imagined
prediction is, the mechanism genuinely uses precise future information when
it has it. If there's no difference, even the real future doesn't help —
meaning the inference-time computation carries no causal weight for action
selection.

An earlier version of this probe replayed `bridge_orig_lerobot` — the exact
dataset F1 was finetuned on, with no held-out split — comparing the oracle's
prediction against the logged expert action. That number is not reported
here: with no held-out split, any gap is confounded with memorization
(the model may reproduce "the right" action because it memorized this
specific trajectory, not because it genuinely used the injected signal),
so it isn't decisive evidence either way. Kept for the record, not cited,
in `OFFLINE_PROBE_BACKLOG.md`. The live probe below replaces it.

## Live oracle probe (closed-loop, randomized)

This probe reuses the identical oracle mechanism
(`F1VLAOracleInference.predict_action_given_true_next_frame`, unmodified)
but sources it from a live, randomized SimplerEnv-Bridge rollout: object
placement is randomized per episode by SimplerEnv itself (the same
mechanism Experiment 1's own closed-loop eval already uses), so verbatim
recall of a specific trajectory is impossible here (distributional
overfitting to the task family is a separate, softer question this doesn't
rule out).

**Design.** A normal closed-loop rollout runs with the REAL (non-oracle)
policy driving the robot — behavior is completely unaffected, this is a pure
side computation. At each real replan, a second oracle-only model instance
is additionally queried with the true next frame, and its predicted action
is compared against what the real policy itself did at that same decision
point.

**Why the comparison is restricted to successful episodes.** There is no
ground truth here the way the offline probe has expert demonstrations —
"the real policy's own action" is the only available reference, and the
real policy only succeeds on part of its episodes. On a failed episode, the
policy's own action wasn't good, so an oracle prediction that *diverges*
from it could be an improvement, not an error — L1-against-a-mediocre-
baseline can't tell the two apart. Restricting to episodes SimplerEnv scored
as successful (`maniskill2_evaluator`'s own `success_arr`) makes the
reference "real behavior that actually worked" — not a full fix (not every
action inside a successful episode is necessarily optimal), but a real
improvement over comparing against unfiltered behavior.
Implementation: `main_inference_live_oracle.py`.

### Results (job 631445, task PutCarrotOnPlateInScene-v0, 24 episodes)

Average success: **45.8%** (11/24).

| | all samples (n=336) | successful episodes only (n=154) |
|---|---|---|
| oracle L1 vs real policy's own action | 0.02021 (sd 0.03532) | **0.01399** (sd 0.02649) |
| magnitude of the real action itself | 0.11030 | 0.11305 |

**How to read it.** Given the true next frame, the oracle's predicted action
stays close to what the real (non-oracle) policy already did on episodes
that worked — about 8x smaller than the action's own magnitude. A perfect
future frame barely moves the prediction away from what the model predicts
without it.

**This number alone is ambiguous, and it's worth being explicit about why.**
A small gap between oracle and the model's own (self-imagined-future)
action has two possible explanations that are opposite in what they'd mean
for the thesis's research question:

- (a) The model's own imagined future is already close to correct, so
  substituting the real one barely changes anything — a ceiling effect. If
  true, this would mean the foresight computation is doing real, accurate
  work; the causal ingredient here would be forecast *quality*, and it's
  already high.
- (b) The action-decoding step doesn't read out the future frame's specific
  content in a fine-grained way at all — a right guess, a wrong guess, and
  the model's own guess all land in roughly the same place, because
  correctness just isn't what this mechanism is sensitive to.

These two stories predict the identical number. Exp2 alone cannot tell
them apart — resolving this needs Exp1.

**Exp1 resolves it, in favor of (b).** Exp1's shuffled condition (a real
frame from a *different, wrong* episode — not a plausible guess, an
outright incorrect one) performs statistically the same as baseline
(50.0% vs. 48.6%), while ablated (no frame at all) performs clearly worse
(34.7%). If (a) were true — the model's own forecasts are already accurate,
so correctness stops mattering once you're near-ceiling — a deliberately
*wrong* frame from an unrelated episode should have hurt, the same way
feeding a language model the wrong context hurts. It doesn't. That rules
out (a) and confirms (b): the mechanism needs *some* frame occupying that
structural slot, and is functionally indifferent to what's actually in it
— correct, wrong, or self-imagined alike.

**Put together, Exp1 + Exp2 give one specific, falsifiable conclusion for
F1-VLA:** the world-model computation is a structural dependency, not a
forecasting one. It is causally load-bearing (Exp1: ablation hurts) but not
on forecast accuracy (Exp1: shuffled doesn't hurt; Exp2: oracle doesn't
move the decision either) — "an expensive ritual that must be performed,
not a signal that must be read." Neither experiment reaches this
conclusion alone; each is genuinely ambiguous in isolation the way Exp2 is
described above.

One more limit on even this resolved reading: it only concerns whether the
model's *decision* is sensitive to forecast accuracy — not whether a more
accurate forecast would lead to *better* outcomes. Oracle never drives the
robot here (side-channel query only), so that closed-loop question has no
number attached to it at all; see "Not pursued" below for why it stays
that way.

**What this does not establish.** This is not a closed-loop oracle
success-rate number — the oracle here is a side-channel query, never
actually driving the robot, so it says nothing about whether an
oracle-driven rollout would succeed more often than the real policy does.
See "Not pursued" below for why that number doesn't exist and isn't
coming.

## Not pursued (a deliberate scope decision, not an oversight)

- **Closed-loop success-rate evaluation** — i.e. the oracle actually
  driving the robot, not just a side-channel query compared against the real
  policy's own action. The precedent this experiment extends (mimic-video's
  own paper, Section III/Fig. 2) reports **closed-loop success rate**, not
  single-step L1 — the live probe above is a cheaper proxy for the causal
  question, not a replication of the paper's own reported metric. Getting a
  genuine closed-loop oracle number is harder than it looks: once the
  model's own action diverges from the logged trajectory, there is no
  pre-recorded "real future" left to inject at the next step. The source
  paper handled this via live human teleoperation (mimic-video's own
  `main_inference_hil.py` / `eval_hil.sh`, "human-in-the-loop evaluation
  (oracle study)") — a real per-episode human in the loop, not a script.
  **Not being run for any of the three models in this thesis** — the
  infrastructure and time cost don't fit the scope here, not a TODO waiting
  on availability. Read the L1 numbers above as a cheap signal that the
  mechanism *can* use real future information, not as evidence about
  closed-loop success rate specifically, permanently — not "for now."

## Not yet done

- [ ] Seed repeats for the live probe (currently one run, 24 episodes, one
  task) — see caveat 2 below, which applies here too.

## Caveats

1. **Distributional overfitting.** Object placement is randomized per
   episode, so verbatim memorization of this exact trajectory is
   impossible — but overfitting to the task *family* (Bridge's specific
   set of tasks/objects/scenes) is a separate, softer question this probe
   doesn't rule out. Weaker and harder to rule out than exact-trajectory
   memorization, but not the same failure mode as the offline probe this
   one replaced (see `OFFLINE_PROBE_BACKLOG.md`).
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
