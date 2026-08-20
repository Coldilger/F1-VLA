# Experiments — index

## Central research question

Does the world-model computation performed at inference time carry causal
weight for action selection — or is the benefit entirely a training-time
representational effect?

Each experiment below attacks this question from a different angle, across
all three models under comparison (F1-VLA, mimic-video, LDA-1B). This repo
is F1-VLA's fork; the same five experiments also live in the mimic-video and
LDA-1B forks, each with a model-specific implementation.

## Experiments

- [`experiment1_ablation/`](experiment1_ablation/) — **Ablation of the
  world-model signal.** One hypothesis, two opposite interventions:
  suppress the foresight signal where the model normally uses it (F1,
  mimic-video), or turn it on where the model normally doesn't (LDA-1B).
- [`experiment2_oracle/`](experiment2_oracle/) — **Oracle injection.**
  Replace the predicted future with the ground-truth future, encoded
  through each model's own pipeline. Tests whether a *perfect* forecast
  would even be used if the model had one. Extends the case study in
  mimic-video's own paper (arXiv:2512.15692, Section III / Fig. 2). A
  **live, randomized-rollout probe** exists for F1 and mimic-video (n=24
  episodes each — memorization-resistant, sourced from live SimplerEnv-
  Bridge rollouts, not a replay of the training set) — for F1, oracle ≈ the
  real policy's own action on successful episodes, reinforcing Experiment
  1's "content doesn't matter much" reading; see each repo's own
  `ORACLE_EXPERIMENT.md`. (An earlier offline replay-based probe existed for
  F1/mimic; retired to `OFFLINE_PROBE_BACKLOG.md` in each repo — no
  held-out split, not a citable result.) LDA-1B's live probe (via RoboCasa)
  is done too — see LDA-1B's own copy. **Closed-loop success-rate evaluation
  with the oracle actually driving the robot — the metric the precedent
  paper itself reports — is still outstanding for all three models**; see
  the "Not yet done" section of `ORACLE_EXPERIMENT.md`.
- [`experiment3_cost/`](experiment3_cost/) — **Cost per decision.**
  Characterizes how much inference-time compute each model actually spends
  on its world-model computation, and at what latency/success-rate
  trade-off. Context for interpreting Experiments 1 and 2: an expensive
  computation that turns out not to matter (per E1/E2) is a stronger result
  than a cheap one. **Done** — all three models' rows measured
  2026-08-19, see `experiment3_cost/README.md`.
- [`experiment4_probing/`](experiment4_probing/) — **Representation
  probing.** Freezes each model's backbone and trains a small probe head to
  predict future end-effector pose from a single frozen hidden state. Tests
  the "training-time representational effect" side of the research question
  directly: does the representation already encode useful future
  information, independent of what the model's own inference-time
  computation does with it? Run for F1 and mimic-video; both fail the same
  way (current pose worse than a constant predictor, confirmed under both
  linear and nonlinear probes — see `experiment4_probing/README.md`). LDA-1B
  differs: its MLP recovers pose where ridge fails, though Experiment 5's
  follow-up narrowed that to "a linear direction ridge's own regularization
  missed" rather than a genuinely nonlinear encoding — see LDA-1B's own copy.
- [`experiment5_erasure/`](experiment5_erasure/) — **Concept erasure
  (LEACE).** Follow-up to Experiment 4's decisive control: surgically erase
  the scene/episode-identity direction from the extracted features and check
  whether pose becomes recoverable in what's left. Tests whether Experiment
  4's failure was pose information being masked by a stronger, irrelevant
  signal, or something else. **Level A run and decisive for F1 and
  mimic-video** — masking is refuted; pose recoverability doesn't improve
  after erasure. Level B (causal downstream re-injection) not started,
  scoped pending a extraction change that makes pose linearly recoverable at
  all. See `experiment5_erasure/README.md`.
