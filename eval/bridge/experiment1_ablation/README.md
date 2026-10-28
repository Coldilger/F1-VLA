# Experiment 1 — Ablation of the world-model signal (F1-VLA)

**Status: done — both variants, real closed-loop; paired rerun 2026-10-28
confirms both numbers (see "Paired rerun" below).** (Offline probes also ran
early on; retired to `OFFLINE_PROBE_BACKLOG.md` — no held-out split, not a
citable result, see below.)

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

## Offline probes (both variants) — retired to backlog

Both variants were first checked with an offline probe (predicted action
vs. logged action on real `bridge_orig_lerobot` moments, same protocol as
`../experiment2_oracle`'s offline probe) before the real closed-loop runs
below. Those numbers are not reported here: with no held-out split, any gap
is confounded with memorization and isn't decisive evidence either way —
same reasoning that makes closed-loop success rate this experiment's actual
metric, not L1 against a logged action. Kept for the record, not cited, in
`OFFLINE_PROBE_BACKLOG.md`.

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

Variant 2's closed-loop run (below) tests directly whether the module cares
*which* episode's real image fills the slot, or merely that a real one is
there at all.

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

**Precisely what "wrong episode" means here (verified against the code,
2026-08-21, not just the design intent):** the 64-frame pool is drawn once
via `rng.choice(meta.total_episodes, size=64, replace=False)`
(`np.random.default_rng(seed=123)`) over **every episode in the entire
`bridge_orig_lerobot` dataset**, with no task filtering at all — task
identity is never read when building the pool. So while evaluating Carrot,
the substituted frame can just as easily come from Spoon, Stack, or
Eggplant (a genuinely different scene, and for Eggplant a different robot
camera rig, `widowx_sink_camera_setup` vs. `widowx`) as from a different
Carrot episode. A fresh index is drawn from this pool on **every control
step** (`self._shuffle_rng.integers(...)`, reseeded from `--f1-seed`), not
fixed once per episode. This makes the finding below stronger than "doesn't
care which episode" — it's "doesn't care which episode, which task, or
which camera rig," provided some real Bridge frame occupies the slot.

| task | baseline | ablated (variant 1) | **shuffled (variant 2)** |
|---|---|---|---|
| Put Carrot on Plate | 34.7% | 20.8% | **36.1%** (37.5/29.2/41.7) |
| Put Spoon on Towel | 50.0% | 41.7% | **50.0%** (50.0/41.7/58.3) |
| Stack Green Cube | 40.3% | 13.9% | **43.1%** (41.7/33.3/54.2) |
| Put Eggplant in Basket | 69.4% | 62.5% | **70.8%** (70.8/70.8/70.8) |
| **average** | **48.6%** | **34.7%** | **50.0%** |

**How to read it.** Shuffled matches or slightly exceeds baseline on *every
single task*, never falling below it — the opposite pattern from ablated,
which fell below baseline on every task. Combined with variant 1's result,
the full picture across all three closed-loop conditions is consistent and
clean:

**no real image in the foresight slot (ablated) < self-imagined foresight
(baseline) ≤ any real Bridge image, right episode, wrong episode, or even a
different task/camera rig entirely (shuffled)**

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

## Paired rerun at the thesis protocol (2026-10-28)

The advisor asked whether small gaps such as 48.6% (baseline) vs 50.0%
(shuffled) mean anything without paired statistics. The scratch purge had
also deleted the per-episode outcomes behind every number above, so all
three conditions were rerun with one script
(`closed_loop/closed_loop.slurm`, save tag `exp1_<condition>_seed<N>`) on
the same 4 tasks × 3 seeds × 24 episodes, and compared episode by episode
with `../paired_stats.py` (pairs = task, seed, episode; Wilson CIs,
task-stratified paired bootstrap, exact McNemar).

The rerun reproduces the August totals exactly: baseline 140/288 = 48.6%,
zero 100/288 = 34.7%, shuffled 144/288 = 50.0%.

| comparison | successes | paired difference [95% CI] | discordant pairs | McNemar p |
|---|---|---|---|---|
| zero vs baseline | 100 vs 140 of 288 | −13.9 pp [−20.1, −7.3] | 29 vs 69 | 0.0001 |
| shuffled vs baseline | 144 vs 140 of 288 | +1.4 pp [−3.1, +5.9] | 24 vs 20 | 0.65 |
| shuffled vs zero | 144 vs 100 of 288 | +15.3 pp [+8.7, +21.5] | 72 vs 28 | <0.0001 |

Per task (baseline / zero / shuffled, of 72): Carrot 25 / 15 / 26, Eggplant
50 / 45 / 51, Spoon 36 / 30 / 36, Stack 29 / 10 / 31.

**What this settles.** Removing the foresight tokens costs a real ~14 pp;
replacing the foresight with a wrong episode's costs nothing measurable —
the interval rules out any effect larger than about ±6 pp, and only 44 of
288 paired episodes change outcome at all. So the "48.6% vs 50.0%" gap is
noise, and the content-blindness reading of variant 2 now has a stated
bound.

**What it does not settle.** Variant 1 removes the tokens, which gives the
action expert a sequence shape it never saw in training (the confound in
the table above) — the same class of problem the mimic-video sanity check
found for its zero-ablation. So the −13.9 pp can mean "the action expert
needs *some* real foresight" or "it breaks on an unseen sequence shape".
The control that would separate them is an on-distribution substitute that
keeps the shape but carries no episode content (e.g. the mean foresight KV
over many episodes), analogous to mimic-video's mean-ablation. Not run yet.

## Not yet done

- [x] Implement variant 1 eval run (offline probe).
- [x] Implement variant 2 (KV-shuffle across episodes, offline probe).
- [x] Run variant 1 on real SimplerEnv-Bridge closed-loop.
- [x] Run variant 2 on real SimplerEnv-Bridge closed-loop.
- [ ] **Confound not yet separated:** the shuffle pool draws from the
      entire `bridge_orig_lerobot` dataset with no task filtering (verified
      2026-08-21 — see the pool description above), so "wrong episode,
      same task" and "wrong task entirely" are currently mixed into one
      condition. A cleaner variant 2b would restrict the pool to
      same-task-wrong-episode only, to check whether F1's "any real image
      works" finding survives at that finer grain or only holds once
      cross-task frames are included.
