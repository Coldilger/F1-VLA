# Experiment 5 — Concept erasure (F1-VLA)

**Status: Level A run and decisive. Level B not started, scoped pending
Level A's outcome.** See mimic-video's copy of this experiment for the full
method writeup (LEACE vs. INLP rationale, O-LEACE caveat, self-test) --
identical method, identical code (`leace.py`, `scene_erasure_diagnostic.py`),
only the input features differ.

## Motivation

F1's Experiment 4 decisive control found the same catastrophic
current-pose failure as mimic-video's (~3.5x worse than a constant
predictor), despite F1's extraction never pooling spatially at all (all 30
real `gen_out` tokens kept individually, only channels compressed). That
cross-model agreement is what motivated testing "is pose information
present but masked by scene dominance" directly, with LEACE, rather than
assuming either model's extraction design was specifically at fault.

## A bug found and fixed along the way

The same leak affected F1's `positive_control.py` as mimic-video's: alpha
was selected by maximizing accuracy directly on the held-out val set. Fixed
identically (k-fold CV within train only). Re-run with the fix:

| variant | episode-identity accuracy (corrected) |
|---|---|
| finetuned | 100.0% (chance 2.5%, 40x) |
| pretrained | 100.0% (chance 2.5%, 40x) |

F1's original numbers happened to land on the same 100%/40x figure the leaky
method reported, so nothing here needed correcting numerically -- but the
method was still wrong and is fixed for consistency and for any future rerun.

## Results

Run 2026-08-18, features from `experiment4_probing/features_{variant}.npz`
(30 real `gen_out` tokens, individually kept, channel-projected), episode
-level split (30 train / 10 val episodes), n=400.

| variant | episode-ID before | episode-ID after erasure | current-pose gain before | current-pose gain after |
|---|---|---|---|---|
| finetuned | 100.0% (chance 3.3%) | **1.3%** | −254.2% | **−257.0%** |
| pretrained | 100.0% (chance 3.3%) | **0.0%** | −253.9% | **−257.0%** |

Erasure works as designed: episode identity collapses to at-or-below chance
on fresh held-out samples of the episodes it was fit to separate, for both
checkpoints. **Current pose does not become recoverable** -- if anything the
gain is marginally worse after erasure (well within what CV/alpha-selection
noise across two independent probe fits would produce, not a real effect).

## Verdict: matches mimic-video's, and this is the more informative half of
the cross-model result

Scene dominance is refuted as the explanation here too. Combined with the
Experiment 4 finding that F1's extraction never pooled spatially at all,
this is the stronger of the two models' results for ruling out "we averaged
away the localized signal": F1's pose information had nowhere to be lost to
pooling, no scene-identity confound was masking it either (verified, not
assumed), and it is still not there. Between the two remaining explanations
in `experiment4_probing/README.md` (extraction/pooling vs. genuine
data/signal scarcity), F1's result weighs more toward scarcity for this
model specifically, since the pooling explanation has now been eliminated
for it twice over.

## Level B: not started

Same reasoning as mimic-video's copy: building a causal re-injection test on
a pose direction that isn't reliably identifiable in the first place isn't
interpretable. Not scoped further pending an extraction change that makes
pose linearly recoverable at all, for either model.

## Files

- `leace.py` -- identical to mimic-video's copy (LEACE fit/erase + self-test)
- `scene_erasure_diagnostic.py` / `scene_erasure.slurm` -- Level A

## Not yet done

- [ ] Extend to LDA-1B once its Experiment 4 extraction point is resolved.
