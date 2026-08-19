# Experiment 4 — Representation probing (F1-VLA)

**Status: implemented and run; the measurement fails the same way
mimic-video's does. No conclusion about F1's representation is available
yet.** See "Cross-model finding" below — this isn't F1-specific, and it
changes what mimic-video's own failure should be attributed to.

## What this tests

Does F1's world-model ("gen") expert already encode useful future
information in its own output, independent of what the action expert's
inference-time cross-attention does with it?

## Design

**Extraction point: `gen_out`, the gen expert's own final-layer hidden
state** — the tensor crossing the boundary from world-model expert to
action expert, analogous to mimic-video's `crossattn_emb`. See the
conversation record for why this, not the multi-layer KV cache the action
expert technically reads at every layer, is the right choice: probing the
full per-layer KV structure would need an attention-based probe head,
inconsistent with the single-frozen-hidden-state design used for every
model in this experiment. mimic's own action decoder reads `crossattn_emb`
at every one of its layers too; we didn't probe its internal per-layer state
either.

**The real structure, confirmed empirically, not from config.**
`gen_expert_config.pn = '1_2_3_4_5_6_8_10_13_16'` names a 10-scale VAR
pyramid (680 tokens if all scales ran), but `num_resolutions = 4` and the
generation loop breaks at `si == num_resolutions - 1`. `probe_true_shape.py`
confirmed on real data: exactly 4 `paligemma_with_expert.forward` calls per
step, giving 1+4+9+16 = **30 real tokens**, not 680. The first call's raw
output has 680 positions (an artifact of how the full nominal prefix gets
vectorized before scales 1-9 exist), but the actual inference code only ever
uses its last position (`gen_out[:, -1:] if si == 0`) — `extract_features.py`
replicates that exact slice, so what's captured is exactly what feeds the
real KV cache, no more.

**Pooling: none needed.** 30 tokens is small enough to keep every one
individually (unlike mimic-video's 19200-token case). Channels are
compressed 1024 → 32 via a fixed (untrained, seed-based) random projection —
same rationale as mimic-video's v2 extraction: a data-fit projection (PCA)
would leak validation-episode structure into the basis. Final feature size:
30 × 32 = 960.

**Checkpoint comparison caveat.** Unlike mimic-video's
`pretrained_cosmos_bridge`/`finetuned_cosmos_bridge` pair (differ ONLY in
the probed component, identical action decoder), F1's pretrained
(`InternRobotics/F1-VLA`) and Bridge-finetuned (`outputs/bridge_finetune`)
checkpoints are one joint model — understanding, generation, and action
experts are presumably all finetuned together. A probe-accuracy gap between
variants would show Bridge finetuning changed what `gen_out` carries, but
doesn't rule out the action expert changing too (irrelevant to what's probed
here, but the isolation isn't as clean as mimic's).

**Setup note.** The pretrained checkpoint's `config.json` hardcoded
`language_tokenizer_path`/`pretrained_path` to an author-machine path
(`/fs-computility/efm/...`). `checkpoints/pretrained_patched/` holds a
corrected copy of `config.json` (repointed at this cluster's local
`paligemma_tokenizer`/`pi0_base`); `model.safetensors` is an untouched
symlink into the original HF snapshot. Verified the weights actually loaded
(not silently reinitialized): the HF "missing/unexpected keys" warning never
appeared, and pretrained vs finetuned smoke features have similar norms
(31.6 vs 32.0) but are genuinely different (cosine similarity 0.89) — not
zero, not identical, not random noise.

## Results

Run 2026-08-18, n=400 (40 episodes × 10 samples), 30 train / 10 val
episodes, identical protocol to mimic-video's re-run.

| variant | ridge L1 | MLP L1 | best constant | no-motion |
|---|---|---|---|---|
| finetuned | 0.06390 | 0.06167 | 0.05355 | 0.05302 |
| pretrained | 0.05782 | 0.06119 | 0.05355 | 0.05302 |

Neither variant beats the best-constant predictor. Unlike mimic-video
(where finetuned edged past the constant by 0.6%), here **pretrained is
closer to the constant than finetuned is** — the opposite ordering. Given
what follows, neither number should be read as meaningful.

**Control 1 — episode identity:** 100% accuracy, 40× chance, both variants
(same as mimic-video).

**Control 2 — current pose (decisive):**

| target | probe L1 | best constant | gain |
|---|---|---|---|
| current pose, finetuned | 0.30784 | 0.08690 | **−254.2%** |
| current pose, pretrained | 0.30756 | 0.08690 | **−253.9%** |
| future delta, finetuned | 0.06390 | 0.05355 | −19.3% |
| future delta, pretrained | 0.05782 | 0.05355 | −8.0% |

The probe is ~3.5× worse than a constant predictor on current pose, for both
variants — the same catastrophic failure mimic-video's re-extraction found
(−251%/−247%), at nearly identical magnitude.

## Cross-model finding: this rules out pooling as mimic-video's explanation

mimic-video's failure was attributed to lossy mean+std pooling over 19200
tokens destroying localized (spatial) information while preserving global
(scene) information. **F1's extraction never did that kind of pooling at
all** — all 30 real tokens are kept individually, only channels are
compressed via a linear random projection. F1 fails identically anyway.

This means pooling design was not the primary cause, or at least not the
only one. The more likely shared culprit, consistent with both models'
identical 100%-episode-identity result: **the representations are dominated
by per-episode/scene identity, and with ~30 training episodes (mimic's own
clustering check found 40 Bridge episodes collapse into ~14 scene families
at a loose threshold), any linear probe learns "which scene → what pose"
and extrapolates catastrophically to unseen scenes** — independent of how
carefully the per-sample features preserve spatial detail.

## What would have to change

Given the above, re-attempting mimic-video's pooling fix (already done, v2)
was a necessary check but is very unlikely to be sufficient on its own — and
this result front-loads that finding for F1 before spending another
extraction cycle finding the same thing. The higher-leverage fix is
**episode/scene diversity**, not pooling granularity: many more distinct
physical scenes in the training split (not just more samples from the same
~14 scene families), so that "which scene" stops being the dominant,
most-exploitable direction of variance for the probe to latch onto.

## Files

- `probe_true_shape.py` / `.slurm` — confirms the real 30-token structure
  before any extraction code was written
- `extract_features.py` / `extract.slurm` — frozen-feature extraction
- `train_probe.py` / `train_probes.slurm` — ridge (primary) + MLP (capacity
  check) probes
- `positive_control.py` / `positive_control.slurm` — control 1, episode
  identity
- `recover_current_pose.py` / `recover_pose.slurm` — replays extraction RNG
  to recover current pose without the model (note: uses `"xyz"` lowercase
  Euler convention, matching F1's own code — NOT mimic-video's `"XYZ"`)
- `compare_targets.py` / `compare_targets.slurm` — control 2, current vs
  future (decisive)

## Update (2026-08-18): 25x more episodes helps in degree, not in kind

Following `experiment5_erasure/`'s refutation of the scene-masking
hypothesis, re-extracted with 1000 episodes x 1 sample (vs. the original 40
x 10) — same total-ish extraction cost (F1's per-sample cost is ~0.4s, so
1000 samples took minutes, not hours), 25x more distinct episodes.

| | 40 episodes (30 train / 10 val) | 1000 episodes (750 train / 250 val) |
|---|---|---|
| current-pose gain | −254.2% / −253.9% | **−144.0% / −144.6%** |
| future-pose gain | −19.3% / −8.0% | **+0.2% / +0.1%** |

Real improvement in *degree* — the probe is now ~2.4x worse than a constant
predictor rather than ~3.5x — but not in *kind*: current pose, a quantity
unambiguously present in the input frame, is still nowhere near recoverable.
25x more episode diversity meaningfully softened the failure without fixing
it. This argues against "just needs a bit more data" and doesn't rule in or
out "needs an order of magnitude more" vs. "the extraction itself caps what's
recoverable regardless of data" -- both remain open.

`positive_control.py`'s episode-identity check is **not meaningful at
samples-per-episode=1**: with exactly one sample per episode, its
sample-level split degenerates into an episode-level one (every episode
lands entirely on one side), so predicted labels (from the 750 train
episodes) can never match held-out val episode IDs by construction — the
resulting 0.0% is a split-design artifact, not a finding about the
representation. The decisive current-pose control is unaffected (it already
uses an episode-level split by design) and remains the number to trust.

## Is current pose nonlinearly recoverable? (2026-08-19, cross-check from LDA-1B)

While probing LDA-1B, an MLP recovered its current pose (+60.2% over
constant) where ridge had failed badly (−229%) — raising the question of
whether F1's own current-pose failure is also a linear-probe artifact
rather than a real absence of information. Tested directly
(`LDA-1B/eval/bridge/experiment4_probing/mlp_current_pose.py`, same
`ProbeHead` architecture this repo's own `train_probe.py` already uses for
the future target, run against this repo's own `features_finetuned_more.npz`
/ `current_pose_more.npz`): **MLP current-pose val L1 0.138 vs. constant
0.138 — a −0.1% gain, i.e. no better than guessing the mean.** Unlike LDA,
nonlinearity is not the explanation for F1's current-pose failure — the
representation (`gen_out` tokens) genuinely does not encode pose here,
under either probe family.

## Not yet done

- [ ] Re-run mimic-video's equivalent more-episodes extraction (in progress:
      250 episodes x 2 samples) for a same-question cross-model comparison.
- [ ] If pursued further: an even larger episode count for F1 (cheap here,
      ~0.4s/sample) to see whether the −144% keeps shrinking or plateaus.
- [ ] Fix or retire positive_control.py's episode-identity check for the
      samples-per-episode=1 case, or document the samples-per-episode>=2
      requirement explicitly in its argparse help.
- [x] Extend to LDA-1B — done 2026-08-19, see
      `LDA-1B/eval/bridge/experiment4_probing/README.md`. Notably different
      result: LDA's shared `vl_embs` backbone *does* encode both current and
      future pose, nonlinearly.
