#!/usr/bin/env python3
"""
Experiment 4 (F1-VLA), step 1: confirm the real per-scale shape of the
world-model expert's hidden states before designing any pooling.

Same discipline as mimic-video's re-extraction: mimic's own reshape lost the
real geometry of a tensor whose docstring described the flattened form, not
the raw one -- confirmed the mistake only by capturing before the reshape on
real data. F1's structure is read from modeling_f1.py's code
(sample_actions_with_world_model, ~line 511-610): the "gen" (world-model)
expert runs the VAR foresight loop autoregressively across
config.patch_nums scales (this checkpoint: pn='1_2_3_4_5_6_8_10_13_16', so
10 scales, 1+4+9+...+256=680 tokens total, hidden_size=1024), producing one
`gen_out` tensor per scale from `self.paligemma_with_expert.forward(...,
inputs_embeds=[..., x, None], ...)`. Those `gen_out` tensors are exactly what
gets written into the shared KV cache the action expert reads -- the same
cache Experiment 1's ablation empties by passing gen_embs=None -- so they are
F1's analog of mimic's crossattn_emb.

This script hooks `paligemma_with_expert.forward` on a live model instance
(no edits to modeling_f1.py, matching every other experiment's discipline)
and records the shape of `gen_out` on every call during one real closed-loop
step, to confirm the per-scale token counts and hidden size directly rather
than trusting the arithmetic above.
"""

import pathlib
import sys

import numpy as np

sys.path.insert(0, "/mnt/beegfsnew/scratch/3295540/F1-VLA")
sys.path.insert(0, "/mnt/beegfsnew/scratch/3295540/F1-VLA/eval/bridge")

from f1_vla_policy import F1VLAInference  # noqa: E402


class _PoseProxy:
    def __init__(self, p, q):
        self.p = p
        self.q = q


def main():
    checkpoint_path = "/mnt/beegfsnew/scratch/3295540/F1-VLA/outputs/bridge_finetune"
    stats_path = "/mnt/beegfsnew/scratch/3295540/F1-VLA/outputs/bridge_finetune/bridge_orig_stats.json"

    model = F1VLAInference(checkpoint_path=checkpoint_path, stats_path=stats_path, seed=0)

    calls = []
    pge = model.policy.model.paligemma_with_expert
    original_forward = pge.forward

    def capturing_forward(*a, **kw):
        out = original_forward(*a, **kw)
        (_, gen_out, _), _past_kv = out
        if gen_out is not None:
            calls.append(tuple(gen_out.shape))
        return out

    pge.forward = capturing_forward

    # One real step, real-shaped dummy frame + state (this is a shape probe,
    # not a correctness check -- values don't matter, only tensor shapes).
    image = np.zeros((224, 224, 3), dtype=np.uint8)
    proprio = _PoseProxy(p=np.zeros(3), q=np.array([1.0, 0.0, 0.0, 0.0]))
    model.reset("pick up the object")
    model.step(image, "pick up the object", proprio, 0.0)

    print(f"\n{len(calls)} paligemma_with_expert.forward calls produced gen_out:", flush=True)
    for i, shape in enumerate(calls):
        print(f"  call {i}: gen_out shape = {shape}", flush=True)

    total_tokens = sum(s[1] for s in calls)
    print(f"\ntotal tokens across all calls: {total_tokens}", flush=True)
    print("expected from patch_nums (1+4+9+16+25+36+64+100+169+256=680):", 680, flush=True)


if __name__ == "__main__":
    main()
