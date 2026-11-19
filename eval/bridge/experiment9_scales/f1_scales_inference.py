"""Experiment 9 (compute vs. performance, advisor's point 4) for F1-VLA:
vary how many VAR scales of the forecast are generated at inference, and
time every decision.

F1 generates its forecast coarse-to-fine over the scales
patch_nums = (1, 2, 3, 4, 5, 6, 8, 10, 13, 16) and stops after
`num_resolutions` of them (4 in training and in the deployed policy). The
inputs of scales 2..S are appended to the generation block, so S scales add
sum(p*p for the first S) - 1 forecast positions to the KV cache that the
action expert reads:

    S = 1: 0    S = 2: 4    S = 4: 29 (deployed)    S = 6: 90

The F1 paper (Lv et al. 2025, Table 4) varies the same quantity at
inference (2 / 4 / 6 scales). Changing S also changes where the action
tokens sit: their RoPE position ids start at the number of valid prefix
positions, so they move by (n_forecast(S) - 29). Experiment 1 showed that a
position shift alone (removing the block, ~709 positions) costs ~14 pp, so
an S effect would be ambiguous. POS_MATCH=1 removes that confound: the
action tokens get exactly the position ids they have at S = 4, and only the
amount of forecast in the cache changes.

Env vars: SCALES (int), POS_MATCH (0/1), TIMING_OUT (path of a JSON file
with every decision's wall-clock latency). modeling_f1.py is not modified;
methods are wrapped on the instance.
"""

from __future__ import annotations

import json
import os
import pathlib
import sys
import time

import torch

sys.path.insert(0, "/home/3295540/F1-VLA")
sys.path.insert(0, "/home/3295540/F1-VLA/eval/bridge")

from f1_vla_policy import F1VLAInference  # noqa: E402
from f1_vla.src.models.modeling_f1 import make_att_2d_masks  # noqa: E402

DEPLOYED_SCALES = 4


def n_forecast(patch_nums, scales: int) -> int:
    return sum(p * p for p in patch_nums[:scales]) - patch_nums[0] ** 2


class F1VLAScales(F1VLAInference):
    def __init__(self, *args, scales: int, pos_match: bool, timing_out: str | None, **kwargs):
        super().__init__(*args, **kwargs)
        m = self.policy.model
        assert 1 <= scales <= len(m.patch_nums), scales
        assert m.num_resolutions == DEPLOYED_SCALES, m.num_resolutions
        m.num_resolutions = scales
        self.scales = scales
        self.pos_match = pos_match
        # action tokens' position shift relative to the deployed layout
        self.pos_offset = n_forecast(m.patch_nums, DEPLOYED_SCALES) - n_forecast(m.patch_nums, scales) if pos_match else 0
        self._timing_out = pathlib.Path(timing_out) if timing_out else None
        self._lat: list[float] = []
        self._prefix_lens: set[int] = set()

        # FORCE_COPY=1 installs the copied denoise_step even at offset 0, to
        # check that the copy reproduces the original exactly
        if pos_match and (self.pos_offset != 0 or os.environ.get("FORCE_COPY") == "1"):
            offset = self.pos_offset

            def denoise_step(state, prefix_pad_masks, past_key_values, x_t, timestep):
                # copy of F1FlowMatching.denoise_step with the action tokens'
                # position ids moved by `offset` (masks and cache unchanged)
                act_embs, act_pad_masks, act_att_masks = m.embed_suffix(state, x_t, timestep)
                act_len = act_pad_masks.shape[1]
                prefix_pad_2d_masks = prefix_pad_masks[:, None, :].repeat(1, act_len, 1)
                act_att_2d_masks = make_att_2d_masks(act_pad_masks, act_att_masks)
                full_att_2d_masks = torch.cat([prefix_pad_2d_masks, act_att_2d_masks], dim=2)
                prefix_offsets = torch.sum(prefix_pad_masks, dim=-1)[:, None] + offset
                position_ids = prefix_offsets + torch.cumsum(act_pad_masks, dim=1) - 1
                inputs_embeds = [None, None, act_embs] if m.config.use_world_model else [None, act_embs]
                outputs_embeds, _ = m.paligemma_with_expert.forward(
                    attention_mask=full_att_2d_masks,
                    position_ids=position_ids,
                    past_key_values=past_key_values,
                    inputs_embeds=inputs_embeds,
                    use_cache=m.config.use_cache,
                    fill_kv_cache=False,
                )
                suffix_out = outputs_embeds[-1][:, -m.config.chunk_size:].to(dtype=torch.float32)
                return m.action_out_proj(suffix_out)

            m.denoise_step = denoise_step

        orig_denoise = m.denoise_step

        def record_prefix(state, prefix_pad_masks, *a, **kw):
            self._prefix_lens.add(int(prefix_pad_masks.shape[1]))
            return orig_denoise(state, prefix_pad_masks, *a, **kw)

        m.denoise_step = record_prefix

    def _predict_new_chunk(self, image, task_description):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        out = super()._predict_new_chunk(image, task_description)
        torch.cuda.synchronize()
        self._lat.append(time.perf_counter() - t0)
        return out

    def close(self):
        if self._timing_out is None:
            return
        self._timing_out.parent.mkdir(parents=True, exist_ok=True)
        self._timing_out.write_text(json.dumps({
            "scales": self.scales, "pos_match": self.pos_match, "pos_offset": self.pos_offset,
            "force_copy": os.environ.get("FORCE_COPY") == "1",
            "prefix_lens": sorted(self._prefix_lens),
            "gpu": torch.cuda.get_device_name(0), "latency_s": self._lat,
        }))
