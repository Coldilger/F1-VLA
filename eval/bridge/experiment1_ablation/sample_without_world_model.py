"""Experiment 1, variant 1 for F1-VLA: "remove the module" (see ../README.md).

Deliberately does NOT modify f1_vla/src/models/modeling_f1.py -- that file
is shared, working, load-bearing code (main eval pipeline, oracle
experiment, training). This is a standalone function reusing the model's
own existing methods unchanged.

How the ablation works without touching the model: `PaliGemmaWithExpertModel.forward`
(f1_vla/src/models/paligemma_with_expert.py) decides which 3 expert
sub-models exist from the model's CONFIG alone (und + gen + act, since our
checkpoint was trained with use_world_model=True) -- but for any given
forward call, it silently SKIPS whichever `inputs_embeds` entries are `None`,
both when filling the KV cache (paligemma_with_expert.py ~line 220: `if
fill_kv_cache: past_key_values[layer_idx] = {...}` only stores what got
concatenated from non-None entries) and when appending to it. So: fill the
KV cache using ONLY the und (image+language) stream (gen slot = None), and
the gen expert's tokens never enter the sequence at all for this call --
functionally "the module isn't there", without reconstructing the model
with a different config or touching its weights.

denoise_step (modeling_f1.py, unchanged, already correctly gated on
self.config.use_world_model) is reused as-is for the actual action-sampling
loop -- since config.use_world_model is still True (we never changed it),
it builds a 3-slot [None, None, act_embs] list every step; both Nones just
mean "nothing new to add to the cache this step", which is exactly correct
here since gen's slot was never populated in the first place.

Confound (matches the slide's Experiment 1 table for variant 1): the
resulting sequence is shorter than anything the model saw in training (no
gen tokens at all, not even a placeholder), so a change in behavior could
reflect "the module matters" or just "this input shape is unfamiliar".
Variant 2 (shuffle real foresight tokens across episodes -- same sequence
shape, no confound) is not yet implemented.
"""

from __future__ import annotations

import torch
from lerobot.policies.pi0.modeling_pi0 import make_att_2d_masks


@torch.no_grad()
def sample_actions_without_world_model(
    model,
    images,
    image_masks,
    lang_tokens,
    lang_masks,
    state,
    noise=None,
):
    """Mirrors F1FlowMatching.sample_actions_with_world_model's structure
    (modeling_f1.py) but skips steps 1-4 (VAR world-model foresight
    sampling) entirely and goes straight to flow-matching action denoising
    using only the und (image+language) prefix. Returns the denoised action
    tensor (bsize, chunk_size, max_action_dim), matching
    sample_actions_with_world_model(..., predict_action_only=True)'s return.
    """
    bsize = state.shape[0]
    device = state.device

    if noise is None:
        actions_shape = (bsize, model.config.chunk_size, model.config.max_action_dim)
        noise = model.sample_noise(actions_shape, device)

    und_embs, und_pad_masks, und_att_masks = model.embed_prefix(
        images, image_masks, lang_tokens, lang_masks
    )

    pad_masks = und_pad_masks
    att_masks = und_att_masks
    att_2d_masks = make_att_2d_masks(pad_masks, att_masks)
    position_ids = torch.cumsum(pad_masks, dim=1) - 1

    cur_position = und_embs.shape[1]

    # Fill the KV cache with ONLY the und stream. gen slot is None -> the
    # gen expert is never queried, its KV cache entries are never created
    # (see module docstring). This IS the ablation.
    _, past_key_values = model.paligemma_with_expert.forward(
        attention_mask=att_2d_masks[:, :cur_position, :cur_position],
        position_ids=position_ids[:, :cur_position],
        past_key_values=None,
        inputs_embeds=[und_embs, None, None],
        use_cache=model.config.use_cache,
        fill_kv_cache=True,
    )

    dt = -1.0 / model.config.num_steps
    dt = torch.tensor(dt, dtype=torch.float32, device=device)

    x_t = noise
    time = torch.tensor(1.0, dtype=torch.float32, device=device)
    while time >= -dt / 2:
        expanded_time = time.expand(bsize)
        v_t = model.denoise_step(
            state,
            pad_masks,  # und-only prefix mask, shorter than the trained-on sequence
            past_key_values,
            x_t,
            expanded_time,
        )
        x_t = x_t + dt * v_t
        time = time + dt

    return x_t
