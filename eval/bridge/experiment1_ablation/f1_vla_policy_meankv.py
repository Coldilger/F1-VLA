"""Closed-loop wrapper for the Experiment 1 mean-foresight control (F1-VLA),
added after the advisor's ablation sanity-check request (2026-10-28).

Why: variant 1 ("remove the module", sample_without_world_model.py) drops
the foresight tokens entirely, so the action expert sees a sequence shape it
never saw in training. Its -13.9 pp could therefore mean "the action expert
needs real foresight" or "it breaks on an unseen shape". mimic-video's
sanity check hit the same confound with zero-ablation and resolved it with
mean-ablation (Heimersheim & Nanda 2024): keep the interface on-distribution
but remove everything specific to the current state. This is that control
for F1.

Mechanism (modeling_f1.py is NOT modified): sample_actions_with_world_model
runs normally -- foresight is generated as usual and fills the shared KV
cache [und tokens | gen tokens: history + VAR scale maps]. Step 5 then calls
model.denoise_step(..., past_key_values, ...) num_steps times; denoise_step
only READS the cache. This wrapper intercepts the first denoise_step call of
each chunk and, per layer, either
  collect: adds the gen slice of key/value states to a running sum (the
           policy behaves exactly like the baseline), or
  mean:    overwrites the gen slice with the stored per-task mean,
so the action expert reads und tokens of the real current observation plus a
gen block with the normal shape, positions and masks but no episode-specific
content. The und length is read from the model's own embed_prefix output and
must be constant across calls (asserted), so mean keys share RoPE positions.
"""

from __future__ import annotations

import sys

import torch

sys.path.insert(0, "/home/3295540/F1-VLA")
sys.path.insert(0, "/home/3295540/F1-VLA/eval/bridge")

from f1_vla_policy import F1VLAInference  # noqa: E402


class F1VLAMeanKVInference(F1VLAInference):
    def __init__(self, *args, meankv_mode: str, meankv_path: str, **kwargs):
        super().__init__(*args, **kwargs)
        assert meankv_mode in ("collect", "mean"), meankv_mode
        self._mode = meankv_mode
        self._path = meankv_path
        self._und_len: int | None = None
        self._fresh_chunk = False
        self._sum = None  # per layer: (key_sum, value_sum), float32
        self._count = 0
        self._mean = None
        if self._mode == "mean":
            blob = torch.load(self._path)
            self._mean = [(k.to(self.device), v.to(self.device)) for k, v in blob["mean"]]
            self._mean_und_len = blob["und_len"]
            print(f"mean foresight KV loaded: {blob['count']} chunks, und_len {blob['und_len']}", flush=True)

        model = self.policy.model
        orig_embed_prefix = model.embed_prefix
        orig_denoise_step = model.denoise_step

        def embed_prefix(*a, **kw):
            out = orig_embed_prefix(*a, **kw)
            n = out[0].shape[1]
            assert self._und_len in (None, n), f"und length changed: {self._und_len} -> {n}"
            self._und_len = n
            self._fresh_chunk = True
            return out

        def denoise_step(state, pad_masks, past_key_values, *a, **kw):
            if self._fresh_chunk:
                self._fresh_chunk = False
                self._transform(past_key_values)
            return orig_denoise_step(state, pad_masks, past_key_values, *a, **kw)

        # instance attributes shadow the class methods; the model class is untouched
        model.embed_prefix = embed_prefix
        model.denoise_step = denoise_step

    @torch.no_grad()
    def _transform(self, past_key_values) -> None:
        u = self._und_len
        layers = [past_key_values[i] for i in sorted(past_key_values)] if isinstance(past_key_values, dict) \
            else list(past_key_values)
        if self._mode == "collect":
            gen = [(l["key_states"][:, u:].float(), l["value_states"][:, u:].float()) for l in layers]
            if self._sum is None:
                self._sum = [(k.clone(), v.clone()) for k, v in gen]
            else:
                assert gen[0][0].shape == self._sum[0][0].shape, "gen block length changed"
                for (ks, vs), (k, v) in zip(self._sum, gen):
                    ks += k
                    vs += v
            self._count += 1
        else:
            assert u == self._mean_und_len, f"und length {u} != collected {self._mean_und_len}"
            for l, (km, vm) in zip(layers, self._mean):
                assert l["key_states"][:, u:].shape[1] == km.shape[1], "gen block length differs from collected mean"
                l["key_states"][:, u:] = km.to(l["key_states"].dtype)
                l["value_states"][:, u:] = vm.to(l["value_states"].dtype)

    def close(self) -> None:
        if self._mode == "collect" and self._count:
            mean = [((k / self._count).cpu(), (v / self._count).cpu()) for k, v in self._sum]
            torch.save({"mean": mean, "count": self._count, "und_len": self._und_len}, self._path)
            print(f"saved mean foresight KV over {self._count} chunks to {self._path}", flush=True)
