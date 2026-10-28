"""Experiment 1 internal diagnostics for F1-VLA (advisor review, 2026-10-28):
for each ablation, check shapes, masks, attention, activation values, and
whether the action branch still runs correctly -- the F1 counterpart of
mimic-video's ablation_sanity/.

The robot drives a NORMAL baseline rollout: the driving action comes from
the unchanged policy.select_action_with_world_model call with the policy's
own RNG, exactly as in ../../main_inference.py. While it runs, the shared KV
cache [und tokens | gen tokens], the prefix pad mask and the state are
captured at the action expert's first denoising step (model.denoise_step
only reads the cache, see modeling_f1.py). Then, with the SAME start noise
for every condition (a separate generator, so the driving RNG stream is
untouched), the action chunk is denoised again under:

  real     captured cache as is
  shuffle  foresight tokens taken from a random real Bridge frame
           (oracle_indices, exactly Exp1 variant 2), cache re-built
  mean     gen block replaced by its per-task mean
           (../closed_loop/meankv/<TASK>.pt, the mean-foresight control)
  removed  cache and pad mask cut to the und block (Exp1 variant 1's
           sequence: sample_without_world_model.py fills only und)

and records per condition: block lengths and how many prefix positions the
action tokens are allowed to see (from the mask the model itself builds),
the action tokens' RoPE position ids, per-layer attention mass of the action
queries on und / gen-history / gen-future / action keys, how much each
condition's gen block differs from the real one (history and future parts), gen-block key/value statistics, and the
decoded actions. On each episode's first replan it also runs an in-situ mask
test: perturb single gen positions of the cache and check that one
denoising step's output changes; control: the same perturbation with that
position switched off in the model's own pad mask must change nothing.
modeling_f1.py and paligemma_with_expert.py are not modified; their methods
are wrapped on the instance.
"""

from __future__ import annotations

import os
import pathlib
import sys

import numpy as np
import torch
from torch import nn

sys.path.insert(0, "/home/3295540/F1-VLA")
sys.path.insert(0, "/home/3295540/F1-VLA/eval/bridge")
sys.path.insert(0, "/home/3295540/F1-VLA/eval/bridge/experiment1_ablation")

from f1_vla_policy import F1VLAInference  # noqa: E402
from f1_vla_policy_shuffled import _load_frame_pool  # noqa: E402

CONDITIONS = ("real", "shuffle", "mean", "removed")
MEANKV_DIR = pathlib.Path("/home/3295540/F1-VLA/eval/bridge/experiment1_ablation/closed_loop/meankv")


def _clone_kv(kv):
    return {i: {"key_states": l["key_states"].clone(), "value_states": l["value_states"].clone()} for i, l in kv.items()}


class F1VLAInternalsInference(F1VLAInference):
    def __init__(self, *args, out_dir: str, task: str,
                 bridge_dataset_root: str = "/mnt/beegfsnew/scratch/3295540/data/bridge_orig_lerobot", **kwargs):
        super().__init__(*args, **kwargs)
        self._out = pathlib.Path(out_dir)
        self._out.mkdir(parents=True, exist_ok=True)
        blob = torch.load(MEANKV_DIR / f"{task}.pt")
        self._mean = [(k.to(self.device), v.to(self.device)) for k, v in blob["mean"]]
        self._pool = _load_frame_pool(pathlib.Path(bridge_dataset_root), pool_size=64)
        self._pool_rng = np.random.default_rng(7)
        self._diag_gen = torch.Generator(device=self.device).manual_seed(1234)
        self._diag_rng = torch.Generator(device=self.device).manual_seed(4321)
        self._episode = -1
        self._records: list[dict] = []
        self._mask_test_done = False

        model = self.policy.model
        pwe = model.paligemma_with_expert
        self._und_len = None
        self._capture = None  # dict filled at first denoise_step when armed
        self._arm = False
        self._denoising = False
        self._attn = None  # list of per-layer [und, gen, act] mass when recording

        orig_embed_prefix = model.embed_prefix
        orig_denoise = model.denoise_step
        orig_eager = pwe.eager_attention_forward
        self._orig_denoise = orig_denoise

        def embed_prefix(*a, **kw):
            out = orig_embed_prefix(*a, **kw)
            self._und_len = out[0].shape[1]
            return out

        def denoise_step(state, prefix_pad_masks, past_key_values, x_t, timestep):
            if self._arm:
                self._arm = False
                self._capture = {"state": state.clone(), "pad": prefix_pad_masks.clone(),
                                 "kv": _clone_kv(past_key_values), "und_len": self._und_len}
            self._denoising = True
            try:
                return orig_denoise(state, prefix_pad_masks, past_key_values, x_t, timestep)
            finally:
                self._denoising = False

        def eager(attention_mask, batch_size, head_dim, q, k, v):
            out = orig_eager(attention_mask, batch_size, head_dim, q, k, v)
            if self._attn is not None and self._denoising:
                self._attn.append(self._mass(attention_mask, head_dim, q, k))
            return out

        model.embed_prefix = embed_prefix
        model.denoise_step = denoise_step
        pwe.eager_attention_forward = eager

    # --- attention mass of action queries on und / gen / act keys ---
    def _mass(self, attention_mask, head_dim, q, k):
        groups = q.shape[2] // k.shape[2]
        k = k.repeat_interleave(groups, dim=2)
        w = torch.einsum("bshd,bnhd->bhsn", q.float(), k.float()) * head_dim ** -0.5
        w = w.masked_fill(~attention_mask[:, None], float("-inf")).softmax(-1)  # [B,H,S_act,N]
        n_prefix = attention_mask.shape[-1] - q.shape[1]
        u, h = self._cur_und, self._cur_hist
        und = w[..., :u].sum(-1).mean().item()
        hist = w[..., u:min(u + h, n_prefix)].sum(-1).mean().item()
        fut = w[..., min(u + h, n_prefix):n_prefix].sum(-1).mean().item()
        act = w[..., n_prefix:].sum(-1).mean().item()
        return (und, hist, fut, act)

    @torch.no_grad()
    def _decode(self, state, pad, kv, noise, record=True):
        m = self.policy.model
        self._cur_und = int(self._capture["und_len"])
        self._attn = [] if record else None
        dt = torch.tensor(-1.0 / m.config.num_steps, device=self.device)
        x_t = noise.clone()
        t = torch.tensor(1.0, device=self.device)
        while t >= -dt / 2:
            self._denoising = True
            v_t = self._orig_denoise(state, pad, kv, x_t, t.expand(state.shape[0]))
            self._denoising = False
            x_t += dt * v_t
            t += dt
        layers = np.array(self._attn) if record else None  # [steps*layers, 4]
        self._attn = None
        return x_t, layers

    def _batch(self, image, task_description):
        main_img = self._preprocess_main_image(image).unsqueeze(0).to(self.device)
        hist = list(self.image_history)
        if len(hist) < self.n_obs_img_steps:
            hist = [hist[0]] * (self.n_obs_img_steps - len(hist)) + hist
        return {
            "observation.images.image0": main_img,
            "observation.images.image0_mask": torch.tensor([True], device=self.device),
            "observation.images.image0_history": torch.stack(hist).unsqueeze(0).to(self.device),
            "observation.state": self._current_state.unsqueeze(0).to(self.device),
            "task": [task_description],
        }

    @torch.no_grad()
    def _predict_new_chunk(self, image, task_description):
        batch = self._batch(image, task_description)
        # 1. driving action: the unchanged baseline call, own RNG
        self._arm = True
        actions = self.policy.select_action_with_world_model(batch, rng=self._rng)
        real_cap = self._capture
        m = self.policy.model
        u = int(real_cap["und_len"])
        n_layers = len(real_cap["kv"])
        # gen block = [encoded history | future tokens]; the future tokens are
        # the VAR scale maps appended for every scale after the first
        self._cur_hist = int(real_cap["pad"].shape[1]) - u - (sum(p * p for p in m.patch_nums) - m.patch_nums[0] ** 2)
        assert self._cur_hist > 0, self._cur_hist
        shape = (1, m.config.chunk_size, m.config.max_action_dim)
        noise = torch.randn(shape, generator=self._diag_gen, device=self.device)

        # 2. condition caches
        caches = {"real": (real_cap["pad"], real_cap["kv"])}
        mean_kv = _clone_kv(real_cap["kv"])
        for i, (km, vm) in zip(sorted(mean_kv), self._mean):
            mean_kv[i]["key_states"][:, u:] = km.to(mean_kv[i]["key_states"].dtype)
            mean_kv[i]["value_states"][:, u:] = vm.to(mean_kv[i]["value_states"].dtype)
        caches["mean"] = (real_cap["pad"], mean_kv)
        removed_kv = {i: {"key_states": l["key_states"][:, :u].clone(), "value_states": l["value_states"][:, :u].clone()}
                      for i, l in real_cap["kv"].items()}
        caches["removed"] = (real_cap["pad"][:, :u].clone(), removed_kv)
        frame = self._pool[self._pool_rng.integers(len(self._pool))]
        oracle = m.vae.img_to_idxBl(self._preprocess_history_image(frame).unsqueeze(0).to(self.device))
        self._arm = True
        self.policy.select_action_with_world_model(batch, rng=self._diag_rng, oracle_indices=oracle, noise=noise.clone())
        caches["shuffle"] = (self._capture["pad"], self._capture["kv"])

        rec = {"global_step": self._step_count, "und_len": u, "hist_len": self._cur_hist, "conditions": {}}
        h = self._cur_hist
        real_action = None
        for name in CONDITIONS:
            pad, kv = caches[name]
            x, mass = self._decode(real_cap["state"], pad, kv, noise)
            a = x[0, :, :7].float().cpu().numpy()
            if name == "real":
                real_action = a
            prefix = int(pad.shape[1])
            allowed = pad[0].bool()
            gk = torch.stack([kv[i]["key_states"][0, u:].float().norm(dim=-1).mean() for i in sorted(kv)]) if prefix > u else None
            gv = torch.stack([kv[i]["value_states"][0, u:].float().norm(dim=-1).mean() for i in sorted(kv)]) if prefix > u else None
            mass = mass.reshape(-1, n_layers, 4).mean(0)  # average over denoise steps -> [layers, 4]
            # how different this condition's gen block actually is from the real one
            kv_diff = {}
            if prefix > u:
                real_kv = caches["real"][1]
                for part, sl in (("hist", slice(u, u + h)), ("future", slice(u + h, prefix))):
                    for key in ("key_states", "value_states"):
                        d = [float((kv[i][key][0, sl].float() - real_kv[i][key][0, sl].float()).norm()
                                   / (real_kv[i][key][0, sl].float().norm() + 1e-8)) for i in sorted(kv)]
                        kv_diff[f"{part}_{key[0]}"] = float(np.mean(d))
            rec["conditions"][name] = {
                "prefix_len": prefix,
                "gen_len": prefix - u,
                "allowed_und": int(allowed[:u].sum()),
                "allowed_gen": int(allowed[u:].sum()) if prefix > u else 0,
                "act_first_position_id": int(allowed.sum()),  # prefix_offsets in denoise_step
                "attn_mass_und_hist_future_act_per_layer": mass,
                "gen_rel_diff_to_real": kv_diff,
                "gen_key_norm_per_layer": None if gk is None else gk.cpu().numpy(),
                "gen_value_norm_per_layer": None if gv is None else gv.cpu().numpy(),
                "action": a,
                "action_nan": int(np.isnan(a).sum()),
            }
        for c in rec["conditions"].values():
            c["l2_to_real"] = float(np.linalg.norm(c["action"] - real_action))
        if not self._mask_test_done:
            rec["mask_test"] = self._mask_test(real_cap, noise)
            self._mask_test_done = True
        self._records.append(rec)

        actions = actions[0].cpu()
        return (actions * self.action_std + self.action_mean).numpy()

    @torch.no_grad()
    def _mask_test(self, cap, noise):
        """One denoising step's output change when single gen cache positions
        are replaced by noise -- allowed (real mask) vs switched off in the
        model's own pad mask (must be exactly 0)."""
        u, pad, kv, state = int(cap["und_len"]), cap["pad"], cap["kv"], cap["state"]
        t = torch.ones(1, device=self.device)
        base = self._orig_denoise(state, pad, kv, noise, t)
        n = pad.shape[1]
        probe = sorted({u, u + (n - u) // 4, u + (n - u) // 2, u + 3 * (n - u) // 4, n - 1})
        out = {"probe_positions": probe, "allowed_change": [], "masked_change": []}
        for j in probe:
            kv2 = _clone_kv(kv)
            for l in kv2.values():
                for key in ("key_states", "value_states"):
                    # own generator: torch.randn_like would draw from the GLOBAL torch RNG,
                    # which the driving policy uses for its flow-matching start noise
                    z = torch.randn(l[key][:, j].shape, generator=self._diag_gen, device=self.device)
                    l[key][:, j] = (z * l[key].float().std() * 3).to(l[key].dtype)
            o = self._orig_denoise(state, pad, kv2, noise, t)
            out["allowed_change"].append(float((o - base).norm() / base.norm()))
            pad_off = pad.clone()
            pad_off[:, j] = False
            b_off = self._orig_denoise(state, pad_off, kv, noise, t)
            o_off = self._orig_denoise(state, pad_off, kv2, noise, t)
            out["masked_change"].append(float((o_off - b_off).norm() / b_off.norm()))
        return out

    def reset(self, task_description):
        self._flush()
        super().reset(task_description)
        self._episode += 1
        self._mask_test_done = False

    def _flush(self):
        if self._episode >= 0 and self._records:
            np.savez_compressed(self._out / f"episode_{self._episode:03d}.npz",
                                records=np.array(self._records, dtype=object))
            self._records = []

    def close(self):
        self._flush()
