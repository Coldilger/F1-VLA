"""Experiment 8 (advisor's point 7), F1-VLA: instruction-foresight conflict,
steps 0-1 (strength check and actions at identical states).

Extends the cue-conflict design of Geirhos et al. (2019) and interchange
interventions (Geiger et al., 2021) to a policy's forecast. The instruction
reaches F1's action expert by two routes: the language tokens in the
understanding ("und") block of the shared KV cache, and the generation
("gen") block, whose tokens attend to the language while the forecast is
produced. This probe separates the two.

The robot drives a NORMAL baseline rollout under the task's own instruction
A (unchanged policy call, own RNG), so the states are the ones the policy
really visits. At every replan, for instruction A and each alternative B
(INSTRUCTIONS below), the shared KV cache is captured at the action
expert's first denoising step with the SAME world-model sampling seed, and
the action chunk is denoised again from the SAME start noise under:

  AA   und(A) + gen(A)                    normal
  AA2  und(A) + gen(A, other sampling seed)   noise floor: same instruction
  BB   und(B) + gen(B)                    normal under the other instruction
  AB   und(A) + gen(B)                    conflict: forecast from B
  BA   und(B) + gen(A)                    conflict: forecast from A
  ABf  und(A) + gen history(A) + future tokens(B)   only the forecast swapped

Layout of the gen block (modeling_f1.py, embed_wm_inputs and the sampling
loop): [sos + 679 history-frame tokens | 29 forecast positions]. The
forecast positions hold the inputs of scales 2-4, which are built from the
tokens sampled for scales 1-3; the tokens sampled for scale 4 are never fed
back. `seed_pos_diff_v` (per-position difference between two sampling
seeds) is stored so this layout can be checked from the data.

Recorded per replan and alternative: the sampled forecast token ids (30
tokens), how much the gen block's keys/values differ between A and B and
between two seeds under A (history and future parts), and every action
chunk. Language is padded to a fixed length, but position ids count only
real tokens, so a B with a different token count shifts the gen block's
positions: `same_len` marks the pairs free of that layout confound.

modeling_f1.py is not modified; methods are wrapped on the instance.
"""

from __future__ import annotations

import pathlib
import sys

import numpy as np
import torch

sys.path.insert(0, "/home/3295540/F1-VLA")
sys.path.insert(0, "/home/3295540/F1-VLA/eval/bridge")

from f1_vla_policy import F1VLAInference  # noqa: E402

CARROT = "put carrot on plate"
SPOON = "put the spoon on the towel"
STACK = "stack the green block on the yellow block"
EGGPLANT = "put eggplant into yellow basket"
# alternative instructions per task: (name, text). "mirror" swaps the two
# objects of the scene's own instruction (same tokens, both objects present);
# the others are the instructions of the other three tasks (objects absent).
INSTRUCTIONS = {
    "PutCarrotOnPlateInScene-v0": [("mirror", "put plate on carrot"), ("spoon", SPOON), ("stack", STACK), ("eggplant", EGGPLANT)],
    "PutSpoonOnTableClothInScene-v0": [("mirror", "put the towel on the spoon"), ("carrot", CARROT), ("stack", STACK), ("eggplant", EGGPLANT)],
    "StackGreenCubeOnYellowCubeBakedTexInScene-v0": [("mirror", "stack the yellow block on the green block"), ("carrot", CARROT), ("spoon", SPOON), ("eggplant", EGGPLANT)],
    "PutEggplantInBasketScene-v0": [("mirror", "put yellow basket into eggplant"), ("carrot", CARROT), ("spoon", SPOON), ("stack", STACK)],
}
WM_SEED, WM_SEED_2 = 111, 222


def _clone_kv(kv):
    return {i: {"key_states": l["key_states"].clone(), "value_states": l["value_states"].clone()} for i, l in kv.items()}


class F1VLAConflictInference(F1VLAInference):
    def __init__(self, *args, out_dir: str, task: str, **kwargs):
        super().__init__(*args, **kwargs)
        self._out = pathlib.Path(out_dir)
        self._out.mkdir(parents=True, exist_ok=True)
        self._alts = INSTRUCTIONS[task]
        self._noise_gen = torch.Generator(device=self.device).manual_seed(1234)
        self._episode = -1
        self._records: list[dict] = []

        model = self.policy.model
        self._und_len = None
        self._arm = False
        self._capture = None
        self._tokens: list[torch.Tensor] | None = None
        orig_embed_prefix = model.embed_prefix
        orig_denoise = model.denoise_step
        self._orig_denoise = orig_denoise

        def embed_prefix(*a, **kw):
            out = orig_embed_prefix(*a, **kw)
            self._und_len = out[0].shape[1]
            self._n_real_und = int(out[1].sum())  # real (non-pad) und tokens
            return out

        def denoise_step(state, prefix_pad_masks, past_key_values, x_t, timestep):
            if self._arm:
                self._arm = False
                self._capture = {"state": state.clone(), "pad": prefix_pad_masks.clone(),
                                 "kv": _clone_kv(past_key_values), "und_len": self._und_len,
                                 "n_real_und": self._n_real_und}
            return orig_denoise(state, prefix_pad_masks, past_key_values, x_t, timestep)

        def record_tokens(_module, inputs):
            if self._tokens is not None:
                self._tokens.append(inputs[0].detach().reshape(-1).cpu())

        model.embed_prefix = embed_prefix
        model.denoise_step = denoise_step
        model.vae.quantize.embedding.register_forward_pre_hook(record_tokens)

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
    def _run(self, batch, instruction, wm_seed, noise):
        """One full policy call under `instruction`; returns the captured cache
        and the forecast token ids sampled for the generated scales."""
        m = self.policy.model
        b = dict(batch)
        b["task"] = [instruction]
        self.policy.reset()  # empty action queue so the call really runs
        self._arm, self._tokens = True, []
        rng = torch.Generator(device=self.device).manual_seed(wm_seed)
        self._last_actions = self.policy.select_action_with_world_model(b, rng=rng, noise=noise.clone())
        n_scales = m.num_resolutions
        sizes = [p * p for p in m.patch_nums[:n_scales]]
        toks = self._tokens[-n_scales:]
        assert [t.numel() for t in toks] == sizes, ([t.numel() for t in self._tokens], sizes)
        self._tokens = None
        return self._capture, torch.cat(toks).numpy()

    @torch.no_grad()
    def _decode(self, state, pad, kv, noise):
        m = self.policy.model
        dt = torch.tensor(-1.0 / m.config.num_steps, device=self.device)
        x_t = noise.clone()
        t = torch.tensor(1.0, device=self.device)
        while t >= -dt / 2:
            v_t = self._orig_denoise(state, pad, kv, x_t, t.expand(state.shape[0]))
            x_t += dt * v_t
            t += dt
        return x_t[0, :, :7].float().cpu().numpy()

    @staticmethod
    def _mix(und_src, gen_src, u, future_from=None, h=None):
        """KV cache with the und block of `und_src` and the gen block of
        `gen_src`; if `future_from` is given, only the future tokens (after
        the h history tokens) come from it."""
        out = _clone_kv(und_src)
        for i in out:
            for key in ("key_states", "value_states"):
                out[i][key][:, u:] = gen_src[i][key][:, u:]
                if future_from is not None:
                    out[i][key][:, u + h:] = future_from[i][key][:, u + h:]
        return out

    @staticmethod
    def _kv_diff(a, b, sl):
        res = {}
        for key in ("key_states", "value_states"):
            d = [float((b[i][key][0, sl].float() - a[i][key][0, sl].float()).norm()
                       / (a[i][key][0, sl].float().norm() + 1e-8)) for i in sorted(a)]
            res[key[0]] = float(np.mean(d))
        return res

    @torch.no_grad()
    def _predict_new_chunk(self, image, task_description):
        batch = self._batch(image, task_description)
        # 1. driving action: the unchanged baseline call, own RNG
        actions = self.policy.select_action_with_world_model(batch, rng=self._rng)
        actions = actions[0].cpu()
        driving = (actions * self.action_std + self.action_mean).numpy()

        m = self.policy.model
        shape = (1, m.config.chunk_size, m.config.max_action_dim)
        noise = torch.randn(shape, generator=self._noise_gen, device=self.device)

        cap_a, tok_a = self._run(batch, task_description, WM_SEED, noise)
        cap_a2, tok_a2 = self._run(batch, task_description, WM_SEED_2, noise)
        u = int(cap_a["und_len"])
        prefix = int(cap_a["pad"].shape[1])
        n_future = sum(p * p for p in m.patch_nums[:m.num_resolutions]) - m.patch_nums[0] ** 2
        h = prefix - u - n_future
        assert h > 0, (prefix, u, n_future)
        hist, fut = slice(u, u + h), slice(u + h, prefix)
        state, pad_a = cap_a["state"], cap_a["pad"]

        # layout check: only positions that depend on the sampled forecast can
        # differ between two sampling seeds under the same instruction
        pos_diff = torch.stack([(cap_a2["kv"][i]["value_states"][0].float() - cap_a["kv"][i]["value_states"][0].float()).norm(dim=-1)
                                / (cap_a["kv"][i]["value_states"][0].float().norm(dim=-1) + 1e-8) for i in sorted(cap_a["kv"])]).mean(0)
        rec = {"seed_pos_diff_v": pos_diff.flatten().cpu().numpy().astype(np.float32),
               "global_step": self._step_count, "instruction": task_description, "und_len": u, "hist_len": h,
               "n_future": n_future, "n_real_und": cap_a["n_real_und"], "tokens_A": tok_a, "tokens_A2": tok_a2,
               "action_AA": self._decode(state, pad_a, cap_a["kv"], noise),
               "action_AA2": self._decode(state, pad_a, self._mix(cap_a["kv"], cap_a2["kv"], u), noise),
               "seed_kv_diff_hist": self._kv_diff(cap_a["kv"], cap_a2["kv"], hist),
               "seed_kv_diff_future": self._kv_diff(cap_a["kv"], cap_a2["kv"], fut),
               "seed_kv_diff_und": self._kv_diff(cap_a["kv"], cap_a2["kv"], slice(0, u)),
               "alts": {}}
        for name, text in self._alts:
            cap_b, tok_b = self._run(batch, text, WM_SEED, noise)
            assert int(cap_b["und_len"]) == u and cap_b["pad"].shape == pad_a.shape
            rec["alts"][name] = {
                "instruction": text,
                "n_real_und": cap_b["n_real_und"],
                "same_len": cap_b["n_real_und"] == cap_a["n_real_und"],
                "tokens_B": tok_b,
                "kv_diff_hist": self._kv_diff(cap_a["kv"], cap_b["kv"], hist),
                "kv_diff_future": self._kv_diff(cap_a["kv"], cap_b["kv"], fut),
                "action_BB": self._decode(state, cap_b["pad"], cap_b["kv"], noise),
                "action_AB": self._decode(state, pad_a, self._mix(cap_a["kv"], cap_b["kv"], u), noise),
                "action_BA": self._decode(state, cap_b["pad"], self._mix(cap_b["kv"], cap_a["kv"], u), noise),
                "action_ABf": self._decode(state, pad_a, self._mix(cap_a["kv"], cap_a["kv"], u, future_from=cap_b["kv"], h=h), noise),
            }
        self._records.append(rec)
        self.policy.reset()  # the probe calls must not leave actions queued
        return driving

    def reset(self, task_description):
        self._flush()
        super().reset(task_description)
        self._episode += 1

    def _flush(self):
        if self._episode >= 0 and self._records:
            np.savez_compressed(self._out / f"episode_{self._episode:03d}.npz",
                                records=np.array(self._records, dtype=object))
            self._records = []

    def close(self):
        self._flush()


class F1VLAConflictClosedLoop(F1VLAConflictInference):
    """Step 2: the robot is driven by a conflict cell. At every replanning
    decision the understanding block comes from a run under instruction
    `und` and the generation block from a run under instruction `gen`
    ("A" = the scene's own instruction, "B" = INSTRUCTIONS[task][alt]); both
    runs share the forecast sampling seed and the action start noise.

      und=A gen=A   baseline through this decode path (control)
      und=B gen=B   normal policy under B
      und=A gen=B   conflict: forecast from B
      und=B gen=A   conflict: forecast from A
    """

    def __init__(self, *args, und: str, gen: str, alt: str = "mirror", **kwargs):
        super().__init__(*args, **kwargs)
        assert und in "AB" and gen in "AB"
        self._und, self._gen = und, gen
        self._text_b = dict(self._alts)[alt]
        seed = kwargs.get("seed") or 0
        self._seed_gen = torch.Generator().manual_seed(seed)
        self._noise_gen = torch.Generator(device=self.device).manual_seed(1234 + seed)
        self._check_max = 0.0

    @torch.no_grad()
    def _predict_new_chunk(self, image, task_description):
        batch = self._batch(image, task_description)
        m = self.policy.model
        noise = torch.randn((1, m.config.chunk_size, m.config.max_action_dim), generator=self._noise_gen, device=self.device)
        wm_seed = int(torch.randint(0, 2**31 - 1, (1,), generator=self._seed_gen))
        text = {"A": task_description, "B": self._text_b}
        cap_u, _ = self._run(batch, text[self._und], wm_seed, noise)
        # sanity: our re-decoding of the unmixed cache must reproduce the
        # policy's own output for the same seed and noise
        own = self._last_actions[0, :, :7].float().cpu().numpy()
        redo = self._decode(cap_u["state"], cap_u["pad"], cap_u["kv"], noise)
        self._check_max = max(self._check_max, float(np.abs(own - redo).max()))
        if self._gen == self._und:
            chunk = redo
        else:
            cap_g, _ = self._run(batch, text[self._gen], wm_seed, noise)
            u = int(cap_u["und_len"])
            assert int(cap_g["und_len"]) == u and cap_g["pad"].shape == cap_u["pad"].shape
            chunk = self._decode(cap_u["state"], cap_u["pad"], self._mix(cap_u["kv"], cap_g["kv"], u), noise)
        self.policy.reset()
        return (torch.from_numpy(chunk) * self.action_std + self.action_mean).numpy()

    def _report(self):
        if self._episode >= 0:
            print(f"[conflict-cl] episode {self._episode} und={self._und} gen={self._gen} "
                  f"B={self._text_b!r} redecode_max_abs_diff={self._check_max:.2e}", flush=True)

    def reset(self, task_description):
        self._report()
        super().reset(task_description)

    def close(self):
        self._report()
        super().close()
