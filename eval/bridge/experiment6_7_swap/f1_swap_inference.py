"""Experiments 6 and 7 (advisor's points 1 and 6), F1-VLA, closed loop.

Experiment 6 (graded foresight replacement) and Experiment 7
(horizon-shifted foresight) replace the forecast the action expert reads
with another one, in the closed loop: the robot executes the actions
decoded with the replaced forecast. Method: interchange interventions
(Geiger et al., 2021), extending Experiment 1's shuffle into a scale of "how
wrong" and Experiment 2's oracle into a scale of "which moment".

What is replaced. F1-VLA's forecast is a next-scale (VAR) token map of the
future frame; the action expert reads the generation block's 29 forecast
positions, which are built from the tokens of scales 1-3 (14 tokens; the
16 tokens of scale 4 are never fed back, Experiment 8). A replacement is a
list of 30 token ids passed as `oracle_indices` to
`sample_actions_with_world_model` (modeling_f1.py), which then builds the
29 positions from them under the current observation and instruction,
exactly as it does for its own sampled tokens. Sources (SWAP):

  bank:same_task     own forecast at the same decision index of another
                     episode of the same task (episode e+1 mod 24)
  bank:other_task    same table (bridge_table_1_v1), another task, same
                     episode index: Carrot -> Spoon -> Stack -> Carrot
                     (not defined for Eggplant, the only sink-scene task)
  bank:other_scene   the other scene: table tasks <- Eggplant (sink),
                     Eggplant <- Carrot; same episode and decision index
  own:<d>            this episode's own forecast from decision t+d
                     (d < 0: computed at earlier decisions of this
                     closed-loop rollout; d > 0: from a lookahead)
  real:<h>           the real camera frame at control step t+h, tokenised
                     by F1's own VAE (img_to_idxBl). h <= 0: frames already
                     seen; h > 0: from a lookahead. h = +3 is the horizon
                     F1 was trained to forecast (+3 frames at 5 fps).
  control            the decision's own sampled tokens fed back through
                     oracle_indices (must reproduce the normal action)

Bank donors come from `bank_tokens.npz` (`build_bank.py`: Experiment 8's
per-decision forecasts on normal rollouts, seed 0). Bank donors are time
matched (same decision index, clamped to the donor episode's length), so
the scale is about scene and task, not about phase of the episode.

Lookahead (own:+d, real:+h). The future does not exist yet in the closed
loop, so it is simulated: a second environment of the same episode is reset
with the same options and replays every action executed so far; its full
simulator state is compared with the main environment's (`replay_max_diff`
in the records; SAPIEN is deterministic on CPU). From there the normal
policy (own forecast, same wrapper state, RNG states saved and restored) is
run for h control steps; its frame at t+h, or the tokens it samples at
decision t+d, are the donor. Nothing of the lookahead reaches the main
environment.

Recorded per decision (`records/<cond>/<task>/seed<S>/episode_NNN.npz`):
the action decoded with the own forecast and with the donor from the same
action start noise (so `d(donor, own)` is the single-decision effect), a
second own forecast from another sampling seed (noise floor), the donor and
own token ids, the relative change of the 29 forecast positions' keys and
values (strength of the intervention), the replay check and timing.
"""

from __future__ import annotations

import pathlib
import sys
import time

import numpy as np
import torch

sys.path.insert(0, "/home/3295540/F1-VLA")
sys.path.insert(0, "/home/3295540/F1-VLA/eval/bridge")
sys.path.insert(0, "/home/3295540/F1-VLA/eval/bridge/SimplerEnv")

from f1_vla_policy import F1VLAInference  # noqa: E402
from simpler_env.utils.env.observation_utils import get_image_from_maniskill2_obs_dict  # noqa: E402

HERE = pathlib.Path(__file__).resolve().parent
TASKS = ["PutCarrotOnPlateInScene-v0", "PutSpoonOnTableClothInScene-v0",
         "StackGreenCubeOnYellowCubeBakedTexInScene-v0", "PutEggplantInBasketScene-v0"]
OTHER_TASK = {TASKS[0]: TASKS[1], TASKS[1]: TASKS[2], TASKS[2]: TASKS[0]}
OTHER_SCENE = {TASKS[0]: TASKS[3], TASKS[1]: TASKS[3], TASKS[2]: TASKS[3], TASKS[3]: TASKS[0]}
SCALE_SIZES = [1, 4, 9, 16]
N_FUTURE = 29
CTRL = "arm_pd_ee_target_delta_pose_align2_gripper_pd_joint_pos"


def proprio(env):
    c = env.agent.controllers[CTRL]
    grip = (c.controllers["gripper"].qpos.mean() - 0.015) / (0.037 - 0.015)
    return c.controllers["arm"].ee_pose_at_base, grip


class EnvTap:
    """Captures the evaluator's environment, its build kwargs, reset options
    and executed actions, and provides the lookahead (a second environment of
    the same episode that replays the executed actions)."""

    def __init__(self):
        self.env = None
        self.build = None
        self.reset_options = None
        self.actions: list[np.ndarray] = []
        self._replay = None

    def wrap_builder(self, builder):
        def build(env_name, **kwargs):
            env = builder(env_name, **kwargs)
            self._close_replay()
            self.env, self.build = env, (builder, env_name, dict(kwargs))
            orig_reset, orig_step = env.reset, env.step

            def reset(*a, **kw):
                self.reset_options = kw.get("options")
                self.actions = []
                return orig_reset(*a, **kw)

            def step(action):
                self.actions.append(np.array(action, copy=True))
                return orig_step(action)

            env.reset, env.step = reset, step
            return env
        return build

    def _close_replay(self):
        if self._replay is not None:
            self._replay.close()
            self._replay = None

    def replay(self):
        """Second env at the main env's current state; returns (env, obs,
        max |state difference| against the main env).

        The replay env is built fresh for every lookahead. Reusing one env
        across resets (the runs before 2026-11-25) left it in a different
        state than the main env at most Eggplant and Spoon decisions after
        the first; a freshly built env was always exact (decision 0)."""
        self._close_replay()
        builder, name, kwargs = self.build
        self._replay = builder(name, **kwargs)
        env = self._replay
        obs, _ = env.reset(options=self.reset_options)
        for a in self.actions:
            obs, *_ = env.step(a)
        diff = float(np.abs(env.unwrapped.get_state() - self.env.unwrapped.get_state()).max())
        return env, obs, diff


class F1VLASwapInference(F1VLAInference):
    def __init__(self, *args, swap: str, task: str, tap: EnvTap, out_dir: str, **kwargs):
        super().__init__(*args, **kwargs)
        self._kind, _, arg = swap.partition(":")
        assert self._kind in ("bank", "own", "real", "control"), swap
        self._arg = arg
        self._shift = int(arg) if self._kind in ("own", "real") else 0
        self._task, self._tap = task, tap
        self._out = pathlib.Path(out_dir)
        self._out.mkdir(parents=True, exist_ok=True)
        if self._kind == "bank":
            donor = {"same_task": task, "other_task": OTHER_TASK.get(task), "other_scene": OTHER_SCENE[task]}[arg]
            assert donor is not None, f"{swap} is not defined for {task}"
            bank = np.load(HERE / "bank_tokens.npz")
            self._donor_task = donor
            self._bank_tok, self._bank_n = bank[f"{donor}/tokens"], bank[f"{donor}/n"]
        seed = kwargs.get("seed") or 0
        self._noise_gen = torch.Generator(device=self.device).manual_seed(1234 + seed)
        self._seed_gen = torch.Generator().manual_seed(seed)
        self._episode = -1
        self._episode_id = None  # SimplerEnv episode id (layout), from the reset options
        self._records: list[dict] = []
        self._lookahead = False
        self._la_capture_at = None  # lookahead step at which to keep the sampled tokens
        self._la_tokens = None
        self._tokens: list[torch.Tensor] | None = None
        self._kv_cap = None
        self._arm_kv = False

        m = self.policy.model
        orig_denoise = m.denoise_step

        def denoise_step(state, prefix_pad_masks, past_key_values, x_t, timestep):
            if self._arm_kv:
                self._arm_kv = False
                self._kv_cap = {i: (l["key_states"][0, -N_FUTURE:].float().clone(), l["value_states"][0, -N_FUTURE:].float().clone())
                                for i, l in past_key_values.items()}
            return orig_denoise(state, prefix_pad_masks, past_key_values, x_t, timestep)

        def record_tokens(_module, inputs):
            if self._tokens is not None:
                self._tokens.append(inputs[0].detach().reshape(-1).cpu())

        m.denoise_step = denoise_step
        m.vae.quantize.embedding.register_forward_pre_hook(record_tokens)

    # ------------------------------------------------------------------ utils
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

    @staticmethod
    def _split(tokens: np.ndarray):
        out, i = [], 0
        for s in SCALE_SIZES:
            out.append(torch.as_tensor(tokens[i:i + s], dtype=torch.long).view(1, s))
            i += s
        return out

    @torch.no_grad()
    def _call(self, batch, noise, wm_seed=None, tokens=None):
        """One policy call; returns (normalised action chunk [chunk, 7],
        the 30 token ids used, forecast-position KV)."""
        m = self.policy.model
        self.policy.reset()
        self._tokens, self._arm_kv = [], True
        rng = None if tokens is not None else torch.Generator(device=self.device).manual_seed(wm_seed)
        oi = None if tokens is None else [t.to(self.device) for t in self._split(tokens)]
        act = self.policy.select_action_with_world_model(batch, rng=rng, noise=noise.clone(), oracle_indices=oi)
        toks = self._tokens[-m.num_resolutions:]
        assert [t.numel() for t in toks] == SCALE_SIZES, [t.numel() for t in self._tokens]
        self._tokens = None
        self.policy.reset()
        return act[0, :, :7].float().cpu().numpy(), torch.cat(toks).numpy().astype(np.int64), self._kv_cap

    @staticmethod
    def _kv_change(a, b):
        res = {}
        for j, key in enumerate("kv"):
            res[key] = float(np.mean([float((b[i][j] - a[i][j]).norm() / (a[i][j].norm() + 1e-8)) for i in sorted(a)]))
        return res

    @torch.no_grad()
    def _frame_tokens(self, image: np.ndarray) -> np.ndarray:
        img = self._preprocess_history_image(image).unsqueeze(0).to(self.device)
        idx = self.policy.model.vae.img_to_idxBl(img)
        return torch.cat([t.reshape(-1).cpu() for t in idx[:len(SCALE_SIZES)]]).numpy().astype(np.int64)

    # -------------------------------------------------------------- lookahead
    def _snapshot(self):
        return dict(hist=list(self.image_history), step=self._step_count, state=self._current_state.clone(),
                    ref=self._ref_rot, buf=self.action_buffer, idx=self.action_buffer_idx,
                    rng=self._rng.get_state() if self._rng is not None else None,
                    cpu=torch.get_rng_state(), cuda=torch.cuda.get_rng_state_all())

    def _restore(self, s):
        self.image_history.clear()
        self.image_history.extend(s["hist"])
        self._step_count, self._current_state, self._ref_rot = s["step"], s["state"], s["ref"]
        self.action_buffer, self.action_buffer_idx = s["buf"], s["idx"]
        if s["rng"] is not None:
            self._rng.set_state(s["rng"])
        torch.set_rng_state(s["cpu"])
        torch.cuda.set_rng_state_all(s["cuda"])

    def _run_lookahead(self, task_description, n_steps: int, capture_decision_at: int | None):
        """From the current decision (t), run the NORMAL policy n_steps
        control steps in the replay env. Returns (frame at t+n_steps, tokens
        the normal policy samples at lookahead step capture_decision_at,
        replay check)."""
        snap = self._snapshot()
        env, obs, diff = self._tap.replay()
        # undo this step's history push so step() redoes it identically on the replay env
        self.image_history.clear()
        self.image_history.extend(snap["hist"][:-1])
        self._step_count -= 1
        if self._step_count == 0:
            self._ref_rot = None
        self.action_buffer, self.action_buffer_idx = None, 0
        self._lookahead, self._la_capture_at, self._la_tokens = True, capture_decision_at, None
        image = get_image_from_maniskill2_obs_dict(env, obs)
        try:
            for j in range(n_steps + (1 if capture_decision_at == n_steps else 0)):
                self._la_step = j
                ee, grip = proprio(env)
                a = self.step(image, task_description, ee, grip)
                if j == n_steps:
                    break
                obs, *_ = env.step(np.concatenate([a["world_vector"], a["rot_axangle"], a["gripper"]]))
                image = get_image_from_maniskill2_obs_dict(env, obs)
        finally:
            self._lookahead = False
            tokens = self._la_tokens
            self._restore(snap)
        return image, tokens, diff

    # --------------------------------------------------------------- decision
    def _predict_new_chunk(self, image, task_description):
        if self._lookahead:  # the normal policy, inside a lookahead
            if self._la_capture_at is not None and self._la_step == self._la_capture_at:
                batch = self._batch(image, task_description)
                self._tokens = []
                with torch.no_grad():
                    act = self.policy.select_action_with_world_model(batch, rng=self._rng)
                self._la_tokens = torch.cat(self._tokens[-len(SCALE_SIZES):]).numpy().astype(np.int64)
                self._tokens = None
                self.policy.reset()
                return (act[0].cpu() * self.action_std + self.action_mean).numpy()
            return super()._predict_new_chunk(image, task_description)

        t0 = time.time()
        k = len(self._records)  # decision index in this episode
        self._frames.append(image)
        batch = self._batch(image, task_description)
        m = self.policy.model
        noise = torch.randn((1, m.config.chunk_size, m.config.max_action_dim), generator=self._noise_gen, device=self.device)
        wm_seed = int(torch.randint(0, 2**31 - 1, (1,), generator=self._seed_gen))
        a_own, tok_own, kv_own = self._call(batch, noise, wm_seed=wm_seed)
        a_own2, tok_own2, _ = self._call(batch, noise, wm_seed=wm_seed + 1)
        self._own_tokens.append(tok_own)

        rec = {"decision": k, "global_step": self._step_count, "episode_id": self._episode_id,
               "tokens_own": tok_own, "tokens_own2": tok_own2}
        donor, la_time = None, 0.0
        if self._kind == "control":
            donor = tok_own
        elif self._kind == "bank":
            e = self._episode_id if self._episode_id is not None else self._episode
            de = (e + 1) % 24 if self._arg == "same_task" else e
            while self._bank_n[de] == 0:  # Stack episode 0 is missing in the bank
                de = (de + 1) % 24
            dk = min(k, int(self._bank_n[de]) - 1)
            donor = self._bank_tok[de, dk]
            rec.update(donor_task=self._donor_task, donor_episode=de, donor_decision=dk)
        elif self._kind == "own" and self._shift < 0:
            dk = max(0, k + self._shift)  # first decisions: the earliest available
            donor = self._own_tokens[dk]
            rec["donor_decision"] = dk
        elif self._kind == "real" and self._shift <= 0:
            fi = len(self._all_frames) - 1 + self._shift
            donor = self._frame_tokens(self._all_frames[max(0, fi)])
            rec["donor_frame_step"] = max(0, fi)
        else:  # lookahead: own:+d or real:+h
            t1 = time.time()
            n = self._shift * self.execute_steps if self._kind == "own" else self._shift
            frame, la_tok, diff = self._run_lookahead(task_description, n, n if self._kind == "own" else None)
            la_time = time.time() - t1
            donor = la_tok if self._kind == "own" else self._frame_tokens(frame)
            rec["replay_max_diff"] = diff
            assert donor is not None
        a_don, tok_used, kv_don = self._call(batch, noise, tokens=donor)
        assert np.array_equal(tok_used[:14], np.asarray(donor)[:14])
        rec.update(tokens_donor=np.asarray(donor), action_own=a_own, action_own2=a_own2, action_donor=a_don,
                   kv_change_donor=self._kv_change(kv_own, kv_don),
                   tok_diff_donor=float((np.asarray(donor)[:14] != tok_own[:14]).mean()),
                   tok_diff_seed=float((tok_own2[:14] != tok_own[:14]).mean()),
                   time_s=time.time() - t0, lookahead_s=la_time)
        self._records.append(rec)
        return (torch.from_numpy(a_don) * self.action_std + self.action_mean).numpy()

    def step(self, image, task_description, ee_pose_proprio, gripper_proprio):
        if not self._lookahead:
            self._all_frames.append(image)
        return super().step(image, task_description, ee_pose_proprio, gripper_proprio)

    def reset(self, task_description):
        self._flush()
        super().reset(task_description)
        self._episode += 1
        opts = self._tap.reset_options or {}
        self._episode_id = (opts.get("obj_init_options") or {}).get("episode_id")
        self._frames, self._all_frames, self._own_tokens = [], [], []

    def _flush(self):
        if self._episode >= 0 and self._records:
            e = self._episode_id if self._episode_id is not None else self._episode
            np.savez_compressed(self._out / f"episode_{e:03d}.npz", records=np.array(self._records, dtype=object))
            diffs = [r["replay_max_diff"] for r in self._records if "replay_max_diff" in r]
            print(f"[swap] episode {e} decisions={len(self._records)} "
                  f"d(donor,own)={np.mean([np.linalg.norm(r['action_donor'] - r['action_own'], axis=-1).mean() for r in self._records]):.4f} "
                  f"d(own2,own)={np.mean([np.linalg.norm(r['action_own2'] - r['action_own'], axis=-1).mean() for r in self._records]):.4f} "
                  f"replay_max_diff={max(diffs) if diffs else float('nan'):.2e} "
                  f"time/decision={np.mean([r['time_s'] for r in self._records]):.2f}s", flush=True)
            self._records = []

    def close(self):
        self._flush()
