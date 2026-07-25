"""SimplerEnv-compatible policy wrapper for the F1-VLA model, modeled on
mimic-video-project's VAMInference (eval/bridge/SimplerEnv/simpler_env/policies/vam/video_action_model.py).

Implements the 3-method interface maniskill2_evaluator.py expects:
    reset(task_description)
    step(image, task_description, ee_pose_proprio, gripper_proprio) -> dict
    visualize_epoch(images, save_path)

Load-bearing assumptions made explicit here (verify empirically once the
SimplerEnv/ManiSkill2 environment is actually installed and runnable):

1. F1's action head has NO built-in normalization (confirmed: f1_policy.py has
   zero references to norm_stats/unnormalize). The bridge_orig training data was
   mean/std-normalized by the *dataset transform* (BridgeV2Inputs), not by the
   policy itself. So both directions must be handled manually here, using the
   exact stats dumped during training (outputs/bridge_finetune/bridge_orig_stats.json):
     - normalize observation.state the same way before feeding the model
     - unnormalize the model's predicted action chunk before use
   observation.state dim 6 ("pad") has std=0 in the stats file; it's forced to
   0 in normalized space to avoid a divide-by-zero.

2. select_action_with_world_model() does NOT do its own action-chunk caching
   despite the policy having a `cache_action_steps`-sized deque — that queue is
   never read/written in the current code. This wrapper does the chunk
   caching/popping itself: call the model once, execute all `chunk_size`
   (=4 for our checkpoint) predicted steps, then call again.

3. Training's world-model history window (n_obs_img_steps=12, n_pred_img_steps=3,
   obs_img_stride=3, fps=5) samples camera_keys at frame offsets
   [-9,-6,-3,0,+3] relative to a training-time anchor, i.e. 4 "observation"
   frames covering the last 1.8s (spaced 0.6s apart) plus 1 "future" frame the
   world-model head learns to predict during training. At *inference* time
   there is no ground-truth future frame — sample_actions_with_world_model
   generates/imagines it internally — so this wrapper only ever supplies the
   4 observation-side frames as `observation.images.image0_history`. This is
   the biggest unverified assumption here: confirm the model accepts a T=4
   history tensor (not T=5) once you can actually run a forward pass.

4. Bridge's raw action (x,y,z,roll,pitch,yaw,gripper) needs two conversions
   before SimplerEnv's widowx controller can use it. An earlier version of this
   file assumed it could be passed through directly; that was WRONG and produced
   0/24 success on PutCarrotOnPlateInScene (the model reached for and brushed
   the correct object but could never hold it). The conversions below now mirror
   SimplerEnv's own reference implementation for this exact action convention
   (simpler_env/policies/octo/octo_model.py:180-238, widowx_bridge branch):
     - rotation: euler (roll,pitch,yaw) -> axis-angle via euler2axangle
     - gripper: 0=close/1=open -> binarize at 0.5 and map to -1=close/+1=open
   Verified against training data: gripper values in bridge_orig are exactly
   {0.0, 1.0}, and rotation deltas are small euler angles (~±0.14 rad).
"""

from __future__ import annotations

import json
from collections import deque

import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
from scipy.spatial.transform import Rotation
from transforms3d.euler import euler2axangle

from f1_vla.src.policies.f1_policy import F1_VLA
from f1_vla.src.utils.image_tools import normalize_01_into_pm1

MID_RESO = 1.125
FINAL_RESO = 256


class F1VLAInference:
    def __init__(
        self,
        checkpoint_path: str,
        stats_path: str,
        device: str = "cuda",
        n_obs_img_steps: int = 4,
        obs_img_stride_s: float = 0.6,
        control_freq: float = 5.0,
        action_dim: int = 7,
        state_dim: int = 8,
        seed: int | None = None,
        execute_steps: int | None = None,
    ):
        self.device = device
        self.policy = F1_VLA.from_pretrained(checkpoint_path).to(device)
        self.policy.eval()
        self.chunk_size = self.policy.config.chunk_size
        # How many of the predicted chunk_size actions are actually executed
        # before re-observing the scene and re-planning. Defaults to the whole
        # chunk, which is how every result up to now was measured. It matters
        # once chunk_size is large: at chunk_size=30 with 60-step episodes the
        # policy only ever looks at the scene twice, i.e. it is effectively
        # open-loop, which confounds 'was the model trained badly' with 'was it
        # driven badly'. Setting execute_steps < chunk_size gives the usual
        # receding-horizon control and separates the two.
        self.execute_steps = self.chunk_size if execute_steps is None else min(execute_steps, self.chunk_size)
        assert self.execute_steps >= 1
        self.main_image_size = tuple(self.policy.config.resize_imgs_with_padding)  # (224, 224)

        stats = json.load(open(stats_path))
        self.state_mean = torch.tensor(stats["observation.state"]["mean"], dtype=torch.float32)
        self.state_std = torch.tensor(stats["observation.state"]["std"], dtype=torch.float32)
        self.action_mean = torch.tensor(stats["action"]["mean"], dtype=torch.float32)
        self.action_std = torch.tensor(stats["action"]["std"], dtype=torch.float32)
        # "pad" dim in observation.state (index 6) has std=0 in training stats;
        # avoid divide-by-zero, it's forced to 0 in normalized space regardless.
        pad_std = self.state_std.clone()
        pad_std[pad_std == 0] = 1.0
        self._state_std_safe = pad_std

        self.n_obs_img_steps = n_obs_img_steps
        self.obs_img_stride_s = obs_img_stride_s
        self.action_dim = action_dim
        self.state_dim = state_dim
        # NOTE (measured, not assumed): training sampled the world-model history
        # at obs_img_stride=3 on a 5 fps dataset, i.e. every 0.6s spanning 1.8s,
        # so matching that at eval (push every 3rd env step at control_freq=5)
        # *looks* like the faithful choice. It is empirically WORSE: on
        # PutCarrotOnPlateInScene, 24 episodes, stride-3 scored 1/24 (4.2%) vs
        # 7/24 (29.2%) for appending every step, with consecutive_grasp 80 vs
        # 154 and src_on_target 36 vs 195. Plausible reason: with stride 3 the
        # history is still mostly repeat-padding until step ~12 of a 60-step
        # episode, degrading exactly the early actions that set up the grasp.
        # Keeping the dense (every-step) history since the numbers decide.
        # Set history_stride_steps > 1 to re-test the sparse variant.
        self.history_stride_steps = 1
        self._step_count = 0

        self._history_transform = T.Compose(
            [
                T.Resize(round(MID_RESO * FINAL_RESO), interpolation=Image.LANCZOS),
                T.CenterCrop(FINAL_RESO),
                T.ToTensor(),  # HWC uint8 [0,255] -> CHW float [0,1]
            ]
        )

        # The world-model head samples (top_k/top_p), so rollouts are stochastic.
        # Seeding it makes a run reproducible and lets us average over seeds to
        # get error bars instead of one noisy 24-episode point estimate.
        self.seed = seed
        self._rng = None
        if seed is not None:
            self._rng = torch.Generator(device=device)
            self._rng.manual_seed(seed)
            # The rng generator above only reaches the world-model token sampler.
            # The flow-matching start noise comes from F1FlowMatching.sample_noise,
            # which calls torch.normal() with no generator, i.e. the *global* torch
            # RNG. Left unseeded, two runs of the same weights with the same
            # --f1-seed gave 38.9% and 36.1% on the carrot task. Seed the global
            # RNG too so a run is reproducible and model comparisons aren't
            # confounded by sampling noise.
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)

        self.task_description = None
        self.image_history: deque = deque(maxlen=n_obs_img_steps)
        self.action_buffer = None
        self.action_buffer_idx = 0
        self._ref_rot = None

    def reset(self, task_description: str) -> None:
        self.policy.reset()
        self.task_description = task_description
        self.image_history.clear()
        self.action_buffer = None
        self.action_buffer_idx = 0
        self._step_count = 0
        self._ref_rot = None

    def _to_training_resolution(self, image: np.ndarray) -> Image.Image:
        """SimplerEnv renders 480x640 (4:3); bridge_orig training frames are
        256x256 (square). Squash to a square first — matching what SimplerEnv's
        own reference policies do (octo_model.py `_resize_image` resizes to a
        square, it does not letterbox) — so the model sees the aspect ratio it
        was trained on. Skipping this leaves the main image letterboxed with
        black bars by F1's internal resize_with_pad, and crops ~1/3 of the
        horizontal FOV off the world-model history frames via CenterCrop; the
        model saw neither during training."""
        return Image.fromarray(image).resize((FINAL_RESO, FINAL_RESO), Image.LANCZOS)

    def _preprocess_main_image(self, image: np.ndarray) -> torch.Tensor:
        """Raw HWC uint8 -> square CHW float [0,1]. F1's own resize_with_pad +
        rescale to [-1,1] happens inside select_action_with_world_model; on a
        square input that resize is a clean no-pad downscale to 224x224."""
        img = torch.from_numpy(np.asarray(self._to_training_resolution(image))).float() / 255.0
        return img.permute(2, 0, 1)

    def _preprocess_history_image(self, image: np.ndarray) -> torch.Tensor:
        """Replicates the training-time gen-expert transform: Resize(288,
        LANCZOS) -> CenterCrop(256) -> ToTensor -> normalize_01_into_pm1,
        applied to a square frame as in training."""
        img = self._history_transform(self._to_training_resolution(image))  # CHW float [0,1]
        return normalize_01_into_pm1(img)  # CHW float [-1,1]

    def _build_state(self, ee_pose_proprio, gripper_proprio: float) -> torch.Tensor:
        pos = np.asarray(ee_pose_proprio.p, dtype=np.float32)
        rot = Rotation.from_quat(ee_pose_proprio.q, scalar_first=True)
        # Bridge records EE orientation as small angles centred on zero (measured
        # over 40 episodes: roll/pitch/yaw means ~0.00/-0.09/0.10, |angle| < 1.6),
        # i.e. relative to the canonical gripper-down home pose. SimplerEnv's
        # ee_pose_at_base is absolute, and at rest yields ~(-3.06, 1.51, -3.08)
        # in scipy 'xyz' euler — a ~93 deg rotation that is nowhere in the
        # training distribution (and sits at the pitch=pi/2 gimbal singularity).
        # No plain euler order fixes this (all 12 were checked); the mismatch is
        # the reference frame, not the ordering. So express rotation relative to
        # the pose captured at episode reset, which reproduces the training
        # structure: ~zero at episode start, drifting as the arm moves.
        if self._ref_rot is None:
            self._ref_rot = rot
        rel_rot = self._ref_rot.inv() * rot
        rpy = rel_rot.as_euler("xyz").astype(np.float32)
        state_raw = np.concatenate([pos, rpy, [0.0], [float(gripper_proprio)]]).astype(np.float32)
        state = torch.from_numpy(state_raw)
        state_norm = (state - self.state_mean) / self._state_std_safe
        state_norm[6] = 0.0  # pad dim
        return state_norm

    def _predict_new_chunk(self, image: np.ndarray, task_description: str) -> np.ndarray:
        main_img = self._preprocess_main_image(image).unsqueeze(0).to(self.device)
        # pad history by repeating the earliest available frame until full,
        # matching VAMInference's approach for the first few steps of an episode
        hist_frames = list(self.image_history)
        if len(hist_frames) < self.n_obs_img_steps:
            hist_frames = [hist_frames[0]] * (self.n_obs_img_steps - len(hist_frames)) + hist_frames
        hist_stack = torch.stack(hist_frames).unsqueeze(0).to(self.device)  # (1, T, C, H, W)

        state = self._current_state.unsqueeze(0).to(self.device)

        batch = {
            "observation.images.image0": main_img,
            "observation.images.image0_mask": torch.tensor([True], device=self.device),
            "observation.images.image0_history": hist_stack,
            "observation.state": state,
            "task": [task_description],
        }
        with torch.no_grad():
            actions = self.policy.select_action_with_world_model(batch, rng=self._rng)  # (1, chunk_size, action_dim)
        actions = actions[0].cpu()
        actions = actions * self.action_std + self.action_mean  # unnormalize to physical units
        return actions.numpy()

    def step(self, image: np.ndarray, task_description: str, ee_pose_proprio, gripper_proprio) -> dict:
        # Push into the world-model history only every Nth env step, so the
        # window matches the 0.6s spacing / 1.8s span the model trained on.
        # (Always seed it on the very first step so it is never empty.)
        if self._step_count % self.history_stride_steps == 0 or not self.image_history:
            self.image_history.append(self._preprocess_history_image(image))
        self._step_count += 1
        self._current_state = self._build_state(ee_pose_proprio, gripper_proprio)

        if self.action_buffer is None:
            self.action_buffer = self._predict_new_chunk(image, task_description)
            self.action_buffer_idx = 0

        pred = self.action_buffer[self.action_buffer_idx]
        self.action_buffer_idx += 1
        if self.action_buffer_idx >= self.execute_steps:
            self.action_buffer = None

        # Convert the raw Bridge action to what SimplerEnv's widowx controller
        # expects. This mirrors SimplerEnv's own octo_model.py widowx_bridge
        # branch exactly (simpler_env/policies/octo/octo_model.py:180-238) —
        # both conversions below are load-bearing:
        #   - rotation: Bridge actions store euler (roll,pitch,yaw); the
        #     controller wants an axis-angle (rotation vector). Passing euler
        #     straight through is wrong (they only coincide for tiny angles).
        #   - gripper: Bridge encodes 0=close / 1=open, but the controller takes
        #     [-1,+1] (normalize_action=True -> scaled into the joint range),
        #     so raw 0 would land mid-range = half-open and the gripper could
        #     never actually close on an object.
        roll, pitch, yaw = np.asarray(pred[3:6], dtype=np.float64)
        rot_ax, rot_angle = euler2axangle(roll, pitch, yaw)

        return {
            "world_vector": pred[0:3].astype(np.float32),
            "rot_axangle": (rot_ax * rot_angle).astype(np.float32),
            "gripper": np.array([2.0 * (pred[6] > 0.5) - 1.0], dtype=np.float32),
            "terminate_episode": np.array([0.0], dtype=np.float32),
        }

    def visualize_epoch(self, images, save_path: str) -> None:
        """Minimal placeholder: dump a strip of sampled frames as a single PNG.
        VAMInference's version plots action-trajectory diagnostics from its own
        internal history; wire that up later if per-episode action plots are
        needed, this just gives you *something* to look at per rollout."""
        if not images:
            return
        n = min(8, len(images))
        idxs = np.linspace(0, len(images) - 1, n).astype(int)
        frames = [Image.fromarray(images[i]) for i in idxs]
        w, h = frames[0].size
        strip = Image.new("RGB", (w * n, h))
        for i, frame in enumerate(frames):
            strip.paste(frame, (i * w, 0))
        strip.save(save_path)
