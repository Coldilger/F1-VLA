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

4. Bridge's raw action convention (x,y,z,roll,pitch,yaw,gripper) is assumed to
   already match the delta-position + delta-rotvec convention SimplerEnv's
   widowx `arm_pd_ee_target_delta_pose_align2_gripper_pd_joint_pos` controller
   expects directly (world_vector=delta xyz, rot_axangle=delta rotvec), since
   this is the standard OXE/RT-1-style action space SimplerEnv's widowx setup
   was built around. Unlike VAMInference, no absolute-pose-tracking/6D-rotation
   conversion is implemented here. If the robot's motion looks wrong (jumpy,
   consistently off in one axis, gripper direction flipped), this is the first
   place to look — start by checking whether roll/pitch/yaw here need reordering
   or sign flips against SimplerEnv's rotvec convention, and whether gripper
   needs inverting/rescaling (Bridge/RT-1 gripper conventions vary: 0=open vs
   0=closed differs across datasets).
"""

from __future__ import annotations

import json
from collections import deque

import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
from scipy.spatial.transform import Rotation

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
        action_dim: int = 7,
        state_dim: int = 8,
    ):
        self.device = device
        self.policy = F1_VLA.from_pretrained(checkpoint_path).to(device)
        self.policy.eval()
        self.chunk_size = self.policy.config.chunk_size
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

        self._history_transform = T.Compose(
            [
                T.Resize(round(MID_RESO * FINAL_RESO), interpolation=Image.LANCZOS),
                T.CenterCrop(FINAL_RESO),
                T.ToTensor(),  # HWC uint8 [0,255] -> CHW float [0,1]
            ]
        )

        self.task_description = None
        self.image_history: deque = deque(maxlen=n_obs_img_steps)
        self.action_buffer = None
        self.action_buffer_idx = 0

    def reset(self, task_description: str) -> None:
        self.policy.reset()
        self.task_description = task_description
        self.image_history.clear()
        self.action_buffer = None
        self.action_buffer_idx = 0

    def _preprocess_main_image(self, image: np.ndarray) -> torch.Tensor:
        """Raw HWC uint8 -> CHW float [0,1]. F1's own resize_with_pad + rescale
        to [-1,1] happens inside select_action_with_world_model; we only need
        to hand it a plain [0,1] CHW tensor."""
        img = torch.from_numpy(image).float() / 255.0
        return img.permute(2, 0, 1)

    def _preprocess_history_image(self, image: np.ndarray) -> torch.Tensor:
        """Replicates the training-time gen-expert transform: Resize(288,
        LANCZOS) -> CenterCrop(256) -> ToTensor -> normalize_01_into_pm1."""
        pil_img = Image.fromarray(image)
        img = self._history_transform(pil_img)  # CHW float [0,1]
        return normalize_01_into_pm1(img)  # CHW float [-1,1]

    def _build_state(self, ee_pose_proprio, gripper_proprio: float) -> torch.Tensor:
        pos = np.asarray(ee_pose_proprio.p, dtype=np.float32)
        rpy = Rotation.from_quat(ee_pose_proprio.q, scalar_first=True).as_euler("xyz").astype(np.float32)
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
            actions = self.policy.select_action_with_world_model(batch)  # (1, chunk_size, action_dim)
        actions = actions[0].cpu()
        actions = actions * self.action_std + self.action_mean  # unnormalize to physical units
        return actions.numpy()

    def step(self, image: np.ndarray, task_description: str, ee_pose_proprio, gripper_proprio) -> dict:
        self.image_history.append(self._preprocess_history_image(image))
        self._current_state = self._build_state(ee_pose_proprio, gripper_proprio)

        if self.action_buffer is None:
            self.action_buffer = self._predict_new_chunk(image, task_description)
            self.action_buffer_idx = 0

        pred = self.action_buffer[self.action_buffer_idx]
        self.action_buffer_idx += 1
        if self.action_buffer_idx >= self.chunk_size:
            self.action_buffer = None

        return {
            "world_vector": pred[0:3].astype(np.float32),
            "rot_axangle": pred[3:6].astype(np.float32),
            "gripper": np.array([pred[6]], dtype=np.float32),
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
