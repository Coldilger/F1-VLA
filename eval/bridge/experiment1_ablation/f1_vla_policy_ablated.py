"""Closed-loop SimplerEnv wrapper for Experiment 1 variant 1 (F1-VLA):
"remove the module". Subclasses the real, working F1VLAInference
(../f1_vla_policy.py) unchanged -- only overrides _predict_new_chunk to call
sample_actions_without_world_model instead of the default
select_action_with_world_model. reset()/step() are inherited as-is, so this
is a drop-in replacement for main_inference.py's closed-loop rollout, not a
separate code path that could silently diverge from the real eval wrapper's
image preprocessing / history handling / gripper-frame conversions.

Why this matters for the central research question, not just the offline
probe: the offline probe (ablation_offline_probe.py) compares a single
predicted action against the REAL LOGGED action from Bridge -- but the
checkpoint was finetuned on the whole dataset with no held-out split, so a
close match there could reflect memorization of that exact (image, action)
pair rather than genuine generalization, for EITHER condition (baseline or
ablated) equally, in a way L1-vs-logged-action can't distinguish. Real
closed-loop rollout doesn't have this problem: SimplerEnv randomizes object
placement per episode (obj_variation_mode: episode), so beyond the first
step the model is looking at states it could not have memorized (they are
the consequence of its OWN actions in this specific simulated instantiation,
never logged anywhere during training). Comparing SUCCESS RATE between
baseline and ablated here is the test that actually isolates "does the
world-model computation carry causal weight," independent of memorization.
"""

from __future__ import annotations

import sys

import torch

sys.path.insert(0, "/mnt/beegfsnew/scratch/3295540/F1-VLA")
sys.path.insert(0, "/mnt/beegfsnew/scratch/3295540/F1-VLA/eval/bridge")
sys.path.insert(0, "/mnt/beegfsnew/scratch/3295540/F1-VLA/eval/bridge/experiment1_ablation")

from f1_vla_policy import F1VLAInference  # noqa: E402
from sample_without_world_model import sample_actions_without_world_model  # noqa: E402


class F1VLAAblatedInference(F1VLAInference):
    def _predict_new_chunk(self, image, task_description: str):
        main_img = self._preprocess_main_image(image).unsqueeze(0).to(self.device)
        hist_frames = list(self.image_history)
        if len(hist_frames) < self.n_obs_img_steps:
            hist_frames = [hist_frames[0]] * (self.n_obs_img_steps - len(hist_frames)) + hist_frames
        hist_stack = torch.stack(hist_frames).unsqueeze(0).to(self.device)

        batch = {
            "observation.images.image0": main_img,
            "observation.images.image0_mask": torch.tensor([True], device=self.device),
            "observation.images.image0_history": hist_stack,
            "observation.state": self._current_state.unsqueeze(0).to(self.device),
            "task": [task_description],
        }
        images, image_masks = self.policy.prepare_mix_images(batch)
        state = self.policy.prepare_state(batch)
        lang_tokens, lang_masks = self.policy.prepare_language(batch)

        with torch.no_grad():
            actions = sample_actions_without_world_model(
                self.policy.model, images, image_masks, lang_tokens, lang_masks, state
            )
        actions = actions[:, :, :7]  # unpad, same as select_action_with_world_model does internally
        actions = actions[0].cpu()
        actions = actions * self.action_std + self.action_mean
        return actions.numpy()
