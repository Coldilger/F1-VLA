"""Make-or-break test: can SAPIEN actually build a Bridge env and render a frame
on this cluster's GPU node? Run on a GPU node via srun/sbatch."""
import os
import sys
import glob

print("=== Vulkan ICD discovery ===", flush=True)
# The NVIDIA Vulkan ICD (driver) must be present on the GPU node for rendering.
icd_candidates = (
    glob.glob("/usr/share/vulkan/icd.d/*.json")
    + glob.glob("/etc/vulkan/icd.d/*.json")
    + ([os.environ["VK_ICD_FILENAMES"]] if os.environ.get("VK_ICD_FILENAMES") else [])
)
print("ICD files found:", icd_candidates, flush=True)
nvidia_libs = glob.glob("/usr/lib*/libGLX_nvidia.so*") + glob.glob("/usr/lib*/*/libGLX_nvidia.so*")
print("NVIDIA GLX libs:", nvidia_libs[:3], flush=True)

print("\n=== build one Bridge env + render one frame ===", flush=True)
import numpy as np
import gymnasium as gym
import mani_skill2_real2sim.envs  # noqa: F401  (registers env ids)

env = gym.make(
    "PutCarrotOnPlateInScene-v0",
    obs_mode="rgbd",
    robot="widowx",
    sim_freq=513,
    control_freq=5,
    max_episode_steps=60,
    scene_name="bridge_table_1_v1",
    control_mode="arm_pd_ee_target_delta_pose_align2_gripper_pd_joint_pos",
)
print("env created", flush=True)
obs, _ = env.reset()
img = obs["image"]["3rd_view_camera"]["rgb"]
print("RENDER OK — rgb frame shape:", img.shape, "dtype:", img.dtype,
      "min/max:", int(img.min()), int(img.max()), flush=True)
print("language instruction:", env.get_language_instruction(), flush=True)
env.close()
print("\nSAPIEN_RENDER_TEST_PASSED", flush=True)
