"""Diagnostic: compare the observation.state our eval wrapper builds from
SimplerEnv against the distribution the model actually trained on.

Training (bridge_orig, measured over 4 episodes):
    x       0.2456 .. 0.4454   mean 0.3451
    y      -0.1100 .. 0.1298   mean 0.0008
    z       0.0065 .. 0.1799   mean 0.0817
    roll   -0.3568 .. 0.4664   mean 0.0014
    pitch  -0.5767 .. 0.2495   mean -0.1106
    yaw    -0.0232 .. 0.9810   mean 0.2751
    pad     0
    gripper 0.0598 .. 1.0086   mean 0.7418

If our eval-time state lands in a visibly different range (esp. sign flips or a
different euler convention), the policy is being fed out-of-distribution proprio.
"""
import numpy as np
import gymnasium as gym
import mani_skill2_real2sim.envs  # noqa: F401
from scipy.spatial.transform import Rotation

CONTROL_MODE = "arm_pd_ee_target_delta_pose_align2_gripper_pd_joint_pos"

env = gym.make(
    "PutCarrotOnPlateInScene-v0",
    obs_mode="rgbd",
    robot="widowx",
    sim_freq=500,
    control_freq=5,
    max_episode_steps=60,
    scene_name="bridge_table_1_v1",
    control_mode=CONTROL_MODE,
)

rows = []
for ep in range(3):
    env.reset(options={"obj_init_options": {"episode_id": ep}})
    for t in range(20):
        ctrl = env.unwrapped.agent.controllers[CONTROL_MODE]
        pose = ctrl.controllers["arm"].ee_pose_at_base
        grip = ctrl.controllers["gripper"].qpos.mean()
        grip = (grip - 0.015) / (0.037 - 0.015)
        rpy = Rotation.from_quat(pose.q, scalar_first=True).as_euler("xyz")
        rows.append(np.concatenate([pose.p, rpy, [0.0], [grip]]))
        # idle action to let the arm settle/move a bit
        env.step(np.zeros(7, dtype=np.float32))

S = np.array(rows)
np.set_printoptions(precision=4, suppress=True)
names = ["x", "y", "z", "roll", "pitch", "yaw", "pad", "gripper"]
print("\n=== EVAL-TIME state produced by our wrapper ===")
print("dim      min      max     mean")
for i, n in enumerate(names):
    print(f"{n:8s} {S[:,i].min():8.4f} {S[:,i].max():8.4f} {S[:,i].mean():8.4f}")
env.close()
print("\nDIAG_DONE")
