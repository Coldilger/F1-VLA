"""Replay-env check for the lookahead conditions (README, "Lookahead").

The full runs recorded replay_max_diff > 0 at most Eggplant and Spoon
decisions after the first, i.e. the replay env reached a different state
than the main env. Decision 0 (fresh replay env) was always exact, so the
suspect is reusing one replay env across resets. This script steps a main
env with fixed pseudo-random actions and, every 4 steps, replays the
actions so far (a) in one reused env, as EnvTap.replay did, and (b) in a
freshly built env. Prints the max |state difference| for both. No model.

    python diag_replay.py TASK [EPISODE]
"""
import sys

import numpy as np

sys.path.insert(0, "/home/3295540/F1-VLA/eval/bridge/SimplerEnv")
from simpler_env.utils.env.env_builder import build_maniskill2_env  # noqa: E402

CFG = {"PutEggplantInBasketScene-v0": ("widowx_sink_camera_setup", "bridge_table_1_v2", "bridge_sink.png", 0.127, 0.06),
       "PutSpoonOnTableClothInScene-v0": ("widowx", "bridge_table_1_v1", "bridge_real_eval_1.png", 0.147, 0.028),
       "PutCarrotOnPlateInScene-v0": ("widowx", "bridge_table_1_v1", "bridge_real_eval_1.png", 0.147, 0.028)}
task, ep = sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 0
robot, scene, ov, x, y = CFG[task]
kw = dict(obs_mode="rgbd", robot=robot, sim_freq=500, control_mode="arm_pd_ee_target_delta_pose_align2_gripper_pd_joint_pos",
          control_freq=5, max_episode_steps=120, scene_name=scene, camera_cfgs={"add_segmentation": True},
          rgb_overlay_path=f"ManiSkill2_real2sim/data/real_inpainting/{ov}")
opts = {"robot_init_options": {"init_xy": np.array([x, y]), "init_rot_quat": np.array([0, 0, 0, 1.0])},
        "obj_init_options": {"episode_id": ep}}
rng = np.random.default_rng(0)
main = build_maniskill2_env(task, **kw)
main.reset(options=opts)
reused = build_maniskill2_env(task, **kw)
acts = []
for t in range(40):
    a = np.concatenate([rng.normal(0, 0.01, 3), rng.normal(0, 0.05, 3), [1.0 if t < 20 else -1.0]])
    main.step(a)
    acts.append(a)
    if (t + 1) % 4:
        continue
    res = {}
    for name in ("reused", "fresh"):
        env = reused if name == "reused" else build_maniskill2_env(task, **kw)
        env.reset(options=opts)
        for b in acts:
            env.step(b)
        res[name] = float(np.abs(env.unwrapped.get_state() - main.unwrapped.get_state()).max())
        if name == "fresh":
            env.close()
    print(f"step {t + 1:3d}: reused {res['reused']:.3g}  fresh {res['fresh']:.3g}", flush=True)
