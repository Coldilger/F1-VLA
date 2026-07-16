"""Entry point for running F1-VLA closed-loop rollout eval in SimplerEnv, modeled
on mimic-video-project's eval/bridge/SimplerEnv/simpler_env/main_inference.py.

Reuses SimplerEnv's evaluator and CLI arg parser as-is (both are policy-agnostic)
rather than duplicating them into this repo.

NOT YET RUNNABLE as-is: requires the SimplerEnv/ManiSkill2_real2sim/sapien stack
to actually be installed and importable (currently only vendored as source under
mimic-video-project, not pip-installed into any environment on this machine —
see conversation notes). Point SIMPLER_ENV_ROOT below at that vendored copy, or
pip-install it properly into (preferably a copy of) the f1_vla conda env once
dependency compatibility with torch/transformers/lerobot is checked.

Example (once environment is set up):
    python main_inference.py \
        --policy-setup widowx_bridge \
        --robot widowx \
        --env-name PutCarrotOnPlateInScene-v0 \
        --scene-name bridge_table_1_v1 \
        --f1-checkpoint-path /mnt/beegfsnew/scratch/3295540/F1-VLA/outputs/bridge_finetune \
        --f1-stats-path /mnt/beegfsnew/scratch/3295540/F1-VLA/outputs/bridge_finetune/bridge_orig_stats.json
"""

import argparse
import os
import sys

import numpy as np

SIMPLER_ENV_ROOT = "/mnt/beegfsnew/scratch/3295540/mimic-video-project/mimic-video/eval/bridge/SimplerEnv"
sys.path.insert(0, SIMPLER_ENV_ROOT)

from simpler_env.evaluation.argparse import get_args  # noqa: E402
from simpler_env.evaluation.maniskill2_evaluator import maniskill2_evaluator  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from eval.bridge.f1_vla_policy import F1VLAInference  # noqa: E402


def parse_f1_args(argv):
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--f1-checkpoint-path", type=str, required=True)
    parser.add_argument("--f1-stats-path", type=str, required=True)
    parser.add_argument("--f1-device", type=str, default="cuda")
    known, remaining = parser.parse_known_args(argv)
    return known, remaining


if __name__ == "__main__":
    f1_args, remaining_argv = parse_f1_args(sys.argv[1:])
    sys.argv = [sys.argv[0]] + remaining_argv
    args = get_args()

    os.environ["DISPLAY"] = ""

    model = F1VLAInference(
        checkpoint_path=f1_args.f1_checkpoint_path,
        stats_path=f1_args.f1_stats_path,
        device=f1_args.f1_device,
    )
    success_arr = maniskill2_evaluator(model, args)
    print(args)
    print(" " * 10, "Average success", np.mean(success_arr))
