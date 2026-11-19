"""Closed-loop entry point for Experiment 9 (F1-VLA, number of forecast
scales; see f1_scales_inference.py). Mirrors ../main_inference.py except
for the policy class. SCALES, POS_MATCH and TIMING_OUT come from the
environment.
"""

import argparse
import os
import sys

import numpy as np

SIMPLER_ENV_ROOT = "/home/3295540/F1-VLA/eval/bridge/SimplerEnv"
sys.path.insert(0, SIMPLER_ENV_ROOT)

from simpler_env.evaluation.argparse import get_args  # noqa: E402
from simpler_env.evaluation.maniskill2_evaluator import maniskill2_evaluator  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from f1_scales_inference import F1VLAScales  # noqa: E402


def parse_f1_args(argv):
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--f1-checkpoint-path", type=str, required=True)
    parser.add_argument("--f1-stats-path", type=str, required=True)
    parser.add_argument("--f1-device", type=str, default="cuda")
    parser.add_argument("--f1-execute-steps", type=int, default=None)
    parser.add_argument("--f1-seed", type=int, default=None)
    known, remaining = parser.parse_known_args(argv)
    return known, remaining


if __name__ == "__main__":
    f1_args, remaining_argv = parse_f1_args(sys.argv[1:])
    sys.argv = [sys.argv[0]] + remaining_argv
    args = get_args()

    os.environ["DISPLAY"] = ""

    model = F1VLAScales(
        checkpoint_path=f1_args.f1_checkpoint_path,
        stats_path=f1_args.f1_stats_path,
        device=f1_args.f1_device,
        seed=f1_args.f1_seed,
        execute_steps=f1_args.f1_execute_steps,
        scales=int(os.environ["SCALES"]),
        pos_match=os.environ.get("POS_MATCH", "0") == "1",
        timing_out=os.environ.get("TIMING_OUT"),
    )
    success_arr = maniskill2_evaluator(model, args)
    model.close()
    print(args)
    print(" " * 10, "Average success", np.mean(success_arr))
