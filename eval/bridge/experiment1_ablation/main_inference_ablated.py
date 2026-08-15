"""Closed-loop eval entrypoint for Experiment 1 variant 1 (F1-VLA), mirrors
../main_inference.py exactly except it instantiates F1VLAAblatedInference
instead of F1VLAInference. Not modifying main_inference.py itself, so the
real (unmodified) baseline eval path stays exactly as it was for
RESULTS.md.
"""

import argparse
import os
import sys

import numpy as np

SIMPLER_ENV_ROOT = "/mnt/beegfsnew/scratch/3295540/F1-VLA/eval/bridge/SimplerEnv"
sys.path.insert(0, SIMPLER_ENV_ROOT)

from simpler_env.evaluation.argparse import get_args  # noqa: E402
from simpler_env.evaluation.maniskill2_evaluator import maniskill2_evaluator  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from f1_vla_policy_ablated import F1VLAAblatedInference  # noqa: E402


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

    model = F1VLAAblatedInference(
        checkpoint_path=f1_args.f1_checkpoint_path,
        stats_path=f1_args.f1_stats_path,
        device=f1_args.f1_device,
        seed=f1_args.f1_seed,
        execute_steps=f1_args.f1_execute_steps,
    )
    success_arr = maniskill2_evaluator(model, args)
    print(args)
    print(" " * 10, "Average success", np.mean(success_arr))
