"""Closed-loop entry point for Experiments 6-7 (F1-VLA, f1_swap_inference.py).
Mirrors ../main_inference.py, but the policy is F1VLASwapInference (SWAP,
SWAP_OUT and TASK from the environment) and the evaluator's environment
builder is wrapped so the policy can run lookaheads in a replay env."""

import argparse
import os
import sys

import numpy as np

SIMPLER_ENV_ROOT = "/home/3295540/F1-VLA/eval/bridge/SimplerEnv"
sys.path.insert(0, SIMPLER_ENV_ROOT)

import simpler_env.evaluation.maniskill2_evaluator as ev  # noqa: E402
from simpler_env.evaluation.argparse import get_args  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from f1_swap_inference import EnvTap, F1VLASwapInference  # noqa: E402


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

    tap = EnvTap()
    ev.build_maniskill2_env = tap.wrap_builder(ev.build_maniskill2_env)
    model = F1VLASwapInference(
        checkpoint_path=f1_args.f1_checkpoint_path,
        stats_path=f1_args.f1_stats_path,
        device=f1_args.f1_device,
        seed=f1_args.f1_seed,
        execute_steps=f1_args.f1_execute_steps,
        swap=os.environ["SWAP"],
        task=os.environ["TASK"],
        tap=tap,
        out_dir=os.environ["SWAP_OUT"],
    )
    success_arr = ev.maniskill2_evaluator(model, args)
    model.close()
    print(args)
    print(" " * 10, "Average success", np.mean(success_arr))
