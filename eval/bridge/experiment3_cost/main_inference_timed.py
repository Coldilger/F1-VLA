"""Closed-loop eval entrypoint for Experiment 3 (F1-VLA): measures real
per-replan wall-clock latency for each of the three conditions already
built for Experiment 1 (baseline, ablated, shuffled), applying
timing_wrapper.add_timing to whichever class --f1-variant selects. Mirrors
../main_inference.py / ../experiment1_ablation/main_inference_ablated.py /
main_inference_shuffled.py exactly otherwise -- same real SimplerEnv
rollout, nothing about the model's actual behavior changes.
"""

import argparse
import os
import sys
import statistics

import numpy as np

SIMPLER_ENV_ROOT = "/mnt/beegfsnew/scratch/3295540/F1-VLA/eval/bridge/SimplerEnv"
sys.path.insert(0, SIMPLER_ENV_ROOT)

from simpler_env.evaluation.argparse import get_args  # noqa: E402
from simpler_env.evaluation.maniskill2_evaluator import maniskill2_evaluator  # noqa: E402

REPO_ROOT = "/mnt/beegfsnew/scratch/3295540/F1-VLA"
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.join(REPO_ROOT, "eval/bridge"))
sys.path.insert(0, os.path.join(REPO_ROOT, "eval/bridge/experiment1_ablation"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from f1_vla_policy import F1VLAInference  # noqa: E402
from f1_vla_policy_ablated import F1VLAAblatedInference  # noqa: E402
from f1_vla_policy_shuffled import F1VLAShuffledInference  # noqa: E402

from timing_wrapper import add_timing  # noqa: E402

add_timing(F1VLAInference)
add_timing(F1VLAAblatedInference)
add_timing(F1VLAShuffledInference)

VARIANTS = {
    "baseline": F1VLAInference,
    "ablated": F1VLAAblatedInference,
    "shuffled": F1VLAShuffledInference,
}


def parse_f1_args(argv):
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--f1-checkpoint-path", type=str, required=True)
    parser.add_argument("--f1-stats-path", type=str, required=True)
    parser.add_argument("--f1-device", type=str, default="cuda")
    parser.add_argument("--f1-execute-steps", type=int, default=None)
    parser.add_argument("--f1-seed", type=int, default=None)
    parser.add_argument("--f1-variant", type=str, required=True, choices=list(VARIANTS.keys()))
    parser.add_argument("--bridge-dataset-root", type=str,
                         default="/mnt/beegfsnew/scratch/3295540/data/bridge_orig_lerobot")
    parser.add_argument("--frame-pool-size", type=int, default=64)
    known, remaining = parser.parse_known_args(argv)
    return known, remaining


if __name__ == "__main__":
    f1_args, remaining_argv = parse_f1_args(sys.argv[1:])
    sys.argv = [sys.argv[0]] + remaining_argv
    args = get_args()

    os.environ["DISPLAY"] = ""

    cls = VARIANTS[f1_args.f1_variant]
    common_kwargs = dict(
        checkpoint_path=f1_args.f1_checkpoint_path,
        stats_path=f1_args.f1_stats_path,
        device=f1_args.f1_device,
        seed=f1_args.f1_seed,
        execute_steps=f1_args.f1_execute_steps,
    )
    if cls is F1VLAShuffledInference:
        common_kwargs["bridge_dataset_root"] = f1_args.bridge_dataset_root
        common_kwargs["frame_pool_size"] = f1_args.frame_pool_size

    model = cls(**common_kwargs)
    success_arr = maniskill2_evaluator(model, args)
    print(args)
    print(" " * 10, "Average success", np.mean(success_arr))

    lat = getattr(model, "_chunk_latencies_s", [])
    if lat:
        lat_ms = sorted(x * 1000 for x in lat)
        n = len(lat_ms)
        median = statistics.median(lat_ms)
        p95 = lat_ms[int(0.95 * (n - 1))]
        mean = statistics.mean(lat_ms)
        print(f"LATENCY variant={f1_args.f1_variant} n={n} mean_ms={mean:.1f} "
              f"median_ms={median:.1f} p95_ms={p95:.1f} min_ms={lat_ms[0]:.1f} max_ms={lat_ms[-1]:.1f}")
    else:
        print("LATENCY: no chunk predictions recorded")
