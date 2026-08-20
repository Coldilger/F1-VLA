"""Live oracle probe for F1-VLA: same oracle mechanism as
oracle_offline_probe.py (F1VLAOracleInference.predict_action_given_true_next_frame,
unmodified), but sourced from a live, randomized SimplerEnv-Bridge rollout
instead of replaying bridge_orig_lerobot -- the exact dataset F1 was
fine-tuned on, with no held-out split. Verbatim recall of a specific
trajectory is impossible here: each episode's object placement is
randomized by SimplerEnv itself (the same mechanism Experiment 1's own
closed-loop eval already uses), so this tests the oracle mechanism against
scenes that cannot literally be memorized (distributional overfitting to
the task family is a separate, softer question this doesn't rule out).

Compares oracle's prediction against what the REAL (non-oracle) policy did
at that same decision point -- but only over SUCCESSFUL episodes (filtered
after maniskill2_evaluator returns success_arr). Without this filter, "the
policy's own action" is not a meaningful target: on the ~52% of episodes
the unmodified policy fails, its own action wasn't good, so an oracle
prediction that *diverges* from it could be an improvement, not an error,
and L1-against-a-mediocre-baseline can't tell the two apart (raised by the
user, 2026-08-19 -- correct). Restricting to successful episodes makes the
reference "real behavior that actually worked," the closed analogue of why
the offline probe's own reference (expert human teleop demonstrations) is
meaningful in the first place. Not a full fix -- not every action inside a
successful episode is necessarily optimal -- but a real improvement over
comparing against unfiltered behavior, and doesn't require new data or a
task-specific heuristic.

Runs a normal closed-loop rollout with the REAL (non-oracle) policy driving
the robot -- behavior is completely unaffected, this is a pure side
computation, same principle as server_policy_oracle_probe.py's approach for
LDA-1B. Wraps F1VLAInference._predict_new_chunk (called once per real
replan, not every control tick) rather than step(), to capture the action
in the same raw physical-unit space oracle_offline_probe.py already compares
against (step()'s own return value is post-SimplerEnv-conversion, not
comparable). Also wraps reset() (called once per episode by
maniskill2_evaluator) purely to count episode boundaries, for the
success-filter join at the end.
"""

import argparse
import os
import sys

import numpy as np

SIMPLER_ENV_ROOT = "/mnt/beegfsnew/scratch/3295540/F1-VLA/eval/bridge/SimplerEnv"
sys.path.insert(0, SIMPLER_ENV_ROOT)

from simpler_env.evaluation.argparse import get_args  # noqa: E402
from simpler_env.evaluation.maniskill2_evaluator import maniskill2_evaluator  # noqa: E402

# repo root (.../F1-VLA) for `from f1_vla...` imports, and eval/bridge/ for
# the local policy/oracle modules -- this file lives one directory deeper
# (eval/bridge/experiment2_oracle/) than main_inference.py, which needs one
# fewer dirname() hop to reach the repo root.
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(_THIS_DIR))))
sys.path.insert(0, os.path.dirname(_THIS_DIR))
from f1_vla_policy import F1VLAInference  # noqa: E402
from experiment2_oracle.oracle_offline_probe import F1VLAOracleInference  # noqa: E402


def add_live_oracle_probe(model: F1VLAInference, oracle_model: F1VLAOracleInference):
    original_predict_new_chunk = model._predict_new_chunk
    original_reset = model.reset
    buf = {"image": None, "task": None, "state": None}
    records = []  # each: dict(oracle_l1, zero_l1, episode_idx)
    episode_idx = {"n": -1}

    def wrapped_reset(task_description):
        episode_idx["n"] += 1
        buf["image"] = None  # never compare across an episode boundary
        return original_reset(task_description)

    def wrapped_predict_new_chunk(image, task_description):
        if buf["image"] is not None and buf["task"] == task_description:
            oracle_model.reset(buf["task"])
            oracle_model.image_history.append(oracle_model._preprocess_history_image(buf["image"]))
            oracle_model._current_state = buf["state"]
            pred_chunk = oracle_model.predict_action_given_true_next_frame(
                buf["image"], buf["task"], image
            )
            oracle_l1 = float(np.abs(pred_chunk[0] - buf["action"]).mean())
            zero_l1 = float(np.abs(buf["action"]).mean())
            records.append(dict(oracle_l1=oracle_l1, zero_l1=zero_l1, episode_idx=episode_idx["n"]))
            print(f"LIVE_ORACLE_SAMPLE n={len(records)} ep={episode_idx['n']} "
                  f"oracle_l1={oracle_l1:.5f} zero_l1={zero_l1:.5f}", flush=True)

        result = original_predict_new_chunk(image, task_description)
        buf["image"] = image
        buf["task"] = task_description
        buf["state"] = model._current_state.clone()
        buf["action"] = result[0].copy()
        return result

    model.reset = wrapped_reset
    model._predict_new_chunk = wrapped_predict_new_chunk
    return records


def report(records, success_arr, label):
    if not records:
        print(f"{label}: no samples recorded")
        return
    o = np.array([r["oracle_l1"] for r in records])
    z = np.array([r["zero_l1"] for r in records])
    print(f"{label} n={len(o)} oracle_mean={o.mean():.5f} oracle_sd={o.std():.5f} zero_mean={z.mean():.5f}")


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

    model = F1VLAInference(
        checkpoint_path=f1_args.f1_checkpoint_path,
        stats_path=f1_args.f1_stats_path,
        device=f1_args.f1_device,
        seed=f1_args.f1_seed,
        execute_steps=f1_args.f1_execute_steps,
    )
    oracle_model = F1VLAOracleInference(
        checkpoint_path=f1_args.f1_checkpoint_path,
        stats_path=f1_args.f1_stats_path,
        device=f1_args.f1_device,
        seed=f1_args.f1_seed,
    )
    records = add_live_oracle_probe(model, oracle_model)

    success_arr = maniskill2_evaluator(model, args)
    print(args)
    print(" " * 10, "Average success", np.mean(success_arr))

    print()
    report(records, success_arr, "LIVE_ORACLE_FINAL_ALL")
    success_arr = np.asarray(success_arr)
    filtered = [r for r in records if r["episode_idx"] < len(success_arr) and success_arr[r["episode_idx"]]]
    report(filtered, success_arr, "LIVE_ORACLE_FINAL_SUCCESSFUL_EPISODES_ONLY")
