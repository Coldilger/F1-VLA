"""Wraps main_inference.py, skipping per-episode video encoding.

Written to unblock the chunk8@100k final eval (2026-08-19): 9 of the first 12
task/seed jobs crashed on `write_video`'s libx264 call
(`RuntimeError: Error writing '...': [vost#0:0/libx264 ...] Error while
opening encoder`), on all three physical nodes in the partition, at varying
episode indices (0 through 10+) -- not concurrency-specific (still failed at
reduced concurrency), so most likely a flaky/oversubscribed ffmpeg install on
this cluster rather than anything about our code or the model. The only
output this comparison actually needs is the printed `Average success` line
and each episode's `success` field (already computed before the video-write
call) -- the rollout videos themselves are not used anywhere in RESULTS.md's
analysis. Skipping the write_video call sidesteps the crash entirely rather
than fighting cluster flakiness.

Does not modify SimplerEnv/ManiSkill2_real2sim or main_inference.py --
monkey-patches `write_video` to a no-op on the already-imported
`maniskill2_evaluator` module (which binds it via `from ...visualization
import write_video`, so patching the module attribute is sufficient), then
runs the unmodified main_inference.py as __main__ via runpy, matching the
non-invasive-patch pattern already used by
robocasa-eval-derived launcher/run_client_agentview.py.
"""

import runpy
import sys

SIMPLER_ENV_ROOT = "/mnt/beegfsnew/scratch/3295540/F1-VLA/eval/bridge/SimplerEnv"
sys.path.insert(0, SIMPLER_ENV_ROOT)

from simpler_env.evaluation import maniskill2_evaluator  # noqa: E402

maniskill2_evaluator.write_video = lambda *args, **kwargs: None
print("PATCHED: write_video is a no-op for this run (video encoding was crashing).", flush=True)

sys.argv[0] = "main_inference.py"
runpy.run_path(
    "/mnt/beegfsnew/scratch/3295540/F1-VLA/eval/bridge/main_inference.py",
    run_name="__main__",
)
