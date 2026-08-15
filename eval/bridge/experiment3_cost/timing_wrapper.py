"""Experiment 3 (cost per decision) instrumentation for F1-VLA. Does not
modify f1_vla_policy.py or any of the Experiment 1 wrapper subclasses --
wraps _predict_new_chunk with a timer via a mixin applied at import time
(monkey-patch on the class, not the shared source file), recording
per-replan wall-clock latency into a list on the instance.

Why _predict_new_chunk specifically: F1VLAInference.step() only calls it
once per replan (F1 predicts a whole action chunk per forward pass, matching
../README.md's Experiment 3 table row -- "VAR foresight loop, re-run every
control step" refers to a full chunk-worth of control steps between
replans, not literally every single 5Hz tick). Timing this one call
captures the real, complete per-decision cost -- foresight sampling +
action flow-matching + everything in between -- without needing to split
the internals of a function this project has deliberately chosen not to
modify.
"""

from __future__ import annotations

import time


def add_timing(cls):
    """Class decorator: wraps cls._predict_new_chunk with a timer. Records
    into self._chunk_latencies_s (created lazily), does not change return
    value or behavior."""
    original = cls._predict_new_chunk

    def timed(self, *args, **kwargs):
        if not hasattr(self, "_chunk_latencies_s"):
            self._chunk_latencies_s = []
        import torch

        torch.cuda.synchronize()
        t0 = time.perf_counter()
        result = original(self, *args, **kwargs)
        torch.cuda.synchronize()
        self._chunk_latencies_s.append(time.perf_counter() - t0)
        return result

    cls._predict_new_chunk = timed
    return cls
