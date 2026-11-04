"""
Last module of the step: waits for the GPU and stamps the step. At the end, writes bench.json with the grid,
the per-step times (step_s, iceflow_s = iceflow + bench_sync) and the TensorFlow memory peak.
"""
import json
import time

import tensorflow as tf

from bench_sync import sync


def initialize(cfg, state):
    sync()
    state.bench_stamps = [time.perf_counter()]


def update(cfg, state):
    sync()
    state.bench_stamps.append(time.perf_counter())


def finalize(cfg, state):
    s = state.bench_stamps
    tc = getattr(state, "tcomp", {})
    flow = [a + b for a, b in zip(tc.get("iceflow", []), tc.get("bench_sync", []))]
    try:
        info = tf.config.experimental.get_memory_info("GPU:0")
        peak = info["peak"] / 2**20
    except Exception:
        peak = None
    out = {
        "nx": int(state.thk.shape[-1]), "ny": int(state.thk.shape[-2]),
        "dx": float(state.dx.numpy() if hasattr(state.dx, "numpy") else state.dx),
        "step_s": [b - a for a, b in zip(s[:-1], s[1:])],
        "iceflow_s": flow,
        "tf_peak_mib": peak,
    }
    with open(cfg.processes.bench_timer.out_file, "w") as f:
        json.dump(out, f)
    print(f"[bench_timer] {len(out['step_s'])} steps, TF peak {peak} MiB -> "
          f"{cfg.processes.bench_timer.out_file}")
