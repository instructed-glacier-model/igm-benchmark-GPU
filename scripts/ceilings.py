#!/usr/bin/env python3
"""
What TensorFlow can get out of this GPU, to put the IGM times in perspective. Same TF install as the benchmark,
vendor-neutral (CUDA or ROCm). A few seconds per test.

  copy_GBps         streaming bandwidth: y = 2x on a 1 GiB float32 tensor (XLA), bytes read + written
  matmul_TFLOPs     float32 8192^2 matrix product
  conv_*            3x3 convolutions shaped like dahunet layers on the 400 m grid (1 x 1505 x 2255, NHWC),
                    C -> C channels, plain tf.function as in IGM's network: forward, backward w.r.t. the
                    input, backward w.r.t. the filter. Reported as achieved TFLOP/s (2 * H * W * 9 * C^2).

  ./scripts/ceilings.py --label rx_7900_xtx      # -> results/<label>/ceilings.json
"""
import argparse
import json
import os
import statistics
import time
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
import tensorflow as tf  # noqa: E402

H, W = 1505, 2255


def timeit(fn, reps=20, warm=3):
    for _ in range(warm):
        r = fn()
    _sync(r)
    ts = []
    for _ in range(reps):
        t = time.perf_counter()
        _sync(fn())
        ts.append(time.perf_counter() - t)
    return statistics.median(ts)


def _sync(r):
    r = r[0] if isinstance(r, (list, tuple)) else r
    float(tf.reduce_sum(tf.reshape(r, [-1])[:1]).numpy())


def copy_bw():
    x = tf.random.uniform([2**28])                       # 1 GiB
    f = tf.function(lambda: x * 2.0, jit_compile=True)
    return 2 * x.shape[0] * 4 / timeit(f) / 1e9


def matmul():
    n = 8192
    a = tf.random.normal([n, n])
    f = tf.function(lambda: tf.matmul(a, a))
    return 2 * n**3 / timeit(f) / 1e12


def conv(c):
    x = tf.random.normal([1, H, W, c])
    k = tf.random.normal([3, 3, c, c]) * 0.1
    flops = 2 * H * W * 9 * c * c
    fwd = tf.function(lambda: tf.nn.conv2d(x, k, 1, "SAME"))
    dy = fwd()
    bwd_in = tf.function(lambda: tf.compat.v1.nn.conv2d_backprop_input(x.shape, k, dy, [1, 1, 1, 1], "SAME"))
    bwd_k = tf.function(lambda: tf.compat.v1.nn.conv2d_backprop_filter(x, k.shape, dy, [1, 1, 1, 1], "SAME"))
    return {f"conv{c}_{n}_TFLOPs": flops / timeit(f) / 1e12
            for n, f in (("fwd", fwd), ("bwd_input", bwd_in), ("bwd_filter", bwd_k))}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--label", required=True, help="results/<label>/ceilings.json")
    a = p.parse_args()
    g = tf.config.list_physical_devices("GPU")
    out = {"tf_gpu": tf.config.experimental.get_device_details(g[0]).get("device_name") if g else None,
           "tensorflow": tf.__version__, "rocm_build": bool(tf.sysconfig.get_build_info().get("is_rocm_build")),
           "copy_GBps": copy_bw(), "matmul_TFLOPs": matmul()}
    for c in (24, 48):
        out.update(conv(c))
    out = {k: round(v, 2) if isinstance(v, float) else v for k, v in out.items()}
    print(json.dumps(out, indent=1))
    d = Path(__file__).resolve().parent.parent / "results" / a.label
    d.mkdir(parents=True, exist_ok=True)
    (d / "ceilings.json").write_text(json.dumps(out, indent=1) + "\n")


if __name__ == "__main__":
    main()
