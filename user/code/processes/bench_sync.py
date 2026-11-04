"""
Waits until the GPU is idle. Placed after `iceflow`, whose GPU work is still in flight when IGM's timer stops:
iceflow + bench_sync = the true ice-flow time.
"""
import tensorflow as tf


def sync():
    try:
        tf.test.experimental.sync_devices()
    except (AttributeError, RuntimeError):      # older TF: a host fetch forces the sync
        tf.constant(0.0).numpy()


def initialize(cfg, state):
    sync()


def update(cfg, state):
    sync()


def finalize(cfg, state):
    pass
