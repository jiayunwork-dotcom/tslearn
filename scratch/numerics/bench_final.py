"""Final comparable benchmark: identical harness for the current code.

The pre-fix numbers were captured with bench_pre.py in the same
single-threaded environment and JIT-warmup convention.
"""
import sys, time, tracemalloc
from unittest import mock
sys.path.insert(0, "/workspace/scratch/numerics")
from bench_common import X
import numpy as np
import tslearn.clustering.kmeans as kmeans_module
import tslearn.barycenters.dba as dba_module
from tslearn.clustering import TimeSeriesKMeans

class Counter:
    def __init__(self):
        self.n = 0
    def wrap(self, fn):
        def w(*a, **k):
            self.n += 1
            return fn(*a, **k)
        return w

def run(name, **params):
    c_dba = Counter(); c_km = Counter()
    patches = [
        mock.patch.object(dba_module, "_robust_assignment",
                          c_dba.wrap(dba_module._robust_assignment)),
        mock.patch.object(kmeans_module, "_robust_assignment",
                          c_km.wrap(kmeans_module._robust_assignment)),
        mock.patch.object(kmeans_module, "cdist_dtw",
                          c_km.wrap(kmeans_module.cdist_dtw)),
    ]
    for p in patches: p.start()
    try:
        TimeSeriesKMeans(**{**params, "n_init": 1, "max_iter": 1}).fit(X[:6])
        c_dba.n = c_km.n = 0
        tracemalloc.start()
        t0 = time.perf_counter()
        km = TimeSeriesKMeans(**params).fit(X)
        elapsed = time.perf_counter() - t0
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
    finally:
        for p in patches: p.stop()
    print("--- %s ---" % name)
    print("  wall time: %.3f s" % elapsed)
    print("  peak memory: %.1f MiB" % (peak / 2**20))
    print("  alignment invocations (dba module): %d" % c_dba.n)
    print("  alignment invocations (kmeans module): %d" % c_km.n)
    print("  n_iter_: %d" % km.n_iter_)
    print("  inertia_: %.6f" % km.inertia_)
    if getattr(km, "objective_", None) is not None:
        print("  objective_: %.6f" % km.objective_)

run("adaptive", n_clusters=3, metric="dtw", channel_weight_regularization=0.05,
    max_iter=20, n_init=2, random_state=0)
run("standard", n_clusters=3, metric="dtw", max_iter=20, n_init=2, random_state=0)
