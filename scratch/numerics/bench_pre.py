import sys, time, tracemalloc
from unittest import mock
import numpy as np
sys.path.insert(0, "/workspace/scratch/numerics")
from bench_common import X, n_ts, sz, d
import tslearn.clustering.kmeans as kmeans_module
import tslearn.barycenters.dba as dba_module
from tslearn.clustering import TimeSeriesKMeans

class AlignmentCounter:
    def __init__(self):
        self.n_dtw_path = 0
        self.n_cdist_pairs = 0
    def wrap_dtw_path(self, orig):
        def wrapped(*args, **kwargs):
            self.n_dtw_path += 1
            return orig(*args, **kwargs)
        return wrapped
    def wrap_cdist(self, orig):
        def wrapped(dataset1, dataset2=None, **kwargs):
            from tslearn.utils import to_time_series_dataset
            n1 = to_time_series_dataset(dataset1).shape[0]
            n2 = (n1 if dataset2 is None else to_time_series_dataset(dataset2).shape[0])
            self.n_cdist_pairs += n1 * n2
            return orig(dataset1, dataset2, **kwargs)
        return wrapped

def run(name, **params):
    counter = AlignmentCounter()
    km = TimeSeriesKMeans(**params)
    patches = [
        mock.patch.object(kmeans_module, "dtw_path", counter.wrap_dtw_path(kmeans_module.dtw_path)),
        mock.patch.object(kmeans_module, "cdist_dtw", counter.wrap_cdist(kmeans_module.cdist_dtw)),
        mock.patch.object(dba_module, "dtw_path", counter.wrap_dtw_path(dba_module.dtw_path)),
    ]
    for p in patches: p.start()
    try:
        # warmup excluded: separate process per run, but trigger JIT compile
        # in a throwaway fit first
        TimeSeriesKMeans(**{**params, "n_init": 1, "max_iter": 1}).fit(X[:6])
        tracemalloc.start()
        t0 = time.perf_counter()
        km.fit(X)
        elapsed = time.perf_counter() - t0
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
    finally:
        for p in patches: p.stop()
    print("--- %s ---" % name)
    print("  wall time: %.3f s" % elapsed)
    print("  peak memory: %.1f MiB" % (peak / 2**20))
    print("  dtw_path calls: %d" % counter.n_dtw_path)
    print("  cdist pair distances: %d" % counter.n_cdist_pairs)
    print("  total alignments: %d" % (counter.n_dtw_path + counter.n_cdist_pairs))
    print("  n_iter_: %d" % km.n_iter_)
    print("  inertia_: %.6f" % km.inertia_)
    if hasattr(km, "objective_") and km.objective_ is not None:
        print("  objective_: %.6f" % km.objective_)
        print("  channel_weights_:\n%s" % np.round(km.channel_weights_, 4))

run("adaptive", n_clusters=3, metric="dtw", channel_weight_regularization=0.05,
    max_iter=20, n_init=2, random_state=0)
run("standard", n_clusters=3, metric="dtw", max_iter=20, n_init=2, random_state=0)
