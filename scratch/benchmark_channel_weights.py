"""Benchmark for adaptive channel-weight k-means.

Input: 80 time series of length 24 with 3 channels (artificial, no download).
Records: wall-clock training time, peak memory (tracemalloc), and the number
of DTW alignment computations (dtw_path calls + pairwise distances inside
cdist_dtw calls), for both the adaptive (channel-weight learning) and the
standard unweighted DTW k-means.
"""
import time
import tracemalloc
from unittest import mock

import numpy as np

import tslearn.clustering.kmeans as kmeans_module
import tslearn.barycenters.dba as dba_module
from tslearn.clustering import TimeSeriesKMeans

rng = np.random.RandomState(0)
n_ts, sz, d = 80, 24, 3
t = np.linspace(0, 4 * np.pi, sz)

# Three latent groups, each characterized by a different channel.
X = []
for i in range(n_ts):
    group = i % 3
    wave = np.sin(t + rng.randint(-3, 4))
    channels = [rng.randn(sz) for _ in range(3)]
    channels[group] = wave + 0.1 * rng.randn(sz)
    X.append(np.column_stack(channels))
X = np.array(X)
print("Dataset: %d series, length %d, %d channels" % X.shape)


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
            n2 = (n1 if dataset2 is None
                  else to_time_series_dataset(dataset2).shape[0])
            self.n_cdist_pairs += n1 * n2
            return orig(dataset1, dataset2, **kwargs)
        return wrapped


def run(name, **params):
    counter = AlignmentCounter()
    km = TimeSeriesKMeans(**params)
    patches = [
        mock.patch.object(kmeans_module, "dtw_path",
                          counter.wrap_dtw_path(kmeans_module.dtw_path)),
        mock.patch.object(kmeans_module, "cdist_dtw",
                          counter.wrap_cdist(kmeans_module.cdist_dtw)),
        # the standard (unweighted) k-means computes its barycenters through
        # dtw_barycenter_averaging, whose DTW calls live in dba's namespace
        mock.patch.object(dba_module, "dtw_path",
                          counter.wrap_dtw_path(dba_module.dtw_path)),
    ]
    for p in patches:
        p.start()
    try:
        tracemalloc.start()
        t0 = time.perf_counter()
        km.fit(X)
        elapsed = time.perf_counter() - t0
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
    finally:
        for p in patches:
            p.stop()
    total_alignments = counter.n_dtw_path + counter.n_cdist_pairs
    print("--- %s ---" % name)
    print("  wall time:            %.2f s" % elapsed)
    print("  peak memory:          %.1f MiB" % (peak / 2 ** 20))
    print("  dtw_path calls:       %d" % counter.n_dtw_path)
    print("  cdist pair distances: %d" % counter.n_cdist_pairs)
    print("  total alignments:     %d" % total_alignments)
    print("  n_iter_:              %d" % km.n_iter_)
    print("  inertia_:             %.4f" % km.inertia_)
    weights = getattr(km, "channel_weights_", None)
    if weights is not None:
        print("  objective_:           %.4f" % km.objective_)
        print("  channel_weights_:\n%s" % np.round(weights, 4))
    return km


run("adaptive (channel_weight_regularization=0.05)",
    n_clusters=3, metric="dtw", channel_weight_regularization=0.05,
    max_iter=20, n_init=2, random_state=0)
run("standard unweighted DTW",
    n_clusters=3, metric="dtw", max_iter=20, n_init=2, random_state=0)
