"""Fair before/after comparison: count actual DP kernel invocations and
time both implementations in the same process layout."""
import sys, time
sys.path.insert(0, "/workspace/scratch/numerics")
from bench_common import X
import numpy as np
from tslearn.clustering import TimeSeriesKMeans
import tslearn.metrics.dtw_variants as dv

# Count kernel invocations at the lowest shared level.
orig_robust = dv._robust_dtw_path_and_score
counter = {"n": 0}
def counting_robust(*a, **k):
    counter["n"] += 1
    return orig_robust(*a, **k)
dv._robust_dtw_path_and_score = counting_robust
# dtw_path/dtw both dispatch through the helper in the new code; the old
# code called the njit kernels directly, so also count those entry points.
orig_njit = dv.njit_accumulated_matrix
orig_njit_w = dv.njit_accumulated_matrix_weighted
def c_njit(*a, **k):
    counter["n"] += 1
    return orig_njit(*a, **k)
def c_njit_w(*a, **k):
    counter["n"] += 1
    return orig_njit_w(*a, **k)

def measure(name, **params):
    counter["n"] = 0
    TimeSeriesKMeans(**{**params, "n_init": 1, "max_iter": 1}).fit(X[:6])  # warmup
    counter["n"] = 0
    t0 = time.perf_counter()
    km = TimeSeriesKMeans(**params).fit(X)
    elapsed = time.perf_counter() - t0
    print("%-10s time=%7.3fs  dp_invocations=%6d  n_iter=%d  inertia=%.4f"
          % (name, elapsed, counter["n"], km.n_iter_, km.inertia_))
    return elapsed, counter["n"]

measure("adaptive", n_clusters=3, metric="dtw", channel_weight_regularization=0.05,
        max_iter=20, n_init=2, random_state=0)
measure("standard", n_clusters=3, metric="dtw", max_iter=20, n_init=2, random_state=0)
