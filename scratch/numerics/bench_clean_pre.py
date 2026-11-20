import sys, time
sys.path.insert(0, "/workspace/scratch/numerics")
from bench_common import X
from tslearn.clustering import TimeSeriesKMeans
def run(name, **params):
    TimeSeriesKMeans(**{**params, "n_init": 1, "max_iter": 1}).fit(X[:6])
    best = float("inf")
    for _ in range(3):
        t0 = time.perf_counter()
        km = TimeSeriesKMeans(**params).fit(X)
        best = min(best, time.perf_counter() - t0)
    print("%-10s best_of_3=%.3fs n_iter=%d inertia=%.4f"
          % (name, best, km.n_iter_, km.inertia_))
run("adaptive", n_clusters=3, metric="dtw", channel_weight_regularization=0.05,
    max_iter=20, n_init=2, random_state=0)
run("standard", n_clusters=3, metric="dtw", max_iter=20, n_init=2, random_state=0)
