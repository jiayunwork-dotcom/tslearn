"""Resource benchmark for the exact pruned DTW k-NN search.

Setup (no downloads, fully synthetic):
  * training set: N=1024 series of length 64 with 2 channels, row i is
    the constant series [20*i, 20*i+1]
  * 24 queries: row (43*j) % 1024 of the training set plus 0.1, j=0..23
  * k=3, unconstrained DTW, channel weights [1, 1]

For each of the three entry points (KNeighborsTimeSeries.kneighbors,
KNeighborsTimeSeriesClassifier.predict_proba,
KNeighborsTimeSeriesRegressor.predict) we record, from fit up to full
consumption of the result:
  * wall-clock time (fit included, split into fit / query phases)
  * extra peak memory: Python allocations (tracemalloc) and process RSS
  * number of complete exact alignments and visited DP cells
  * time spent in lower-bound evaluation vs exact DTW fallback
and we check the returned values against an exact brute-force reference
(full pairwise cdist_dtw + stable sort, and the dtw_search='brute'
estimators). JIT compilation and a small-sample warmup happen before
any measurement. A second, hard-to-prune dataset of overlapping
waveforms is included to report the degraded cost.

Run with: python scratch/benchmark_pruned_knn.py
"""

import gc
import resource
import time
import tracemalloc

import numpy as np

from tslearn.metrics import cdist_dtw
from tslearn.neighbors import (
    KNeighborsTimeSeries,
    KNeighborsTimeSeriesClassifier,
    KNeighborsTimeSeriesRegressor,
)

N, L, D, K, NQ = 1024, 64, 2, 3, 24
WEIGHTS = [1.0, 1.0]
MIB = 1024.0 * 1024.0


def make_main_data():
    X = np.empty((N, L, D))
    idx = np.arange(N, dtype=np.float64)
    X[:, :, 0] = idx[:, None]
    X[:, :, 0] *= 20.0
    X[:, :, 1] = idx[:, None]
    X[:, :, 1] *= 20.0
    X[:, :, 1] += 1.0
    queries = X[[(43 * j) % N for j in range(NQ)]].copy() + 0.1
    y_class = np.arange(N) % 3
    y_reg = np.sin(idx / 10.0)
    return X, queries, y_class, y_reg


def make_overlapping_data(seed=0):
    """Hard-to-prune set: all series share essentially the same value
    range, so envelope lower bounds are weak."""
    rng = np.random.RandomState(seed)
    X = rng.randn(N, L, D).cumsum(axis=1)
    X -= X.min(axis=1, keepdims=True)
    X /= X.max(axis=1, keepdims=True)  # every series spans [0, 1]
    queries = rng.randn(NQ, L, D).cumsum(axis=1)
    queries -= queries.min(axis=1, keepdims=True)
    queries /= queries.max(axis=1, keepdims=True)
    return X, queries


def rss_mib():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def warmup():
    """Compile all numba kernels and warm the small-sample paths."""
    rng = np.random.RandomState(1)
    Xw = rng.randn(8, 6, D)
    Qw = rng.randn(3, 6, D)
    mp = {"channel_weights": WEIGHTS}
    knn = KNeighborsTimeSeries(n_neighbors=2, dtw_search="pruned",
                               metric_params=dict(mp)).fit(Xw)
    knn.kneighbors(Qw)
    clf = KNeighborsTimeSeriesClassifier(
        n_neighbors=2, dtw_search="pruned", metric_params=dict(mp))
    clf.fit(Xw, [0, 1, 0, 1, 0, 1, 0, 1]).predict_proba(Qw)
    reg = KNeighborsTimeSeriesRegressor(
        n_neighbors=2, dtw_search="pruned", metric_params=dict(mp))
    reg.fit(Xw, np.arange(8.0)).predict(Qw)
    cdist_dtw(Qw, Xw, channel_weights=WEIGHTS)  # reference path kernels


def measure(entry_name, fit_and_consume):
    """Run fit+entry under tracemalloc; return timing/memory/stats."""
    gc.collect()
    gc.collect()
    rss_before = rss_mib()
    tracemalloc.start()
    t0 = time.perf_counter()
    result, stats, t_fit = fit_and_consume()
    t1 = time.perf_counter()
    _, peak_py = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    rss_after = rss_mib()
    return {
        "entry": entry_name,
        "result": result,
        "stats": stats,
        "time_total": t1 - t0,
        "time_fit": t_fit,
        "py_peak_mib": peak_py / MIB,
        "rss_delta_mib": max(0.0, rss_after - rss_before),
    }


def fit_knn(X, Q):
    mp = {"channel_weights": WEIGHTS}
    t0 = time.perf_counter()
    knn = KNeighborsTimeSeries(n_neighbors=K, dtw_search="pruned",
                               metric_params=mp).fit(X)
    t_fit = time.perf_counter() - t0
    out = knn.kneighbors(Q)
    return out, dict(knn.search_stats_), t_fit


def fit_clf(X, Q, y):
    mp = {"channel_weights": WEIGHTS}
    t0 = time.perf_counter()
    clf = KNeighborsTimeSeriesClassifier(n_neighbors=K, dtw_search="pruned",
                                         metric_params=mp).fit(X, y)
    t_fit = time.perf_counter() - t0
    out = clf.predict_proba(Q)
    return out, dict(clf.search_stats_), t_fit


def fit_reg(X, Q, y):
    mp = {"channel_weights": WEIGHTS}
    t0 = time.perf_counter()
    reg = KNeighborsTimeSeriesRegressor(n_neighbors=K, dtw_search="pruned",
                                        metric_params=mp).fit(X, y)
    t_fit = time.perf_counter() - t0
    out = reg.predict(Q)
    return out, dict(reg.search_stats_), t_fit


def reference_neighbors(X, Q):
    """Exact reference: full pairwise cdist_dtw + stable (dist, row) sort."""
    cd = cdist_dtw(Q, X, channel_weights=WEIGHTS)
    order = np.lexsort((np.arange(N)[None, :].repeat(NQ, axis=0), cd),
                       axis=1)
    ind = order[:, :K]
    return cd[np.arange(NQ)[:, None], ind], ind


def report(title, m, ref_cells):
    s = m["stats"]
    print("  %-26s time=%7.3fs (fit=%6.3fs)  mem: py-peak=%6.2f MiB, "
          "rss-delta=%6.2f MiB" % (m["entry"], m["time_total"],
                                   m["time_fit"], m["py_peak_mib"],
                                   m["rss_delta_mib"]))
    print("  %-26s exact DTW calls: %6d / %d candidates   DP cells: "
          "%9d / %d (%.2f%%)" % (title, s["n_exact"], s["n_candidates"],
                                 s["n_dp_cells"], ref_cells,
                                 100.0 * s["n_dp_cells"] / ref_cells))
    print("  %-26s phase times (CPU-sum): lower bounds=%.4fs, "
          "exact fallback=%.4fs" % (title, s["time_lb"], s["time_exact"]))


def main():
    print("=" * 78)
    print("Pruned DTW k-NN benchmark: N=%d, L=%d, d=%d, Q=%d, k=%d, "
          "unconstrained DTW, weights=%s" % (N, L, D, NQ, K, WEIGHTS))
    print("=" * 78)
    X, Q, y_class, y_reg = make_main_data()
    warmup()
    gc.collect()

    ref_cells = NQ * N * L * L
    print("Reference (unpruned pairwise) work: %d alignments, %d DP cells"
          % (NQ * N, ref_cells))

    # Exact references (computed outside any measurement)
    t0 = time.perf_counter()
    ref_dist, ref_ind = reference_neighbors(X, Q)
    t_ref = time.perf_counter() - t0
    print("Reference cdist_dtw (24 x 1024) took %.2fs" % t_ref)
    mp = {"channel_weights": WEIGHTS}
    t0 = time.perf_counter()
    clf_b = KNeighborsTimeSeriesClassifier(n_neighbors=K, metric_params=mp)
    proba_ref = clf_b.fit(X, y_class).predict_proba(Q)
    t_brute_proba = time.perf_counter() - t0
    t0 = time.perf_counter()
    reg_b = KNeighborsTimeSeriesRegressor(n_neighbors=K, metric_params=mp)
    reg_ref = reg_b.fit(X, y_reg).predict(Q)
    t_brute_reg = time.perf_counter() - t0
    t0 = time.perf_counter()
    knn_b = KNeighborsTimeSeries(n_neighbors=K, metric_params=mp).fit(X)
    knn_b.kneighbors(Q)
    t_brute_knn = time.perf_counter() - t0
    print("Brute-force entry points (fit + call, incl. full distance "
          "matrix): kneighbors=%.2fs, predict_proba=%.2fs, predict=%.2fs"
          % (t_brute_knn, t_brute_proba, t_brute_reg))

    print("\n-- Main dataset (well-separated constant series) --")
    m_knn = measure("kneighbors", lambda: fit_knn(X, Q))
    m_clf = measure("predict_proba", lambda: fit_clf(X, Q, y_class))
    m_reg = measure("predict", lambda: fit_reg(X, Q, y_reg))
    for title, m in (("kneighbors", m_knn), ("predict_proba", m_clf),
                     ("predict", m_reg)):
        report(title, m, ref_cells)

    # Correctness against the exact references
    dist, ind = m_knn["result"]
    np.testing.assert_array_equal(ind, ref_ind)
    np.testing.assert_allclose(dist, ref_dist, rtol=1e-12)
    np.testing.assert_allclose(m_clf["result"], proba_ref, rtol=1e-12)
    np.testing.assert_allclose(m_reg["result"], reg_ref, rtol=1e-12)
    print("\n  Returned values match the exact brute-force references.")

    # Acceptance criteria
    ok = True
    for m in (m_knn, m_clf, m_reg):
        if m["py_peak_mib"] > 16.0:
            ok = False
            print("  FAIL: %s Python peak %.2f MiB > 16 MiB"
                  % (m["entry"], m["py_peak_mib"]))
        ratio = m["stats"]["n_dp_cells"] / ref_cells
        if ratio > 0.25:
            ok = False
            print("  FAIL: %s visited %.1f%% of the unpruned DP cells"
                  % (m["entry"], 100.0 * ratio))
    print("  Criteria (py-peak <= 16 MiB, cells <= 25%%): %s"
          % ("PASS" if ok else "FAIL"))

    print("\n-- Hard-to-prune dataset (overlapping waveforms) --")
    Xo, Qo = make_overlapping_data()
    warmup_done_note = "(JIT already warm)"
    m_hard = measure("kneighbors", lambda: fit_knn(Xo, Qo))
    report("overlapping", m_hard, ref_cells)
    dist_o, ind_o = m_hard["result"]
    ref_dist_o, ref_ind_o = reference_neighbors(Xo, Qo)
    np.testing.assert_array_equal(ind_o, ref_ind_o)
    np.testing.assert_allclose(dist_o, ref_dist_o, rtol=1e-12)
    print("  Returned values still match the exact reference %s."
          % warmup_done_note)

    print("\nDone.")


if __name__ == "__main__":
    main()
