"""Tests for the exact pruned DTW nearest-neighbor search
(``dtw_search='pruned'``) and for ``dtw(..., max_dist=...)``."""

import pickle

import numpy as np
import pytest
from joblib import Parallel, delayed
from sklearn.base import clone

import tslearn.metrics
from tslearn.backend.backend import Backend
from tslearn.metrics import cdist_dtw, dtw
from tslearn.metrics.dtw_variants import (
    GLOBAL_CONSTRAINT_CODE,
    _check_max_dist,
    compute_mask,
)
from tslearn.neighbors import (
    KNeighborsTimeSeries,
    KNeighborsTimeSeriesClassifier,
    KNeighborsTimeSeriesRegressor,
)
from tslearn.utils import to_time_series_dataset

__author__ = "tslearn contributors"


# ---------------------------------------------------------------------------
# Reference implementations (independent of the pruned search machinery)
# ---------------------------------------------------------------------------

def _enum_legal_paths(n, m, mask=None):
    """Enumerate all legal warping paths between (0, 0) and (n-1, m-1)."""
    paths = []

    def rec(i, j, path):
        if i == n - 1 and j == m - 1:
            paths.append(path + [(i, j)])
            return
        for di, dj in ((0, 1), (1, 0), (1, 1)):
            ni, nj = i + di, j + dj
            if ni < n and nj < m and (
                    mask is None or np.isfinite(mask[ni, nj])):
                rec(ni, nj, path + [(i, j)])

    if mask is None or np.isfinite(mask[0, 0]):
        rec(0, 0, [])
    return paths


def _enum_dtw(s1, s2, mask=None, weights=None):
    """DTW computed by exhaustive enumeration of legal warping paths."""
    s1 = np.asarray(s1, dtype=float).reshape((-1, 1) if np.ndim(s1) == 1
                                             else (len(s1), -1))
    s2 = np.asarray(s2, dtype=float).reshape((-1, 1) if np.ndim(s2) == 1
                                             else (len(s2), -1))
    if weights is None:
        weights = np.ones(s1.shape[1])
    best = np.inf
    for path in _enum_legal_paths(len(s1), len(s2), mask):
        cost = 0.0
        for i, j in path:
            cost += np.sum(weights * (s1[i] - s2[j]) ** 2)
        best = min(best, cost)
    return np.sqrt(best)


def _reference_kneighbors(train, queries, k, exclude_self=False, **dtw_kw):
    """Exact k-NN by explicit pairwise DTW + stable sort on (dist, row).

    This reference never touches the pruned search: every pairwise
    distance is computed with tslearn.metrics.dtw and the k smallest
    (distance, training row) pairs are kept. With exclude_self=True,
    the training row with the query's index is removed from the
    candidates entirely (and only that row).
    """
    n = len(train)
    all_dist, all_ind = [], []
    for qi, q in enumerate(queries):
        dists = np.empty(n)
        for i in range(n):
            dists[i] = dtw(q, train[i], **dtw_kw)
        if exclude_self:
            cand = np.array([i for i in range(n) if i != qi])
        else:
            cand = np.arange(n)
        order = cand[np.lexsort((cand, dists[cand]))]
        all_ind.append(order[:k])
        all_dist.append(dists[order[:k]])
    return np.array(all_dist), np.array(all_ind)


def _trimmed(dataset):
    """Strip trailing all-NaN padding rows from each series of a dataset."""
    from tslearn.utils import ts_size
    return [np.asarray(ts)[:ts_size(ts)] for ts in np.asarray(dataset)]


# ---------------------------------------------------------------------------
# Datasets
# ---------------------------------------------------------------------------

def _mixed_dataset():
    """Multivariate, variable-length (NaN padded), with a duplicate pair."""
    X = to_time_series_dataset([
        [[0.0, 1.0], [1.0, 2.0], [2.0, 1.0], [3.0, 0.0]],
        [[0.5, 1.5], [1.5, 2.5], [2.5, 1.5]],
        [[0.0, 1.0], [1.0, 2.0], [2.0, 1.0], [3.0, 0.0]],  # duplicate of 0
        [[-1.0, 0.0], [0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0]],
        [[4.0, 4.0], [4.0, 4.0]],
    ])
    return X


def _random_dataset(n=12, sz=8, d=2, seed=0):
    rng = np.random.RandomState(seed)
    return rng.randn(n, sz, d)


# ---------------------------------------------------------------------------
# dtw(max_dist=...)
# ---------------------------------------------------------------------------

def test_max_dist_against_path_enumeration():
    rng = np.random.RandomState(42)
    cases = [
        ([1, 2, 3], [1.0, 2.0, 2.0, 3.0]),
        ([1, 2, 3], [1.0, 2.0, 2.0, 3.0, 4.0]),
        ([0.5, -1.0], [2.0, 0.0, -3.0]),
        (rng.randn(4), rng.randn(3)),
        (rng.randn(4, 2), rng.randn(4, 2)),
        (rng.randn(2, 3), rng.randn(4, 3)),
    ]
    for s1, s2 in cases:
        s1_arr = np.asarray(s1, dtype=float)
        s2_arr = np.asarray(s2, dtype=float)
        exact = _enum_dtw(s1_arr, s2_arr)
        # dtw() agrees with the enumeration of legal paths
        np.testing.assert_allclose(dtw(s1, s2), exact, rtol=1e-12)
        # cutoff below / equal / above the true distance, and 0
        if exact > 0:
            assert dtw(s1, s2, max_dist=exact * 0.5) == np.inf
        assert dtw(s1, s2, max_dist=exact) == exact
        assert dtw(s1, s2, max_dist=exact + 1.0) == exact
        if exact == 0.0:
            assert dtw(s1, s2, max_dist=0.0) == 0.0
        else:
            assert dtw(s1, s2, max_dist=0.0) == np.inf


def test_max_dist_with_constraints_and_weights():
    rng = np.random.RandomState(1)
    s1 = rng.randn(4, 2)
    s2 = rng.randn(5, 2)
    for gc, kw in [("sakoe_chiba", {"sakoe_chiba_radius": 1}),
                   ("itakura", {"itakura_max_slope": 2.0})]:
        mask = compute_mask(s1, s2, GLOBAL_CONSTRAINT_CODE[gc], be=None,
                            **kw)
        exact = _enum_dtw(s1, s2, mask=mask)
        got = dtw(s1, s2, global_constraint=gc, **kw)
        np.testing.assert_allclose(got, exact, rtol=1e-12)
        assert dtw(s1, s2, global_constraint=gc, max_dist=exact, **kw) == exact
        if exact > 0:
            assert dtw(s1, s2, global_constraint=gc,
                       max_dist=exact * 0.99, **kw) == np.inf
    # Weighted variant, checked against weighted path enumeration
    w = np.array([1.0, 0.5])
    exact = _enum_dtw(s1, s2, weights=w)
    np.testing.assert_allclose(dtw(s1, s2, channel_weights=w), exact,
                               rtol=1e-12)
    assert dtw(s1, s2, channel_weights=w, max_dist=exact) == exact
    if exact > 0:
        assert dtw(s1, s2, channel_weights=w,
                   max_dist=exact * 0.99) == np.inf
    # Zero-weight channel: only channel 0 matters
    w0 = np.array([1.0, 0.0])
    exact0 = _enum_dtw(s1[:, :1], s2[:, :1])
    np.testing.assert_allclose(dtw(s1, s2, channel_weights=w0), exact0,
                               rtol=1e-12)
    assert dtw(s1, s2, channel_weights=w0, max_dist=exact0) == exact0


def test_max_dist_validation():
    for bad in (-1.0, np.inf, -np.inf, np.nan, True, np.bool_(False),
                "1.0", [1.0], np.array([0.5, 0.6]), complex(1, 1)):
        with pytest.raises(ValueError):
            dtw([1, 2], [1.0, 2.0], max_dist=bad)
    # None keeps the standard behavior
    assert dtw([1, 2], [1.0, 2.0], max_dist=None) == 0.0
    # integer cutoffs are accepted
    assert dtw([1, 2], [1.0, 2.0], max_dist=0) == 0.0
    # Non-NumPy backends reject a non-None cutoff (simulated here since
    # torch is not necessarily installed)
    fake_be = Backend("numpy")
    fake_be.is_numpy = False
    with pytest.raises(ValueError):
        _check_max_dist(1.0, be=fake_be)
    try:
        import torch  # noqa: F401
    except ImportError:
        pass
    else:
        with pytest.raises(ValueError):
            dtw([1, 2], [1.0, 2.0], max_dist=1.0, be="pytorch")


def test_max_dist_does_not_leak_into_cdist():
    rng = np.random.RandomState(3)
    X = rng.randn(4, 5, 2)
    Y = rng.randn(3, 6, 2)
    ref = cdist_dtw(X, Y)
    # cdist_dtw has no max_dist parameter: the cutoff used by the pruned
    # search cannot leak into the full-matrix API
    with pytest.raises(TypeError):
        cdist_dtw(X, Y, max_dist=1.0)
    # Exhaustive pruned searches do not affect later cdist_dtw calls
    knn = KNeighborsTimeSeries(n_neighbors=2, dtw_search="pruned").fit(X)
    knn.kneighbors(Y)
    np.testing.assert_array_equal(cdist_dtw(X, Y), ref)


# ---------------------------------------------------------------------------
# Pruned kneighbors
# ---------------------------------------------------------------------------

def test_pruned_kneighbors_matches_pairwise_reference():
    X = _mixed_dataset()
    queries = to_time_series_dataset([
        [[0.1, 1.1], [1.1, 2.1], [2.1, 1.1]],
        [[-1.0, 0.0], [0.0, 0.0], [1.0, 0.0], [2.0, 0.0]],
    ])
    for k in (1, 2, 3, 5):
        knn = KNeighborsTimeSeries(n_neighbors=k, dtw_search="pruned")
        knn.fit(X)
        dist, ind = knn.kneighbors(queries)
        ref_dist, ref_ind = _reference_kneighbors(
            _trimmed(X), _trimmed(queries), k)
        np.testing.assert_array_equal(ind, ref_ind)
        np.testing.assert_allclose(dist, ref_dist, rtol=1e-12)
        # distances are sorted, ties broken by row index
        for row_d, row_i in zip(dist, ind):
            assert list(zip(row_d, row_i)) == sorted(zip(row_d, row_i))


def test_pruned_kneighbors_self_query_excludes_only_same_row():
    X = _mixed_dataset()  # rows 0 and 2 are identical
    knn = KNeighborsTimeSeries(n_neighbors=2, dtw_search="pruned").fit(X)
    dist, ind = knn.kneighbors()
    ref_dist, ref_ind = _reference_kneighbors(
        _trimmed(X), _trimmed(X), 2, exclude_self=True)
    np.testing.assert_array_equal(ind, ref_ind)
    np.testing.assert_allclose(dist, ref_dist, rtol=1e-12)
    # the identical twin row is returned (distance 0), the row itself is not
    np.testing.assert_array_equal(ind[0, 0], 2)
    np.testing.assert_array_equal(ind[2, 0], 0)
    assert dist[0, 0] == 0.0 and dist[2, 0] == 0.0
    for qi in range(len(X)):
        assert qi not in ind[qi]

    # Passing the training array explicitly disables self-exclusion:
    # every row finds a zero-distance neighbor (itself or its duplicate)
    dist2, ind2 = knn.kneighbors(X)
    ref_dist2, ref_ind2 = _reference_kneighbors(
        _trimmed(X), _trimmed(X), 2, exclude_self=False)
    np.testing.assert_array_equal(ind2, ref_ind2)
    np.testing.assert_allclose(dist2, ref_dist2, rtol=1e-12)
    np.testing.assert_allclose(dist2[:, 0], 0.0, atol=0.0)
    # and the tie between identical rows 0 and 2 is broken by row index
    np.testing.assert_array_equal(ind2[0, 0], 0)
    np.testing.assert_array_equal(ind2[2, 0], 0)


def test_pruned_kneighbors_truncates_large_k():
    X = _mixed_dataset()
    n = len(X)
    knn = KNeighborsTimeSeries(n_neighbors=n + 3, dtw_search="pruned")
    knn.fit(X)
    dist, ind = knn.kneighbors(X)
    assert dist.shape == (n, n) and ind.shape == (n, n)
    # self queries are truncated to n - 1 neighbors
    dist_self, ind_self = knn.kneighbors()
    assert dist_self.shape == (n, n - 1) and ind_self.shape == (n, n - 1)


def test_pruned_kneighbors_invalid_k():
    X = _mixed_dataset()
    knn = KNeighborsTimeSeries(n_neighbors=2, dtw_search="pruned").fit(X)
    with pytest.raises(ValueError):
        knn.kneighbors(X, n_neighbors=0)
    with pytest.raises(ValueError):
        knn.kneighbors(X, n_neighbors=-2)
    with pytest.raises(TypeError):
        knn.kneighbors(X, n_neighbors=1.5)


def test_pruned_kneighbors_with_constraints_and_weights():
    rng = np.random.RandomState(7)
    X = rng.randn(8, 6, 2)
    Q = rng.randn(3, 5, 2)
    param_sets = [
        {"channel_weights": [1.0, 0.0]},
        {"channel_weights": [2.0, 0.5]},
        {"global_constraint": "sakoe_chiba", "sakoe_chiba_radius": 1},
        {"global_constraint": "itakura", "itakura_max_slope": 2.0},
        {"sakoe_chiba_radius": 2, "channel_weights": [1.0, 3.0]},
    ]
    for mp in param_sets:
        knn = KNeighborsTimeSeries(n_neighbors=3, dtw_search="pruned",
                                   metric_params=dict(mp))
        knn.fit(X)
        dist, ind = knn.kneighbors(Q)
        ref_dist, ref_ind = _reference_kneighbors(X, Q, 3, **mp)
        np.testing.assert_array_equal(ind, ref_ind)
        np.testing.assert_allclose(dist, ref_dist, rtol=1e-12)


def test_pruned_kneighbors_variable_length_and_weights():
    X = _mixed_dataset()
    Q = to_time_series_dataset([
        [[0.2, 1.2], [1.2, 2.2], [2.2, 1.2], [3.2, 0.2]],
        [[4.1, 4.1]],
    ])
    mp = {"channel_weights": [1.0, 0.25]}
    knn = KNeighborsTimeSeries(n_neighbors=3, dtw_search="pruned",
                               metric_params=dict(mp)).fit(X)
    dist, ind = knn.kneighbors(Q)
    ref_dist, ref_ind = _reference_kneighbors(
        _trimmed(X), _trimmed(Q), 3, **mp)
    np.testing.assert_array_equal(ind, ref_ind)
    np.testing.assert_allclose(dist, ref_dist, rtol=1e-12)


def test_pruned_kneighbors_n_jobs_consistency():
    X = _random_dataset()
    Q = _random_dataset(n=5, seed=5)
    results = {}
    for n_jobs in (1, 2, -1):
        knn = KNeighborsTimeSeries(n_neighbors=4, dtw_search="pruned",
                                   n_jobs=n_jobs).fit(X)
        results[n_jobs] = knn.kneighbors(Q)
        # single-threaded and parallel runs prune for real
        stats = knn.search_stats_
        assert stats["n_exact"] + stats["n_pruned"] == stats["n_candidates"]
    np.testing.assert_array_equal(results[1][1], results[2][1])
    np.testing.assert_array_equal(results[1][1], results[-1][1])
    np.testing.assert_allclose(results[1][0], results[2][0])
    np.testing.assert_allclose(results[1][0], results[-1][0])


def test_pruned_search_actually_prunes_single_threaded():
    # Well-separated constant series: the lower bound is nearly exact, so
    # only a handful of exact DTW computations should survive per query.
    n, sz, d = 64, 16, 2
    X = np.empty((n, sz, d))
    for i in range(n):
        X[i, :, 0] = 20.0 * i
        X[i, :, 1] = 20.0 * i + 1.0
    Q = X[::16].copy() + 0.1
    knn = KNeighborsTimeSeries(n_neighbors=3, dtw_search="pruned",
                               n_jobs=1).fit(X)
    dist, ind = knn.kneighbors(Q)
    stats = knn.search_stats_
    assert stats["n_exact"] <= 4 * len(Q)  # << 64 candidates per query
    ref_dist, ref_ind = _reference_kneighbors(X, Q, 3)
    np.testing.assert_array_equal(ind, ref_ind)
    np.testing.assert_allclose(dist, ref_dist, rtol=1e-12)
    # no N x N placeholder matrix was allocated at fit time
    assert knn._X_fit.size == n


def test_pruned_fit_rejects_invalid_metric_combinations():
    X = _mixed_dataset()
    y = [0, 1, 0, 1, 0]
    for est in (KNeighborsTimeSeries(dtw_search="pruned", metric="softdtw"),
                KNeighborsTimeSeries(dtw_search="pruned", metric="euclidean"),
                KNeighborsTimeSeries(dtw_search="pruned",
                                     metric=tslearn.metrics.cdist_dtw),
                KNeighborsTimeSeries(dtw_search="bogus")):
        with pytest.raises(ValueError):
            est.fit(X)
    with pytest.raises(ValueError):
        KNeighborsTimeSeriesClassifier(
            dtw_search="pruned", metric="ctw").fit(X, y)
    with pytest.raises(ValueError):
        KNeighborsTimeSeriesRegressor(
            dtw_search="pruned", metric="softdtw").fit(X, y)
    # invalid metric_params are rejected at fit time
    with pytest.raises(ValueError):
        KNeighborsTimeSeries(dtw_search="pruned",
                             metric_params={"unknown_key": 1}).fit(X)
    with pytest.raises(ValueError):
        KNeighborsTimeSeries(
            dtw_search="pruned",
            metric_params={"channel_weights": [1.0]}).fit(X)  # wrong length


def test_pruned_fit_rejects_invalid_data():
    y = [0, 1, 0, 1]
    # internal NaN (NaN row followed by valid rows)
    X_internal_nan = to_time_series_dataset(
        [[1, 2], [np.nan], [3, 4], [1, 2]])
    # partially NaN time step
    X_partial_nan = np.full((2, 3, 2), 1.0)
    X_partial_nan[0, 1, 0] = np.nan
    # infinite values
    X_inf = np.full((2, 3, 1), 1.0)
    X_inf[1, 2, 0] = np.inf
    # empty series (all-NaN padding)
    X_empty = to_time_series_dataset([[1, 2, 3], [np.nan, np.nan]])
    X_empty[1, :] = np.nan
    for bad_X in (X_internal_nan, X_partial_nan, X_inf, X_empty):
        with pytest.raises(ValueError):
            KNeighborsTimeSeries(dtw_search="pruned").fit(bad_X)
        with pytest.raises(ValueError):
            KNeighborsTimeSeriesClassifier(dtw_search="pruned").fit(
                bad_X, y[:len(bad_X)])
    # dimension mismatch between fit and query
    knn = KNeighborsTimeSeries(dtw_search="pruned").fit(_mixed_dataset())
    with pytest.raises(ValueError):
        knn.kneighbors(np.ones((2, 4, 3)))
    # invalid query data raises and does not pollute the estimator
    with pytest.raises(ValueError):
        knn.kneighbors(X_partial_nan)
    assert knn.metric == "dtw"
    dist, ind = knn.kneighbors(_mixed_dataset()[:2])
    assert dist.shape == (2, 5)


def test_pruned_refit_replaces_state_and_failed_refit_is_safe():
    X1 = _random_dataset(n=8, seed=1)
    X2 = _random_dataset(n=10, seed=2)
    Q = _random_dataset(n=3, seed=3)
    knn = KNeighborsTimeSeries(n_neighbors=2, dtw_search="pruned").fit(X1)
    knn.fit(X2)
    assert knn._pruned_index_.n_samples_ == 10
    fresh = KNeighborsTimeSeries(n_neighbors=2, dtw_search="pruned").fit(X2)
    np.testing.assert_array_equal(knn.kneighbors(Q)[1],
                                  fresh.kneighbors(Q)[1])
    # a failed refit keeps the previous, fully usable model
    clf = KNeighborsTimeSeriesClassifier(n_neighbors=2,
                                         dtw_search="pruned")
    clf.fit(X1, [0, 1, 0, 1, 0, 1, 0, 1])
    pred_before = clf.predict(Q)
    with pytest.raises(ValueError):
        clf.fit(np.full((4, 5, 2), np.inf), [0, 1, 0, 1])
    np.testing.assert_array_equal(clf.predict(Q), pred_before)
    assert clf.metric == "dtw"


def test_pruned_set_params_semantics():
    X = _random_dataset(n=10, seed=4)
    Q = _random_dataset(n=3, seed=6)
    knn = KNeighborsTimeSeries(n_neighbors=3, dtw_search="pruned").fit(X)
    # Changing the metric away from dtw invalidates the pruned path
    knn.set_params(metric="softdtw")
    with pytest.raises(ValueError):
        knn.kneighbors(Q)
    knn.set_params(metric="dtw")
    # Distance parameters are re-read at query time (the index only
    # depends on the training data), exactly like the brute path
    knn.set_params(metric_params={"sakoe_chiba_radius": 1})
    dist, ind = knn.kneighbors(Q)
    ref_dist, ref_ind = _reference_kneighbors(
        X, Q, 3, sakoe_chiba_radius=1)
    np.testing.assert_array_equal(ind, ref_ind)
    np.testing.assert_allclose(dist, ref_dist, rtol=1e-12)
    # invalid new parameters are rejected at query time
    knn.set_params(metric_params={"channel_weights": [1.0, 2.0, 3.0]})
    with pytest.raises(ValueError):
        knn.kneighbors(Q)


def test_pruned_index_rebuilt_after_deserialization():
    X = _random_dataset(n=8, seed=8)
    Q = _random_dataset(n=2, seed=9)
    knn = KNeighborsTimeSeries(n_neighbors=2, dtw_search="pruned").fit(X)
    expected = knn.kneighbors(Q)
    # simulate a serialization format that only carries _ts_fit
    del knn._pruned_index_
    np.testing.assert_allclose(knn.kneighbors(Q)[0], expected[0])
    # pickle round-trip keeps behavior
    knn2 = KNeighborsTimeSeriesClassifier(n_neighbors=2,
                                          dtw_search="pruned")
    knn2.fit(X, [0, 1, 0, 1, 0, 1, 0, 1])
    pred = knn2.predict(Q)
    knn2_roundtrip = pickle.loads(pickle.dumps(knn2))
    np.testing.assert_array_equal(knn2_roundtrip.predict(Q), pred)
    np.testing.assert_array_equal(
        knn2_roundtrip.kneighbors(Q)[1], knn2.kneighbors(Q)[1])


def test_pruned_clone_and_no_caller_mutation():
    est = KNeighborsTimeSeriesClassifier(
        n_neighbors=3, dtw_search="pruned", metric="dtw",
        metric_params={"channel_weights": [1.0, 0.5]})
    cloned = clone(est)
    assert cloned.get_params()["dtw_search"] == "pruned"

    X = _random_dataset(n=9, seed=11)
    Q = _random_dataset(n=2, seed=12)
    X_copy = X.copy()
    Q_copy = Q.copy()
    mp = {"channel_weights": np.array([1.0, 0.5])}
    mp_copy = {"channel_weights": mp["channel_weights"].copy()}
    est2 = KNeighborsTimeSeriesClassifier(n_neighbors=3,
                                          dtw_search="pruned",
                                          metric_params=mp)
    est2.fit(X, [0, 1, 0, 1, 0, 1, 0, 1, 0])
    est2.predict(Q)
    est2.predict_proba(Q)
    est2.kneighbors(Q)
    np.testing.assert_array_equal(X, X_copy)
    np.testing.assert_array_equal(Q, Q_copy)
    np.testing.assert_array_equal(mp["channel_weights"],
                                  mp_copy["channel_weights"])
    assert set(mp.keys()) == set(mp_copy.keys())


# ---------------------------------------------------------------------------
# Supervised estimators
# ---------------------------------------------------------------------------

def _clf_data(seed=0):
    rng = np.random.RandomState(seed)
    X = rng.randn(14, 7, 2)
    y = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1])
    Q = rng.randn(5, 6, 2)
    return X, y, Q


@pytest.mark.parametrize("weights", ["uniform", "distance",
                                     lambda d: 1.0 / (np.abs(d) + 1.0)])
def test_pruned_classifier_matches_brute(weights):
    X, y, Q = _clf_data()
    clf_p = KNeighborsTimeSeriesClassifier(n_neighbors=4, weights=weights,
                                           dtw_search="pruned").fit(X, y)
    clf_b = KNeighborsTimeSeriesClassifier(n_neighbors=4, weights=weights,
                                           dtw_search="brute").fit(X, y)
    np.testing.assert_array_equal(clf_p.predict(Q), clf_b.predict(Q))
    np.testing.assert_allclose(clf_p.predict_proba(Q),
                               clf_b.predict_proba(Q), rtol=1e-12)
    np.testing.assert_allclose(clf_p.predict_proba(Q).sum(axis=1), 1.0)


def test_pruned_classifier_zero_distance_and_duplicates():
    X = _mixed_dataset()
    y = np.array([0, 1, 2, 1, 2])
    # query exactly equal to training rows 0 and 2 (duplicates)
    Q = X[:1].copy()
    for weights in ("uniform", "distance"):
        clf_p = KNeighborsTimeSeriesClassifier(
            n_neighbors=3, weights=weights, dtw_search="pruned").fit(X, y)
        clf_b = KNeighborsTimeSeriesClassifier(
            n_neighbors=3, weights=weights, dtw_search="brute").fit(X, y)
        np.testing.assert_array_equal(clf_p.predict(Q), clf_b.predict(Q))
        np.testing.assert_allclose(clf_p.predict_proba(Q),
                                   clf_b.predict_proba(Q))
        # zero-distance neighbors are present and dominate
        assert clf_p.predict(Q)[0] == 0


@pytest.mark.parametrize("weights", ["uniform", "distance",
                                     lambda d: np.exp(-np.abs(d))])
def test_pruned_regressor_matches_brute(weights):
    rng = np.random.RandomState(10)
    X = rng.randn(12, 6, 2)
    y = rng.randn(12)
    Q = rng.randn(4, 6, 2)
    reg_p = KNeighborsTimeSeriesRegressor(n_neighbors=3, weights=weights,
                                          dtw_search="pruned").fit(X, y)
    reg_b = KNeighborsTimeSeriesRegressor(n_neighbors=3, weights=weights,
                                          dtw_search="brute").fit(X, y)
    np.testing.assert_allclose(reg_p.predict(Q), reg_b.predict(Q),
                               rtol=1e-12)


def test_pruned_regressor_multioutput():
    rng = np.random.RandomState(11)
    X = rng.randn(10, 5, 2)
    Y = rng.randn(10, 3)
    Q = rng.randn(4, 5, 2)
    for weights in ("uniform", "distance"):
        reg_p = KNeighborsTimeSeriesRegressor(
            n_neighbors=3, weights=weights, dtw_search="pruned").fit(X, Y)
        reg_b = KNeighborsTimeSeriesRegressor(
            n_neighbors=3, weights=weights, dtw_search="brute").fit(X, Y)
        pred_p = reg_p.predict(Q)
        assert pred_p.shape == (4, 3)
        np.testing.assert_allclose(pred_p, reg_b.predict(Q), rtol=1e-12)


def test_pruned_classifier_multioutput():
    rng = np.random.RandomState(16)
    X = rng.randn(10, 5, 2)
    Y = np.column_stack([rng.randint(0, 2, 10), rng.randint(0, 3, 10)])
    Q = rng.randn(4, 5, 2)
    clf_p = KNeighborsTimeSeriesClassifier(n_neighbors=3,
                                           dtw_search="pruned").fit(X, Y)
    clf_b = KNeighborsTimeSeriesClassifier(n_neighbors=3,
                                           dtw_search="brute").fit(X, Y)
    np.testing.assert_array_equal(clf_p.predict(Q), clf_b.predict(Q))
    proba_p = clf_p.predict_proba(Q)
    proba_b = clf_b.predict_proba(Q)
    assert isinstance(proba_p, list) and len(proba_p) == 2
    for p_, b_ in zip(proba_p, proba_b):
        np.testing.assert_allclose(p_, b_, rtol=1e-12)


def test_pruned_supervised_k_exceeding_samples_raises():
    X, y, Q = _clf_data()
    clf = KNeighborsTimeSeriesClassifier(n_neighbors=len(X) + 1,
                                         dtw_search="pruned").fit(X, y)
    with pytest.raises(ValueError):
        clf.predict(Q)
    with pytest.raises(ValueError):
        clf.predict_proba(Q)
    reg = KNeighborsTimeSeriesRegressor(n_neighbors=len(X) + 1,
                                        dtw_search="pruned")
    reg.fit(X, np.arange(len(X), dtype=float))
    with pytest.raises(ValueError):
        reg.predict(Q)


def _infeasible_setup():
    # itakura_max_slope=1.0 is infeasible between length-3 queries and
    # length-10 training series: every pairwise DTW distance is inf
    import warnings
    X = np.random.RandomState(0).randn(4, 10, 1)
    Q = np.random.RandomState(1).randn(2, 3, 1)
    mp = {"global_constraint": "itakura", "itakura_max_slope": 1.0}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        ref = cdist_dtw(Q, X, **mp)
    assert np.isinf(ref).all()
    return X, Q, mp


def test_pruned_all_inf_distance_weights_raise():
    X, Q, mp = _infeasible_setup()
    y = [0, 1, 0, 1]
    clf = KNeighborsTimeSeriesClassifier(n_neighbors=2, weights="distance",
                                         dtw_search="pruned",
                                         metric_params=mp).fit(X, y)
    with pytest.warns(RuntimeWarning), pytest.raises(ValueError):
        clf.predict(Q)
    with pytest.warns(RuntimeWarning), pytest.raises(ValueError):
        clf.predict_proba(Q)
    reg = KNeighborsTimeSeriesRegressor(n_neighbors=2, weights="distance",
                                        dtw_search="pruned",
                                        metric_params=mp)
    reg.fit(X, [0.0, 1.0, 0.0, 1.0])
    with pytest.warns(RuntimeWarning), pytest.raises(ValueError):
        reg.predict(Q)
    # uniform weights still deliver predictions (inf distances are
    # sorted by training row, exactly like the reference)
    clf_u = KNeighborsTimeSeriesClassifier(n_neighbors=2, weights="uniform",
                                           dtw_search="pruned",
                                           metric_params=mp).fit(X, y)
    with pytest.warns(RuntimeWarning):
        pred = clf_u.predict(Q)
    assert set(pred) <= set(y)


def test_pruned_inf_distances_sorted_by_row():
    # Only training row 1 (length 3, identical to the query) is at finite
    # distance: pairs with the length-10 rows are infeasible under
    # itakura_max_slope=1.0, hence inf, and are selected ordered by row.
    rng = np.random.RandomState(2)
    X = np.full((4, 10, 1), np.nan)
    for i in (0, 2, 3):
        X[i] = rng.randn(10, 1)
    X[1, :3, 0] = rng.randn(3)
    Q = X[1:2, :3].copy()
    mp = {"global_constraint": "itakura", "itakura_max_slope": 1.0}
    knn = KNeighborsTimeSeries(n_neighbors=3, dtw_search="pruned",
                               metric_params=mp).fit(X)
    with pytest.warns(RuntimeWarning):
        dist, ind = knn.kneighbors(Q)
    with pytest.warns(RuntimeWarning):
        ref_dist, ref_ind = _reference_kneighbors(X, Q, 3, **mp)
    np.testing.assert_array_equal(ind, ref_ind)
    np.testing.assert_array_equal(dist, ref_dist)
    assert ind[0, 0] == 1 and dist[0, 0] == 0.0
    assert np.isinf(dist[0, 1:]).all()
    assert list(ind[0, 1:]) == sorted(ind[0, 1:])


def test_pruned_self_query_with_infeasible_pairs():
    # Self-queries under a constraint that makes most pairs infeasible:
    # the query's own row must be excluded even when inf-distance
    # neighbors are selected (regression test: an infinite lower bound
    # alone does not exclude the self row, it must be skipped).
    rng = np.random.RandomState(5)
    X = np.full((4, 10, 1), np.nan)
    X[0] = rng.randn(10, 1)          # length 10
    X[1, :3, 0] = rng.randn(3)       # length 3
    X[2] = rng.randn(10, 1)          # length 10
    X[3, :3, 0] = rng.randn(3)       # length 3
    mp = {"global_constraint": "itakura", "itakura_max_slope": 1.0}
    knn = KNeighborsTimeSeries(n_neighbors=2, dtw_search="pruned",
                               metric_params=mp).fit(X)
    with pytest.warns(RuntimeWarning):
        dist, ind = knn.kneighbors()
    with pytest.warns(RuntimeWarning):
        ref_dist, ref_ind = _reference_kneighbors(
            _trimmed(X), _trimmed(X), 2, exclude_self=True, **mp)
    np.testing.assert_array_equal(ind, ref_ind)
    np.testing.assert_array_equal(dist, ref_dist)
    for qi in range(4):
        assert qi not in ind[qi]
    # length-3 rows 1 and 3 are each other's only finite-distance
    # neighbor; their other neighbors are at infinite distance
    assert np.isfinite(dist[1]).sum() == 1
    assert np.isfinite(dist[3]).sum() == 1
    assert ind[1, 0] == 3 and ind[3, 0] == 1


def test_pruned_supervised_parallel_and_batched_consistency():
    X, y, Q = _clf_data()
    clf = KNeighborsTimeSeriesClassifier(n_neighbors=3, dtw_search="pruned",
                                         n_jobs=2).fit(X, y)
    proba_full = clf.predict_proba(Q)
    pred_full = clf.predict(Q)
    # batched inference dispatched through joblib gives identical results
    chunks = [Q[i:i + 2] for i in range(0, len(Q), 2)]
    proba_chunks = Parallel(n_jobs=2, prefer="threads")(
        delayed(clf.predict_proba)(c) for c in chunks)
    np.testing.assert_allclose(np.vstack(proba_chunks), proba_full)
    reg = KNeighborsTimeSeriesRegressor(n_neighbors=3, dtw_search="pruned",
                                        n_jobs=2)
    reg.fit(X, np.arange(len(X), dtype=float))
    reg_pred_full = reg.predict(Q)
    pred_chunks = Parallel(n_jobs=2, prefer="threads")(
        delayed(reg.predict)(c) for c in chunks)
    np.testing.assert_allclose(np.hstack(pred_chunks), reg_pred_full)


def test_pruned_kneighbors_graph_and_return_distance_false():
    X = _random_dataset(n=8, seed=13)
    knn = KNeighborsTimeSeries(n_neighbors=3, dtw_search="pruned").fit(X)
    ind_only = knn.kneighbors(X, return_distance=False)
    dist, ind = knn.kneighbors(X)
    np.testing.assert_array_equal(ind_only, ind)
    assert dist.shape == ind.shape == (8, 3)
    # kneighbors_graph goes through the same exact pruned search
    graph = knn.kneighbors_graph(X, mode="distance")
    np.testing.assert_allclose(
        np.asarray(graph.todense()).max(axis=1), dist[:, -1], rtol=1e-12)


def test_pruned_sklearn_format_2d_input():
    X = _random_dataset(n=8, sz=6, d=1, seed=14)
    knn = KNeighborsTimeSeries(n_neighbors=2, dtw_search="pruned").fit(X)
    # 2-D (n_ts, sz) queries are accepted, as in the brute path
    Q2d = _random_dataset(n=3, sz=6, d=1, seed=15)[:, :, 0]
    dist, ind = knn.kneighbors(Q2d)
    ref_dist, ref_ind = _reference_kneighbors(
        X, Q2d[:, :, None], 2)
    np.testing.assert_array_equal(ind, ref_ind)
    np.testing.assert_allclose(dist, ref_dist, rtol=1e-12)
    # 2-D training data works as well
    knn2d = KNeighborsTimeSeries(n_neighbors=2, dtw_search="pruned")
    knn2d.fit(X[:, :, 0])
    np.testing.assert_array_equal(knn2d.kneighbors(Q2d)[1],
                                  knn.kneighbors(Q2d)[1])
