"""Tests for channel-weighted DTW (fixed weights) and for adaptive
per-cluster channel-weight learning (TimeSeriesKMeans with
channel_weight_regularization and dtw_barycenter_averaging with the same
regularization).
"""

import pickle
from unittest import mock

import numpy as np
import pytest
from sklearn.base import clone
from sklearn.exceptions import NotFittedError

import tslearn.clustering.kmeans as kmeans_module
import tslearn.barycenters.dba as dba_module
from tslearn.barycenters import dtw_barycenter_averaging
from tslearn.barycenters.dba import (
    _check_min_channel_weight,
    _check_channel_weight_regularization,
    _check_sample_weights_strict,
    _check_init_channel_weights,
    _entropy,
    _solve_channel_weights_entropy,
)
from tslearn.clustering import TimeSeriesKMeans
from tslearn.clustering.utils import EmptyClusterError
from tslearn.metrics import cdist_dtw, dtw, dtw_path
from tslearn.metrics.dtw_variants import (
    _check_channel_weights,
    itakura_mask,
    njit_sakoe_chiba_mask,
)
from tslearn.utils import to_time_series_dataset

__author__ = "Romain Tavenard romain.tavenard[at]univ-rennes2.fr"


# ---------------------------------------------------------------------------
# Independent reference implementations (pure Python, no numba)
# ---------------------------------------------------------------------------

def _enum_paths(n1, n2, mask=None):
    """Enumerate all legal DTW paths from (0, 0) to (n1-1, n2-1).

    A path is a sequence of index pairs starting at (0, 0), ending at
    (n1-1, n2-1), with steps in {(1, 0), (0, 1), (1, 1)}. If a mask is
    given, only cells with finite mask values may be visited.
    """
    paths = []

    def rec(i, j, acc):
        if i == n1 - 1 and j == n2 - 1:
            paths.append(acc + [(i, j)])
            return
        for di, dj in ((1, 0), (0, 1), (1, 1)):
            ni, nj = i + di, j + dj
            if ni < n1 and nj < n2 and (
                mask is None or np.isfinite(mask[ni, nj])
            ):
                rec(ni, nj, acc + [(i, j)])

    if mask is None or np.isfinite(mask[0, 0]):
        rec(0, 0, [])
    return paths


def _path_cost(x, y, w, path):
    """Cost of a path: sum over aligned positions and channels of
    w[c] * (x[i, c] - y[j, c])**2."""
    cost = 0.0
    for i, j in path:
        for c in range(x.shape[1]):
            cost += w[c] * (x[i, c] - y[j, c]) ** 2
    return cost


def _brute_force_dtw_sq(x, y, w, mask=None):
    """Minimum weighted path cost over all legal paths (exact enumeration)."""
    best = np.inf
    for path in _enum_paths(x.shape[0], y.shape[0], mask=mask):
        best = min(best, _path_cost(x, y, w, path))
    return best


def _py_weighted_dtw(x, y, w, mask=None):
    """Pure-Python weighted DTW (dynamic programming), independent of the
    numba kernels used inside tslearn."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    w = np.asarray(w, dtype=np.float64)
    n1, n2 = len(x), len(y)
    acc = np.full((n1 + 1, n2 + 1), np.inf)
    acc[0, 0] = 0.0
    for i in range(n1):
        for j in range(n2):
            if mask is not None and not np.isfinite(mask[i, j]):
                continue
            cost = float(np.sum(w * (x[i] - y[j]) ** 2))
            acc[i + 1, j + 1] = cost + min(
                acc[i, j + 1], acc[i + 1, j], acc[i, j]
            )
    return np.sqrt(acc[n1, n2])


def _py_dba_objective(X, sample_weights, barycenter, channel_weights, lam):
    """Independent recomputation of the adaptive DBA objective."""
    sw = np.asarray(sample_weights, dtype=np.float64)
    total = 0.0
    for i in range(len(X)):
        d = _py_weighted_dtw(barycenter, X[i], channel_weights)
        total += sw[i] * d ** 2
    return total / sw.sum() + lam * _entropy(channel_weights)


def _py_kmeans_objective(X, labels, centers, channel_weights, lam):
    """Independent recomputation of the adaptive k-means objective J."""
    n = len(X)
    total = 0.0
    for i in range(n):
        k = labels[i]
        d = _py_weighted_dtw(centers[k], X[i], channel_weights[k])
        total += d ** 2
    return total / n + lam * _entropy(channel_weights.ravel())


def _make_two_group_dataset(rng, n_per=6, sz=8):
    """Two groups of 2-channel series: group 0 has structure in channel 0
    and noise in channel 1, group 1 the other way around."""
    t = np.linspace(0, 2 * np.pi, sz)
    xs = []
    ys = []
    for i in range(n_per):
        sig = np.sin(t + 0.2 * rng.randn())
        xs.append(np.column_stack([sig + 0.05 * rng.randn(sz),
                                   2.0 * rng.randn(sz)]))
        ys.append(np.column_stack([2.0 * rng.randn(sz),
                                   sig + 0.05 * rng.randn(sz)]))
    return np.array(xs + ys), np.array([0] * n_per + [1] * n_per)


# ---------------------------------------------------------------------------
# Fixed channel weights: validation
# ---------------------------------------------------------------------------

def test_check_channel_weights_validation():
    # None means no weighting
    assert _check_channel_weights(None, 3) is None

    w = _check_channel_weights([1, 2, 0.5], 3)
    np.testing.assert_allclose(w, [1.0, 2.0, 0.5])
    assert w.dtype == np.float64

    # integer arrays are accepted and converted
    w = _check_channel_weights(np.array([1, 0, 2]), 3)
    np.testing.assert_allclose(w, [1.0, 0.0, 2.0])

    # the caller's array must not be modified
    arr = np.array([1.0, 2.0])
    w = _check_channel_weights(arr, 2)
    w[0] = 99.0
    np.testing.assert_allclose(arr, [1.0, 2.0])

    # wrong length
    with pytest.raises(ValueError):
        _check_channel_weights([1.0, 2.0], 3)
    # 2-d array
    with pytest.raises(ValueError):
        _check_channel_weights([[1.0, 2.0]], 2)
    # scalar
    with pytest.raises(ValueError):
        _check_channel_weights(1.5, 1)
    # negative entries
    with pytest.raises(ValueError):
        _check_channel_weights([1.0, -0.5], 2)
    # all zero
    with pytest.raises(ValueError):
        _check_channel_weights([0.0, 0.0], 2)
    # non-finite
    with pytest.raises(ValueError):
        _check_channel_weights([1.0, np.nan], 2)
    with pytest.raises(ValueError):
        _check_channel_weights([1.0, np.inf], 2)
    # booleans are not numeric weights
    with pytest.raises(ValueError):
        _check_channel_weights([True, False], 2)
    with pytest.raises(ValueError):
        _check_channel_weights(True, 1)
    # non-numeric types
    with pytest.raises(ValueError):
        _check_channel_weights(["a", "b"], 2)
    with pytest.raises(ValueError):
        _check_channel_weights([1 + 2j, 0], 2)
    with pytest.raises(ValueError):
        _check_channel_weights(None if False else object(), 1)


def test_check_channel_weights_non_numpy_backend():
    from tslearn.backend.backend import Backend

    class NonNumpyBackend(Backend):
        """A stand-in non-NumPy backend (torch is not installed here)."""

        def __init__(self):
            super().__init__("numpy")
            self.is_numpy = False
            self.backend_string = "notnumpy"

    with pytest.raises(ValueError):
        _check_channel_weights([1.0, 1.0], 2, be=NonNumpyBackend())


def test_fixed_weight_entry_points_validation():
    rng = np.random.RandomState(0)
    x = rng.randn(4, 2)
    y = rng.randn(5, 2)
    for bad in ([1.0], [1.0, -1.0], [0.0, 0.0], [1.0, np.nan], [True, True]):
        with pytest.raises(ValueError):
            dtw(x, y, channel_weights=bad)
        with pytest.raises(ValueError):
            dtw_path(x, y, channel_weights=bad)
        with pytest.raises(ValueError):
            cdist_dtw(x[None], y[None], channel_weights=bad)
    # dimension mismatch between series is still reported
    with pytest.raises(ValueError):
        dtw(x, rng.randn(5, 3), channel_weights=[1.0, 1.0])


# ---------------------------------------------------------------------------
# Fixed channel weights: exactness against brute-force path enumeration
# ---------------------------------------------------------------------------

def test_fixed_weights_against_brute_force():
    rng = np.random.RandomState(42)
    d = 2
    weight_sets = [
        np.array([1.0, 1.0]),
        np.array([1.0, 0.0]),
        np.array([0.0, 1.0]),
        np.array([2.5, 0.25]),
        np.array([0.0, 3.0]),
    ]
    sizes = [(3, 4), (4, 4), (2, 5), (5, 3)]
    for sz1, sz2 in sizes:
        x = rng.randn(sz1, d)
        y = rng.randn(sz2, d)
        for w in weight_sets:
            expected = _brute_force_dtw_sq(x, y, w)
            # dtw
            got = dtw(x, y, channel_weights=w) ** 2
            np.testing.assert_allclose(got, expected, rtol=1e-10, atol=1e-12)
            # dtw_path: the returned path is legal and optimal, and its
            # cost matches the returned distance
            path, dist = dtw_path(x, y, channel_weights=w)
            assert path[0] == (0, 0)
            assert path[-1] == (sz1 - 1, sz2 - 1)
            for (i0, j0), (i1, j1) in zip(path[:-1], path[1:]):
                assert (i1 - i0, j1 - j0) in [(0, 1), (1, 0), (1, 1)]
            np.testing.assert_allclose(
                _path_cost(x, y, w, path), dist ** 2, rtol=1e-10
            )
            np.testing.assert_allclose(dist ** 2, expected, rtol=1e-10)
            # independent pure-Python DP agrees too
            np.testing.assert_allclose(
                _py_weighted_dtw(x, y, w) ** 2, expected, rtol=1e-10
            )


def test_fixed_weights_against_brute_force_constrained():
    rng = np.random.RandomState(7)
    d = 2
    w = np.array([1.5, 0.5])
    for sz1, sz2 in [(4, 4), (5, 4), (3, 5)]:
        x = rng.randn(sz1, d)
        y = rng.randn(sz2, d)
        # Sakoe-Chiba
        mask = njit_sakoe_chiba_mask(sz1, sz2, radius=1)
        expected = _brute_force_dtw_sq(x, y, w, mask=mask)
        got = dtw(x, y, global_constraint="sakoe_chiba",
                  sakoe_chiba_radius=1, channel_weights=w) ** 2
        np.testing.assert_allclose(got, expected, rtol=1e-10)
        path, dist = dtw_path(x, y, global_constraint="sakoe_chiba",
                              sakoe_chiba_radius=1, channel_weights=w)
        for i, j in path:
            assert np.isfinite(mask[i, j])
        np.testing.assert_allclose(dist ** 2, expected, rtol=1e-10)
        # Itakura
        mask = itakura_mask(sz1, sz2, max_slope=2.0)
        expected = _brute_force_dtw_sq(x, y, w, mask=mask)
        got = dtw(x, y, global_constraint="itakura",
                  itakura_max_slope=2.0, channel_weights=w) ** 2
        np.testing.assert_allclose(got, expected, rtol=1e-10)


def test_fixed_weights_cdist_and_semantics():
    rng = np.random.RandomState(1)
    X = rng.randn(4, 5, 2)
    Y = rng.randn(3, 6, 2)
    w = np.array([2.0, 0.5])
    cd = cdist_dtw(X, Y, channel_weights=w)
    for i in range(X.shape[0]):
        for j in range(Y.shape[0]):
            np.testing.assert_allclose(
                cd[i, j], dtw(X[i], Y[j], channel_weights=w), rtol=1e-12
            )
    # self-similarity
    cd_self = cdist_dtw(X, channel_weights=w)
    np.testing.assert_allclose(np.diag(cd_self), 0.0, atol=1e-12)
    np.testing.assert_allclose(cd_self, cd_self.T, rtol=1e-12)

    # None and all-ones are compatible with the original distance
    x, y = X[0], Y[0]
    np.testing.assert_allclose(dtw(x, y), dtw(x, y, channel_weights=None))
    np.testing.assert_allclose(
        dtw(x, y), dtw(x, y, channel_weights=np.ones(2)), rtol=1e-12
    )
    np.testing.assert_allclose(
        dtw_path(x, y)[1], dtw_path(x, y, channel_weights=np.ones(2))[1],
        rtol=1e-12
    )

    # zero-weight channels do not participate in the path cost
    x2 = x.copy()
    x2[:, 1] = 1000.0  # huge values in channel 1
    np.testing.assert_allclose(
        dtw(x, y, channel_weights=[1.0, 0.0]),
        dtw(x2, y, channel_weights=[1.0, 0.0]),
        rtol=1e-12
    )

    # multiplying all weights by a positive constant scales the distance
    # by its square root
    for c in [0.5, 2.0, 7.0]:
        np.testing.assert_allclose(
            dtw(x, y, channel_weights=c * w),
            np.sqrt(c) * dtw(x, y, channel_weights=w),
            rtol=1e-12,
        )

    # channel permutation symmetry
    perm = [1, 0]
    np.testing.assert_allclose(
        dtw(x[:, perm], y[:, perm], channel_weights=w[perm]),
        dtw(x, y, channel_weights=w),
        rtol=1e-12,
    )


def test_shared_path_vs_per_channel_independent():
    """A counterexample proving that channels share a single alignment path
    (summing per-channel independent DTWs would give a different value)."""
    # Channel 0 of x matches channel 0 of y with zero cost via its own
    # warping path, same for channel 1, but no single shared path matches
    # both channels at zero cost.
    x = np.array([[0.0, 1.0], [1.0, 1.0], [1.0, 0.0]])
    y = np.array([[0.0, 1.0], [0.0, 0.0], [1.0, 0.0]])
    w = np.ones(2)

    # Per-channel independent DTWs are both zero:
    np.testing.assert_allclose(dtw(x[:, 0], y[:, 0]), 0.0, atol=1e-12)
    np.testing.assert_allclose(dtw(x[:, 1], y[:, 1]), 0.0, atol=1e-12)
    per_channel_sum = dtw(x[:, 0], y[:, 0]) + dtw(x[:, 1], y[:, 1])

    # The shared-path weighted DTW is strictly positive:
    shared = dtw(x, y, channel_weights=w)
    expected = np.sqrt(_brute_force_dtw_sq(x, y, w))
    np.testing.assert_allclose(shared, expected, rtol=1e-12)
    assert shared > 0.5
    assert shared != pytest.approx(per_channel_sum, abs=1e-8)

    # With a zero weight on either channel, the cost vanishes (the other
    # channel aligns perfectly on its own):
    np.testing.assert_allclose(
        dtw(x, y, channel_weights=[1.0, 0.0]), 0.0, atol=1e-12
    )
    np.testing.assert_allclose(
        dtw(x, y, channel_weights=[0.0, 1.0]), 0.0, atol=1e-12
    )


def test_fixed_weights_variable_length_and_nans():
    # Unequal lengths with trailing all-NaN padding keep working
    X = to_time_series_dataset(
        [[[1.0, 0.0], [2.0, 1.0], [3.0, 2.0]],
         [[1.0, 1.0], [2.0, 0.0]]]
    )
    assert np.isnan(X[1, 2]).all()
    w = [1.0, 2.0]
    d_padded = dtw(X[0], X[1], channel_weights=w)
    d_stripped = dtw(X[0], X[1][:2], channel_weights=w)
    np.testing.assert_allclose(d_padded, d_stripped, rtol=1e-12)
    cd = cdist_dtw(X, channel_weights=w)
    np.testing.assert_allclose(cd[0, 1], d_padded, rtol=1e-12)


# ---------------------------------------------------------------------------
# Weight solver and hyper-parameter validation
# ---------------------------------------------------------------------------

def test_solve_channel_weights_entropy():
    rng = np.random.RandomState(0)
    lam = 0.3
    eps = 1e-6
    for d in [1, 2, 3, 7]:
        for _ in range(20):
            A = rng.rand(d) * 10
            A[rng.randint(d)] = 0.0  # include exact zeros
            w = _solve_channel_weights_entropy(A, lam, eps)
            assert w.shape == (d,)
            assert np.all(w >= eps)
            np.testing.assert_allclose(w.sum(), d, rtol=1e-12, atol=1e-12)
            # KKT optimality check: for the objective
            # f(w) = sum w_c A_c + lam * sum w_c log w_c on the simplex,
            # the gradient components at free coordinates are all equal
            # and pinned coordinates have a larger gradient.
            grad = A + lam * (np.log(w) + 1.0)
            free = w > eps + 1e-12
            if d > 1 and np.any(free):
                np.testing.assert_allclose(
                    grad[free], grad[free][0], rtol=1e-6, atol=1e-8
                )
                assert np.all(grad[~free] >= grad[free].min() - 1e-6)
    # uniform costs give uniform weights
    w = _solve_channel_weights_entropy(np.ones(4), 0.5, eps)
    np.testing.assert_allclose(w, np.ones(4), rtol=1e-10)
    # d == 1 is forced to 1
    np.testing.assert_allclose(
        _solve_channel_weights_entropy(np.array([5.0]), 0.1, eps), [1.0]
    )


def test_hyperparameter_validation():
    for bad in [0.0, -1.0, np.nan, np.inf, "x", True, np.bool_(True), None]:
        with pytest.raises(ValueError):
            _check_channel_weight_regularization(bad)
    for good in [0.5, 1, 2.5, np.float64(0.1), np.int64(3)]:
        assert _check_channel_weight_regularization(good) == float(good)
    for bad in [0.0, 1.0, -0.5, 1.5, np.nan, np.inf, "x", True]:
        with pytest.raises(ValueError):
            _check_min_channel_weight(bad)
    for good in [0.5, 1e-6, 0.999]:
        assert _check_min_channel_weight(good) == float(good)


def test_sample_weights_strict_validation():
    # None means uniform
    np.testing.assert_allclose(_check_sample_weights_strict(None, 3),
                               [1.0, 1.0, 1.0])
    w = _check_sample_weights_strict([0.5, 0.5, 0.0], 3)
    np.testing.assert_allclose(w, [0.5, 0.5, 0.0])
    for bad in ([1.0, 1.0],          # wrong length
                [[1.0, 1.0, 1.0]],   # not 1-d
                [1.0, -1.0, 1.0],    # negative
                [0.0, 0.0, 0.0],     # all zero
                [1.0, np.nan, 1.0],  # non-finite
                [1.0, np.inf, 1.0],
                [True, False, True],  # bool
                "abc"):
        with pytest.raises(ValueError):
            _check_sample_weights_strict(bad, 3)


def test_init_channel_weights_validation():
    w = _check_init_channel_weights(None, 3, 1e-6)
    np.testing.assert_allclose(w, [1.0, 1.0, 1.0])
    w = _check_init_channel_weights([2.0, 0.5, 0.5], 3, 1e-6)
    np.testing.assert_allclose(w, [2.0, 0.5, 0.5])
    # sum must be d
    with pytest.raises(ValueError):
        _check_init_channel_weights([2.0, 2.0, 2.0], 3, 1e-6)
    with pytest.raises(ValueError):
        _check_init_channel_weights([1.0, 1.0, 0.5], 3, 1e-6)
    # entries below the lower bound
    with pytest.raises(ValueError):
        _check_init_channel_weights([2.9, 0.05, 0.05], 3, 0.1)
    # negative / non-finite / bool
    with pytest.raises(ValueError):
        _check_init_channel_weights([4.0, -0.5, -0.5], 3, 1e-6)
    with pytest.raises(ValueError):
        _check_init_channel_weights([1.0, np.nan, 2.0], 3, 1e-6)


# ---------------------------------------------------------------------------
# Adaptive DBA (dtw_barycenter_averaging with channel_weight_regularization)
# ---------------------------------------------------------------------------

def test_dba_adaptive_default_mode_unchanged():
    # Without the regularization parameter, the original return type and
    # values are preserved.
    b = dtw_barycenter_averaging([[1, 2, 3, 4], [1, 2, 4, 5]], max_iter=5)
    assert isinstance(b, np.ndarray)
    np.testing.assert_allclose(b.ravel(), [1.0, 2.0, 3.5, 4.5])


def test_dba_adaptive_weights_move_and_objective_decreases():
    rng = np.random.RandomState(0)
    n, sz = 6, 8
    t = np.linspace(0, 2 * np.pi, sz)
    X = np.array([
        np.column_stack([np.sin(t) + 0.05 * rng.randn(sz),
                         2.0 * rng.randn(sz)])
        for _ in range(n)
    ])
    lam = 0.05
    bary, w, history = dtw_barycenter_averaging(
        X, channel_weight_regularization=lam, max_iter=30
    )
    # constraints on the learned weights
    assert w.shape == (2,)
    np.testing.assert_allclose(w.sum(), 2.0, rtol=1e-10, atol=1e-12)
    assert np.all(w >= 1e-6)
    # weights moved towards the informative channel
    assert w[0] > 1.5 > w[1]
    # history is finite and non-increasing (with the specified tolerance)
    assert all(np.isfinite(history))
    for prev, nxt in zip(history[:-1], history[1:]):
        assert nxt <= prev + 1e-8 * max(1.0, abs(prev))
    # the final objective is better than the starting point (same initial
    # center, all-ones weights)
    assert history[-1] < history[0]
    # independent recomputation of the final objective
    j_final = _py_dba_objective(X, np.ones(n), bary, w, lam)
    np.testing.assert_allclose(history[-1], j_final, rtol=1e-8)
    # and of the initial state (all-ones weights)
    j_init = _py_dba_objective(
        X, np.ones(n),
        dtw_barycenter_averaging_one_init_barycenter(X),
        np.ones(2), lam,
    )
    np.testing.assert_allclose(history[0], j_init, rtol=1e-8)


def dtw_barycenter_averaging_one_init_barycenter(X):
    """Reproduce the default deterministic initial barycenter used by DBA."""
    from tslearn.barycenters.dba import _init_avg
    X_ = to_time_series_dataset(X)
    return _init_avg(X_, X_.shape[1])


def test_dba_adaptive_history_matches_independent_recomputation():
    # Record every accepted state through a spy on the overflow-safe
    # per-series alignment entry point and recompute each recorded
    # objective independently (pure-Python weighted DTW).
    rng = np.random.RandomState(3)
    n, sz, d = 4, 5, 2
    X = rng.randn(n, sz, d) * 0.1
    X[:, :, 0] += np.sin(np.linspace(0, 3, sz))[None, :]
    lam = 0.2
    states = []

    # The spy intercepts the state evaluation wrapper so that one recorded
    # state corresponds to each entry of the objective history.
    orig_evaluate = dba_module._evaluate_scaled

    def spy_evaluate(bary, w, X_, weights, metric_params, regularization):
        states.append((bary.copy(), w.copy(),
                       np.asarray(weights).copy()))
        return orig_evaluate(bary, w, X_, weights, metric_params,
                             regularization)

    with mock.patch.object(dba_module, "_evaluate_scaled", spy_evaluate):
        bary, w, history = dtw_barycenter_averaging(
            X, channel_weight_regularization=lam, max_iter=10
        )
    assert len(states) == len(history)
    for (b_state, w_state, sw_state), recorded in zip(states, history):
        j_indep = _py_dba_objective(X, sw_state, b_state, w_state, lam)
        np.testing.assert_allclose(recorded, j_indep, rtol=1e-8)
    # final state consistency
    np.testing.assert_allclose(bary, states[-1][0], rtol=1e-12)
    np.testing.assert_allclose(w, states[-1][1], rtol=1e-12)


def test_dba_adaptive_tiny_problem_global_optimum():
    """On a tiny single-cluster problem, compare the delivered objective
    with an exact enumeration of all path combinations (global optimum)."""
    # Two samples of lengths 2 and 3, two channels.
    X = np.array([
        [[0.0, 0.0], [1.0, 0.2]],
        [[0.1, 0.0], [0.9, 0.1], [1.8, 0.0]],
    ], dtype=object)
    X = np.array([np.asarray(xi) for xi in X], dtype=object)
    # to_time_series_dataset padding
    X_ = to_time_series_dataset(list(X))
    lam = 0.4
    eps = 1e-6
    d = 2

    bary, w, history = dtw_barycenter_averaging(
        X_, channel_weight_regularization=lam, min_channel_weight=eps,
        max_iter=50, tol=1e-12
    )

    # Exact reference: enumerate all pairs of legal paths (barycenter size
    # is the max series length, as in DBA's default), for each compute the
    # optimal barycenter in closed form and the optimal weights on a fine
    # 1-d grid refined by golden-section search.
    from scipy.optimize import minimize_scalar

    sz_bary = bary.shape[0]
    x0, x1 = X_[0][:2], X_[1]
    paths0 = _enum_paths(sz_bary, 2)
    paths1 = _enum_paths(sz_bary, 3)
    best_j = np.inf
    for p0 in paths0:
        for p1 in paths1:
            # optimal barycenter for these paths (uniform sample weights)
            b = np.zeros((sz_bary, d))
            for t in range(sz_bary):
                vals = [x0[i] for (tt, i) in p0 if tt == t]
                vals += [x1[i] for (tt, i) in p1 if tt == t]
                b[t] = np.mean(vals, axis=0)
            # per-channel costs A_c for this barycenter and these paths
            A = np.zeros(d)
            for c in range(d):
                for (tt, i) in p0:
                    A[c] += (b[tt, c] - x0[i, c]) ** 2
                for (tt, i) in p1:
                    A[c] += (b[tt, c] - x1[i, c]) ** 2
            A /= 2.0

            def obj(t_, A=A):
                w_ = np.array([t_, d - t_])
                return (A * w_).sum() + lam * _entropy(w_)

            res = minimize_scalar(obj, bounds=(eps, d - eps),
                                  method="bounded",
                                  options={"xatol": 1e-15})
            best_j = min(best_j, res.fun)
    # Our delivered objective cannot beat the exact global optimum, and
    # for this tiny problem it should essentially reach it.
    assert history[-1] >= best_j - 1e-10
    np.testing.assert_allclose(history[-1], best_j, rtol=1e-6, atol=1e-8)


def test_dba_adaptive_sample_weights_and_validation():
    rng = np.random.RandomState(0)
    X = rng.randn(4, 5, 2)
    lam = 0.2
    # sample weights are used in the objective (weighted average)
    w_sample = np.array([2.0, 1.0, 1.0, 0.5])
    bary, w, history = dtw_barycenter_averaging(
        X, channel_weight_regularization=lam, weights=w_sample, max_iter=5
    )
    j_indep = _py_dba_objective(X, w_sample, bary, w, lam)
    np.testing.assert_allclose(history[-1], j_indep, rtol=1e-8)

    # invalid sample weights are rejected in adaptive mode
    for bad in ([1.0, 1.0], [1.0, -1.0, 1.0, 1.0], [0.0] * 4,
                [1.0, np.nan, 1.0, 1.0], [True, True, False, True]):
        with pytest.raises(ValueError):
            dtw_barycenter_averaging(
                X, channel_weight_regularization=lam, weights=bad
            )
    # invalid regularization / lower bound / init weights
    with pytest.raises(ValueError):
        dtw_barycenter_averaging(X, channel_weight_regularization=-1.0)
    with pytest.raises(ValueError):
        dtw_barycenter_averaging(X, channel_weight_regularization=True)
    with pytest.raises(ValueError):
        dtw_barycenter_averaging(X, channel_weight_regularization=0.5,
                                 min_channel_weight=1.5)
    with pytest.raises(ValueError):
        dtw_barycenter_averaging(X, channel_weight_regularization=0.5,
                                 init_channel_weights=[2.0, 2.0])  # sum != 2
    with pytest.raises(ValueError):
        dtw_barycenter_averaging(
            X, channel_weight_regularization=0.5, min_channel_weight=0.4,
            init_channel_weights=[1.9, 0.1])  # below lower bound
    # fixed channel_weights in metric_params conflict with learning
    with pytest.raises(ValueError):
        dtw_barycenter_averaging(
            X, channel_weight_regularization=0.5,
            metric_params={"channel_weights": [1.0, 1.0]})
    # valid init weights are accepted and not modified
    init_w = np.array([1.5, 0.5])
    init_w_copy = init_w.copy()
    dtw_barycenter_averaging(X, channel_weight_regularization=0.5,
                             init_channel_weights=init_w, max_iter=2)
    np.testing.assert_allclose(init_w, init_w_copy)


def test_dba_adaptive_constraints_and_n_init():
    rng = np.random.RandomState(11)
    X = rng.randn(5, 6, 3)
    bary, w, history = dtw_barycenter_averaging(
        X, channel_weight_regularization=0.1, min_channel_weight=0.01,
        n_init=2, max_iter=5
    )
    np.testing.assert_allclose(w.sum(), 3.0, rtol=1e-10, atol=1e-12)
    assert np.all(w >= 0.01)
    assert all(np.isfinite(history))


# ---------------------------------------------------------------------------
# Adaptive TimeSeriesKMeans
# ---------------------------------------------------------------------------

def test_kmeans_adaptive_invalid_usage():
    rng = np.random.RandomState(0)
    X = rng.randn(6, 5, 2)
    # regularization with non-dtw metrics
    for metric in ["euclidean", "softdtw"]:
        km = TimeSeriesKMeans(n_clusters=2, metric=metric,
                              channel_weight_regularization=0.5)
        with pytest.raises(ValueError):
            km.fit(X)
    # fixed channel_weights in metric_params conflict with learning
    km = TimeSeriesKMeans(n_clusters=2, metric="dtw",
                          channel_weight_regularization=0.5,
                          metric_params={"channel_weights": [1.0, 1.0]})
    with pytest.raises(ValueError):
        km.fit(X)
    # invalid regularization / lower bound (incl. booleans)
    for bad in [0.0, -2.0, np.nan, np.inf, True, "x"]:
        km = TimeSeriesKMeans(n_clusters=2, metric="dtw",
                              channel_weight_regularization=bad)
        with pytest.raises(ValueError):
            km.fit(X)
    for bad in [0.0, 1.0, -1e-3, np.nan, True]:
        km = TimeSeriesKMeans(n_clusters=2, metric="dtw",
                              channel_weight_regularization=0.5,
                              min_channel_weight=bad)
        with pytest.raises(ValueError):
            km.fit(X)


def test_kmeans_adaptive_invalid_data():
    lam = 0.5
    # internal NaN
    X = np.ones((3, 4, 2))
    X[1, 2, :] = np.nan
    with pytest.raises(ValueError):
        TimeSeriesKMeans(n_clusters=2, metric="dtw",
                         channel_weight_regularization=lam).fit(X)
    # partial-channel NaN
    X = np.ones((3, 4, 2))
    X[0, 1, 0] = np.nan
    with pytest.raises(ValueError):
        TimeSeriesKMeans(n_clusters=2, metric="dtw",
                         channel_weight_regularization=lam).fit(X)
    # inf
    X = np.ones((3, 4, 2))
    X[0, 1, 0] = np.inf
    with pytest.raises(ValueError):
        TimeSeriesKMeans(n_clusters=2, metric="dtw",
                         channel_weight_regularization=lam).fit(X)
    # empty series (all NaNs)
    X = np.ones((3, 4, 2))
    X[2, :, :] = np.nan
    with pytest.raises(ValueError):
        TimeSeriesKMeans(n_clusters=2, metric="dtw",
                         channel_weight_regularization=lam).fit(X)
    # trailing all-NaN padding rows are fine (variable-length series)
    X = to_time_series_dataset(
        [[[1.0, 0.0], [2.0, 1.0], [3.0, 0.0]],
         [[1.0, 1.0], [2.0, 0.0]],
         [[3.0, 1.0], [2.0, 2.0], [1.0, 1.0]]]
    )
    TimeSeriesKMeans(n_clusters=2, metric="dtw", max_iter=3,
                     channel_weight_regularization=lam,
                     random_state=0).fit(X)


def test_kmeans_adaptive_two_clusters_different_channels():
    rng = np.random.RandomState(0)
    X, y = _make_two_group_dataset(rng)
    lam = 0.05
    km = TimeSeriesKMeans(n_clusters=2, metric="dtw", n_init=2, max_iter=20,
                          channel_weight_regularization=lam,
                          random_state=0).fit(X)
    # the two clusters focus on different channels
    w = km.channel_weights_
    assert w.shape == (2, 2)
    np.testing.assert_allclose(w.sum(axis=1), [2.0, 2.0], rtol=1e-8)
    assert np.all(w >= 1e-6)
    focus = w.argmax(axis=1)
    assert focus[0] != focus[1]
    assert w[0, focus[0]] > 1.5
    assert w[1, focus[1]] > 1.5
    # clusters recover the two groups (up to permutation)
    pred = km.labels_
    agree = max(np.mean(pred == y), np.mean(pred == 1 - y))
    assert agree > 0.9
    # objective bookkeeping
    assert km.objective_history_[-1] == pytest.approx(km.objective_)
    assert all(np.isfinite(km.objective_history_))
    hist = km.objective_history_
    for prev, nxt in zip(hist[:-1], hist[1:]):
        assert nxt <= prev + 1e-8 * max(1.0, abs(prev))
    assert hist[-1] < hist[0]
    # inertia_ keeps its meaning: average squared (weighted) distance
    dists = km.transform(X)
    np.testing.assert_allclose(
        km.inertia_, np.mean(dists[np.arange(len(X)), km.labels_] ** 2),
        rtol=1e-10,
    )
    # objective = inertia + lam * entropy
    np.testing.assert_allclose(
        km.objective_,
        km.inertia_ + lam * _entropy(km.channel_weights_.ravel()),
        rtol=1e-10,
    )
    # labels_ equal predict on the training set
    np.testing.assert_array_equal(km.labels_, km.predict(X))


def test_kmeans_adaptive_history_independent_recomputation():
    # Record every evaluated candidate state and recompute the recorded
    # objectives independently with the pure-Python weighted DTW.
    rng = np.random.RandomState(5)
    X, _ = _make_two_group_dataset(rng, n_per=4, sz=6)
    lam = 0.1
    states = []

    orig = TimeSeriesKMeans._scaled_dists_with_centers

    def spy(self, X_, centers, channel_weights):
        scaled_sq, scales = orig(self, X_, centers, channel_weights)
        labels = self._assign_scaled(scaled_sq, scales)
        states.append((centers.copy(), channel_weights.copy(), labels))
        return scaled_sq, scales

    with mock.patch.object(TimeSeriesKMeans,
                           "_scaled_dists_with_centers", spy):
        km = TimeSeriesKMeans(n_clusters=2, metric="dtw", max_iter=8,
                              channel_weight_regularization=lam,
                              random_state=0).fit(X)
    # every recorded objective is independently reproducible
    assert len(states) >= len(km.objective_history_)
    for (centers, w, labels), recorded in zip(states, km.objective_history_):
        j_indep = _py_kmeans_objective(X, labels, centers, w, lam)
        np.testing.assert_allclose(recorded, j_indep, rtol=1e-8)
    # final model state matches the last recorded objective
    j_final = _py_kmeans_objective(X, km.labels_, km.cluster_centers_,
                                   km.channel_weights_, lam)
    np.testing.assert_allclose(km.objective_, j_final, rtol=1e-8)
    np.testing.assert_allclose(km.objective_history_[-1], j_final, rtol=1e-8)


def test_kmeans_adaptive_matches_dba_single_cluster():
    # With one cluster, the k-means objective and trajectory coincide with
    # the standalone adaptive DBA objective (same initial center).
    rng = np.random.RandomState(2)
    n, sz, d = 5, 6, 2
    X = rng.randn(n, sz, d) * 0.2
    X[:, :, 0] += np.sin(np.linspace(0, 3, sz))[None, :]
    lam = 0.3
    init_center = X[0:1].copy()

    bary, w_dba, hist_dba = dtw_barycenter_averaging(
        X, init_barycenter=init_center[0].copy(),
        channel_weight_regularization=lam, max_iter=10, tol=1e-9
    )
    km = TimeSeriesKMeans(n_clusters=1, metric="dtw", init=init_center,
                          channel_weight_regularization=lam, max_iter=10,
                          tol=1e-9).fit(X)
    np.testing.assert_allclose(km.objective_history_, hist_dba, rtol=1e-8)
    np.testing.assert_allclose(km.channel_weights_[0], w_dba, rtol=1e-8)
    np.testing.assert_allclose(km.cluster_centers_[0], bary, rtol=1e-8)
    np.testing.assert_allclose(km.objective_, hist_dba[-1], rtol=1e-10)


def test_kmeans_adaptive_n_init_selects_best_objective():
    rng = np.random.RandomState(0)
    X, _ = _make_two_group_dataset(rng, n_per=5, sz=6)
    recorded = []
    orig = TimeSeriesKMeans._fit_one_init_adaptive

    def spy(self, X_, metric_params):
        out = orig(self, X_, metric_params)
        recorded.append(self.objective_history_[-1])
        return out

    with mock.patch.object(TimeSeriesKMeans, "_fit_one_init_adaptive", spy):
        km = TimeSeriesKMeans(n_clusters=2, metric="dtw", n_init=3,
                              max_iter=6, channel_weight_regularization=0.1,
                              random_state=0).fit(X)
    assert len(recorded) == 3
    # the delivered model is the initialization with the lowest final J
    np.testing.assert_allclose(km.objective_, min(recorded), rtol=1e-12)
    # and the history is the one of that initialization
    np.testing.assert_allclose(km.objective_history_[-1], km.objective_)


def test_kmeans_adaptive_empty_cluster_recovery():
    rng = np.random.RandomState(0)
    X, _ = _make_two_group_dataset(rng, n_per=5, sz=6)

    # More clusters than samples: every attempt fails, no half-trained
    # state may be left behind.
    km = TimeSeriesKMeans(n_clusters=20, metric="dtw",
                          channel_weight_regularization=0.1,
                          random_state=0).fit(X)
    assert km._X_fit is None
    assert km.cluster_centers_ is None
    assert km.channel_weights_ is None
    assert km.objective_ is None
    assert km.objective_history_ is None
    assert km.labels_ is None
    with pytest.raises(NotFittedError):
        km.predict(X)

    # A failed attempt is not recorded as a successful descent: force the
    # first attempt to raise EmptyClusterError, the fit then recovers.
    calls = {"n": 0}
    orig_check = kmeans_module._check_no_empty_cluster

    def flaky_check(labels, n_clusters):
        calls["n"] += 1
        if calls["n"] == 1:
            raise EmptyClusterError
        return orig_check(labels, n_clusters)

    with mock.patch.object(kmeans_module, "_check_no_empty_cluster",
                           flaky_check):
        km = TimeSeriesKMeans(n_clusters=2, metric="dtw", n_init=1,
                              max_iter=5, channel_weight_regularization=0.1,
                              random_state=0).fit(X)
    assert km._X_fit is not None
    assert km.channel_weights_ is not None
    hist = km.objective_history_
    assert all(np.isfinite(hist))
    for prev, nxt in zip(hist[:-1], hist[1:]):
        assert nxt <= prev + 1e-8 * max(1.0, abs(prev))
    np.testing.assert_array_equal(km.labels_, km.predict(X))


def test_kmeans_adaptive_max_iter_and_n_iter():
    rng = np.random.RandomState(0)
    X, _ = _make_two_group_dataset(rng, n_per=4, sz=6)
    km = TimeSeriesKMeans(n_clusters=2, metric="dtw", max_iter=1,
                          channel_weight_regularization=0.1,
                          random_state=0).fit(X)
    assert km.n_iter_ == 1
    # one iteration: initial state + one accepted update
    assert len(km.objective_history_) == 2
    np.testing.assert_array_equal(km.labels_, km.predict(X))


def test_kmeans_adaptive_inference_invariance():
    rng = np.random.RandomState(0)
    X, _ = _make_two_group_dataset(rng)
    km = TimeSeriesKMeans(n_clusters=2, metric="dtw", max_iter=10,
                          channel_weight_regularization=0.05,
                          random_state=0).fit(X)
    w_before = km.channel_weights_.copy()
    # batched vs one-by-one predictions are identical
    full = km.predict(X)
    pieces = np.concatenate([km.predict(X[i:i + 3])
                             for i in range(0, len(X), 3)])
    np.testing.assert_array_equal(full, pieces)
    # transform rows do not depend on batch composition
    np.testing.assert_allclose(km.transform(X)[:5], km.transform(X[:5]))
    # inference does not update the learned weights
    km.predict(X)
    km.transform(X)
    np.testing.assert_allclose(km.channel_weights_, w_before)
    # transform column k uses the k-th center with its own learned weights
    dists = km.transform(X)
    for k in range(km.n_clusters):
        col = cdist_dtw(X, km.cluster_centers_[k:k + 1],
                        channel_weights=km.channel_weights_[k])[:, 0]
        np.testing.assert_allclose(dists[:, k], col, rtol=1e-12)
    # predict picks the argmin of transform
    np.testing.assert_array_equal(km.predict(X), dists.argmin(axis=1))


def test_kmeans_adaptive_reproducibility():
    rng = np.random.RandomState(0)
    X, _ = _make_two_group_dataset(rng)
    params = dict(n_clusters=2, metric="dtw", n_init=2, max_iter=10,
                  channel_weight_regularization=0.05, random_state=42)
    km1 = TimeSeriesKMeans(**params).fit(X)
    km2 = TimeSeriesKMeans(**params).fit(X)
    np.testing.assert_array_equal(km1.labels_, km2.labels_)
    np.testing.assert_allclose(km1.channel_weights_, km2.channel_weights_)
    np.testing.assert_allclose(km1.objective_history_, km2.objective_history_)
    np.testing.assert_allclose(km1.cluster_centers_, km2.cluster_centers_)


def test_kmeans_adaptive_input_immutability():
    rng = np.random.RandomState(0)
    X, _ = _make_two_group_dataset(rng, n_per=4, sz=6)
    X_copy = X.copy()
    metric_params = {"sakoe_chiba_radius": 2}
    metric_params_copy = dict(metric_params)
    init = X[:2].copy()
    init_copy = init.copy()
    km = TimeSeriesKMeans(n_clusters=2, metric="dtw", max_iter=3,
                          channel_weight_regularization=0.1,
                          metric_params=metric_params, init=init,
                          random_state=0)
    km.fit(X)
    np.testing.assert_array_equal(X, X_copy)
    assert metric_params == metric_params_copy
    np.testing.assert_array_equal(init, init_copy)


def test_kmeans_adaptive_clone_pickle_roundtrip(tmp_path):
    rng = np.random.RandomState(0)
    X, _ = _make_two_group_dataset(rng, n_per=4, sz=6)
    km = TimeSeriesKMeans(n_clusters=2, metric="dtw", max_iter=5,
                          channel_weight_regularization=0.1,
                          random_state=0).fit(X)

    # sklearn clone copies the new hyper-parameters
    km_clone = clone(km)
    assert km_clone.channel_weight_regularization == 0.1
    assert km_clone.min_channel_weight == km.min_channel_weight

    # plain pickle round-trip
    km2 = pickle.loads(pickle.dumps(km))
    np.testing.assert_array_equal(km2.predict(X), km.predict(X))
    np.testing.assert_allclose(km2.channel_weights_, km.channel_weights_)
    assert km2.objective_ == km.objective_

    # tslearn to_pickle / from_pickle round-trip
    path = str(tmp_path / "km.pkl")
    km.to_pickle(path)
    km3 = TimeSeriesKMeans.from_pickle(path)
    np.testing.assert_array_equal(km3.predict(X), km.predict(X))
    np.testing.assert_allclose(km3.channel_weights_, km.channel_weights_)
    np.testing.assert_allclose(km3.cluster_centers_, km.cluster_centers_)
    assert km3.objective_ == pytest.approx(km.objective_)
    assert list(km3.objective_history_) == list(km.objective_history_)

    # json round-trip
    path = str(tmp_path / "km.json")
    km.to_json(path)
    km4 = TimeSeriesKMeans.from_json(path)
    np.testing.assert_array_equal(km4.predict(X), km.predict(X))
    np.testing.assert_allclose(km4.channel_weights_, km.channel_weights_)

    # hdf5 round-trip (if h5py is available)
    pytest.importorskip("h5py")
    path = str(tmp_path / "km.h5")
    km.to_hdf5(path)
    km5 = TimeSeriesKMeans.from_hdf5(path)
    np.testing.assert_array_equal(km5.predict(X), km.predict(X))
    np.testing.assert_allclose(km5.channel_weights_, km.channel_weights_)
    assert km5.objective_ == pytest.approx(km.objective_)

    # refitting the same instance relearns the weights from scratch
    km.fit(X)
    assert km.channel_weights_ is not None
    np.testing.assert_allclose(km.channel_weights_.sum(axis=1),
                               [2.0, 2.0], rtol=1e-8)


def test_kmeans_adaptive_variable_length():
    # Multivariate, unequal-length series with trailing NaN padding
    rng = np.random.RandomState(0)
    t = np.linspace(0, 2 * np.pi, 10)
    series = []
    for i in range(8):
        sig = np.sin(t[: 6 + (i % 4)]) + 0.05 * rng.randn(6 + (i % 4))
        noise = rng.randn(6 + (i % 4))
        series.append(np.column_stack([sig, noise]))
    X = to_time_series_dataset(series)
    km = TimeSeriesKMeans(n_clusters=2, metric="dtw", max_iter=5,
                          channel_weight_regularization=0.1,
                          random_state=0).fit(X)
    assert km.channel_weights_.shape == (2, 2)
    np.testing.assert_array_equal(km.labels_, km.predict(X))
    # prediction accepts NaN-padded variable-length input
    km.predict(X[:3])


def test_kmeans_adaptive_univariate():
    # With d=1 the only feasible weight vector is [1.]: the model must
    # still train consistently (entropy term is then constant).
    rng = np.random.RandomState(0)
    X = rng.randn(10, 6, 1)
    km = TimeSeriesKMeans(n_clusters=2, metric="dtw", max_iter=5,
                          channel_weight_regularization=0.2,
                          random_state=0).fit(X)
    np.testing.assert_allclose(km.channel_weights_, [[1.0], [1.0]])
    hist = km.objective_history_
    for prev, nxt in zip(hist[:-1], hist[1:]):
        assert nxt <= prev + 1e-8 * max(1.0, abs(prev))
    np.testing.assert_array_equal(km.labels_, km.predict(X))
    # objective reduces to the plain inertia (entropy of [1] is 0)
    np.testing.assert_allclose(km.objective_, km.inertia_, rtol=1e-10)


def test_kmeans_default_mode_has_no_new_attributes():
    # Default behavior is unchanged: no learned weights, no objective, and
    # the v0.6.4 attribute set is preserved (serialization included).
    rng = np.random.RandomState(0)
    X = rng.randn(10, 6, 2)
    for metric in ["dtw", "euclidean", "softdtw"]:
        km = TimeSeriesKMeans(n_clusters=2, metric=metric, max_iter=3,
                              random_state=0).fit(X)
        assert not hasattr(km, "channel_weights_")
        assert not hasattr(km, "objective_")
        assert not hasattr(km, "objective_history_")
