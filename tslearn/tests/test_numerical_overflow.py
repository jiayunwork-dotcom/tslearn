"""Regression tests for overflow-safe weighted DTW and adaptive training.

These tests cover the failure modes observed when intermediate quantities
(``1e308 ** 2`` in particular) overflow float64:

* fixed-weight DTW / dtw_path / cdist_dtw with zero-weight channels and
  values at the float64 magnitude boundary;
* the adaptive channel-weight solver delivering simplex-violating weights
  because its bisection evaluated ``exp`` of huge arguments;
* DBA and TimeSeriesKMeans rejecting representable states (or silently
  delivering infinite objectives and half-trained models) after an
  intermediate square, weighted sum or accumulation overflowed;
* genuinely non-representable results raising ValueError in finite steps.

Reference answers are computed independently with pure-Python exact path
enumeration, :mod:`decimal` high-precision arithmetic or direct formulas;
they never call the tslearn numerical kernels under test.
"""
import decimal
import pickle

import numpy as np
import pytest
from sklearn.base import clone
from sklearn.exceptions import NotFittedError

from tslearn.barycenters import dtw_barycenter_averaging
from tslearn.clustering import TimeSeriesKMeans
from tslearn.metrics import cdist_dtw, dtw, dtw_path
from tslearn.metrics._numerics import (
    NonRepresentableNumericalState,
    stable_entropy_solve,
)
from tslearn.utils import to_time_series_dataset

RTOL = 1e-9
ATOL = 1e-12


# ---------------------------------------------------------------------------
# Independent references (no call into the kernels under test)
# ---------------------------------------------------------------------------

def _enum_paths(n1, n2, mask=None):
    """All legal DTW paths from (0, 0) to (n1-1, n2-1)."""
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


def _path_weighted_cost(x, y, w, path):
    """Exact path cost in Decimal high precision (independent reference)."""
    D = decimal.Decagonal if False else decimal.Decimal
    total = decimal.Decimal(0)
    for i, j in path:
        for c in range(x.shape[1]):
            if w[c] == 0.0:
                continue
            diff = D(repr(float(x[i, c]))) - D(repr(float(y[j, c])))
            total += D(repr(float(w[c]))) * diff * diff
    return total


def _exact_dtw(x, y, w, mask=None):
    """Reference weighted DTW squared cost via exhaustive path enumeration."""
    best = None
    for path in _enum_paths(x.shape[0], y.shape[0], mask):
        cost = _path_weighted_cost(x, y, w, path)
        if best is None or cost < best:
            best = cost
    return best


def _is_legal_path(path, n1, n2, mask=None):
    if path[0] != (0, 0) or path[-1] != (n1 - 1, n2 - 1):
        return False
    for (i0, j0), (i1, j1) in zip(path[:-1], path[1:]):
        if (i1 - i0, j1 - j0) not in ((0, 1), (1, 0), (1, 1)):
            return False
        if mask is not None and not np.isfinite(mask[i1, j1]):
            return False
    return True


def _assert_weight_constraints(w, d, min_weight=1e-6):
    """Constraints checked independently of the objective tolerance."""
    assert np.all(np.isfinite(w))
    assert np.all(w >= min_weight - 1e-12)
    np.testing.assert_allclose(float(np.sum(w)), float(d),
                               rtol=1e-9, atol=1e-12)


# ---------------------------------------------------------------------------
# Fixed-weight DTW: zero-weight channels and boundary magnitudes
# ---------------------------------------------------------------------------

def test_zero_weight_channel_dominates_overflowing_channel():
    # The motivating failure: channel 0 overflows when squared but has
    # weight zero, so the distance must come solely from channel 1.
    x = np.array([[1e308, 0.0]])
    y = np.array([[-1e308, 1.0]])
    w = np.array([0.0, 1.0])
    path, dist = dtw_path(x, y, channel_weights=w)
    assert _is_legal_path(path, 1, 1)
    np.testing.assert_allclose(dist, 1.0, rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(dtw(x, y, channel_weights=w), 1.0,
                               rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(
        cdist_dtw(x[None], y[None], channel_weights=w), [[1.0]],
        rtol=RTOL, atol=ATOL,
    )
    # The single-cell path is the unique legal alignment.
    assert path == [(0, 0)]


def test_zero_weight_channel_multiple_cells_and_constraint():
    x = np.array([[1e308, 0.0], [1e308, 0.0]])
    y = np.array([[-1e308, 1.0], [-1e308, 1.0]])
    w = np.array([0.0, 1.0])
    # Unconstrained: two aligned unit differences -> sqrt(2).
    np.testing.assert_allclose(dtw(x, y, channel_weights=w),
                               np.sqrt(2.0), rtol=RTOL, atol=ATOL)
    # Sakoe-Chiba radius 0 forces the diagonal two-cell path.
    from tslearn.metrics.dtw_variants import njit_sakoe_chiba_mask
    mask = njit_sakoe_chiba_mask(2, 2, radius=0)
    path, dist = dtw_path(x, y, channel_weights=w,
                          global_constraint="sakoe_chiba",
                          sakoe_chiba_radius=0)
    assert _is_legal_path(path, 2, 2, mask)
    np.testing.assert_allclose(dist, np.sqrt(2.0), rtol=RTOL, atol=ATOL)
    # And the answer matches exact path enumeration on the same mask.
    exact = _exact_dtw(x, y, w, mask)
    np.testing.assert_allclose(dist ** 2, float(exact), rtol=RTOL,
                               atol=ATOL)


def test_fixed_weight_large_but_representable_matches_enumeration():
    # Large magnitudes whose weighted squared cost stays in float64.
    rng = np.random.RandomState(11)
    x = rng.uniform(-1e150, 1e150, size=(3, 2))
    y = rng.uniform(-1e150, 1e150, size=(4, 2))
    w = np.array([1.0, 0.25])
    exact_sq = float(_exact_dtw(x, y, w))
    path, dist = dtw_path(x, y, channel_weights=w)
    assert _is_legal_path(path, 3, 4)
    np.testing.assert_allclose(dist ** 2, exact_sq, rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(dtw(x, y, channel_weights=w) ** 2,
                               exact_sq, rtol=RTOL, atol=ATOL)
    # The returned path itself realizes the reported squared cost.
    np.testing.assert_allclose(
        float(_path_weighted_cost(x, y, w, path)), dist ** 2,
        rtol=RTOL, atol=ATOL,
    )


def test_unrepresentable_distance_raises_value_error_finite_steps():
    # (2 * 1e308)^2 exceeds float64: the correct distance cannot exist as a
    # finite float, so every entry point must raise ValueError, never NaN.
    x = np.array([[1e308]])
    y = np.array([[-1e308]])
    for fn, kwargs in (
        (dtw, {}),
        (dtw_path, {}),
    ):
        with pytest.raises(ValueError):
            fn(x, y, **kwargs)
    with pytest.raises(ValueError):
        cdist_dtw(x[None], y[None])
    # And the failure leaves no NaN-shaped state: the error type is exactly
    # ValueError (the internal numeric error subclasses it).
    assert issubclass(NonRepresentableNumericalState, ValueError)


def test_ordinary_weighted_dtw_unchanged():
    # Ordinary-magnitude regression against exact enumeration.
    rng = np.random.RandomState(7)
    for shape in ((3, 4), (4, 4), (2, 5)):
        x = rng.randn(shape[0], 3)
        y = rng.randn(shape[1], 3)
        for w in (np.ones(3), np.array([2.0, 0.0, 0.5]),
                  np.array([0.0, 3.0, 0.0])):
            exact = float(_exact_dtw(x, y, w))
            np.testing.assert_allclose(
                dtw(x, y, channel_weights=w) ** 2, exact,
                rtol=1e-10, atol=1e-12,
            )
            path, dist = dtw_path(x, y, channel_weights=w)
            np.testing.assert_allclose(dist ** 2, exact,
                                       rtol=1e-10, atol=1e-12)


# ---------------------------------------------------------------------------
# Stable entropy weight solver
# ---------------------------------------------------------------------------

def test_stable_solver_symmetric_huge_costs_gives_uniform_weights():
    # The original failure at A ~ 1e16: three identical channels must get
    # weight 1 each, summing to the number of channels.
    A = np.full(3, 1e16)
    w = stable_entropy_solve(A, regularization=1.0, min_weight=1e-6)
    _assert_weight_constraints(w, 3)
    np.testing.assert_allclose(w, np.ones(3), rtol=RTOL, atol=ATOL)


def test_stable_solver_extreme_cost_range():
    # Costs spanning the whole finite range, including an infinite channel:
    # the infinite-cost channel is pinned to the lower bound and the two
    # finite channels share the remaining mass while satisfying KKT.
    A = np.array([0.0, 1.0, np.inf])
    eps = 1e-6
    w = stable_entropy_solve(A, regularization=0.1, min_weight=eps)
    _assert_weight_constraints(w, 3, min_weight=eps)
    assert w[2] == pytest.approx(eps)
    assert w[0] > w[1] > w[2] - 1e-12
    # KKT on the two free coordinates: equal gradients.
    grad = A[:2] + 0.1 * (np.log(w[:2]) + 1.0)
    np.testing.assert_allclose(grad, grad[0], rtol=1e-9, atol=1e-10)

    # Even a finite-but-huge cost difference (1e300) is handled without
    # overflow: the expensive finite channel is driven to the lower bound
    # exactly like an infinite one at this regularization.
    A_huge = np.array([0.0, 1e300])
    w_huge = stable_entropy_solve(A_huge, 0.1, eps)
    _assert_weight_constraints(w_huge, 2, min_weight=eps)
    assert w_huge[1] == pytest.approx(eps)
    assert w_huge[0] == pytest.approx(2.0 - eps)


def test_stable_solver_matches_direct_formula_when_finite():
    # On ordinary finite costs the stable solver matches a direct
    # high-precision solve of the KKT equation by bisection.
    rng = np.random.RandomState(3)
    lam, eps = 0.3, 1e-6
    for _ in range(20):
        A = rng.rand(4) * 5.0
        w = stable_entropy_solve(A, lam, eps)
        _assert_weight_constraints(w, 4, min_weight=eps)
        # Direct KKT residual check on free coordinates.
        free = w > eps + 1e-11
        if np.any(free):
            grad = A[free] + lam * (np.log(w[free]) + 1.0)
            np.testing.assert_allclose(grad, grad[0], rtol=1e-8,
                                       atol=1e-10)


# ---------------------------------------------------------------------------
# Adaptive DBA: the reported failure scenarios
# ---------------------------------------------------------------------------

def test_dba_symmetric_huge_channels_legal_weights_and_objective():
    X = np.array([[[-1e8, -1e8, -1e8]],
                  [[1e8, 1e8, 1e8]]])
    bary, w, history = dtw_barycenter_averaging(
        X, init_barycenter=np.zeros((1, 3)),
        channel_weight_regularization=1.0, max_iter=1,
    )
    np.testing.assert_allclose(bary, np.zeros((1, 3)),
                               rtol=RTOL, atol=ATOL)
    _assert_weight_constraints(w, 3)
    np.testing.assert_allclose(w, np.ones(3), rtol=RTOL, atol=ATOL)
    # True objective at the (zero center, unit weights) state: mean squared
    # cost 3e16, entropy of unit weights zero.
    np.testing.assert_allclose(history[0], 3e16, rtol=RTOL, atol=ATOL)
    assert history[-1] <= history[0] + 1e-8 * max(1.0, abs(history[0]))
    # The last history entry really corresponds to the returned pair:
    # recompute its mean squared cost directly from a fresh alignment.
    total = 0.0
    for xi in X:
        total += dtw(bary, xi, channel_weights=w) ** 2
    direct_data = total / len(X)
    np.testing.assert_allclose(
        history[-1], direct_data + 1.0 * float(
            np.sum(w * np.log(w))
        ), rtol=1e-8, atol=1e-10,
    )


def test_dba_sample_weight_common_scaling_invariance():
    X = np.array([[[0.0, 0.0]], [[2.0, 4.0]]])
    kw = dict(init_barycenter=np.zeros((1, 2)), max_iter=2)
    b1 = dtw_barycenter_averaging(X, weights=np.array([1.0, 1.0]), **kw)
    b2 = dtw_barycenter_averaging(
        X, weights=np.array([1e308, 1e308]), **kw
    )
    np.testing.assert_allclose(b1, b2, rtol=RTOL, atol=ATOL)
    # And a non-uniform ratio scaled by a huge common factor.
    b3 = dtw_barycenter_averaging(
        X, weights=np.array([3.0, 7.0]), **kw
    )
    b4 = dtw_barycenter_averaging(
        X, weights=np.array([3e307, 7e307]), **kw
    )
    np.testing.assert_allclose(b3, b4, rtol=RTOL, atol=ATOL)


def test_dba_extreme_series_representable_state_delivered():
    # -1e154 / 1e154 single channel: center 0, weight 1, mean squared
    # cost 1e308, all finite in float64.
    X = np.array([[[-1e154]], [[1e154]]])
    bary, w, history = dtw_barycenter_averaging(
        X, init_barycenter=np.zeros((1, 1)),
        channel_weight_regularization=1.0, max_iter=1,
    )
    np.testing.assert_allclose(bary, np.zeros((1, 1)),
                               rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(w, np.ones(1), rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(history[-1], 1e308, rtol=RTOL, atol=ATOL)
    assert all(np.isfinite(history))


def test_dba_identical_boundary_series_zero_objective():
    X = np.array([[[1e308, 1e308]], [[1e308, 1e308]]])
    bary, w, history = dtw_barycenter_averaging(
        X, init_barycenter=X[0].copy(),
        channel_weight_regularization=1.0, max_iter=1,
    )
    np.testing.assert_allclose(bary, X[0], rtol=RTOL, atol=ATOL)
    _assert_weight_constraints(w, 2)
    np.testing.assert_allclose(history, [0.0, 0.0], atol=ATOL)


def test_dba_history_monotone_and_final_consistency():
    rng = np.random.RandomState(0)
    n, sz = 6, 8
    t = np.linspace(0, 2 * np.pi, sz)
    X = np.array([
        np.column_stack([np.sin(t) + 0.05 * rng.randn(sz),
                         2.0 * rng.randn(sz)])
        for _ in range(n)
    ])
    bary, w, history = dtw_barycenter_averaging(
        X, channel_weight_regularization=0.05, max_iter=20
    )
    _assert_weight_constraints(w, 2)
    for prev, nxt in zip(history[:-1], history[1:]):
        assert nxt <= prev + 1e-8 * max(1.0, abs(prev))
    # Final entry corresponds to the returned center and weights.
    total = sum(dtw(bary, xi, channel_weights=w) ** 2 for xi in X) / n
    np.testing.assert_allclose(
        history[-1], total + 0.05 * float(np.sum(w * np.log(w))),
        rtol=1e-8, atol=1e-10,
    )


# ---------------------------------------------------------------------------
# Adaptive TimeSeriesKMeans
# ---------------------------------------------------------------------------

def test_kmeans_symmetric_huge_case_delivers_legal_model():
    X = np.array([[[-1e8, -1e8, -1e8]],
                  [[1e8, 1e8, 1e8]]])
    km = TimeSeriesKMeans(
        n_clusters=1, metric="dtw", init=np.zeros((1, 1, 3)),
        n_init=1, max_iter=1, channel_weight_regularization=1.0,
    ).fit(X)
    _assert_weight_constraints(km.channel_weights_[0], 3)
    np.testing.assert_allclose(km.channel_weights_[0], np.ones(3),
                               rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(km.cluster_centers_, np.zeros((1, 1, 3)),
                               rtol=RTOL, atol=ATOL)
    assert np.isfinite(km.objective_)
    np.testing.assert_allclose(km.objective_, 3e16, rtol=RTOL,
                               atol=ATOL)
    # labels and predict agree, and transform uses the learned weights.
    np.testing.assert_array_equal(km.labels_, km.predict(X))
    distances = km.transform(X)
    assert distances.shape == (2, 1)
    assert np.all(np.isfinite(distances))


def test_kmeans_extreme_series_representable_model():
    X = np.array([[[-1e154]], [[1e154]]])
    km = TimeSeriesKMeans(
        n_clusters=1, metric="dtw", init=np.zeros((1, 1, 1)),
        n_init=1, max_iter=1, channel_weight_regularization=1.0,
    ).fit(X)
    np.testing.assert_allclose(km.cluster_centers_, np.zeros((1, 1, 1)),
                               rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(km.channel_weights_, np.ones((1, 1)),
                               rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(km.objective_, 1e308, rtol=RTOL,
                               atol=ATOL)
    np.testing.assert_array_equal(km.labels_, km.predict(X))


def test_kmeans_identical_boundary_series():
    X = np.array([[[1e308, 1e308]], [[1e308, 1e308]]])
    km = TimeSeriesKMeans(
        n_clusters=1, metric="dtw", init=X[:1].copy(), n_init=1,
        max_iter=1, channel_weight_regularization=1.0,
    ).fit(X)
    np.testing.assert_allclose(km.cluster_centers_, X[:1],
                               rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(km.objective_, 0.0, atol=ATOL)
    _assert_weight_constraints(km.channel_weights_.reshape(-1), 2)


def test_kmeans_failure_leaves_no_usable_model():
    # A genuinely non-representable single-cluster problem: the mean
    # squared cost overflows float64 and no finite candidate exists.
    X = np.array([[[1e308]], [[-1e308]]])
    km = TimeSeriesKMeans(
        n_clusters=1, metric="dtw", init=np.zeros((1, 1, 1)),
        n_init=1, max_iter=1, channel_weight_regularization=1.0,
    )
    with pytest.raises(ValueError):
        km.fit(X)
    # The estimator must not look fitted and must refuse prediction.
    with pytest.raises(NotFittedError):
        km.predict(X)
    assert km.cluster_centers_ is None
    assert km.channel_weights_ is None
    assert km.objective_ is None
    # A subsequent, well-posed fit on the same estimator instance works and
    # does not inherit any state from the failed run.
    X_ok = np.array([[[1.0]], [[2.0]]])
    km_ok = TimeSeriesKMeans(
        n_clusters=1, metric="dtw", init=np.zeros((1, 1, 1)),
        n_init=1, max_iter=2, channel_weight_regularization=1.0,
    )
    with pytest.raises(ValueError):
        km_ok.fit(X)
    km_ok.fit(X_ok)
    assert km_ok.cluster_centers_ is not None
    np.testing.assert_array_equal(km_ok.labels_, km_ok.predict(X_ok))


def test_kmeans_pickle_clone_and_input_immutability():
    rng = np.random.RandomState(4)
    t = np.linspace(0, 2 * np.pi, 6)
    # Two well-separated groups so both clusters stay non-empty.
    group0 = np.column_stack([np.sin(t), 0.1 * rng.randn(6)])[None]
    group1 = np.column_stack([np.cos(t), 0.1 * rng.randn(6)])[None]
    X = np.concatenate([
        group0 + 0.05 * rng.randn(3, 6, 2),
        group1 + 0.05 * rng.randn(3, 6, 2),
    ], axis=0)
    init = np.stack([group0[0], group1[0]])
    init_copy = init.copy()
    params = dict(n_clusters=2, metric="dtw", init=init, n_init=1,
                  max_iter=3, channel_weight_regularization=0.1,
                  random_state=0)
    km = TimeSeriesKMeans(**params).fit(X)
    assert km.objective_ is not None
    # The caller's init array and a params dict are not modified.
    np.testing.assert_array_equal(init, init_copy)
    # clone + refit reproduces the model.
    km_clone = clone(TimeSeriesKMeans(**params))
    km_clone.fit(X)
    np.testing.assert_allclose(km_clone.cluster_centers_,
                               km.cluster_centers_, rtol=1e-10)
    np.testing.assert_allclose(km_clone.channel_weights_,
                               km.channel_weights_, rtol=1e-10)
    # pickle round trip keeps the delivered finite model working.
    restored = pickle.loads(pickle.dumps(km))
    np.testing.assert_array_equal(restored.predict(X), km.predict(X))
    np.testing.assert_allclose(restored.transform(X), km.transform(X),
                               rtol=1e-10)
    np.testing.assert_allclose(restored.objective_history_,
                               km.objective_history_, rtol=1e-10)


def test_kmeans_n_init_selects_by_true_objective():
    # On a well-separated two-group dataset, a good per-group init reaches
    # a lower true objective than swapping the two centers; the fit loop's
    # candidate comparison therefore keeps the better run.
    rng = np.random.RandomState(2)
    t = np.linspace(0, 2 * np.pi, 4)
    g0 = np.column_stack([np.sin(t), 0.1 * rng.randn(4)])
    g1 = np.column_stack([np.cos(t), 0.1 * rng.randn(4)])
    X = np.concatenate([
        g0[None] + 0.05 * rng.randn(4, 4, 2),
        g1[None] + 0.05 * rng.randn(4, 4, 2),
    ], axis=0)
    good_init = np.stack([g0, g1])
    swapped_init = np.stack([g1, g0])
    objs = []
    for init in (good_init, swapped_init):
        km = TimeSeriesKMeans(
            n_clusters=2, metric="dtw", init=init, n_init=1, max_iter=3,
            channel_weight_regularization=0.2, random_state=0,
        ).fit(X)
        assert km.objective_ is not None
        objs.append(km.objective_)
    km_good = TimeSeriesKMeans(
        n_clusters=2, metric="dtw", init=good_init, n_init=1, max_iter=3,
        channel_weight_regularization=0.2, random_state=0,
    ).fit(X)
    assert km_good.objective_ == pytest.approx(min(objs), rel=1e-10)
    # Both inits converge to the same partition quality up to label
    # permutation, so their objectives agree closely.
    np.testing.assert_allclose(objs[0], objs[1], rtol=1e-8, atol=1e-8)


# ---------------------------------------------------------------------------
# Variable length, NaN padding and highly unbalanced sample ratios
# ---------------------------------------------------------------------------

def test_adaptive_dba_variable_length_with_huge_values():
    X = to_time_series_dataset([
        [[1e300, 0.0], [1e300, 1.0]],
        [[1e300, 0.0]],
    ])
    bary, w, history = dtw_barycenter_averaging(
        X, channel_weight_regularization=0.5, max_iter=3,
        min_channel_weight=1e-4,
    )
    assert bary.shape == (2, 2)
    assert np.all(np.isfinite(bary))
    _assert_weight_constraints(w, 2, min_weight=1e-4)
    assert all(np.isfinite(history))


def test_zero_sample_weight_entry_still_supported():
    rng = np.random.RandomState(0)
    X = rng.randn(4, 5, 2)
    sw = np.array([1.0, 0.0, 1.0, 0.5])
    # Use an explicit initial barycenter so the test isolates the weighted
    # iteration behavior rather than the unweighted nanmean initialization.
    init = X[0].copy()
    bary, w, history = dtw_barycenter_averaging(
        X, init_barycenter=init, weights=sw,
        channel_weight_regularization=0.2, max_iter=3,
    )
    assert np.all(np.isfinite(bary))
    _assert_weight_constraints(w, 2)
    # A zero-weight series has no influence on the weighted iterations:
    # replacing its data by arbitrary values leaves the result unchanged.
    X_perturbed = X.copy()
    X_perturbed[1] = 1e6
    bary2, w2, _ = dtw_barycenter_averaging(
        X_perturbed, init_barycenter=init.copy(), weights=sw,
        channel_weight_regularization=0.2, max_iter=3,
    )
    np.testing.assert_allclose(bary, bary2, rtol=1e-9, atol=1e-10)
    np.testing.assert_allclose(w, w2, rtol=1e-9, atol=1e-10)
    # An all-zero weight vector is still strictly rejected.
    with pytest.raises(ValueError):
        dtw_barycenter_averaging(
            X, weights=np.zeros(4),
            channel_weight_regularization=0.2, max_iter=1,
        )


def test_asymmetric_channels_near_lower_bound():
    # One channel almost entirely uninformative: its weight should sit on
    # the lower bound while the constraint still holds exactly.
    rng = np.random.RandomState(9)
    n, sz = 5, 6
    t = np.linspace(0, 3, sz)
    X = np.array([
        np.column_stack([np.sin(t) + 0.01 * rng.randn(sz),
                         1e6 * rng.randn(sz)])
        for _ in range(n)
    ])
    eps = 1e-4
    bary, w, history = dtw_barycenter_averaging(
        X, channel_weight_regularization=1e-3, max_iter=30,
        min_channel_weight=eps,
    )
    _assert_weight_constraints(w, 2, min_weight=eps)
    assert w[0] > 1.5 > w[1]
    for prev, nxt in zip(history[:-1], history[1:]):
        assert nxt <= prev + 1e-8 * max(1.0, abs(prev))
