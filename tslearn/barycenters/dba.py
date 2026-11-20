import math
import numbers

import numpy
import warnings
from scipy.interpolate import interp1d
from sklearn.exceptions import ConvergenceWarning
from sklearn.utils import check_random_state

from ..metrics import dtw_path
from ..metrics._numerics import (
    NonRepresentableNumericalState,
    _SQUARE_SAFE_BOUND,
    _scale_exponent_from_log2_bound,
    per_channel_costs_scaled,
    stable_entropy_solve,
    unscale_or_inf,
    unscale_value,
)
from ..metrics.dtw_variants import (
    GLOBAL_CONSTRAINT_CODE,
    _check_channel_weights,
    _njit_return_path,
    compute_mask,
    njit_accumulated_matrix,
    njit_accumulated_matrix_weighted,
    needs_scaled_kernel,
    scaled_dtw_pair,
)
from ..utils import to_time_series, to_time_series_dataset, ts_size
from .utils import _set_weights

__author__ = 'Romain Tavenard romain.tavenard[at]univ-rennes2.fr'


def _is_real_scalar(value):
    """Return True for real (non-boolean) numeric scalars."""
    return isinstance(value, numbers.Real) and not isinstance(
        value, (bool, numpy.bool_)
    )


def _check_channel_weight_regularization(value):
    """Validate the entropy regularization coefficient lambda."""
    if not _is_real_scalar(value) or not numpy.isfinite(value) or value <= 0:
        raise ValueError(
            "channel_weight_regularization must be a finite positive real "
            "number, got %r." % (value,)
        )
    return float(value)


def _check_min_channel_weight(value):
    """Validate the lower bound on learned channel weights."""
    if (
        not _is_real_scalar(value)
        or not numpy.isfinite(value)
        or not 0.0 < value < 1.0
    ):
        raise ValueError(
            "min_channel_weight must be a finite real number in the open "
            "interval (0, 1), got %r." % (value,)
        )
    return float(value)


def _check_sample_weights_strict(weights, n):
    """Validate sample weights for adaptive channel-weight learning.

    Unlike :func:`_set_weights`, this is strict: anything else than None or a
    one-dimensional real-valued vector of length `n` with finite non-negative
    entries and at least one positive entry is rejected.
    """
    if weights is None:
        return numpy.ones((n,))
    if isinstance(weights, (bool, numpy.bool_)):
        raise ValueError(
            "weights must be an array-like of real numbers, not a boolean."
        )
    arr = numpy.asarray(weights)
    if arr.dtype == numpy.bool_ or not (
        numpy.issubdtype(arr.dtype, numpy.integer)
        or numpy.issubdtype(arr.dtype, numpy.floating)
    ):
        raise ValueError(
            "weights must be a real-valued numeric vector, got dtype %r."
            % arr.dtype
        )
    if arr.ndim != 1 or arr.shape[0] != n:
        raise ValueError(
            "weights must be a one-dimensional array of length equal to the "
            "number of time series (%d), got shape %r."
            % (n, (arr.shape,))
        )
    arr = arr.astype(numpy.float64)  # copy: caller's array is left untouched
    if not numpy.all(numpy.isfinite(arr)):
        raise ValueError("weights must contain only finite values.")
    if numpy.any(arr < 0.0):
        raise ValueError("weights must be non-negative.")
    if not numpy.any(arr > 0.0):
        raise ValueError("weights must have at least one positive entry.")
    return arr


def _check_init_channel_weights(init_channel_weights, d, min_channel_weight):
    """Validate initial channel weights for adaptive channel-weight learning.

    Initial weights must be a valid fixed weight vector (finite, non-negative,
    not all zero), sum to `d` and have all entries at or above
    `min_channel_weight`. If None, a vector of ones is returned.
    """
    if init_channel_weights is None:
        return numpy.ones((d,))
    w = _check_channel_weights(init_channel_weights, d)
    if numpy.any(w < min_channel_weight):
        raise ValueError(
            "init_channel_weights entries must all be greater than or equal "
            "to min_channel_weight (%r)." % min_channel_weight
        )
    if not numpy.isclose(w.sum(), float(d), rtol=1e-8, atol=1e-12):
        raise ValueError(
            "init_channel_weights must sum to d=%d, got sum %r."
            % (d, w.sum())
        )
    return w


def _entropy(w):
    """Entropy-like regularizer sum_c w_c * log(w_c) (w_c > 0)."""
    return float(numpy.sum(w * numpy.log(w)))


def _solve_channel_weights_entropy(per_channel_costs, regularization,
                                   min_weight):
    """Minimize a linear-plus-entropy objective over a capped simplex.

    Solves, for non-negative per-channel costs ``A_c`` (finite or
    ``+inf``),

    .. math::

        \\min_w \\sum_c w_c A_c + \\lambda \\sum_c w_c \\log w_c
        \\quad \\text{s.t.} \\quad \\sum_c w_c = d,\\ w_c \\geq \\varepsilon

    The KKT conditions give :math:`w_c = \\max(\\varepsilon,
    \\exp((-\\mu - A_c)/\\lambda - 1))` where the dual variable
    :math:`\\mu` is the unique root of :math:`\\sum_c w_c(\\mu) = d`.

    The numerical work is delegated to
    :func:`tslearn.metrics._numerics.stable_entropy_solve`, a stable
    KKT reformulation (shifted costs, bounded bisection variable) that
    stays exact when ``A_c`` spans the whole float64 range: the previous
    direct implementation evaluated ``exp((-mu - A) / lam - 1)`` and
    silently produced weights whose sum violated the simplex constraint as
    soon as ``A`` reached ~1e16.

    Parameters
    ----------
    per_channel_costs : numpy.ndarray of shape (d,)
        Non-negative per-channel alignment costs ``A_c`` (entries may be
        ``+inf`` for channels whose true cost is not representable; such
        channels are pinned to the lower bound).
    regularization : float
        Positive entropy regularization coefficient :math:`\\lambda`.
    min_weight : float
        Lower bound :math:`\\varepsilon \\in (0, 1)` on each weight.

    Returns
    -------
    w : numpy.ndarray of shape (d,)
        Optimal weights with ``w.sum() == d`` exactly (up to machine
        precision) and ``w >= min_weight``.
    """
    return stable_entropy_solve(per_channel_costs, regularization,
                                min_weight)


def _per_channel_costs(X, barycenter, list_p_k, weights):
    """Accumulate per-channel alignment costs for fixed warping paths.

    Parameters
    ----------
    X : numpy.array of shape (n, sz, d)
        Time-series to be averaged.
    barycenter : numpy.array of shape (barycenter_size, d)
        Barycenter as computed at the current step of the algorithm.
    list_p_k : list of index pairs
        Warping paths (each path aligns the barycenter to one time series).
    weights : array
        Weights of each X[i]. Must be the same size as len(X).

    Returns
    -------
    numpy.array of shape (d,)
        Per-channel costs ``A_c`` such that ``sum_c w_c A_c`` is the
        (sample-weighted) average of the fixed-path alignment costs, i.e.
        ``A_c = sum_k weights[k] * sum_{(t, j) in p_k}
        (barycenter[t, c] - X[k, j, c])**2 / sum(weights)``.

    Notes
    -----
    The accumulation runs through
    :func:`tslearn.metrics._numerics.per_channel_costs_scaled`: each
    channel gets a power-of-two rescaling chosen from its actual
    differences, so intermediate squares and weighted sums cannot overflow
    even when the inputs sit at the float64 magnitude limit. Entries whose
    true cost is not representable come back as ``+inf`` and are handled by
    the capped-simplex solver (the corresponding weight is pinned to its
    lower bound).
    """
    costs, _ = per_channel_costs_scaled(X, barycenter, list_p_k, weights)
    return costs


def _robust_assignment(barycenter, x_i, metric_params):
    """DTW alignment of one series against the barycenter, overflow-safe.

    Mirrors :func:`dtw_path` but returns the optimal path together with the
    squared cost kept in its scaled representation
    ``(scaled_sq_cost, scale_log2)`` so that callers can average several
    such costs without ever materializing an overflowing individual term.
    The path is identical to the unscaled optimum because a common positive
    scaling preserves arg-min decisions.

    On ordinary-magnitude finite data this takes the classic numba fast
    path directly; the rescaled machinery is only entered when an
    intermediate square or accumulation could actually overflow.
    """
    channel_weights = metric_params.get("channel_weights")
    constraint_code = GLOBAL_CONSTRAINT_CODE[
        metric_params.get("global_constraint")
    ]
    # Fast path: plain float64 arrays with no NaN padding and no overflow
    # risk. Avoids the per-call to_time_series bookkeeping cost on the hot
    # training loop while still using the canonical constraint mask builder.
    plain = (
        isinstance(barycenter, numpy.ndarray)
        and isinstance(x_i, numpy.ndarray)
        and barycenter.dtype == numpy.float64
        and x_i.dtype == numpy.float64
        and numpy.all(numpy.isfinite(barycenter))
        and numpy.all(numpy.isfinite(x_i))
    )
    if plain:
        s1 = barycenter
        s2 = x_i
    else:
        s1 = numpy.ascontiguousarray(
            to_time_series(barycenter, remove_nans=True),
            dtype=numpy.float64,
        )
        s2 = numpy.ascontiguousarray(
            to_time_series(x_i, remove_nans=True), dtype=numpy.float64
        )
    mask = compute_mask(
        s1,
        s2,
        constraint_code,
        sakoe_chiba_radius=metric_params.get("sakoe_chiba_radius"),
        itakura_max_slope=metric_params.get("itakura_max_slope"),
    )
    if not needs_scaled_kernel(s1, s2, channel_weights):
        if channel_weights is None:
            acc = njit_accumulated_matrix(s1, s2, mask=mask)
        else:
            w = numpy.ascontiguousarray(channel_weights,
                                        dtype=numpy.float64)
            acc = njit_accumulated_matrix_weighted(
                s1, s2, mask=mask, weights=w
            )
        path = _njit_return_path(acc)
        return path, float(acc[-1, -1]), 0.0

    path, scaled_sq, scale_log2, overflowed = scaled_dtw_pair(
        s1, s2, mask, channel_weights
    )
    if overflowed:
        raise NonRepresentableNumericalState(
            "An alignment cost used during barycenter training exceeds the "
            "float64 range and cannot be averaged into a finite objective."
        )
    return path, scaled_sq, scale_log2


def _scaled_weighted_average(sq_costs_scaled, scales_log2, sample_weights):
    """Weighted mean of squared DTW costs kept in scaled form.

    ``sq_costs_scaled[i]`` is the squared cost of series ``i`` divided by
    ``2 ** (2 * scales_log2[i])``. The mean is evaluated at a common scale
    (the largest of the per-pair scales) in log-space-safe exponent
    arithmetic, so neither the individual terms nor their weighted sum can
    overflow. Returns ``(mean_scaled, common_scale_log2)``; the true mean is
    recovered with :func:`unscale_value` (or flagged non-representable).
    """
    common_scale = float(numpy.max(scales_log2))
    sample_weights = numpy.asarray(sample_weights, dtype=numpy.float64)
    # Normalize the sample weights by their own maximum: a common positive
    # rescaling of the weights leaves a weighted mean unchanged (the
    # factors cancel) and keeps every accumulator product finite even when
    # the raw weights sit at the float64 magnitude limit. Input weights are
    # strictly validated as finite with at least one positive entry.
    w_max = float(numpy.max(sample_weights))
    if w_max <= 0.0 or not numpy.isfinite(w_max):
        raise NonRepresentableNumericalState(
            "Weighted averaging requires at least one finite positive "
            "sample weight."
        )
    normalized_w = sample_weights / w_max
    total_w = float(numpy.sum(normalized_w))
    # Fast path for ordinary data: every cost was evaluated at the natural
    # scale. Even then their raw sum can overflow (e.g. two finite squared
    # costs of 1e308 average to a representable 1e308), so suppress the
    # numpy overflow warning and fall through to the rescaled accumulation
    # when a direct reduction saturates.
    if common_scale == 0.0:
        sq_arr = numpy.asarray(sq_costs_scaled, dtype=numpy.float64)
        with numpy.errstate(over="ignore"):
            direct = float(
                numpy.sum(normalized_w * sq_arr)
            ) / total_w
        if numpy.isfinite(direct):
            return direct, 0.0
    # Choose an additional accumulation scale from the largest term as
    # expressed at the common scale: averaging several finite terms can
    # otherwise overflow even when their mean is perfectly representable
    # (e.g. the mean of two 1e308 costs is 1e308, while their raw sum
    # saturates). Measured in log space so the bound itself cannot overflow.
    max_term_log2 = -numpy.inf
    for scaled_sq, e in zip(sq_costs_scaled, scales_log2):
        if scaled_sq > 0.0 and numpy.isfinite(scaled_sq):
            term_log2 = math.log2(float(scaled_sq)) - 2.0 * (
                common_scale - float(e)
            )
            if term_log2 > max_term_log2:
                max_term_log2 = term_log2
    if numpy.isinf(max_term_log2):
        extra_scale = 0.0
    else:
        # The numerator adds at most n finite terms; cover that worst case,
        # the final division by total_w can only reduce the magnitude.
        extra_scale = _scale_exponent_from_log2_bound(
            max_term_log2 + math.log2(float(len(sq_costs_scaled)))
        )
    acc = 0.0
    for scaled_sq, e, w in zip(sq_costs_scaled, scales_log2,
                               normalized_w):
        if w == 0.0:
            continue
        # Bring this term to the common scale and then apply the extra
        # accumulation scale (pure exponent changes, never overflowing).
        delta = common_scale - float(e)
        at_common = scaled_sq * (
            1.0 if delta == 0.0 else numpy.float64(2.0 ** (-delta))
        )
        if extra_scale > 0.0:
            at_common = at_common * numpy.float64(2.0 ** (-extra_scale))
        acc += w * at_common
    mean_scaled = acc / total_w
    if not numpy.isfinite(mean_scaled):
        raise NonRepresentableNumericalState(
            "The weighted average of alignment costs is not finite; the "
            "training objective cannot be represented in float64."
        )
    # The common scale divides squared terms by 2**(2 * common_scale); the
    # extra accumulation rescaling divides the already-scaled squared
    # numerator once more by 2**extra_scale (note the single, not doubled,
    # exponent because it is applied after the squared terms are formed).
    return mean_scaled, 2.0 * common_scale + extra_scale


def _true_or_inf_scaled(mean_scaled, scale_log2):
    """Materialize a scaled objective value, returning +inf on overflow."""
    if not numpy.isfinite(mean_scaled):
        return numpy.inf
    return unscale_or_inf(mean_scaled, scale_log2)


def _init_avg(X, barycenter_size):
    if X.shape[1] == barycenter_size:
        return numpy.nanmean(X, axis=0)
    else:
        X_avg = numpy.nanmean(X, axis=0)
        xnew = numpy.linspace(0, 1, barycenter_size)
        f = interp1d(numpy.linspace(0, 1, X_avg.shape[0]), X_avg,
                     kind="linear", axis=0)
        return f(xnew)


def _petitjean_assignment(X, barycenter, metric_params=None):
    if metric_params is None:
        metric_params = {}
    n = X.shape[0]
    barycenter_size = barycenter.shape[0]
    assign = ([[] for _ in range(barycenter_size)],
              [[] for _ in range(barycenter_size)])
    for i in range(n):
        path, _ = dtw_path(X[i], barycenter, **metric_params)
        for pair in path:
            assign[0][pair[1]].append(i)
            assign[1][pair[1]].append(pair[0])
    return assign


def _petitjean_update_barycenter(X, assign, barycenter_size, weights):
    barycenter = numpy.zeros((barycenter_size, X.shape[-1]))
    for t in range(barycenter_size):
        barycenter[t] = numpy.average(X[assign[0][t], assign[1][t]], axis=0,
                                      weights=weights[assign[0][t]])
    return barycenter


def _petitjean_cost(X, barycenter, assign, weights):
    cost = 0.
    barycenter_size = barycenter.shape[0]
    for t_barycenter in range(barycenter_size):
        for i_ts, t_ts in zip(assign[0][t_barycenter],
                              assign[1][t_barycenter]):
            sq_norm = numpy.linalg.norm(X[i_ts, t_ts] -
                                        barycenter[t_barycenter]) ** 2
            cost += weights[i_ts] * sq_norm
    return cost / weights.sum()


def dtw_barycenter_averaging_petitjean(X, barycenter_size=None,
                                       init_barycenter=None,
                                       max_iter=30, tol=1e-5, weights=None,
                                       metric_params=None,
                                       verbose=False):
    """DTW Barycenter Averaging (DBA) method.

    DBA was originally presented in [1]_.
    This implementation is not the one documented in the API, but is kept
    in the codebase to check the documented one for non-regression.

    Parameters
    ----------
    X : array-like, shape=(n_ts, sz, d)
        Time series dataset.

    barycenter_size : int or None (default: None)
        Size of the barycenter to generate. If None, the size of the barycenter
        is that of the data provided at fit
        time or that of the initial barycenter if specified.

    init_barycenter : array or None (default: None)
        Initial barycenter to start from for the optimization process.

    max_iter : int (default: 30)
        Number of iterations of the Expectation-Maximization optimization
        procedure.

    tol : float (default: 1e-5)
        Tolerance to use for early stopping: if the decrease in cost is lower
        than this value, the
        Expectation-Maximization procedure stops.

    weights: None or array
        Weights of each X[i]. Must be the same size as len(X).
        If None, uniform weights are used.

    metric_params: dict or None (default: None)
        DTW constraint parameters to be used.
        See :ref:`tslearn.metrics.dtw_path <fun-tslearn.metrics.dtw_path>` for
        a list of accepted parameters
        If None, no constraint is used for DTW computations.

    verbose : boolean (default: False)
        Whether to print information about the cost at each iteration or not.

    Returns
    -------
    numpy.array of shape (barycenter_size, d) or (sz, d) if barycenter_size \
            is None
        DBA barycenter of the provided time series dataset.

    Examples
    --------
    >>> time_series = [[1, 2, 3, 4], [1, 2, 4, 5]]
    >>> dtw_barycenter_averaging_petitjean(time_series, max_iter=5)
    array([[1. ],
           [2. ],
           [3.5],
           [4.5]])
    >>> time_series = [[1, 2, 3, 4], [1, 2, 3, 4, 5]]
    >>> dtw_barycenter_averaging_petitjean(time_series, max_iter=5)
    array([[1. ],
           [2. ],
           [3. ],
           [4. ],
           [4.5]])
    >>> dtw_barycenter_averaging_petitjean(time_series, max_iter=5,
    ...                          metric_params={"itakura_max_slope": 2})
    array([[1. ],
           [2. ],
           [3. ],
           [3.5],
           [4.5]])
    >>> dtw_barycenter_averaging_petitjean(time_series, max_iter=5,
    ...                                    barycenter_size=3)
    array([[1.5       ],
           [3.        ],
           [4.33333333]])
    >>> dtw_barycenter_averaging_petitjean([[0, 0, 0], [10, 10, 10]],
    ...                                    weights=numpy.array([0.75, 0.25]))
    array([[2.5],
           [2.5],
           [2.5]])

    References
    ----------
    .. [1] F. Petitjean, A. Ketterlin & P. Gancarski. A global averaging method
       for dynamic time warping, with applications to clustering. Pattern
       Recognition, Elsevier, 2011, Vol. 44, Num. 3, pp. 678-693
    """
    X_ = to_time_series_dataset(X)
    if barycenter_size is None:
        barycenter_size = X_.shape[1]
    weights = _set_weights(weights, X_.shape[0])
    if init_barycenter is None:
        barycenter = _init_avg(X_, barycenter_size)
    else:
        barycenter_size = init_barycenter.shape[0]
        barycenter = init_barycenter
    cost_prev, cost = numpy.inf, numpy.inf
    for it in range(max_iter):
        assign = _petitjean_assignment(X_, barycenter, metric_params)
        cost = _petitjean_cost(X_, barycenter, assign, weights)
        if verbose:
            print("[DBA] epoch %d, cost: %.3f" % (it + 1, cost))
        barycenter = _petitjean_update_barycenter(X_, assign, barycenter_size,
                                                  weights)
        if abs(cost_prev - cost) < tol:
            break
        elif cost_prev < cost:
            warnings.warn("DBA loss is increasing while it should not be. "
                          "Stopping optimization.", ConvergenceWarning)
            break
        else:
            cost_prev = cost
    return barycenter


def _mm_assignment(X, barycenter, weights, metric_params=None):
    """Computes item assignement based on DTW alignments and return cost as a
    bonus.

    Parameters
    ----------
    X : numpy.array of shape (n, sz, d)
        Time-series to be averaged

    barycenter : numpy.array of shape (barycenter_size, d)
        Barycenter as computed at the current step of the algorithm.

    weights: array
        Weights of each X[i]. Must be the same size as len(X).

    metric_params: dict or None (default: None)
        DTW constraint parameters to be used.
        See :ref:`tslearn.metrics.dtw_path <fun-tslearn.metrics.dtw_path>` for
        a list of accepted parameters
        If None, no constraint is used for DTW computations.

    Returns
    -------
    list of index pairs
        Warping paths

    float
        Current alignment cost
    """
    if metric_params is None:
        metric_params = {}
    list_p_k = []
    sq_scaled = []
    scales = []
    for i in range(X.shape[0]):
        path, scaled_sq, scale_log2 = _robust_assignment(
            barycenter, X[i], metric_params
        )
        list_p_k.append(path)
        if weights[i] == 0.0:
            sq_scaled.append(0.0)
            scales.append(0.0)
        else:
            sq_scaled.append(scaled_sq)
            scales.append(scale_log2)
    mean_scaled, mean_scale = _scaled_weighted_average(
        sq_scaled, scales, numpy.asarray(weights, dtype=numpy.float64)
    )
    cost = _true_or_inf_scaled(mean_scaled, mean_scale)
    return list_p_k, cost


def _normalize_sample_weights(weights):
    """Divide finite sample weights by their maximum.

    A weighted mean is invariant to multiplying every weight by the same
    positive constant, so this preserves every barycenter objective while
    keeping the internal valence/warping coefficients bounded by one (the
    strict validators guarantee finite non-negative weights with a positive
    entry). The returned array is a fresh copy.
    """
    weights = numpy.asarray(weights, dtype=numpy.float64).copy()
    w_max = float(numpy.max(weights))
    if w_max <= 0.0 or not numpy.isfinite(w_max):
        raise NonRepresentableNumericalState(
            "Sample weights must contain a finite positive entry."
        )
    return weights / w_max


def _subgradient_valence_warping(list_p_k, barycenter_size, weights):
    """Compute Valence and Warping matrices from paths.

    Valence matrices are denoted :math:`V^{(k)}` and Warping matrices are
    :math:`W^{(k)}` in [1]_.

    This function returns a list of :math:`V^{(k)}` diagonals (as a vector)
    and a list of :math:`W^{(k)}` matrices.

    Parameters
    ----------
    list_p_k : list of index pairs
        Warping paths

    barycenter_size : int
        Size of the barycenter to generate.

    weights: array
        Weights of each X[i]. Must be the same size as len(X).

    Returns
    -------
    list of numpy.array of shape (barycenter_size, )
        list of weighted :math:`V^{(k)}` diagonals (as a vector)

    list of numpy.array of shape (barycenter_size, sz_k)
        list of weighted :math:`W^{(k)}` matrices

    References
    ----------

    .. [1] D. Schultz and B. Jain. Nonsmooth Analysis and Subgradient Methods
       for Averaging in Dynamic Time Warping Spaces.
       Pattern Recognition, 74, 340-358.
    """
    list_v_k = []
    list_w_k = []
    for k, p_k in enumerate(list_p_k):
        sz_k = p_k[-1][1] + 1
        w_k = numpy.zeros((barycenter_size, sz_k))
        if weights[k] != 0.0:
            for i, j in p_k:
                w_k[i, j] = 1.
        list_w_k.append(w_k * weights[k])
        list_v_k.append(w_k.sum(axis=1) * weights[k])
    return list_v_k, list_w_k


def _mm_valence_warping(list_p_k, barycenter_size, weights):
    """Compute Valence and Warping matrices from paths.

    Valence matrices are denoted :math:`V^{(k)}` and Warping matrices are
    :math:`W^{(k)}` in [1]_.

    This function returns the sum of :math:`V^{(k)}` diagonals (as a vector)
    and a list of :math:`W^{(k)}` matrices.

    Parameters
    ----------
    list_p_k : list of index pairs
        Warping paths

    barycenter_size : int
        Size of the barycenter to generate.

    weights: array
        Weights of each X[i]. Must be the same size as len(X).

    Returns
    -------
    numpy.array of shape (barycenter_size, )
        sum of weighted :math:`V^{(k)}` diagonals (as a vector)

    list of numpy.array of shape (barycenter_size, sz_k)
        list of weighted :math:`W^{(k)}` matrices

    References
    ----------

    .. [1] D. Schultz and B. Jain. Nonsmooth Analysis and Subgradient Methods
       for Averaging in Dynamic Time Warping Spaces.
       Pattern Recognition, 74, 340-358.
    """
    list_v_k, list_w_k = _subgradient_valence_warping(
        list_p_k=list_p_k,
        barycenter_size=barycenter_size,
        weights=weights)
    diag_sum_v_k = numpy.zeros(list_v_k[0].shape)
    for v_k in list_v_k:
        diag_sum_v_k += v_k
    return diag_sum_v_k, list_w_k


def _mm_update_barycenter(X, diag_sum_v_k, list_w_k):
    """Update barycenters using the formula from Algorithm 2 in [1]_.

    Parameters
    ----------
    X : numpy.array of shape (n, sz, d)
        Time-series to be averaged

    diag_sum_v_k : numpy.array of shape (barycenter_size, )
        sum of weighted :math:`V^{(k)}` diagonals (as a vector)

    list_w_k : list of numpy.array of shape (barycenter_size, sz_k)
        list of weighted :math:`W^{(k)}` matrices

    Returns
    -------
    numpy.array of shape (barycenter_size, d)
        Updated barycenter

    References
    ----------

    .. [1] D. Schultz and B. Jain. Nonsmooth Analysis and Subgradient Methods
       for Averaging in Dynamic Time Warping Spaces.
       Pattern Recognition, 74, 340-358.
    """
    d = X.shape[2]
    barycenter_size = diag_sum_v_k.shape[0]
    # Fast path for ordinary-magnitude data: the original vectorized MM
    # update, bit-for-bit identical to the v0.6.4 arithmetic.
    if (
        numpy.all(numpy.isfinite(X))
        and float(numpy.max(numpy.abs(X))) <= _SQUARE_SAFE_BOUND
    ):
        sum_w_x = numpy.zeros((barycenter_size, d))
        for k, (w_k, x_k) in enumerate(zip(list_w_k, X)):
            sum_w_x += w_k.dot(x_k[:ts_size(x_k)])
        barycenter = numpy.diag(1. / diag_sum_v_k).dot(sum_w_x)
        if numpy.all(numpy.isfinite(barycenter)):
            return barycenter
    sum_w_x = numpy.zeros((barycenter_size, d))
    # Each barycenter coordinate is a convex combination of finite input
    # values, hence it is representable whenever the inputs are. The
    # numerator and denominator are nevertheless accumulated under a
    # per-coordinate power-of-two rescaling so intermediate matrix products
    # cannot overflow (e.g. several 1e308-valued warping entries).
    for t in range(barycenter_size):
        if diag_sum_v_k[t] <= 0.0:
            continue
        max_abs = 0.0
        for k, w_k in enumerate(list_w_k):
            x_k = X[k][:ts_size(X[k])]
            nz = numpy.flatnonzero(w_k[t])
            if nz.size:
                m = float(numpy.max(numpy.abs(x_k[nz])))
                if m > max_abs:
                    max_abs = m
        if max_abs <= 1.0:
            scale_log2 = 0.0
            inv_scale = 1.0
        else:
            # A convex combination of values bounded by max_abs cannot
            # exceed max_abs, so a scale making max_abs itself finite is
            # sufficient for both the numerator and the final coordinate.
            scale_log2 = _scale_exponent_from_log2_bound(
                math.log2(max_abs)
            )
            inv_scale = numpy.float64(2.0 ** (-scale_log2))
        numerator = numpy.zeros((d,))
        for k, w_k in enumerate(list_w_k):
            x_k = X[k][:ts_size(X[k])]
            numerator += w_k[t].dot(x_k * inv_scale)
        if not numpy.all(numpy.isfinite(numerator)):
            raise NonRepresentableNumericalState(
                "Barycenter coordinate %d cannot be evaluated at finite "
                "float64 magnitude." % t
            )
        # The numerator already holds sum(w * (x / 2**e)); multiplying the
        # coordinate by 2**e restores its true magnitude, and the division
        # by diag_sum_v_k is scale-invariant (both numerator and denominator
        # live at the same scale). Apply the division first while values
        # are still scaled so the final matrix product cannot re-overflow.
        if scale_log2 > 0.0:
            coord = numpy.array([
                unscale_value(float(v) / float(diag_sum_v_k[t]),
                              scale_log2)
                for v in numerator
            ])
        else:
            coord = numerator / diag_sum_v_k[t]
        sum_w_x[t] = coord
    # diag_sum_v_k is the per-row valence total, which equals the sum of
    # warping coefficients entering each barycenter coordinate: sum_w_x is
    # already divided by it above, so the barycenter is sum_w_x itself.
    barycenter = sum_w_x
    if not numpy.all(numpy.isfinite(barycenter)):
        raise NonRepresentableNumericalState(
            "The updated barycenter contains non-finite entries; the "
            "requested barycenter is not representable in float64."
        )
    return barycenter


def _subgradient_update_barycenter(X, list_diag_v_k, list_w_k, weights_sum,
                                   barycenter, eta):
    """Update barycenters using the formula from Algorithm 1 in [1]_.

    Parameters
    ----------
    X : numpy.array of shape (n, sz, d)
        Time-series to be averaged

    list_diag_v_k : list of numpy.array of shape (barycenter_size, )
        list of weighted :math:`V^{(k)}` diagonals (as vectors)

    list_w_k : list of numpy.array of shape (barycenter_size, sz_k)
        list of weighted :math:`W^{(k)}` matrices

    weights_sum : float
        sum of weights applied to matrices :math:`V^{(k)}` and :math:`W^{(k)}`

    barycenter : numpy.array of shape (barycenter_size, d)
        Barycenter as computed at the previous iteration of the algorithm

    eta : float
        Step-size for the subgradient descent algorithm

    Returns
    -------
    numpy.array of shape (barycenter_size, d)
        Updated barycenter

    References
    ----------

    .. [1] D. Schultz and B. Jain. Nonsmooth Analysis and Subgradient Methods
       for Averaging in Dynamic Time Warping Spaces.
       Pattern Recognition, 74, 340-358.
    """
    d = X.shape[2]
    barycenter_size = barycenter.shape[0]
    delta_bar = numpy.zeros((barycenter_size, d))
    for k, (v_k, w_k, x_k) in enumerate(zip(list_diag_v_k, list_w_k, X)):
        delta_bar += v_k.reshape((-1, 1)) * barycenter
        delta_bar -= w_k.dot(x_k[:ts_size(x_k)])
    barycenter -= (2. * eta / weights_sum) * delta_bar
    return barycenter


def _evaluate_scaled(bary, w, X_, weights, metric_params, regularization):
    """Evaluate one adaptive-DBA state, keeping the data term scaled.

    Returns
    -------
    list_p_k : list of warping paths
    data_scaled, data_scale_log2 : float
        Weighted mean squared DTW cost of the state, divided by
        ``2 ** data_scale_log2`` (never materialized at its true magnitude
        while it could overflow).
    entropy : float
        Entropy regularizer value (data-scale independent).
    """
    mp = dict(metric_params)
    mp["channel_weights"] = w
    sq_scaled, scales = [], []
    list_p_k = []
    for i in range(X_.shape[0]):
        if weights[i] == 0.0:
            # Zero-weight series still get a real alignment so all path
            # matrices keep their true shapes, but their zero sample weight
            # makes every downstream contribution vanish identically.
            path, _, _ = _robust_assignment(bary, X_[i], mp)
            list_p_k.append(path)
            sq_scaled.append(0.0)
            scales.append(0.0)
            continue
        path, scaled_sq, scale_log2 = _robust_assignment(bary, X_[i], mp)
        list_p_k.append(path)
        sq_scaled.append(scaled_sq)
        scales.append(scale_log2)
    data_scaled, data_scale_log2 = _scaled_weighted_average(
        sq_scaled, scales, weights
    )
    entropy = _entropy(w)
    return list_p_k, data_scaled, data_scale_log2, entropy


def _objective_scaled(data_scaled, data_scale_log2, entropy,
                      regularization):
    """Pack the regularized objective into a (scaled value, scale) pair.

    The entropy term is finite by construction (weights are finite and
    bounded); it is expressed in the data term's scale by dividing it by
    the same power of two, which is exact in exponent arithmetic.
    """
    if data_scale_log2 == 0.0:
        return data_scaled + regularization * entropy, 0.0
    entropy_at_scale = regularization * entropy * numpy.float64(
        2.0 ** (-data_scale_log2)
    )
    return data_scaled + entropy_at_scale, data_scale_log2


def _accept_scaled(new_pair, old_pair, tol_factor=1e-8):
    """Scale-invariant version of the public acceptance rule.

    ``new <= old + tol_factor * max(1, |old|)`` evaluated after aligning
    both states to a common power-of-two scale. Returns True when the new
    state does not increase the objective beyond the tolerance.
    """
    new_scaled, new_scale = new_pair
    old_scaled, old_scale = old_pair
    common = max(new_scale, old_scale)
    old_at_common = old_scaled * numpy.float64(
        2.0 ** (-(common - old_scale))
    )
    new_at_common = new_scaled * numpy.float64(
        2.0 ** (-(common - new_scale))
    )
    if not (numpy.isfinite(old_at_common)
            and numpy.isfinite(new_at_common)):
        # Only a strict finite improvement can be accepted once one side
        # saturates at the common scale.
        return new_at_common <= old_at_common
    tolerance = tol_factor * max(1.0, abs(old_at_common))
    return new_at_common <= old_at_common + tolerance


def _materialize_objective(scaled_pair):
    """True float64 objective value, or +inf when it is not representable."""
    scaled, scale_log2 = scaled_pair
    return _true_or_inf_scaled(scaled, scale_log2)


def _dba_weighted_one_init(X_, barycenter, weights, init_channel_weights,
                           regularization, min_channel_weight, max_iter, tol,
                           metric_params, verbose):
    """Jointly optimize a DBA barycenter and its channel weights (one init).

    Alternating Majorize-Minimize scheme: at each iteration, (i) optimal
    warping paths are recomputed for the current barycenter and channel
    weights, (ii) the barycenter is updated with the classic MM update
    (which is invariant to positive channel weights), (iii) the channel
    weights are updated by exactly minimizing the fixed-path surrogate plus
    the entropy regularizer. Each of these three steps does not increase
    the true objective

        J(b, w) = weighted average of dtw_w(b, x_k)**2
                  + regularization * sum_c w_c log(w_c),

    hence the recorded objective values form a non-increasing sequence (up
    to a numerical acceptance tolerance of 1e-8 * max(1, |J_previous|)).

    All costs that can overflow float64 are carried in scaled
    representations: states whose center, weights and true objective are
    representable are accepted even when individual per-series squared
    distances are not, while genuinely non-representable states raise
    :class:`ValueError` instead of being delivered.

    Parameters
    ----------
    X_ : numpy.array of shape (n, sz, d)
        Time series dataset (already converted with to_time_series_dataset).
    barycenter : numpy.array of shape (barycenter_size, d)
        Initial barycenter.
    weights : numpy.array of shape (n,)
        Validated sample weights.
    init_channel_weights : numpy.array of shape (d,)
        Validated initial channel weights (sum to d, entries >=
        min_channel_weight).
    regularization : float
        Positive entropy regularization coefficient.
    min_channel_weight : float
        Lower bound in (0, 1) for learned channel weights.
    max_iter : int
        Maximum number of alternating iterations.
    tol : float
        Stop when the objective decrease between two accepted states is
        lower than this value.
    metric_params: dict or None
        DTW constraint parameters (must not contain "channel_weights").
    verbose : boolean
        Whether to print the objective at each iteration.

    Returns
    -------
    barycenter : numpy.array of shape (barycenter_size, d)
        Learned barycenter.
    channel_weights : numpy.array of shape (d,)
        Learned channel weights.
    objective_history : list of float
        True objective values of the accepted states, in chronological
        order. The last entry is the objective of the returned
        (barycenter, channel_weights) pair and is always finite.
    """
    barycenter_size = barycenter.shape[0]
    channel_weights = init_channel_weights.copy()
    if metric_params is None:
        metric_params = {}
    else:
        metric_params = dict(metric_params)

    list_p_k, data_scaled, data_scale, entropy = _evaluate_scaled(
        barycenter, channel_weights, X_, weights, metric_params,
        regularization
    )
    objective_pair = _objective_scaled(
        data_scaled, data_scale, entropy, regularization
    )
    objective_history = [_materialize_objective(objective_pair)]
    for it in range(max_iter):
        diag_sum_v_k, list_w_k = _mm_valence_warping(list_p_k,
                                                     barycenter_size,
                                                     weights)
        new_barycenter = _mm_update_barycenter(X_, diag_sum_v_k, list_w_k)
        per_channel = _per_channel_costs(X_, new_barycenter, list_p_k,
                                         weights)
        new_channel_weights = _solve_channel_weights_entropy(
            per_channel, regularization, min_channel_weight
        )
        # Defensive constraint check: the solver repair step guarantees
        # these, but never deliver a state that violates the simplex.
        _assert_valid_learned_weights(new_channel_weights,
                                      min_channel_weight)
        (new_list_p_k, new_data_scaled, new_data_scale,
         new_entropy) = _evaluate_scaled(
            new_barycenter, new_channel_weights, X_, weights,
            metric_params, regularization
        )
        new_objective_pair = _objective_scaled(
            new_data_scaled, new_data_scale, new_entropy, regularization
        )
        if not _accept_scaled(new_objective_pair, objective_pair):
            # Should not happen (each update decreases the true objective in
            # exact arithmetic): refuse the increasing state and stop.
            warnings.warn(
                "Adaptive DBA objective increased by more than the "
                "acceptance tolerance (numerical issue). Stopping "
                "optimization.",
                ConvergenceWarning,
            )
            break
        barycenter = new_barycenter
        channel_weights = new_channel_weights
        list_p_k = new_list_p_k
        new_objective = _materialize_objective(new_objective_pair)
        if not numpy.isfinite(new_objective):
            raise NonRepresentableNumericalState(
                "The adaptive DBA objective of an otherwise acceptable "
                "state exceeds the float64 range; no finite model can be "
                "delivered."
            )
        objective_history.append(new_objective)
        if verbose:
            print("[Weighted DBA] epoch %d, objective: %.5f"
                  % (it + 1, new_objective))
        if abs(objective_history[-2] - new_objective) < tol:
            objective_pair = new_objective_pair
            break
        objective_pair = new_objective_pair
    # Final delivery contract: the last history entry is the true objective
    # of the returned (barycenter, weights) pair evaluated against them.
    # The alternating loop already evaluated this exact state when it was
    # accepted, so its scaled pair is reused: no extra alignment pass (and
    # no extra observable state evaluation) is needed.
    final_objective = _materialize_objective(objective_pair)
    if not numpy.isfinite(final_objective):
        raise NonRepresentableNumericalState(
            "The final adaptive DBA objective is not representable in "
            "float64; refusing to deliver an infinite-objective model."
        )
    if not numpy.all(numpy.isfinite(barycenter)):
        raise NonRepresentableNumericalState(
            "The learned barycenter contains non-finite entries."
        )
    _assert_valid_learned_weights(channel_weights, min_channel_weight)
    objective_history[-1] = final_objective
    return barycenter, channel_weights, objective_history


def _assert_valid_learned_weights(channel_weights, min_channel_weight):
    """Raise ValueError if a learned weight vector violates its constraints."""
    w = numpy.asarray(channel_weights, dtype=numpy.float64)
    if not numpy.all(numpy.isfinite(w)):
        raise NonRepresentableNumericalState(
            "Learned channel weights must all be finite, got %r." % (w,)
        )
    if numpy.any(w < min_channel_weight - 1e-12):
        raise NonRepresentableNumericalState(
            "Learned channel weights must be at least min_channel_weight "
            "(%r), got %r." % (min_channel_weight, w)
        )
    if not numpy.isclose(float(numpy.sum(w)), float(w.shape[0]),
                         rtol=1e-9, atol=1e-12):
        raise NonRepresentableNumericalState(
            "Learned channel weights must sum to the number of channels "
            "(%d), got sum %.15g." % (w.shape[0], float(numpy.sum(w)))
        )


def dtw_barycenter_averaging(X, barycenter_size=None, init_barycenter=None,
                             max_iter=30, tol=1e-5, weights=None,
                             metric_params=None,
                             verbose=False, n_init=1,
                             channel_weight_regularization=None,
                             min_channel_weight=1e-6,
                             init_channel_weights=None):
    """DTW Barycenter Averaging (DBA) method estimated through
    Expectation-Maximization algorithm.

    DBA was originally presented in [1]_.
    This implementation is based on a idea from [2]_ (Majorize-Minimize Mean
    Algorithm).

    If `channel_weight_regularization` is set, channel weights are learned
    jointly with the barycenter: the optimized objective is the
    (sample-weighted) average of the channel-weighted squared DTW distances
    plus `channel_weight_regularization` times an entropy regularizer
    :math:`\\sum_c w_c \\log w_c`, where the weights are constrained to sum
    to `d` and to be at least `min_channel_weight`. In that case, a tuple
    `(barycenter, channel_weights, objective_history)` is returned.

    Parameters
    ----------
    X : array-like, shape=(n_ts, sz, d)
        Time series dataset.

    barycenter_size : int or None (default: None)
        Size of the barycenter to generate. If None, the size of the barycenter
        is that of the data provided at fit
        time or that of the initial barycenter if specified.

    init_barycenter : array or None (default: None)
        Initial barycenter to start from for the optimization process.

    max_iter : int (default: 30)
        Number of iterations of the Expectation-Maximization optimization
        procedure.

    tol : float (default: 1e-5)
        Tolerance to use for early stopping: if the decrease in cost is lower
        than this value, the
        Expectation-Maximization procedure stops.

    weights: None or array
        Weights of each X[i]. Must be the same size as len(X).
        If None, uniform weights are used.
        If `channel_weight_regularization` is set, these are strictly
        validated: they must be a one-dimensional real-valued vector of
        length `n_ts` with finite non-negative entries and at least one
        positive entry.

    metric_params: dict or None (default: None)
        DTW constraint parameters to be used.
        See :ref:`tslearn.metrics.dtw_path <fun-tslearn.metrics.dtw_path>` for
        a list of accepted parameters
        If None, no constraint is used for DTW computations.
        Must not contain a "channel_weights" entry when
        `channel_weight_regularization` is set (channel weights are then
        learned, not fixed).

    verbose : boolean (default: False)
        Whether to print information about the cost at each iteration or not.

    n_init : int (default: 1)
        Number of different initializations to be tried (useful only is
        init_barycenter is set to None, otherwise, all trials will reach the
        same performance)

    channel_weight_regularization : None or float (default: None)
        If None, the standard DBA barycenter is computed and returned as
        before. If a finite positive real number :math:`\\lambda`, channel
        weights are learned together with the barycenter by minimizing the
        average weighted squared DTW distance plus
        :math:`\\lambda \\sum_c w_c \\log w_c`.

    min_channel_weight : float (default: 1e-6)
        Lower bound :math:`\\varepsilon \\in (0, 1)` on each learned channel
        weight. Only used when `channel_weight_regularization` is set.

    init_channel_weights : array-like, shape=(d,) or None (default: None)
        Initial channel weights for the adaptive mode. Must sum to `d` and
        have all entries at or above `min_channel_weight`. If None, a vector
        of ones is used. Only used when `channel_weight_regularization` is
        set.

    Returns
    -------
    numpy.array of shape (barycenter_size, d) or (sz, d) if barycenter_size \
            is None
        DBA barycenter of the provided time series dataset.
        This is the only returned value when
        `channel_weight_regularization` is None.

    tuple (barycenter, channel_weights, objective_history)
        Returned instead when `channel_weight_regularization` is set:
        `barycenter` is the learned barycenter (same shape as above),
        `channel_weights` is the learned weight vector of shape (d,) with
        entries in `[min_channel_weight, d]` summing to `d`, and
        `objective_history` is the list of true objective values of the
        accepted states (non-increasing), whose last entry is the objective
        of the returned (barycenter, channel_weights) pair. With `n_init`
        greater than 1, the results of the initialization with the lowest
        final objective are returned.

    Examples
    --------
    >>> time_series = [[1, 2, 3, 4], [1, 2, 4, 5]]
    >>> dtw_barycenter_averaging(time_series, max_iter=5)
    array([[1. ],
           [2. ],
           [3.5],
           [4.5]])
    >>> time_series = [[1, 2, 3, 4], [1, 2, 3, 4, 5]]
    >>> dtw_barycenter_averaging(time_series, max_iter=5)
    array([[1. ],
           [2. ],
           [3. ],
           [4. ],
           [4.5]])
    >>> dtw_barycenter_averaging(time_series, max_iter=5,
    ...                          metric_params={"itakura_max_slope": 2})
    array([[1. ],
           [2. ],
           [3. ],
           [3.5],
           [4.5]])
    >>> dtw_barycenter_averaging(time_series, max_iter=5, barycenter_size=3)
    array([[1.5       ],
           [3.        ],
           [4.33333333]])
    >>> dtw_barycenter_averaging([[0, 0, 0], [10, 10, 10]], max_iter=1,
    ...                          weights=numpy.array([0.75, 0.25]))
    array([[2.5],
           [2.5],
           [2.5]])
    >>> bary, w, hist = dtw_barycenter_averaging(
    ...     [[[0., 0.], [1., 0.]], [[0., 1.], [1., 1.]]],
    ...     channel_weight_regularization=0.5, max_iter=5)
    >>> round(float(w.sum()), 12)
    2.0

    References
    ----------
    .. [1] F. Petitjean, A. Ketterlin & P. Gancarski. A global averaging method
       for dynamic time warping, with applications to clustering. Pattern
       Recognition, Elsevier, 2011, Vol. 44, Num. 3, pp. 678-693

    .. [2] D. Schultz and B. Jain. Nonsmooth Analysis and Subgradient Methods
       for Averaging in Dynamic Time Warping Spaces.
       Pattern Recognition, 74, 340-358.
    """
    if channel_weight_regularization is not None:
        # Validate everything before any optimization starts.
        regularization = _check_channel_weight_regularization(
            channel_weight_regularization
        )
        min_channel_weight = _check_min_channel_weight(min_channel_weight)
        if metric_params is not None and "channel_weights" in metric_params:
            raise ValueError(
                "metric_params must not contain 'channel_weights' when "
                "channel_weight_regularization is set: channel weights are "
                "then learned, not fixed."
            )
        X_ = to_time_series_dataset(X)
        weights_ = _check_sample_weights_strict(weights, X_.shape[0])
        init_channel_weights_ = _check_init_channel_weights(
            init_channel_weights, X_.shape[2], min_channel_weight
        )
        best_objective = numpy.inf
        best_result = None
        for i in range(n_init):
            if verbose:
                print("Attempt {}".format(i + 1))
            bary, w, history = dtw_barycenter_averaging_one_init(
                X=X_,
                barycenter_size=barycenter_size,
                init_barycenter=init_barycenter,
                max_iter=max_iter,
                tol=tol,
                weights=weights_,
                metric_params=metric_params,
                verbose=verbose,
                channel_weight_regularization=regularization,
                min_channel_weight=min_channel_weight,
                init_channel_weights=init_channel_weights_,
            )
            if history[-1] < best_objective:
                best_objective = history[-1]
                best_result = (bary, w, history)
        return best_result

    best_cost = numpy.inf
    best_barycenter = None
    for i in range(n_init):
        if verbose:
            print("Attempt {}".format(i + 1))
        bary, loss = dtw_barycenter_averaging_one_init(
            X=X,
            barycenter_size=barycenter_size,
            init_barycenter=init_barycenter,
            max_iter=max_iter,
            tol=tol,
            weights=weights,
            metric_params=metric_params,
            verbose=verbose
        )
        if loss < best_cost:
            best_cost = loss
            best_barycenter = bary
    return best_barycenter


def dtw_barycenter_averaging_one_init(X, barycenter_size=None,
                                      init_barycenter=None,
                                      max_iter=30, tol=1e-5, weights=None,
                                      metric_params=None,
                                      verbose=False,
                                      channel_weight_regularization=None,
                                      min_channel_weight=1e-6,
                                      init_channel_weights=None):
    """DTW Barycenter Averaging (DBA) method estimated through
    Expectation-Maximization algorithm.

    DBA was originally presented in [1]_.
    This implementation is based on a idea from [2]_ (Majorize-Minimize Mean
    Algorithm).

    Parameters
    ----------
    X : array-like, shape=(n_ts, sz, d)
        Time series dataset.

    barycenter_size : int or None (default: None)
        Size of the barycenter to generate. If None, the size of the barycenter
        is that of the data provided at fit
        time or that of the initial barycenter if specified.

    init_barycenter : array or None (default: None)
        Initial barycenter to start from for the optimization process.

    max_iter : int (default: 30)
        Number of iterations of the Expectation-Maximization optimization
        procedure.

    tol : float (default: 1e-5)
        Tolerance to use for early stopping: if the decrease in cost is lower
        than this value, the
        Expectation-Maximization procedure stops.

    weights: None or array
        Weights of each X[i]. Must be the same size as len(X).
        If None, uniform weights are used.

    metric_params: dict or None (default: None)
        DTW constraint parameters to be used.
        See :ref:`tslearn.metrics.dtw_path <fun-tslearn.metrics.dtw_path>` for
        a list of accepted parameters
        If None, no constraint is used for DTW computations.

    verbose : boolean (default: False)
        Whether to print information about the cost at each iteration or not.

    channel_weight_regularization : None or float (default: None)
        If None, the standard DBA barycenter is computed. If a finite
        positive real number, channel weights are learned jointly with the
        barycenter (see :func:`dtw_barycenter_averaging`).

    min_channel_weight : float (default: 1e-6)
        Lower bound in (0, 1) on each learned channel weight. Only used when
        `channel_weight_regularization` is set.

    init_channel_weights : array-like, shape=(d,) or None (default: None)
        Initial channel weights for the adaptive mode. If None, a vector of
        ones is used.

    Returns
    -------
    numpy.array of shape (barycenter_size, d) or (sz, d) if barycenter_size \
            is None
        DBA barycenter of the provided time series dataset.
    float
        Associated inertia

    If `channel_weight_regularization` is set, returns instead a tuple
    (barycenter, channel_weights, objective_history) where
    objective_history[-1] is the final value of the regularized objective.

    References
    ----------
    .. [1] F. Petitjean, A. Ketterlin & P. Gancarski. A global averaging method
       for dynamic time warping, with applications to clustering. Pattern
       Recognition, Elsevier, 2011, Vol. 44, Num. 3, pp. 678-693

    .. [2] D. Schultz and B. Jain. Nonsmooth Analysis and Subgradient Methods
       for Averaging in Dynamic Time Warping Spaces.
       Pattern Recognition, 74, 340-358.
    """
    X_ = to_time_series_dataset(X)
    if barycenter_size is None:
        barycenter_size = X_.shape[1]
    if channel_weight_regularization is not None:
        regularization = _check_channel_weight_regularization(
            channel_weight_regularization
        )
        min_channel_weight = _check_min_channel_weight(min_channel_weight)
        if metric_params is not None and "channel_weights" in metric_params:
            raise ValueError(
                "metric_params must not contain 'channel_weights' when "
                "channel_weight_regularization is set: channel weights are "
                "then learned, not fixed."
            )
        weights = _normalize_sample_weights(
            _check_sample_weights_strict(weights, X_.shape[0])
        )
        init_channel_weights = _check_init_channel_weights(
            init_channel_weights, X_.shape[2], min_channel_weight
        )
        if init_barycenter is None:
            barycenter = _init_avg(X_, barycenter_size)
        else:
            barycenter_size = init_barycenter.shape[0]
            barycenter = init_barycenter
        return _dba_weighted_one_init(
            X_, barycenter, weights, init_channel_weights,
            regularization, min_channel_weight, max_iter, tol,
            metric_params, verbose
        )
    weights = _set_weights(weights, X_.shape[0])
    if numpy.isfinite(numpy.max(weights)) and numpy.max(weights) > 0.0:
        weights = _normalize_sample_weights(weights)
    if init_barycenter is None:
        barycenter = _init_avg(X_, barycenter_size)
    else:
        barycenter_size = init_barycenter.shape[0]
        barycenter = init_barycenter
    cost_prev, cost = numpy.inf, numpy.inf
    for it in range(max_iter):
        list_p_k, cost = _mm_assignment(X_, barycenter, weights, metric_params)
        diag_sum_v_k, list_w_k = _mm_valence_warping(list_p_k, barycenter_size,
                                                     weights)
        if verbose:
            print("[DBA] epoch %d, cost: %.3f" % (it + 1, cost))
        barycenter = _mm_update_barycenter(X_, diag_sum_v_k, list_w_k)
        if abs(cost_prev - cost) < tol:
            break
        elif cost_prev < cost:
            warnings.warn("DBA loss is increasing while it should not be. "
                          "Stopping optimization.", ConvergenceWarning)
            break
        else:
            cost_prev = cost
    return barycenter, cost


def dtw_barycenter_averaging_subgradient(X, barycenter_size=None,
                                         init_barycenter=None, max_iter=30,
                                         initial_step_size=.05,
                                         final_step_size=.005,
                                         tol=1e-5, random_state=None,
                                         weights=None,
                                         metric_params=None, verbose=False):
    """DTW Barycenter Averaging (DBA) method estimated through subgradient
    descent algorithm.

    DBA was originally presented in [1]_.
    This implementation is based on a idea from [2]_ (Stochastic Subgradient
    Mean Algorithm).

    Parameters
    ----------
    X : array-like, shape=(n_ts, sz, d)
        Time series dataset.

    barycenter_size : int or None (default: None)
        Size of the barycenter to generate. If None, the size of the barycenter
        is that of the data provided at fit
        time or that of the initial barycenter if specified.

    init_barycenter : array or None (default: None)
        Initial barycenter to start from for the optimization process.

    max_iter : int (default: 30)
        Number of iterations of the Expectation-Maximization optimization
        procedure.

    initial_step_size : float (default: 0.05)
        Initial step size for the subgradient descent algorithm.
        Default value is the one suggested in [2]_.

    final_step_size : float (default: 0.005)
        Final step size for the subgradient descent algorithm.
        Default value is the one suggested in [2]_.

    tol : float (default: 1e-5)
        Tolerance to use for early stopping: if the decrease in cost is lower
        than this value, the
        Expectation-Maximization procedure stops.

    random_state : int, RandomState instance or None, optional (default=None)
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.

    weights: None or array
        Weights of each X[i]. Must be the same size as len(X).
        If None, uniform weights are used.

    metric_params: dict or None (default: None)
        DTW constraint parameters to be used.
        See :ref:`tslearn.metrics.dtw_path <fun-tslearn.metrics.dtw_path>` for
        a list of accepted parameters
        If None, no constraint is used for DTW computations.

    verbose : boolean (default: False)
        Whether to print information about the cost at each iteration or not.

    Returns
    -------
    numpy.array of shape (barycenter_size, d) or (sz, d) if barycenter_size \
            is None
        DBA barycenter of the provided time series dataset.

    Examples
    --------
    >>> time_series = [[1, 2, 3, 4], [1, 2, 4, 5]]
    >>> dtw_barycenter_averaging_subgradient(
    ...     time_series,
    ...     max_iter=10,
    ...     random_state=0
    ... )  # doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
    array([[1. ],
           [2. ],
           [3.5...],
           [4.5...]])

    References
    ----------
    .. [1] F. Petitjean, A. Ketterlin & P. Gancarski. A global averaging method
       for dynamic time warping, with applications to clustering. Pattern
       Recognition, Elsevier, 2011, Vol. 44, Num. 3, pp. 678-693

    .. [2] D. Schultz and B. Jain. Nonsmooth Analysis and Subgradient Methods
       for Averaging in Dynamic Time Warping Spaces.
       Pattern Recognition, 74, 340-358.
    """
    rng = check_random_state(random_state)

    X_ = to_time_series_dataset(X)
    if barycenter_size is None:
        barycenter_size = X_.shape[1]
    weights = _set_weights(weights, X_.shape[0])
    if init_barycenter is None:
        barycenter = _init_avg(X_, barycenter_size)
    else:
        barycenter_size = init_barycenter.shape[0]
        barycenter = init_barycenter
    cost_prev, cost = numpy.inf, numpy.inf
    eta = initial_step_size
    n = X_.shape[0]
    for it in range(max_iter):
        shuffled_indices = rng.permutation(n)
        for idx in shuffled_indices:
            Xi = X_[idx:idx+1]
            wi = weights[idx:idx+1]
            list_p_k, cost = _mm_assignment(Xi, barycenter, weights,
                                            metric_params)
            list_diag_v_k, list_w_k = _subgradient_valence_warping(
                list_p_k,
                barycenter_size,
                wi
            )
            if verbose:
                print("[DBA] epoch %d, cost: %.3f" % (it + 1, cost))
            barycenter = _subgradient_update_barycenter(Xi, list_diag_v_k,
                                                        list_w_k, wi.sum(),
                                                        barycenter, eta)
            if it == 0:
                eta -= (initial_step_size - final_step_size) / n
        if abs(cost_prev - cost) < tol:
            break
        elif cost_prev < cost:
            warnings.warn("DBA loss is increasing while it should not be. "
                          "Stopping optimization.", ConvergenceWarning)
            break
        else:
            cost_prev = cost
    return barycenter
