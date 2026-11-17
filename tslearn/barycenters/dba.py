import numbers

import numpy
import warnings
from scipy.interpolate import interp1d
from sklearn.exceptions import ConvergenceWarning
from sklearn.utils import check_random_state

from ..metrics import dtw_path
from ..metrics.dtw_variants import _check_channel_weights
from ..utils import to_time_series_dataset, ts_size
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

    Solves, for non-negative per-channel costs ``A_c``,

    .. math::

        \\min_w \\sum_c w_c A_c + \\lambda \\sum_c w_c \\log w_c
        \\quad \\text{s.t.} \\quad \\sum_c w_c = d,\\ w_c \\geq \\varepsilon

    The KKT conditions give :math:`w_c = \\max(\\varepsilon,
    \\exp((-\\mu - A_c)/\\lambda - 1))` where the dual variable
    :math:`\\mu` is the unique root of :math:`\\sum_c w_c(\\mu) = d`, found
    here by bisection.

    Parameters
    ----------
    per_channel_costs : numpy.ndarray of shape (d,)
        Non-negative per-channel alignment costs ``A_c``.
    regularization : float
        Positive entropy regularization coefficient :math:`\\lambda`.
    min_weight : float
        Lower bound :math:`\\varepsilon \\in (0, 1)` on each weight.

    Returns
    -------
    w : numpy.ndarray of shape (d,)
        Optimal weights with ``w.sum() == d`` (up to machine precision) and
        ``w >= min_weight``.
    """
    A = numpy.asarray(per_channel_costs, dtype=numpy.float64)
    d = A.shape[0]
    if d == 1:
        # The only feasible weight is 1 (sum must be d = 1).
        return numpy.ones((1,))
    eps = float(min_weight)
    lam = float(regularization)

    def capped_weights(mu):
        w = numpy.exp((-mu - A) / lam - 1.0)
        numpy.maximum(w, eps, out=w)
        return w

    # total(mu) is continuous and decreasing in mu, from +inf (mu -> -inf)
    # to d * eps < d (mu -> +inf): a unique mu with total(mu) == d exists.
    step = max(lam, 1.0)
    mu_lo = 0.0
    while capped_weights(mu_lo).sum() < d:
        mu_lo -= step
        step *= 2.0
    step = max(lam, 1.0)
    mu_hi = 0.0
    while capped_weights(mu_hi).sum() > d:
        mu_hi += step
        step *= 2.0
    for _ in range(200):  # 200 halvings: far beyond float64 resolution
        mu_mid = 0.5 * (mu_lo + mu_hi)
        if capped_weights(mu_mid).sum() > d:
            mu_lo = mu_mid
        else:
            mu_hi = mu_mid
    w = capped_weights(mu_hi)
    # Enforce the constraints exactly (up to machine precision): clip below
    # eps and absorb the residual on the free (non-pinned) entries.
    for _ in range(3):
        numpy.maximum(w, eps, out=w)
        excess = w.sum() - d
        if excess <= 0.0:
            break
        free = w > eps
        if not numpy.any(free):
            break
        w[free] *= (w[free].sum() - excess) / w[free].sum()
    return w


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
    """
    d = X.shape[2]
    acc = numpy.zeros((d,))
    for k, p_k in enumerate(list_p_k):
        if weights[k] == 0.0:
            continue
        x_k = X[k][:ts_size(X[k])]
        bary_idx = numpy.array([t for (t, _) in p_k], dtype=numpy.intp)
        samp_idx = numpy.array([j for (_, j) in p_k], dtype=numpy.intp)
        diff = barycenter[bary_idx] - x_k[samp_idx]
        acc += weights[k] * (diff * diff).sum(axis=0)
    return acc / weights.sum()


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
    n = X.shape[0]
    cost = 0.
    list_p_k = []
    for i in range(n):
        path, dist_i = dtw_path(barycenter, X[i], **metric_params)
        cost += dist_i ** 2 * weights[i]
        list_p_k.append(path)
    cost /= weights.sum()
    return list_p_k, cost


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
    sum_w_x = numpy.zeros((barycenter_size, d))
    for k, (w_k, x_k) in enumerate(zip(list_w_k, X)):
        sum_w_x += w_k.dot(x_k[:ts_size(x_k)])
    barycenter = numpy.diag(1. / diag_sum_v_k).dot(sum_w_x)
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


def _dba_weighted_one_init(X_, barycenter, weights, init_channel_weights,
                           regularization, min_channel_weight, max_iter, tol,
                           metric_params, verbose):
    """Jointly optimize a DBA barycenter and its channel weights (one init).

    Alternating Majorize-Minimize scheme: at each iteration, (i) optimal
    warping paths are recomputed for the current barycenter and channel
    weights, (ii) the barycenter is updated with the classic MM update
    (which is invariant to positive channel weights), (iii) the channel
    weights are updated by exactly minimizing the fixed-path surrogate plus
    the entropy regularizer. Each of these three steps does not increase the
    true objective

        J(b, w) = weighted average of dtw_w(b, x_k)**2
                  + regularization * sum_c w_c log(w_c),

    hence the recorded objective values form a non-increasing sequence (up
    to a numerical acceptance tolerance of 1e-8 * max(1, |J_previous|)).

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
        (barycenter, channel_weights) pair.
    """
    barycenter_size = barycenter.shape[0]
    channel_weights = init_channel_weights.copy()
    if metric_params is None:
        metric_params = {}
    else:
        metric_params = dict(metric_params)

    def evaluate(bary, w):
        mp = dict(metric_params)
        mp["channel_weights"] = w
        list_p_k, data_cost = _mm_assignment(X_, bary, weights, mp)
        return list_p_k, data_cost + regularization * _entropy(w)

    list_p_k, objective = evaluate(barycenter, channel_weights)
    objective_history = [objective]
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
        new_list_p_k, new_objective = evaluate(new_barycenter,
                                               new_channel_weights)
        accept_tol = 1e-8 * max(1.0, abs(objective))
        if new_objective > objective + accept_tol:
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
        objective_history.append(new_objective)
        if verbose:
            print("[Weighted DBA] epoch %d, objective: %.5f"
                  % (it + 1, new_objective))
        if abs(objective - new_objective) < tol:
            objective = new_objective
            break
        objective = new_objective
    return barycenter, channel_weights, objective_history


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
        weights = _check_sample_weights_strict(weights, X_.shape[0])
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
