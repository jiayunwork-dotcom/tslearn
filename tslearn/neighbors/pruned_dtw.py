"""Exact k-nearest-neighbor search for DTW with conservative pruning.

This module implements the retrieval engine behind
``dtw_search='pruned'`` in :mod:`tslearn.neighbors`. The search is
*exact*: for every query it returns the k training rows with the
smallest true DTW distances, ties being broken by training row index,
exactly as if all pairwise DTW distances had been computed and stably
sorted. Pruning only skips computations whose outcome cannot influence
the result; it never returns approximate distances.

Design notes (see also the user guide section "Nearest neighbor
searches"):

* At fit time, the index stores the (trimmed) training time series and,
  for each of them, a per-channel summary (minimum and maximum over
  time). Memory is O(N * d) on top of the training data itself: no
  N x N structure is ever allocated.
* For each query, a lower bound on the DTW distance to every training
  series is computed from these summaries in O(N * d * L) vectorized
  operations (no dynamic programming). Candidates are then visited in
  increasing lower-bound order, and an exact DTW with a distance cutoff
  (see ``dtw(..., max_dist=...)``) is computed only for candidates whose
  lower bound does not exceed the current k-th best exact distance. As
  soon as a candidate's lower bound exceeds that k-th distance, every
  remaining candidate is certified to be worse and the scan stops.
* The lower bound is the maximum of two one-sided "envelope" bounds:
  the sum, over query points, of the squared weighted distance to the
  candidate's per-channel value range, and the symmetric sum over
  candidate points relative to the query's range. Both are valid for
  unconstrained DTW with non-negative channel weights, for any series
  lengths and dimensionality. Since constraining the warping paths
  (Sakoe-Chiba, Itakura) can only increase the optimal cost, the bound
  is also valid (though looser) under those constraints.
"""

import numbers
import time
from bisect import insort

import numpy
from joblib import Parallel, delayed

from tslearn.backend import instantiate_backend
from tslearn.metrics.dtw_variants import (
    GLOBAL_CONSTRAINT_CODE,
    _check_channel_weights,
    _dtw_cells_exact,
    itakura_mask,
    njit_sakoe_chiba_mask,
)

__author__ = "tslearn contributors"

# Keys of `metric_params` that the pruned search understands and forwards
# to the exact DTW computations.
PRUNED_METRIC_PARAM_KEYS = frozenset(
    {
        "global_constraint",
        "sakoe_chiba_radius",
        "itakura_max_slope",
        "channel_weights",
        "be",
    }
)

# Safety margins used when comparing a lower bound to the current k-th
# exact distance. Both quantities are float64 values for which the
# mathematical ordering (lower bound <= true distance) could be off by a
# few ulps due to floating-point summation; the margin only ever makes
# the search *more* conservative (a few extra exact computations), never
# approximate.
_LB_RTOL = 1e-12
_LB_ATOL = 1e-15


def _trim_and_validate(X, d_expected=None, name="X"):
    """Validate a time series dataset and trim trailing NaN padding.

    Parameters
    ----------
    X : array-like, shape=(n_ts, sz, d)
        Dataset of time series, possibly padded with trailing rows that
        are NaN on all channels (tslearn's variable-length
        representation).
    d_expected : int or None
        Expected number of channels. If provided and different from the
        actual number of channels, a ValueError is raised.
    name : str
        Name of the dataset used in error messages.

    Returns
    -------
    trimmed : list of numpy.ndarray
        List of views ``X[i, :length_i]`` (no copy is made).
    lengths : numpy.ndarray of shape=(n_ts,), dtype=int
        Effective length of each series.

    Raises
    ------
    ValueError
        If the dataset is not 3-dimensional, if the number of channels
        does not match `d_expected`, or if any series contains infinite
        values, NaN values outside of trailing all-NaN padding rows
        (internal missing values or rows with only some channels NaN),
        or has zero valid time steps.
    """
    X = numpy.asarray(X)
    if X.ndim != 3:
        raise ValueError(
            "%s must be a 3-dimensional array (n_ts, sz, d), got %d "
            "dimensions." % (name, X.ndim)
        )
    n, _, d = X.shape
    if d_expected is not None and d != d_expected:
        raise ValueError(
            "Number of features of the provided timeseries (last "
            "dimension) must match the one of the fitted data! "
            "(%d and %d)" % (d_expected, d)
        )
    if n == 0:
        raise ValueError("%s must contain at least one time series." % name)
    if numpy.isinf(X).any():
        raise ValueError(
            "%s contains infinite values, which are not supported by the "
            "pruned DTW search." % name
        )
    isnan = numpy.isnan(X)
    trimmed = []
    lengths = numpy.empty(n, dtype=numpy.intp)
    for i in range(n):
        row_any_nan = isnan[i].any(axis=1)
        if not row_any_nan.any():
            length = X.shape[1]
        else:
            row_all_nan = isnan[i].all(axis=1)
            if (row_any_nan & ~row_all_nan).any():
                raise ValueError(
                    "%s: time series at index %d contains NaN values on "
                    "some but not all channels of a time step, which is "
                    "not supported by the pruned DTW search." % (name, i)
                )
            first_pad = int(numpy.argmax(row_all_nan))
            if not row_all_nan[first_pad:].all():
                raise ValueError(
                    "%s: time series at index %d contains internal "
                    "missing values (NaN rows outside of the trailing "
                    "padding), which is not supported by the pruned DTW "
                    "search." % (name, i)
                )
            length = first_pad
        if length == 0:
            raise ValueError(
                "%s: time series at index %d is empty (only NaN padding "
                "or zero length), which is not supported by the pruned "
                "DTW search." % (name, i)
            )
        lengths[i] = length
        trimmed.append(X[i, :length])
    return trimmed, lengths


def _lb_query_side(q, weights, cand_min, cand_max):
    """Lower bound on DTW(q, c) for every candidate c, from query side.

    For each query time step and channel, accumulate the squared
    weighted distance to the candidate's per-channel value range
    [cand_min, cand_max]. Every query index appears in at least one cell
    of any warping path, and the cheapest possible alignment of a query
    point is at least its distance to the candidate's range, hence the
    sum lower-bounds the squared DTW (unconstrained, weights >= 0).

    Parameters
    ----------
    q : numpy.ndarray, shape=(Lq, d)
        Query time series.
    weights : numpy.ndarray, shape=(d,)
        Non-negative channel weights.
    cand_min, cand_max : numpy.ndarray, shape=(n, d)
        Per-channel minima and maxima of the candidates.

    Returns
    -------
    lb : numpy.ndarray, shape=(n,)
        Lower bounds on the DTW distances.
    """
    lb2 = numpy.zeros(cand_min.shape[0])
    for t in range(q.shape[0]):
        qt = q[t]
        excess = numpy.maximum(qt - cand_max, cand_min - qt)
        numpy.maximum(excess, 0.0, out=excess)
        lb2 += (excess * excess * weights).sum(axis=1)
    return numpy.sqrt(lb2)


def _lb_cand_side(q, weights, padded, lengths):
    """Lower bound on DTW(q, c) for every candidate c, from candidate side.

    Symmetric of `_lb_query_side`: for each candidate time step and
    channel, accumulate the squared weighted distance to the query's
    per-channel value range. Every candidate index appears in at least
    one cell of any warping path, hence the bound is valid for
    unconstrained DTW with non-negative weights.

    Parameters
    ----------
    q : numpy.ndarray, shape=(Lq, d)
        Query time series.
    weights : numpy.ndarray, shape=(d,)
        Non-negative channel weights.
    padded : numpy.ndarray, shape=(n, L, d)
        Candidates, padded with trailing NaN rows.
    lengths : numpy.ndarray, shape=(n,)
        Effective length of each candidate.

    Returns
    -------
    lb : numpy.ndarray, shape=(n,)
        Lower bounds on the DTW distances.
    """
    q_min = q.min(axis=0)
    q_max = q.max(axis=0)
    n, max_len, _ = padded.shape
    lb2 = numpy.zeros(n)
    for t in range(max_len):
        active = lengths > t
        if not active.any():
            break
        vals = padded[:, t, :]
        excess = numpy.maximum(vals - q_max, q_min - vals)
        numpy.maximum(excess, 0.0, out=excess)
        contrib = (excess * excess * weights).sum(axis=1)
        # Padded (NaN) positions produce NaN contributions that are
        # discarded here for series shorter than `t`.
        lb2 += numpy.where(active, contrib, 0.0)
    return numpy.sqrt(lb2)


def _constraint_mask(sz1, sz2, gc_code, sakoe_chiba_radius,
                     itakura_max_slope):
    """Constraint mask for one pair, with compute_mask's exact semantics.

    This is a lean re-implementation of
    ``tslearn.metrics.dtw_variants.compute_mask`` for the NumPy backend
    (identical masks, identical constraint inference, identical
    warnings), avoiding per-call backend dispatch in the search hot
    loop. Returns None when no constraint applies (fully unconstrained
    pair), which the exact-DTW kernels accept as "all cells allowed".
    """
    if (gc_code == 0 and sakoe_chiba_radius is not None
            and itakura_max_slope is not None):
        raise RuntimeWarning(
            "global_constraint is not set for DTW, but both "
            "sakoe_chiba_radius and itakura_max_slope are "
            "set, hence global_constraint cannot be inferred "
            "and no global constraint will be used."
        )
    if gc_code == 2 or (gc_code == 0 and sakoe_chiba_radius is not None):
        if sakoe_chiba_radius is None:
            sakoe_chiba_radius = 1
        return njit_sakoe_chiba_mask(sz1, sz2, radius=sakoe_chiba_radius)
    if gc_code == 1 or (gc_code == 0 and itakura_max_slope is not None):
        if itakura_max_slope is None:
            itakura_max_slope = 2.0
        return itakura_mask(sz1, sz2, max_slope=itakura_max_slope)
    return None


def _exact_dtw_one(q, candidate, weights, gc_code, sakoe_chiba_radius,
                   itakura_max_slope, max_dist):
    """Exact DTW between one query and one candidate, with cutoff.

    Returns
    -------
    dist : float
        Exact DTW distance if it does not exceed `max_dist`, else inf.
    n_cells : int
        Number of dynamic-programming cells computed.
    """
    mask = _constraint_mask(q.shape[0], candidate.shape[0], gc_code,
                            sakoe_chiba_radius, itakura_max_slope)
    return _dtw_cells_exact(q, candidate, mask, weights, max_dist)


def _query_one(q, k, exclude_index, data, lengths, mins, maxs, padded,
               weights, lb_weights, gc_code, sakoe_chiba_radius,
               itakura_max_slope):
    """Exact k-NN of one query against the index, with pruning.

    Returns (distances, indices, n_exact, n_cells, time_lb, time_exact)
    where the first two are arrays of shape (k,) sorted by
    (distance, training row index).
    """
    t_lb0 = time.perf_counter()
    lb = numpy.maximum(
        _lb_query_side(q, lb_weights, mins, maxs),
        _lb_cand_side(q, lb_weights, padded, lengths),
    )
    t_lb1 = time.perf_counter()

    # Visit candidates in increasing lower-bound order; stable sort so
    # that equal bounds are visited in row order (deterministic).
    order = numpy.argsort(lb, kind="stable")
    best = []  # sorted list of (distance, row), at most k entries
    kth = numpy.inf
    n_exact = 0
    n_cells = 0
    t_exact = 0.0
    for cand in order:
        if cand == exclude_index:
            # Self-query: skip the query's own row (and only that row;
            # duplicate series living at other rows are kept).
            continue
        if lb[cand] > kth + _LB_ATOL + _LB_RTOL * kth:
            # Every remaining candidate has a lower bound (hence a true
            # distance) strictly larger than the current k-th exact
            # distance: the top-k is final.
            break
        max_dist = kth if numpy.isfinite(kth) else None
        t0 = time.perf_counter()
        dist, cells = _exact_dtw_one(
            q, data[cand], weights, gc_code, sakoe_chiba_radius,
            itakura_max_slope, max_dist,
        )
        t_exact += time.perf_counter() - t0
        n_exact += 1
        n_cells += cells
        if dist > kth:
            # Can only happen with dist == inf (the cutoff semantics
            # guarantee an exact value whenever it is <= kth).
            continue
        insort(best, (dist, cand))
        if len(best) > k:
            best.pop()
        if len(best) == k:
            kth = best[-1][0]
    dist_out = numpy.array([b[0] for b in best])
    ind_out = numpy.array([b[1] for b in best], dtype=numpy.intp)
    return dist_out, ind_out, n_exact, n_cells, t_lb1 - t_lb0, t_exact


def normalize_pruned_metric_params(metric_params, d):
    """Validate metric_params for the pruned search and normalize them.

    A copy of the parameters is made; the caller's dictionary is never
    modified.

    Parameters
    ----------
    metric_params : dict or None
        Distance parameters as given to the estimators (may contain
        "global_constraint", "sakoe_chiba_radius", "itakura_max_slope",
        "channel_weights" and "be"; "n_jobs"/"verbose" entries are
        ignored).
    d : int
        Number of channels of the data (used to validate
        channel_weights).

    Returns
    -------
    params : dict
        Normalized parameters with keys "gc_code",
        "sakoe_chiba_radius", "itakura_max_slope" and "channel_weights"
        (a validated float64 copy, or None).
    """
    mp = {} if metric_params is None else dict(metric_params)
    mp.pop("n_jobs", None)
    mp.pop("verbose", None)
    unknown = set(mp) - PRUNED_METRIC_PARAM_KEYS
    if unknown:
        raise ValueError(
            "Unsupported metric_params for dtw_search='pruned': %s. "
            "Supported keys are: %s."
            % (sorted(unknown), sorted(PRUNED_METRIC_PARAM_KEYS))
        )
    be = mp.pop("be", None)
    if be is not None and not instantiate_backend(be).is_numpy:
        raise ValueError(
            "dtw_search='pruned' only supports the NumPy backend, "
            "got %r." % (be,)
        )
    global_constraint = mp.get("global_constraint", None)
    if global_constraint not in GLOBAL_CONSTRAINT_CODE:
        raise ValueError(
            "Invalid global_constraint %r for DTW. Valid values are "
            "'itakura' and 'sakoe_chiba'." % (global_constraint,)
        )
    weights = _check_channel_weights(
        mp.get("channel_weights", None), d, be="numpy"
    )
    return {
        "gc_code": GLOBAL_CONSTRAINT_CODE[global_constraint],
        "sakoe_chiba_radius": mp.get("sakoe_chiba_radius", None),
        "itakura_max_slope": mp.get("itakura_max_slope", None),
        "channel_weights": weights,
    }


class PrunedDtwIndex:
    """Exact DTW k-nearest-neighbor index with lower-bound pruning.

    The index stores the trimmed training series together with
    per-channel min/max summaries (O(N * d) memory on top of the data).
    All distance-related parameters (channel weights, path constraints)
    are provided at query time, so the index built at fit time stays
    valid when those parameters change; it only depends on the training
    data itself.

    Attributes
    ----------
    data_ : list of numpy.ndarray
        Trimmed training time series (views into the fitted data).
    lengths_ : numpy.ndarray of shape=(n_samples,)
        Effective length of each training series.
    min_, max_ : numpy.ndarray of shape=(n_samples, d)
        Per-channel minima and maxima of each training series.
    n_samples_ : int
        Number of training series.
    d_ : int
        Number of channels.
    """

    def fit(self, X):
        """Build the index from a time series dataset.

        Parameters
        ----------
        X : array-like, shape=(n_ts, sz, d)
            Training time series, possibly padded with trailing all-NaN
            rows. The array is referenced, not copied, and never
            modified.
        """
        X = numpy.asarray(X)
        trimmed, lengths = _trim_and_validate(X, name="training data")
        # Per-channel summaries. Padding rows are NaN and valid regions
        # were validated to be NaN-free, so NaN-aware reductions over
        # whole series give exactly the per-channel min/max of the
        # trimmed series (every series has at least one valid row).
        with numpy.errstate(all="ignore"):
            mins = numpy.nanmin(X, axis=1)
            maxs = numpy.nanmax(X, axis=1)
        self.data_ = trimmed
        self.lengths_ = lengths
        self.min_ = numpy.asarray(mins, dtype=numpy.float64)
        self.max_ = numpy.asarray(maxs, dtype=numpy.float64)
        self.n_samples_ = len(trimmed)
        self.d_ = X.shape[2]
        self._padded = X
        return self

    def kneighbors(self, Q, k, exclude_self=False, metric_params=None,
                   n_jobs=None, verbose=0):
        """Exact k nearest training rows for each query.

        Parameters
        ----------
        Q : array-like, shape=(n_queries, sz, d) or None
            Query time series. If None, the training set itself is
            queried and `exclude_self` must be True.
        k : int
            Number of neighbors. Must not exceed the number of available
            candidates (n_samples, or n_samples - 1 for self-queries).
        exclude_self : bool
            If True, query i ignores the training row with the same
            index i (other rows with identical values are kept).
        metric_params : dict or None
            Distance parameters (channel_weights, global_constraint,
            sakoe_chiba_radius, itakura_max_slope).
        n_jobs : int or None
            Number of parallel jobs (over queries). Does not change the
            result.
        verbose : int
            Verbosity level passed to joblib.

        Returns
        -------
        dist : numpy.ndarray, shape=(n_queries, k)
            Exact DTW distances, sorted by (distance, row index).
        ind : numpy.ndarray, shape=(n_queries, k)
            Training row indices of the neighbors.
        stats : dict
            Work counters (number of lower-bound evaluations, exact DTW
            computations, visited dynamic-programming cells, timings).
        """
        params = normalize_pruned_metric_params(metric_params, self.d_)
        if Q is None:
            queries = self.data_
        else:
            queries, _ = _trim_and_validate(Q, d_expected=self.d_,
                                            name="query data")
        n_queries = len(queries)
        n_avail = self.n_samples_ - 1 if exclude_self else self.n_samples_
        if k > n_avail:
            raise ValueError(
                "Cannot select k=%d neighbors among %d available "
                "training samples." % (k, n_avail)
            )
        if k == 0 or n_queries == 0:
            stats = {
                "n_queries": n_queries, "k": k,
                "n_candidates": 0, "n_lb": 0, "n_exact": 0,
                "n_pruned": 0, "n_dp_cells": 0,
                "time_lb": 0.0, "time_exact": 0.0,
            }
            return (numpy.zeros((n_queries, 0)),
                    numpy.zeros((n_queries, 0), dtype=numpy.intp), stats)

        weights = params["channel_weights"]
        lb_weights = numpy.ones(self.d_) if weights is None else weights
        results = Parallel(n_jobs=n_jobs, prefer="threads", verbose=verbose)(
            delayed(_query_one)(
                queries[qi], k, qi if exclude_self else None,
                self.data_, self.lengths_, self.min_, self.max_,
                self._padded, weights, lb_weights, params["gc_code"],
                params["sakoe_chiba_radius"], params["itakura_max_slope"],
            )
            for qi in range(n_queries)
        )
        dist = numpy.empty((n_queries, k))
        ind = numpy.empty((n_queries, k), dtype=numpy.intp)
        n_exact = 0
        n_cells = 0
        time_lb = 0.0
        time_exact = 0.0
        for qi, (d_row, i_row, ne, nc, t_lb, t_ex) in enumerate(results):
            dist[qi] = d_row
            ind[qi] = i_row
            n_exact += ne
            n_cells += nc
            time_lb += t_lb
            time_exact += t_ex
        n_candidates = n_queries * n_avail
        stats = {
            "n_queries": n_queries,
            "k": k,
            "n_candidates": n_candidates,
            "n_lb": n_candidates,
            "n_exact": n_exact,
            "n_pruned": n_candidates - n_exact,
            "n_dp_cells": n_cells,
            "time_lb": time_lb,
            "time_exact": time_exact,
        }
        return dist, ind, stats


def check_pruned_k(k):
    """Validate a requested number of neighbors for the pruned entries.

    Follows scikit-learn's conventions: non-integer values raise a
    TypeError, values below 1 raise a ValueError. Truncation (or
    rejection) of values larger than the number of available samples is
    left to the caller.
    """
    if not isinstance(k, numbers.Integral):
        raise TypeError(
            "n_neighbors must be an integer, got %r." % (k,)
        )
    if k < 1:
        raise ValueError(
            "n_neighbors must be a positive integer, got %d." % k
        )
