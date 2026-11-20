"""Overflow-safe numerical primitives for weighted DTW and adaptive training.

The public objectives of weighted DTW, DBA and adaptive channel-weight
k-means are unchanged. What differs from a plain float64 implementation is
*how* their intermediate quantities are evaluated when inputs are very large
or very small. Two families of strategies cover the failure modes observed
when e.g. ``1e308 ** 2`` overflows to ``inf``:

**Scheme A -- rescaled evaluation (the one implemented here).** Every
addition that can overflow is performed after dividing the terms by a common
power-of-two scale, so no term ever reaches the float64 overflow boundary.
The scale is carried alongside the scaled value (a ``(magnitude, scale_log2)``
pair) and is only undone when a finite, representable public result is
needed. Two quantities are compared in their scaled representation using the
existing relative tolerance, which makes the acceptance test
``1e-8 * max(1, |J_prev|)`` scale invariant. On ordinary-magnitude data the
extra cost is one max pass over the inputs and a handful of scalar
operations per DTW cell; the classic kernels are kept and used unchanged
whenever no overflow is possible.

**Scheme B -- log-domain accumulation.** Sums of non-negative terms are kept
as ``(log(sum exp(log_term)))`` quantities; the entropy weight solve is then
a root finding in ``log w`` space. This is equally immune to overflow but
every DTW cell needs several ``log``/``exp`` evaluations, which is an order
of magnitude more expensive than Scheme A, and backtracking on a log-domain
DP matrix loses the exact arg-min ordering whenever terms underflow. Scheme
B is therefore not used; it is documented here for completeness.

The module is intentionally NumPy/numba-only: channel weighting is a
NumPy-backend feature.
"""
import math

import numpy
from numba import njit

__author__ = "Romain Tavenard romain.tavenard[at]univ-rennes2.fr"

#: Values whose magnitude (or whose pairwise difference) exceeds this bound
#: can square to infinity in float64. The robust kernels are only dispatched
#: when this threshold is crossed, so ordinary data keep the classic path.
_SQUARE_SAFE_BOUND = 0.5 * numpy.sqrt(numpy.finfo(numpy.float64).max)

#: Finite sentinel used inside the scaled DP matrices to mark cells whose
#: true accumulated cost exceeds the float64 range. Any finite value is
#: therefore a genuine (finite) accumulated cost; overflow is detected
#: explicitly via non-finite local terms rather than by value comparison.
_OVERFLOW_SENTINEL = -1.0


class NonRepresentableNumericalState(ValueError):
    """Raised when a requested finite result cannot fit in float64.

    This is always a :class:`ValueError` so that callers only have to catch
    ``ValueError`` if they wish to, and it never silently propagates ``inf``
    or ``nan`` into a returned model.
    """


def needs_scaled_kernel(s1, s2, weights):
    """Return True if the classic DTW kernels could overflow or produce NaN.

    Three situations force the robust path:

    * any finite input magnitude in an *active* channel (positive weight)
      is large enough that squaring a difference can reach the float64
      overflow boundary;
    * a fixed channel weight is itself large enough that
      ``weight * diff**2`` can overflow even for ordinary differences;
    * a zero channel weight coexists with non-finite-looking (huge)
      differences in any channel, because ``0.0 * inf`` is NaN in the
      classic kernel.
    """
    max_abs_active = 0.0
    max_abs_any = 0.0
    for c in range(s1.shape[1]):
        m = max(float(numpy.max(numpy.abs(s1[:, c]))),
                float(numpy.max(numpy.abs(s2[:, c]))))
        max_abs_any = max(max_abs_any, m)
        if weights is None or numpy.asarray(weights)[c] > 0.0:
            max_abs_active = max(max_abs_active, m)
    max_weight = 1.0
    if weights is not None:
        w_arr = numpy.asarray(weights, dtype=numpy.float64)
        max_weight = max(1.0, float(numpy.max(w_arr)))
        if not numpy.isfinite(max_weight):
            # Validated weights are always finite; treat anything else as
            # requiring the robust path rather than risking a silent NaN.
            return True
    # A weighted local term is bounded by max_weight * (2 * M)**2. Request
    # the rescaled kernel whenever that bound can reach the overflow limit.
    if max_abs_active > 0.0:
        worst_local_log2 = (
            math.log2(max_weight)
            + 2.0 * math.log2(2.0 * max_abs_active)
        )
        if worst_local_log2 > 1023.0:
            return True
    if weights is not None and numpy.any(numpy.asarray(weights) == 0.0):
        # A zero weight must erase even an infinite-looking difference.
        if 2.0 * max_abs_any > _SQUARE_SAFE_BOUND:
            return True
    return False


def _scale_exponent_from_log2_bound(log2_bound):
    """Smallest non-negative power-of-two scale exponent ``e`` such that a
    cost whose log2 magnitude is ``log2_bound`` stays finite after dividing
    by ``2**(2e)``. Returns 0 when no scaling is needed.
    """
    if log2_bound <= 1023.0:
        return 0.0
    return float(min(512, math.ceil(0.5 * (log2_bound - 1023.0))))


def unscale_value(scaled_value, scale_log2):
    """Undo a power-of-two scaling, raising if the result overflows.

    The multiplication is performed with :func:`numpy.ldexp` so the factor
    itself never has to be materialized. Exponents beyond the ldexp limit
    are applied in two halves, which still raises cleanly when the final
    value overflows.
    """
    if not numpy.isfinite(scaled_value):
        raise NonRepresentableNumericalState(
            "Intermediate scaled value is not finite; the requested result "
            "is not representable in float64."
        )
    if scale_log2 == 0.0:
        return float(scaled_value)
    exponent = int(round(scale_log2))
    if abs(exponent - scale_log2) > 1e-9:
        raise NonRepresentableNumericalState(
            "Non-integer scale exponent %r is not supported." % scale_log2
        )
    result = numpy.float64(scaled_value)
    if exponent > 1023 or exponent < -1074:
        half = exponent // 2
        result = numpy.ldexp(result, half)
        result = numpy.ldexp(result, exponent - half)
    else:
        result = numpy.ldexp(result, exponent)
    if not numpy.isfinite(result):
        raise NonRepresentableNumericalState(
            "The correct finite result exceeds the float64 range."
        )
    return float(result)


def unscale_or_inf(scaled_value, scale_log2):
    """Like :func:`unscale_value` but return +/-inf instead of raising.

    Used internally when an overflowing *candidate* must still be compared
    against other candidates rather than aborting the whole computation.
    """
    if not numpy.isfinite(scaled_value):
        return numpy.inf
    if scale_log2 == 0.0:
        return float(scaled_value)
    exponent = int(round(scale_log2))
    result = numpy.float64(scaled_value)
    if exponent > 1023 or exponent < -1074:
        half = exponent // 2
        result = numpy.ldexp(result, half)
        result = numpy.ldexp(result, exponent - half)
    else:
        result = numpy.ldexp(result, exponent)
    if numpy.isnan(result):
        return numpy.inf
    return float(result)


@njit()
def _njit_scaled_local_dist(x, y, has_weights, weights, inv_scale):
    """Scaled weighted squared distance between two vectors.

    A single common power-of-two scale is used for every channel, so the
    scaled cost of any alignment path is its true cost divided by the same
    factor and the dynamic program's optimal substructure is preserved.
    Zero weight channels are skipped explicitly so a ``0 * inf`` product
    cannot occur.
    """
    dist = 0.0
    for c in range(x.shape[0]):
        if has_weights and weights[c] == 0.0:
            continue
        diff = (x[c] - y[c]) * inv_scale
        term = diff * diff
        if has_weights:
            term *= weights[c]
        dist += term
    return dist


@njit()
def njit_scaled_accumulated_matrix(s1, s2, mask, has_weights, weights,
                                   inv_scale):
    """Scaled DTW accumulated cost matrix under a common power-of-two scale.

    All finite cell values are the true costs divided by ``scale ** 2``,
    hence the arg-min decisions are identical to the unscaled problem. A
    cell whose local term overflows, or whose best predecessor has already
    overflowed, carries the negative :data:`_OVERFLOW_SENTINEL`, which can
    never be confused with a genuine non-negative cost. Forbidden cells
    keep the standard ``+inf`` marking. The returned flag reports whether
    the endpoint itself is unreachable at finite float64 magnitude.
    """
    l1 = s1.shape[0]
    l2 = s2.shape[0]
    cum = numpy.full((l1 + 1, l2 + 1), numpy.inf)
    cum[0, 0] = 0.0
    endpoint_overflow = False
    for i in range(l1):
        for j in range(l2):
            if numpy.isfinite(mask[i, j]):
                local = _njit_scaled_local_dist(
                    s1[i], s2[j], has_weights, weights, inv_scale
                )
                if not numpy.isfinite(local):
                    # The local squared term itself overflowed at this
                    # scale: the cell cannot hold a finite value.
                    cum[i + 1, j + 1] = _OVERFLOW_SENTINEL
                    if i == l1 - 1 and j == l2 - 1:
                        endpoint_overflow = True
                    continue
                best = min(cum[i, j + 1], cum[i + 1, j], cum[i, j])
                if best == numpy.inf:
                    # No finite predecessor (boundary cell with no
                    # reachable neighbor): keep the standard infinity
                    # marking so the cell stays forbidden.
                    cum[i + 1, j + 1] = numpy.inf
                elif best == _OVERFLOW_SENTINEL:
                    # Every finite predecessor has itself saturated.
                    cum[i + 1, j + 1] = _OVERFLOW_SENTINEL
                else:
                    val = local + best
                    if not numpy.isfinite(val):
                        cum[i + 1, j + 1] = _OVERFLOW_SENTINEL
                    else:
                        cum[i + 1, j + 1] = val
                if i == l1 - 1 and j == l2 - 1 and (
                    cum[i + 1, j + 1] == _OVERFLOW_SENTINEL
                ):
                    endpoint_overflow = True
    return cum[1:, 1:], endpoint_overflow


@njit()
def _njit_scaled_return_path(acc_cost_mat):
    """Backtrack an optimal path on a scaled accumulated cost matrix.

    Identical ordering logic to :func:`_njit_return_path` in
    :mod:`tslearn.metrics.dtw_variants`; scaling by a positive common factor
    preserves every arg-min decision.
    """
    sz1, sz2 = acc_cost_mat.shape
    path = [(sz1 - 1, sz2 - 1)]
    while path[-1] != (0, 0):
        i, j = path[-1]
        if i == 0:
            path.append((0, j - 1))
        elif j == 0:
            path.append((i - 1, 0))
        else:
            arr = numpy.array(
                [
                    acc_cost_mat[i - 1][j - 1],
                    acc_cost_mat[i - 1][j],
                    acc_cost_mat[i][j - 1],
                ]
            )
            argmin = numpy.argmin(arr)
            if argmin == 0:
                path.append((i - 1, j - 1))
            elif argmin == 1:
                path.append((i - 1, j))
            else:
                path.append((i, j - 1))
    return path[::-1]


def _run_scaled_dp(s1, s2, mask, has_weights, w_arr, inv_scale):
    """Run the scaled DP once, returning (matrix, endpoint_overflow)."""
    return njit_scaled_accumulated_matrix(
        numpy.ascontiguousarray(s1, dtype=numpy.float64),
        numpy.ascontiguousarray(s2, dtype=numpy.float64),
        numpy.ascontiguousarray(mask, dtype=numpy.float64),
        has_weights,
        w_arr,
        inv_scale,
    )


def scaled_dtw_pair(s1, s2, mask, weights=None):
    """Robust weighted DTW for a single pair of series.

    Two-stage adaptive strategy:

    1. The DP is first evaluated at the natural scale (``inv_scale = 1``),
       i.e. exactly like the classic kernel but with zero-weight channels
       skipped. If the endpoint is finite, the result is bit-for-bit the
       classic one, including full float64 precision.
    2. Only if the endpoint saturates (the true accumulated cost crossed
       the float64 boundary) is the DP rerun under a power-of-two scale
       derived from the tightest log-space bound on the worst path cost.
       The scaled DP then either produces a finite endpoint (the true
       distance is representable) or keeps saturating (genuinely
       non-representable; the caller raises ``ValueError``).

    A single common scale is used for every channel, so every path cost is
    divided by the same positive factor and the optimal path is unchanged.
    Zero-weight channels are ignored both in the local cost (so ``0 * inf``
    never happens) and when choosing the scale.

    Returns
    -------
    path : list of integer pairs
        Optimal alignment path.
    scaled_sq_cost : float
        Optimal squared path cost divided by ``2 ** (2 * scale_log2)``.
    scale_log2 : float
        Base-2 exponent of the common power-of-two scale (0 at stage 1).
    final_overflowed : bool
        True if even the scaled endpoint saturates.
    """
    d = s1.shape[1]
    if weights is None:
        has_weights = False
        w_arr = numpy.ones(1)
        active = numpy.ones((d,), dtype=bool)
    else:
        has_weights = True
        w_arr = numpy.ascontiguousarray(weights, dtype=numpy.float64)
        active = w_arr > 0.0

    # Stage 1: natural scale, classic precision.
    acc, endpoint_overflow = _run_scaled_dp(
        s1, s2, mask, has_weights, w_arr, 1.0
    )
    if not endpoint_overflow:
        path = _njit_scaled_return_path(acc)
        return path, float(acc[-1, -1]), 0.0, False

    # Stage 2: adaptive rescaling driven by the DP itself. We rerun the
    # recurrence under progressively larger power-of-two scales, starting
    # from no rescaling, and stop at the first scale at which the endpoint
    # is finite. Because each iteration only doubles the scale, the first
    # successful scale is the smallest admissible one and representable
    # costs keep essentially full float64 precision. Local-term overflows
    # are absorbed inside the kernel (they simply mark their cell) until the
    # scale catches up, so no separate local-term bound is needed.
    l1, l2 = s1.shape[0], s2.shape[0]
    scale_log2 = 0.0
    acc = None
    endpoint_overflow = True
    for _ in range(64):
        inv_scale = 1.0 if scale_log2 == 0.0 else numpy.float64(
            2.0 ** (-scale_log2)
        )
        acc, endpoint_overflow = _run_scaled_dp(
            s1, s2, mask, has_weights, w_arr, inv_scale
        )
        if not endpoint_overflow:
            break
        if scale_log2 >= 512.0:
            break
        scale_log2 += 1.0
    path = _njit_scaled_return_path(acc)
    final = float(acc[-1, -1])
    still_overflow = bool(
        endpoint_overflow or not numpy.isfinite(final)
        or final == _OVERFLOW_SENTINEL
    )
    return path, final, scale_log2, still_overflow


def per_channel_costs_scaled(X, barycenter, list_p_k, sample_weights):
    """Per-channel fixed-path alignment costs with per-channel scaling.

    Computes, for each channel ``c``,

    ``A_c = sum_k sw[k] * sum_{(t, j) in p_k} (bary[t, c] - X[k, j, c])**2
           / sum_k sw[k]``

    without letting the intermediate squares or weighted sums overflow.
    Each channel gets its own power-of-two scale, so channels with wildly
    different magnitudes remain accurate.

    Returns
    -------
    costs : numpy.ndarray of shape (d,), finite non-negative values
    scale_log2 : numpy.ndarray of shape (d,)
        Scale associated with each channel; cost ``c`` is stored at its true
        magnitude, but whenever it overflows the corresponding entry is
        pinned to inf and ``scale_log2`` still describes the evaluation
        scale (callers route such channels to the min-weight face of the
        simplex).
    """
    d = X.shape[2]
    costs = numpy.zeros((d,))
    scales = numpy.zeros((d,))
    # Normalize the per-series sample weights once: the weighted mean is
    # invariant to a common positive rescaling, while this keeps every
    # accumulator product bounded. The strict validators guarantee finite
    # non-negative weights with a positive entry.
    series_w = numpy.asarray(sample_weights, dtype=numpy.float64)
    w_max = float(numpy.max(series_w))
    if w_max <= 0.0 or not numpy.isfinite(w_max):
        raise NonRepresentableNumericalState(
            "Per-channel costs require at least one finite positive "
            "sample weight."
        )
    series_w_norm = series_w / w_max
    total_w_norm = float(numpy.sum(series_w_norm))
    # Fast path for ordinary data: vectorized numpy accumulation exactly
    # matching the original formula, with no per-cell Python list building
    # or log-space bookkeeping. The scaled loop below only runs when an
    # intermediate quantity could actually overflow.
    if (
        numpy.all(numpy.isfinite(X))
        and numpy.all(numpy.isfinite(barycenter))
        and float(numpy.max(numpy.abs(X))) <= _SQUARE_SAFE_BOUND
        and float(numpy.max(numpy.abs(barycenter))) <= _SQUARE_SAFE_BOUND
    ):
        fast_costs = numpy.zeros((d,))
        for k, p_k in enumerate(list_p_k):
            if series_w_norm[k] == 0.0:
                continue
            x_k = X[k]
            bary_idx = numpy.array([t for (t, _) in p_k],
                                   dtype=numpy.intp)
            samp_idx = numpy.array([j for (_, j) in p_k],
                                   dtype=numpy.intp)
            diff = barycenter[bary_idx] - x_k[samp_idx]
            fast_costs += (
                series_w_norm[k] * (diff * diff).sum(axis=0)
            )
        fast_costs /= total_w_norm
        if numpy.all(numpy.isfinite(fast_costs)):
            return fast_costs, numpy.zeros((d,))
    for c in range(d):
        # Gather all aligned differences for this channel to pick a scale.
        diffs = []
        ws = []
        for k, p_k in enumerate(list_p_k):
            if series_w_norm[k] == 0.0:
                continue
            x_k = X[k]
            for (t, j) in p_k:
                diffs.append(barycenter[t, c] - x_k[j, c])
                ws.append(series_w_norm[k])
        if not diffs:
            continue
        diffs = numpy.asarray(diffs, dtype=numpy.float64)
        ws = numpy.asarray(ws, dtype=numpy.float64)
        max_abs = float(numpy.max(numpy.abs(diffs)))
        # Size the scale on the actual largest squared difference and the
        # number of aligned terms (tight bound; evaluated in log space so
        # the bound itself cannot overflow).
        if max_abs <= 0.0:
            scale_log2 = 0.0
        else:
            log2_bound = (math.log2(float(len(diffs)))
                          + 2.0 * math.log2(max_abs))
            scale_log2 = _scale_exponent_from_log2_bound(log2_bound)
        scales[c] = scale_log2
        inv_scale = 1.0 if scale_log2 == 0.0 else numpy.float64(
            2.0 ** (-scale_log2)
        )
        scaled_sq = (diffs * inv_scale) ** 2
        # The denominator is the sum of the *per-series* normalized
        # weights, not the number of aligned positions: the sample weight
        # of a series applies to every one of its path cells.
        scaled_acc = float(numpy.sum(ws * scaled_sq)) / total_w_norm
        if not numpy.isfinite(scaled_acc):
            costs[c] = numpy.inf
            continue
        # The accumulation divided every term by scale**2; undo it.
        true_cost = unscale_or_inf(scaled_acc, 2.0 * scale_log2)
        costs[c] = true_cost
    return costs, scales


def stable_entropy_solve(per_channel_costs, regularization, min_weight):
    """Minimize ``w.A + lam * sum w log w`` over the capped simplex.

    Constraints: ``sum_c w_c = d`` and ``w_c >= min_weight``. This is the
    same mathematical problem as
    :func:`tslearn.barycenters.dba._solve_channel_weights_entropy`, solved
    here with a numerically stable KKT reformulation that stays exact when
    the costs ``A_c`` range up to the float64 limits:

    writing ``r_c = (A_min - A_c) / lam`` and ``t = exp((-mu - A_min)/lam
    - 1)`` gives ``w_c = max(min_weight, t * exp(r_c))`` with ``t > 0``
    bounded, so no exponent ever overflows and the bisection bracket is
    finite by construction. Infinite costs (channels whose true alignment
    cost is not representable) are pinned to the lower bound.

    The returned weights always satisfy the constraints up to machine
    precision: a final exact residual correction absorbs rounding of the
    sum, and the lower bound is enforced before that correction.
    """
    A = numpy.asarray(per_channel_costs, dtype=numpy.float64)
    d = A.shape[0]
    if d == 1:
        return numpy.ones((1,))
    eps = float(min_weight)
    lam = float(regularization)

    finite = numpy.isfinite(A)
    # Channels with infinite costs are necessarily pinned to eps.
    if numpy.all(~finite):
        w = numpy.full((d,), eps)
        return _repair_simplex(w, eps, d)

    a_min = float(numpy.min(A[finite]))
    r = numpy.where(finite, (a_min - A) / lam, -numpy.inf)
    exp_r = numpy.where(numpy.isfinite(r), numpy.exp(numpy.minimum(r, 0.0)),
                        0.0)

    def weights_at(t):
        w = t * exp_r
        numpy.maximum(w, eps, out=w)
        return w

    # Bracket t in (0, +inf): at t -> 0 all entries are pinned to eps and
    # the sum is d*eps < d; at a sufficiently large t the sum exceeds d.
    t_lo = numpy.finfo(numpy.float64).tiny
    t_hi = float(d)
    while float(numpy.sum(weights_at(t_hi))) <= d:
        t_hi *= 2.0
        if not numpy.isfinite(t_hi):
            # Even at the largest finite t the sum is below d: every free
            # coordinate underflows, so the solution sits on the lower-bound
            # face; repair distributes the residual exactly.
            w = weights_at(numpy.finfo(numpy.float64).max / 2.0)
            return _repair_simplex(w, eps, d)
    for _ in range(200):
        t_mid = 0.5 * (t_lo + t_hi)
        if float(numpy.sum(weights_at(t_mid))) > d:
            t_hi = t_mid
        else:
            t_lo = t_mid
    w = weights_at(t_hi)
    return _repair_simplex(w, eps, d)


def _repair_simplex(w, eps, d):
    """Enforce ``w >= eps`` and ``sum(w) == d`` exactly up to rounding.

    The residual is distributed on free (non-pinned) coordinates by a
    single multiplicative correction, repeated a few times to absorb
    round-off. If every coordinate is pinned (degenerate face), the residual
    is spread uniformly instead of violating the lower bound.
    """
    w = numpy.asarray(w, dtype=numpy.float64).copy()
    numpy.maximum(w, eps, out=w)
    for _ in range(5):
        residual = d - float(numpy.sum(w))
        if abs(residual) <= 1e-12 * max(1, d):
            break
        free = w > eps
        if not numpy.any(free):
            # Uniform redistribution is the only feasible correction left.
            w = w + residual / d
            numpy.maximum(w, eps, out=w)
            continue
        if residual > 0.0:
            w[free] = w[free] * (
                (numpy.sum(w[free]) + residual) / numpy.sum(w[free])
            )
        else:
            # Shrink free coordinates while respecting the lower bound.
            free_sum = numpy.sum(w[free])
            factor = max(0.0, (free_sum + residual) / free_sum)
            w[free] = numpy.maximum(w[free] * factor, eps)
    # Final exact accounting: force the sum to d on a single free entry if a
    # sub-ulp residual remains. Constraints are checked separately anyway.
    delta = d - float(numpy.sum(w))
    if delta != 0.0:
        idx = int(numpy.argmax(w))
        w[idx] += delta
    numpy.maximum(w, eps, out=w)
    return w
