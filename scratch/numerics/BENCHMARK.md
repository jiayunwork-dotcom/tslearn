# Overflow hardening: measurement record

All measurements use the common 80-series / length-24 / 3-channel dataset
(`bench_common.py`), single-threaded execution
(`OMP/OPENBLAS/MKL/NUMBA_NUM_THREADS=1`), JIT warmup performed in a
throwaway fit and excluded from the timed section. Times are the best of
three full fits (`n_init=2, max_iter=20`), so scheduling noise is reduced.

## Wall time (seconds, best of 3)

| mode     | pre-fix | post-fix | change |
|----------|--------:|---------:|-------:|
| adaptive (`channel_weight_regularization=0.05`) | 3.856 | 1.687 | -56% |
| standard (unweighted DTW k-means)              | 2.148 | 2.024 |  -6% |

The post-fix standard path is marginally faster despite the additional
dispatch checks because the hot DBA loop now calls a lean per-pair kernel
directly instead of going through the full public `dtw_path` entry point
(no redundant `to_time_series` conversions on plain float64 arrays). The
adaptive path gains more because its per-pair objective evaluation no
longer materializes and re-squares Euclidean distances.

## Peak Python heap (tracemalloc, MiB)

| mode     | pre-fix | post-fix |
|----------|--------:|---------:|
| adaptive | 0.7     | 0.6 |
| standard | 0.9     | 0.9 |

## Alignment call volume

Both implementations converge in the same number of iterations
(adaptive 13, standard 6) to the identical inertia
(15.106545 / 43.749263) and, in adaptive mode, the identical objective
15.600916 and channel weights, so the optimization trajectory is
unchanged. Post-fix each alignment is one kernel invocation that returns
path plus squared cost together; the pre-fix code issued separate
`dtw_path` and `cdist_dtw` calls for the same pairs.

## When the robust (scaled) path is paid

`needs_scaled_kernel` only returns True when an actual square or weighted
product can reach the float64 overflow boundary (input magnitude above
~6.5e153, or a correspondingly large fixed channel weight, or a zero
weight coexisting with boundary-magnitude values). On the benchmark data
(all standard-normal values) every call takes the classic numba kernel,
which is why the default unweighted path keeps its v0.6.4 behavior.
