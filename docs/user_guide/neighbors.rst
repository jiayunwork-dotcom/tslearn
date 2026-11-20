.. _neighbors:

==========================
Nearest neighbor searches
==========================

The :mod:`tslearn.neighbors` module provides k-nearest-neighbor
estimators for time series:

* :class:`tslearn.neighbors.KNeighborsTimeSeries` (unsupervised
  retrieval),
* :class:`tslearn.neighbors.KNeighborsTimeSeriesClassifier`,
* :class:`tslearn.neighbors.KNeighborsTimeSeriesRegressor`.

All three accept a ``dtw_search`` parameter selecting how DTW neighbors
are computed:

* ``dtw_search='brute'`` (default): the historical behavior. Every
  query computes the full Q x N cross-distance matrix against the
  training set (through :func:`tslearn.metrics.cdist_dtw`) and selects
  the k smallest entries.
* ``dtw_search='pruned'``: an *exact* search that avoids most of those
  computations. It is only valid with the string ``metric='dtw'`` (any
  other metric, including callables, is rejected with a ``ValueError``
  at fit time) and currently supports the NumPy backend only.

The pruned search returns exactly the same neighbors, distances,
probabilities and predictions as the brute-force path: neighbors are
the k training rows with the smallest true DTW distances, ties being
broken by training row index. Pruning is *conservative*: it only skips
computations whose outcome cannot influence the result, and never
returns approximate distances.

Exactness with less work
------------------------

At fit time, the pruned index stores the training series (trimmed of
their trailing NaN padding) together with per-channel summaries
(minimum and maximum over time) of each series. That is O(N * d) extra
memory on top of the data itself — no N x N matrix is ever allocated.

For each query, the search then proceeds as follows:

1. **Lower bounds for all candidates.** A lower bound on the DTW
   distance between the query and every training series is computed
   from the summaries, in vectorized O(N * d * L) operations and O(N)
   memory — no dynamic programming and no Q x N distance table. The
   bound is the maximum of two one-sided envelope bounds: the sum, over
   query points, of the squared weighted distance to each candidate's
   per-channel value range, and the symmetric sum over candidate points
   relative to the query's range. Both are valid for unconstrained DTW
   with non-negative ``channel_weights``, for any series lengths and
   dimensionality: every query (resp. candidate) index appears in at
   least one cell of any warping path, and the cheapest alignment of a
   point is at least its distance to the other series' value range.
   Since constraining the warping paths (Sakoe-Chiba band, Itakura
   parallelogram) can only *increase* the optimal cost, the bound
   remains valid — though looser — under those constraints.
2. **Best-first exact evaluation.** Candidates are visited in
   increasing lower-bound order. An exact DTW with a distance cutoff
   (``dtw(..., max_dist=...)``, see :ref:`dtw-max-dist`) is computed
   only for candidates whose lower bound does not exceed the current
   k-th best exact distance: the cutoff lets the dynamic program stop
   as soon as the candidate is certified to be worse. As soon as a
   candidate's lower bound exceeds the k-th exact distance, every
   remaining candidate is certified to be worse and the scan stops.
   (A small floating-point safety margin is used when comparing lower
   bounds to the cutoff, so ties at the k-th distance are always
   resolved with exact distances and row indices.)

Only O(k) distances per query are kept, and there is no Q x N distance
matrix at any point. The remaining per-pair workspace is the dynamic
programming buffer of a single DTW computation — O(L_candidate) for
the two accumulated-cost rows, plus O(L_query x L_candidate) for the
constraint mask when a Sakoe-Chiba or Itakura constraint is active
(no mask is allocated for unconstrained pairs). Queries are independent
and are dispatched over ``n_jobs`` threads; ``n_jobs`` changes neither
the results nor the memory profile. After each pruned search, the
estimator's ``search_stats_`` attribute reports the number of
lower-bound evaluations, exact DTW computations and visited
dynamic-programming cells.

Design alternatives considered
------------------------------

Two other schemes were evaluated before settling on the cascade above:

* **LB_Keogh envelope index.** Storing per-series envelopes
  (upper/lower bounds over a window) gives a tighter lower bound, but
  only when query and candidate have equal lengths *and* the
  Sakoe-Chiba radius used for the envelopes matches the constraint of
  the query. For unconstrained DTW the envelope degenerates to the
  global min/max (a weaker version of our bound), and for
  variable-length data — a core tslearn use case — the bound is not
  applicable at all, forcing a per-pair exact fallback. Storage is
  O(N * L * d), i.e. a full copy of the dataset.
* **Early-abandon only (no index).** Computing exact DTW with a
  distance cutoff for every candidate, in arbitrary order, needs no
  summaries and no extra memory, but still evaluates every pair: the
  cutoff only shortens each individual dynamic program. It cannot
  guarantee that any candidate is skipped entirely.

The implemented cascade trades a small, linear fit-time preprocessing
(O(N * d) summaries) for the ability to skip whole candidates, and it
stays valid under every supported combination of lengths, weights and
path constraints. When the lower bounds are intrinsically weak (e.g.
heavily overlapping waveforms), the search gracefully degrades to
computing most pairs exactly — with the early-termination cutoff still
reducing the per-pair cost — while remaining exact.

Scope and fallbacks
-------------------

* ``metric`` must be the string ``'dtw'``; ``channel_weights``,
  ``global_constraint``, ``sakoe_chiba_radius`` and
  ``itakura_max_slope`` (passed through ``metric_params``) are honored
  with the same semantics as the brute-force path, including
  ``inf`` distances for infeasible constrained pairs (selected last,
  ordered by training row).
* Training and query series may be multivariate and of variable length
  (trailing all-NaN padding rows). Series with internal missing values,
  partially-NaN time steps, infinite values, or zero valid length are
  rejected with a ``ValueError``, as are inconsistent feature
  dimensions.
* The unsupervised ``kneighbors`` keeps the historical truncation
  habit when ``n_neighbors`` exceeds the number of available samples;
  the supervised entries (``predict`` / ``predict_proba``) raise a
  ``ValueError`` in that case. When ``kneighbors`` is called without
  queries, only the training row with the same index is excluded from
  each search (duplicate series living at other rows are kept).
* If all selected neighbors of a sample lie at an infinite DTW distance
  and ``weights='distance'``, classification and regression raise a
  ``ValueError`` instead of returning fabricated probabilities or NaN
  predictions. ``weights='uniform'`` and callables are unaffected.
* The index depends only on the training data, so changing
  distance-related parameters (``metric_params``) through
  ``set_params`` takes effect at the next query, as with the brute
  path; changing ``metric`` away from ``'dtw'`` makes pruned queries
  raise a ``ValueError`` until the estimator is refit. A failed fit
  leaves any previously fitted state untouched.

Example
-------

.. code-block:: python

    from tslearn.neighbors import KNeighborsTimeSeriesClassifier

    clf = KNeighborsTimeSeriesClassifier(n_neighbors=3,
                                         metric="dtw",
                                         dtw_search="pruned")
    clf.fit(X_train, y_train)
    labels = clf.predict(X_queries)          # exact same labels as brute
    clf.search_stats_["n_exact"]             # how many DTW were computed

A fully runnable retrieval and label-completion example is available in
:ref:`sphx_glr_examples_neighbors_plot_pruned_knn.py`.
