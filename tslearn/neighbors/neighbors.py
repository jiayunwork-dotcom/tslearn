import numpy
from sklearn import neighbors
from sklearn.neighbors import (KNeighborsClassifier, NearestNeighbors,
                               KNeighborsRegressor)
from sklearn.neighbors._base import _get_weights
from sklearn.utils.arrayfuncs import _all_with_any_reduction_axis_1
from sklearn.utils.extmath import weighted_mode
from sklearn.utils.fixes import _mode
from sklearn.utils.validation import check_is_fitted
from scipy.spatial.distance import cdist as scipy_cdist

from tslearn.metrics import cdist_dtw, cdist_ctw, cdist_soft_dtw, \
    cdist_sax, TSLEARN_VALID_METRICS
from tslearn.neighbors.pruned_dtw import (PrunedDtwIndex, check_pruned_k,
                                          normalize_pruned_metric_params)
from tslearn.piecewise import SymbolicAggregateApproximation
from tslearn.utils import (to_time_series_dataset, to_sklearn_dataset,
                           check_array, check_dims)
from tslearn.bases import BaseModelPackage

neighbors.VALID_METRICS['brute'].extend(['dtw', 'softdtw', 'sax', 'ctw'])

_ALL_INF_DISTANCES_MSG = (
    "All nearest neighbors of at least one sample are at an infinite "
    "DTW distance (for example because no admissible warping path "
    "exists under the current global constraint), so "
    "weights='distance' would produce undefined predictions. Use "
    "weights='uniform' or relax the DTW constraints."
)


class KNeighborsTimeSeriesMixin():
    """Mixin for k-neighbors searches on Time Series."""

    def _sax_preprocess(self, X, n_segments=10, alphabet_size_avg=4,
                        scale=False):
        # Now SAX-transform the time series
        if not hasattr(self, '_sax') or self._sax is None:
            self._sax = SymbolicAggregateApproximation(
                n_segments=n_segments,
                alphabet_size_avg=alphabet_size_avg,
                scale=scale
            )

        X = to_time_series_dataset(X)
        X_sax = self._sax.fit_transform(X)

        return X_sax

    def _get_metric_params(self):
        if self.metric_params is None:
            metric_params = {}
        else:
            metric_params = self.metric_params.copy()
        if "gamma_sdtw" in metric_params.keys():
            metric_params["gamma"] = metric_params["gamma_sdtw"]
            del metric_params["gamma_sdtw"]
        if "n_jobs" in metric_params.keys():
            del metric_params["n_jobs"]
        if "verbose" in metric_params.keys():
            del metric_params["verbose"]
        return metric_params

    def _validate_dtw_search(self):
        """Validate the dtw_search/metric combination (called at fit)."""
        if self.dtw_search not in ("brute", "pruned"):
            raise ValueError(
                "dtw_search must be 'brute' or 'pruned', got %r."
                % (self.dtw_search,)
            )
        if self.dtw_search == "pruned" and self.metric != "dtw":
            raise ValueError(
                "dtw_search='pruned' is only supported with the string "
                "metric='dtw', got metric=%r." % (self.metric,)
            )

    def _check_pruned_metric(self):
        """Ensure the (possibly set_params-modified) metric is still
        compatible with the pruned search at query time."""
        if self.metric != "dtw":
            raise ValueError(
                "dtw_search='pruned' requires metric='dtw', got %r. If "
                "'metric' was changed through set_params after fitting, "
                "restore metric='dtw' or refit the estimator."
                % (self.metric,)
            )

    def _fit_pruned(self, X, y=None):
        """Fit in pruned mode: validate data, build the DTW index.

        No N x N placeholder matrix is allocated: scikit-learn's fit is
        called on a minimal (n_samples, 1) placeholder, the actual
        training time series are kept in `_ts_fit` and summarized in
        `_pruned_index_`.
        """
        X = check_array(X, allow_nd=True, force_all_finite=False)
        X = to_time_series_dataset(X)
        X = check_dims(X)
        # Validate the distance parameters early (they are re-validated
        # at query time since set_params may change them).
        normalize_pruned_metric_params(self._get_metric_params(),
                                       X.shape[2])
        # Build the index first: if the data is invalid, no attribute is
        # modified and a previously fitted state (if any) is preserved.
        index = PrunedDtwIndex().fit(X)
        placeholder = numpy.zeros((X.shape[0], 1))
        # scikit-learn's parameter validation rejects metric='dtw' (and
        # metric='precomputed' would require a square, hence N x N,
        # placeholder): validate/fit the base estimator with a metric it
        # accepts. None of the attributes derived from the placeholder
        # are used by the pruned query paths.
        self._ts_metric = self.metric
        self.metric = "euclidean"
        try:
            super().fit(placeholder, y)
        finally:
            self.metric = self._ts_metric
        # Commit the pruned state only once everything succeeded, so a
        # failed fit never leaves a half-new model behind.
        self._ts_fit = X
        self._d = X.shape[2]
        self._X_fit = placeholder
        self._pruned_index_ = index
        return self

    def _get_pruned_index(self):
        """Return the fitted pruned index, rebuilding it if needed.

        The index only depends on the training data stored in `_ts_fit`,
        so it can be transparently rebuilt after deserializing a model
        that does not carry it.
        """
        check_is_fitted(self, "_ts_fit")
        index = getattr(self, "_pruned_index_", None)
        if index is None:
            index = PrunedDtwIndex().fit(self._ts_fit)
            self._pruned_index_ = index
        return index

    def _kneighbors_pruned(self, X, n_neighbors, return_distance):
        """kneighbors implementation for dtw_search='pruned'."""
        self._check_pruned_metric()
        index = self._get_pruned_index()
        if n_neighbors is None:
            n_neighbors = self.n_neighbors
        check_pruned_k(n_neighbors)
        if X is None:
            # Query the training set itself: only the row with the same
            # index is excluded from each search.
            Q = None
            exclude_self = True
            n_available = index.n_samples_ - 1
        else:
            X = check_array(X, allow_nd=True, force_all_finite=False)
            X = to_time_series_dataset(X)
            X = check_dims(X, X_fit_dims=self._ts_fit.shape, extend=True,
                           check_n_features_only=True)
            Q = X
            exclude_self = False
            n_available = index.n_samples_
        # Unsupervised entry point: keep the historical truncation habit
        # when n_neighbors exceeds the number of available samples.
        k = min(n_neighbors, max(n_available, 0))
        dist, ind, stats = index.kneighbors(
            Q, k, exclude_self=exclude_self,
            metric_params=self._get_metric_params(),
            n_jobs=self.n_jobs, verbose=self.verbose
        )
        self.search_stats_ = stats
        if return_distance:
            return dist, ind
        return ind

    def _pruned_neighbors_for_predict(self, X):
        """Exact neighbors used by the supervised pruned predictions."""
        self._check_pruned_metric()
        index = self._get_pruned_index()
        check_pruned_k(self.n_neighbors)
        if self.n_neighbors > index.n_samples_:
            raise ValueError(
                "n_neighbors=%d exceeds the number of training samples "
                "(%d): cannot compute exact neighbors for prediction. "
                "Reduce n_neighbors or fit on more samples."
                % (self.n_neighbors, index.n_samples_)
            )
        X = check_array(X, allow_nd=True, force_all_finite=False)
        X = to_time_series_dataset(X)
        X = check_dims(X, X_fit_dims=self._ts_fit.shape, extend=True,
                       check_n_features_only=True)
        dist, ind, stats = index.kneighbors(
            X, self.n_neighbors, exclude_self=False,
            metric_params=self._get_metric_params(),
            n_jobs=self.n_jobs, verbose=self.verbose
        )
        self.search_stats_ = stats
        return dist, ind

    def _check_inf_distance_weights(self, neigh_dist):
        """Reject all-infinite neighbor distances with weights='distance'."""
        if self.weights == "distance" and numpy.isinf(neigh_dist).all(
                axis=1).any():
            raise ValueError(_ALL_INF_DISTANCES_MSG)

    def _precompute_cross_dist(self, X, other_X=None):
        if other_X is None:
            other_X = self._ts_fit

        self._ts_metric = self.metric
        self.metric = "precomputed"

        metric_params = self._get_metric_params()

        X = check_array(X, allow_nd=True, force_all_finite=False)
        X = to_time_series_dataset(X)

        if self._ts_metric == "dtw":
            X_ = cdist_dtw(X, other_X, n_jobs=self.n_jobs,
                           **metric_params)
        elif self._ts_metric == "ctw":
            X_ = cdist_ctw(X, other_X, **metric_params)
        elif self._ts_metric == "softdtw":
            X_ = cdist_soft_dtw(X, other_X, **metric_params)
        elif self._ts_metric == "sax":
            X = self._sax_preprocess(X, **metric_params)
            X_ = cdist_sax(X, self._sax.breakpoints_avg_,
                           self._sax._X_fit_dims_[1], other_X,
                           n_jobs=self.n_jobs)
        else:
            raise ValueError("Invalid metric recorded: %s" %
                             self._ts_metric)

        return X_

    def kneighbors(self, X=None, n_neighbors=None, return_distance=True):
        """Finds the K-neighbors of a point.

        Returns indices of and distances to the neighbors of each point.

        Parameters
        ----------
        X : array-like, shape (n_ts, sz, d)
            The query time series.
            If not provided, neighbors of each indexed point are returned.
            In this case, the query point is not considered its own neighbor.
        n_neighbors : int
            Number of neighbors to get (default is the value passed to the
            constructor).
        return_distance : boolean, optional. Defaults to True.
            If False, distances will not be returned

        Returns
        -------
        dist : array
            Array representing the distance to points, only present if
            return_distance=True
        ind : array
            Indices of the nearest points in the population matrix.
        """
        if getattr(self, "dtw_search", "brute") == "pruned":
            return self._kneighbors_pruned(X, n_neighbors, return_distance)
        self_neighbors = False
        if n_neighbors is None:
            n_neighbors = self.n_neighbors
        if X is None:
            X = self._X_fit
            self_neighbors = True
        if self.metric == "precomputed":
            full_dist_matrix = X
        else:

            if X.ndim == 2:  # sklearn-format case
                X = X.reshape((X.shape[0], -1, self._d))
                fit_X = self._X_fit.reshape((self._X_fit.shape[0],
                                             -1,
                                             self._d))
            elif hasattr(self, '_ts_fit') and self._ts_fit is not None:
                fit_X = self._ts_fit
            else:
                fit_X = self._X_fit

            if (self.metric in TSLEARN_VALID_METRICS or
                    self.metric in [cdist_dtw, cdist_ctw,
                                    cdist_soft_dtw, cdist_sax]):
                full_dist_matrix = self._precompute_cross_dist(X,
                                                               other_X=fit_X)
            elif self.metric in ["euclidean", "sqeuclidean", "cityblock"]:
                full_dist_matrix = scipy_cdist(X.reshape((X.shape[0], -1)),
                                               fit_X.reshape((fit_X.shape[0],
                                                              -1)),
                                               metric=self.metric)
            else:
                raise ValueError("Unrecognized time series metric string: %s "
                                 "(should be one of 'dtw', 'softdtw', "
                                 "'sax', 'euclidean', 'sqeuclidean' "
                                 "or 'cityblock')" % self.metric)

        # Code similar to sklearn (sklearn/neighbors/base.py), to make sure
        # that TimeSeriesKNeighbor~(metric='euclidean') has the same results as
        # feeding a distance matrix to sklearn.KNeighbors~(metric='euclidean')
        kbin = min(n_neighbors - 1, full_dist_matrix.shape[1] - 1)
        # argpartition will make sure the first `kbin` entries are the
        # `kbin` smallest ones (but in arbitrary order) --> complexity: O(n)
        ind = numpy.argpartition(full_dist_matrix, kbin, axis=1)

        if self_neighbors:
            ind = ind[:, 1:]
        if n_neighbors > full_dist_matrix.shape[1]:
            n_neighbors = full_dist_matrix.shape[1]
        ind = ind[:, :n_neighbors]

        n_ts = X.shape[0]
        sample_range = numpy.arange(n_ts)[:, None]
        # Sort the `kbin` nearest neighbors according to distance
        ind = ind[
            sample_range, numpy.argsort(full_dist_matrix[sample_range, ind])]
        dist = full_dist_matrix[sample_range, ind]

        if hasattr(self, '_ts_metric'):
            self.metric = self._ts_metric

        if return_distance:
            return dist, ind
        else:
            return ind

    def _get_tags(self):
        # sklearn < 1.6 super()._get_tags() returns dict based on _more_tags
        # sklearn >= 1.6 super()._get_tags() returns dict based on __sklearn_tags__
        tags = super()._get_tags()

        # Make sure update tags based on _more_tags for sklearn > 1.6
        tags.update(self._more_tags())
        return tags

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.input_tags.allow_nan = True
        tags.input_tags.sparse = False
        return tags

class KNeighborsTimeSeries(KNeighborsTimeSeriesMixin, NearestNeighbors,
                           BaseModelPackage):
    """Unsupervised learner for implementing neighbor searches for Time Series.

    Parameters
    ----------
    n_neighbors : int (default: 5)
        Number of nearest neighbors to be considered for the decision.

    metric : {'dtw', 'softdtw', 'ctw', 'euclidean', 'sqeuclidean', \
              'cityblock',  'sax'} (default: 'dtw')
        Metric to be used at the core of the nearest neighbor procedure.
        DTW and SAX are described in more detail in :mod:`tslearn.metrics`.
        When SAX is provided as a metric, the data is expected to be
        normalized such that each time series has zero mean and unit
        variance. Other metrics are described in `scipy.spatial.distance doc
        <https://docs.scipy.org/doc/scipy/reference/spatial.distance.html>`_.

    metric_params : dict or None (default: None)
        Dictionary of metric parameters.
        For metrics that accept parallelization of the cross-distance matrix
        computations, `n_jobs` and `verbose` keys passed in `metric_params`
        are overridden by the `n_jobs` and `verbose` arguments.
        For 'sax' metric, these are hyper-parameters to be passed at the
        creation of the `SymbolicAggregateApproximation` object.

    dtw_search : {'brute', 'pruned'} (default: 'brute')
        Search strategy used when ``metric='dtw'``. 'brute' computes the
        full cross-distance matrix between queries and training samples
        (historical behavior). 'pruned' performs an *exact* search that
        avoids most of those computations: a lower bound on the DTW
        distance to every training sample is derived from per-channel
        summaries built at fit time, candidates are visited in
        increasing lower-bound order, and exact DTW (with early
        termination through ``max_dist``) is only computed for
        candidates that can still make it into the k nearest ones. The
        returned neighbors and distances are exactly those that 'brute'
        would return (ties are broken by training row index), only
        faster and with less memory: no Q x N distance matrix is built.
        'pruned' is only valid with the string ``metric='dtw'`` and
        raises a ValueError at fit time for any other metric. It
        supports multivariate and variable-length time series (with
        trailing NaN padding), ``channel_weights`` and the
        Sakoe-Chiba/Itakura global constraints. Series with internal
        NaNs, partially-NaN time steps, infinite values or zero valid
        length are rejected with a ValueError. After each pruned search,
        the ``search_stats_`` attribute holds work counters (number of
        exact DTW computations, visited dynamic-programming cells, ...).

    n_jobs : int or None, optional (default=None)
        The number of jobs to run in parallel for cross-distance matrix
        computations.
        Ignored if the cross-distance matrix cannot be computed using
        parallelization.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See scikit-learns'
        `Glossary <https://scikit-learn.org/stable/glossary.html#term-n-jobs>`__
        for more details.

    Notes
    -----
        The training data are saved to disk if this model is
        serialized and may result in a large model file if the training
        dataset is large.

    Examples
    --------
    >>> time_series = to_time_series_dataset([[1, 2, 3, 4],
    ...                                       [3, 3, 2, 0],
    ...                                       [1, 2, 2, 4]])
    >>> knn = KNeighborsTimeSeries(n_neighbors=1).fit(time_series)
    >>> dataset = to_time_series_dataset([[1, 1, 2, 2, 2, 3, 4]])
    >>> dist, ind = knn.kneighbors(dataset, return_distance=True)
    >>> dist
    array([[0.]])
    >>> print(ind)
    [[0]]
    >>> knn2 = KNeighborsTimeSeries(n_neighbors=10,
    ...                             metric="euclidean").fit(time_series)
    >>> print(knn2.kneighbors(return_distance=False))
    [[2 1]
     [2 0]
     [0 1]]
    >>> knn3 = KNeighborsTimeSeries(n_neighbors=1,
    ...                             dtw_search="pruned").fit(time_series)
    >>> knn3.kneighbors(dataset, return_distance=False)
    array([[0]])
    """  # noqa: E501
    def __init__(self, n_neighbors=5, metric="dtw", metric_params=None,
                 n_jobs=None, verbose=0, dtw_search="brute"):
        NearestNeighbors.__init__(self,
                                  n_neighbors=n_neighbors,
                                  algorithm='brute')
        self.metric = metric
        self.metric_params = metric_params
        self.n_jobs = n_jobs
        self.verbose = verbose
        self.dtw_search = dtw_search

    def _is_fitted(self):
        if self.metric in TSLEARN_VALID_METRICS:
            check_is_fitted(self, '_ts_fit')
        else:
            check_is_fitted(self, '_X_fit')

        return True

    def _get_model_params(self):
        if self.metric in TSLEARN_VALID_METRICS:
            return {'_ts_fit': self._ts_fit}
        else:
            return {'_X_fit': self._X_fit}

    def fit(self, X, y=None):
        """Fit the model using X as training data

        Parameters
        ----------
        X : array-like, shape (n_ts, sz, d)
            Training data.
        """
        self._validate_dtw_search()
        if self.dtw_search == "pruned":
            return self._fit_pruned(X, y)
        if self.metric in TSLEARN_VALID_METRICS:
            self._ts_metric = self.metric
            self.metric = "precomputed"

        X = check_array(X,
                        allow_nd=True,
                        force_all_finite=(self.metric != "precomputed"))
        X = to_time_series_dataset(X)
        X = check_dims(X)
        if self.metric == "precomputed" and hasattr(self, '_ts_metric'):
            self._ts_fit = X
            self._d = X.shape[2]
            self._X_fit = numpy.zeros((self._ts_fit.shape[0],
                                       self._ts_fit.shape[0]))
        else:
            self._X_fit, self._d = to_sklearn_dataset(X, return_dim=True)
        super().fit(self._X_fit, y)
        if hasattr(self, '_ts_metric'):
            self.metric = self._ts_metric
        return self

    def kneighbors(self, X=None, n_neighbors=None, return_distance=True):
        """Finds the K-neighbors of a point.

        Returns indices of and distances to the neighbors of each point.

        Parameters
        ----------
        X : array-like, shape (n_ts, sz, d)
            The query time series.
            If not provided, neighbors of each indexed point are returned.
            In this case, the query point is not considered its own neighbor.
        n_neighbors : int
            Number of neighbors to get (default is the value passed to the
            constructor).
        return_distance : boolean, optional. Defaults to True.
            If False, distances will not be returned

        Returns
        -------
        dist : array
            Array representing the distance to points, only present if
            return_distance=True
        ind : array
            Indices of the nearest points in the population matrix.
        """
        if getattr(self, "dtw_search", "brute") == "pruned":
            return KNeighborsTimeSeriesMixin.kneighbors(
                self, X=X, n_neighbors=n_neighbors,
                return_distance=return_distance)
        if self.metric in TSLEARN_VALID_METRICS:
            self._ts_metric = self.metric
            self.metric = "precomputed"

            metric_params = self._get_metric_params()
            check_is_fitted(self, '_ts_fit')
            X = check_array(X, allow_nd=True, force_all_finite=False)
            X = check_dims(X, X_fit_dims=self._ts_fit.shape, extend=True,
                           check_n_features_only=True)
            if self._ts_metric == "dtw":
                X_ = cdist_dtw(X, self._ts_fit, n_jobs=self.n_jobs,
                               verbose=self.verbose, **metric_params)
            elif self._ts_metric == "ctw":
                X_ = cdist_ctw(X, self._ts_fit, **metric_params)
            elif self._ts_metric == "softdtw":
                X_ = cdist_soft_dtw(X, self._ts_fit, **metric_params)
            else:
                raise ValueError("Invalid metric recorded: %s" %
                                 self._ts_metric)
            pred = KNeighborsTimeSeriesMixin.kneighbors(
                self,
                X=X_,
                n_neighbors=n_neighbors,
                return_distance=return_distance)
            self.metric = self._ts_metric
            return pred
        else:
            check_is_fitted(self, '_X_fit')
            if X is None:
                X_ = None
            else:
                X = check_array(X, allow_nd=True)
                X = to_time_series_dataset(X)
                X_ = to_sklearn_dataset(X)
                X_ = check_dims(X_, X_fit_dims=self._X_fit.shape, extend=False)
            return KNeighborsTimeSeriesMixin.kneighbors(
                self,
                X=X_,
                n_neighbors=n_neighbors,
                return_distance=return_distance)


class KNeighborsTimeSeriesClassifier(KNeighborsTimeSeriesMixin,
                                     BaseModelPackage,
                                     KNeighborsClassifier):
    """Classifier implementing the k-nearest neighbors vote for Time Series.

    Parameters
    ----------
    n_neighbors : int (default: 5)
        Number of nearest neighbors to be considered for the decision.
    weights : str or callable, optional (default: 'uniform')
        Weight function used in prediction. Possible values:

        - 'uniform' : uniform weights. All points in each neighborhood are
          weighted equally.
        - 'distance' : weight points by the inverse of their distance. in this
          case, closer neighbors of a query point
          will have a greater influence than neighbors which are further away.
        - [callable] : a user-defined function which accepts an array of
          distances, and returns an array of the same
          shape containing the weights.
    metric : one of the metrics allowed for :class:`.KNeighborsTimeSeries`
    class (default: 'dtw')
        Metric to be used at the core of the nearest neighbor procedure
    metric_params : dict or None (default: None)
        Dictionnary of metric parameters.
        For metrics that accept parallelization of the cross-distance matrix
        computations, `n_jobs` and `verbose` keys passed in `metric_params`
        are overridden by the `n_jobs` and `verbose` arguments.
        For 'sax' metric, these are hyper-parameters to be passed at the
        creation of the `SymbolicAggregateApproximation` object.

    dtw_search : {'brute', 'pruned'} (default: 'brute')
        Search strategy used when ``metric='dtw'``. See
        :class:`.KNeighborsTimeSeries` for details. With 'pruned',
        predictions are computed from the same exact k nearest neighbors
        as with 'brute' (no full distance matrix is ever built), and a
        ValueError is raised at predict time if `n_neighbors` exceeds
        the number of training samples, or if all selected neighbors of
        a sample lie at an infinite DTW distance while
        ``weights='distance'``.

    n_jobs : int or None, optional (default=None)
        The number of jobs to run in parallel for cross-distance matrix
        computations.
        Ignored if the cross-distance matrix cannot be computed using
        parallelization.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See scikit-learns'
        `Glossary <https://scikit-learn.org/stable/glossary.html#term-n-jobs>`__
        for more details.

    verbose : int, optional (default=0)
        The verbosity level: if non zero, progress messages are printed.
        Above 50, the output is sent to stdout.
        The frequency of the messages increases with the verbosity level.
        If it more than 10, all iterations are reported.
        `Glossary <https://joblib.readthedocs.io/en/latest/parallel.html#parallel-reference-documentation>`__
        for more details.

    Notes
    -----
        The training data are saved to disk if this model is
        serialized and may result in a large model file if the training
        dataset is large.

    Examples
    --------
    >>> clf = KNeighborsTimeSeriesClassifier(n_neighbors=2, metric="dtw")
    >>> clf.fit([[1, 2, 3], [1, 1.2, 3.2], [3, 2, 1]],
    ...         y=[0, 0, 1]).predict([[1, 2.2, 3.5]])
    array([0])
    >>> clf = KNeighborsTimeSeriesClassifier(n_neighbors=2,
    ...                                      metric="dtw",
    ...                                      n_jobs=2)
    >>> clf.fit([[1, 2, 3], [1, 1.2, 3.2], [3, 2, 1]],
    ...         y=[0, 0, 1]).predict([[1, 2.2, 3.5]])
    array([0])
    >>> clf = KNeighborsTimeSeriesClassifier(n_neighbors=2,
    ...                                      metric="dtw",
    ...                                      metric_params={
    ...                                          "itakura_max_slope": 2.},
    ...                                      n_jobs=2)
    >>> clf.fit([[1, 2, 3], [1, 1.2, 3.2], [3, 2, 1]],
    ...         y=[0, 0, 1]).predict([[1, 2.2, 3.5]])
    array([0])
    """  # noqa: E501
    def __init__(self,
                 n_neighbors=5,
                 weights='uniform',
                 metric='dtw',
                 metric_params=None,
                 n_jobs=None,
                 verbose=0,
                 dtw_search='brute'):
        KNeighborsClassifier.__init__(self,
                                      n_neighbors=n_neighbors,
                                      weights=weights,
                                      algorithm='brute')
        self.metric = metric
        self.metric_params = metric_params
        self.n_jobs = n_jobs
        self.verbose = verbose
        self.dtw_search = dtw_search

    def _is_fitted(self):
        check_is_fitted(self, '_ts_fit')
        return True

    def _get_model_params(self):
        return {
            '_X_fit': self._X_fit,
            '_ts_fit': self._ts_fit,
            '_d': self._d,
            'classes_': self.classes_,
            '_y': self._y,
            'outputs_2d_': self.outputs_2d_,
            '_fit_X': self._fit_X,
            '_fit_method': self._fit_method
        }

    def fit(self, X, y):
        """Fit the model using X as training data and y as target values

        Parameters
        ----------
        X : array-like, shape (n_ts, sz, d)
            Training data.
        y : array-like, shape (n_ts, )
            Target values.

        Returns
        -------
        KNeighborsTimeSeriesClassifier
            The fitted estimator
        """
        self._validate_dtw_search()
        if self.dtw_search == "pruned":
            return self._fit_pruned(X, y)
        if self.metric in TSLEARN_VALID_METRICS:
            self._ts_metric = self.metric
            self.metric = "precomputed"

        X = check_array(X,
                        allow_nd=True,
                        force_all_finite=(self.metric != "precomputed"))
        X = to_time_series_dataset(X)
        X = check_dims(X)
        if self.metric == "precomputed" and hasattr(self, '_ts_metric'):
            self._ts_fit = X
            if self._ts_metric == 'sax':
                if self.metric_params is not None:
                    self._ts_fit = self._sax_preprocess(X,
                                                        **self.metric_params)
                else:
                    self._ts_fit = self._sax_preprocess(X)

            self._d = X.shape[2]
            self._X_fit = numpy.zeros((self._ts_fit.shape[0],
                                       self._ts_fit.shape[0]))
        else:
            self._X_fit, self._d = to_sklearn_dataset(X, return_dim=True)
        super().fit(self._X_fit, y)
        if hasattr(self, '_ts_metric'):
            self.metric = self._ts_metric
        return self

    def predict(self, X):
        """Predict the class labels for the provided data

        Parameters
        ----------
        X : array-like, shape (n_ts, sz, d)
            Test samples.

        Returns
        -------
        array, shape = (n_ts, )
            Array of predicted class labels
        """
        if getattr(self, "dtw_search", "brute") == "pruned":
            return self._predict_pruned(X)
        if self.metric in TSLEARN_VALID_METRICS:
            check_is_fitted(self, '_ts_fit')
            X = to_time_series_dataset(X)
            X = check_dims(X, X_fit_dims=self._ts_fit.shape, extend=True,
                           check_n_features_only=True)
            X_ = self._precompute_cross_dist(X)
            pred = super().predict(X_)
            self.metric = self._ts_metric
            return pred
        else:
            check_is_fitted(self, '_X_fit')
            X = check_array(X, allow_nd=True)
            X = to_time_series_dataset(X)
            X_ = to_sklearn_dataset(X)
            X_ = check_dims(X_, X_fit_dims=self._X_fit.shape, extend=False)
            return super().predict(X_)

    def predict_proba(self, X):
        """Predict the class probabilities for the provided data

        Parameters
        ----------
        X : array-like, shape (n_ts, sz, d)
            Test samples.

        Returns
        -------
        array, shape = (n_ts, n_classes)
            Array of predicted class probabilities
        """
        if getattr(self, "dtw_search", "brute") == "pruned":
            return self._predict_proba_pruned(X)
        if self.metric in TSLEARN_VALID_METRICS:
            check_is_fitted(self, '_ts_fit')
            X = check_dims(X, X_fit_dims=self._ts_fit.shape, extend=True,
                           check_n_features_only=True)
            X_ = self._precompute_cross_dist(X)
            pred = super().predict_proba(X_)
            self.metric = self._ts_metric
            return pred
        else:
            check_is_fitted(self, '_X_fit')
            X = check_array(X, allow_nd=True)
            X = to_time_series_dataset(X)
            X_ = to_sklearn_dataset(X)
            X_ = check_dims(X_, X_fit_dims=self._X_fit.shape, extend=False)
            return super().predict_proba(X_)

    def _predict_pruned(self, X):
        """predict for dtw_search='pruned'.

        Mirrors sklearn.neighbors.KNeighborsClassifier.predict
        (scikit-learn 1.5.2) but sources the neighbors from the exact
        pruned search instead of a precomputed distance matrix.
        """
        neigh_dist, neigh_ind = self._pruned_neighbors_for_predict(X)
        self._check_inf_distance_weights(neigh_dist)

        classes_ = self.classes_
        _y = self._y
        if not self.outputs_2d_:
            _y = self._y.reshape((-1, 1))
            classes_ = [self.classes_]

        n_outputs = len(classes_)
        n_queries = neigh_dist.shape[0]
        weights = _get_weights(neigh_dist, self.weights)
        if weights is not None and _all_with_any_reduction_axis_1(
                weights, value=0):
            raise ValueError(
                "All neighbors of some sample is getting zero weights. "
                "Please modify 'weights' to avoid this case if you are "
                "using a user-defined function."
            )

        y_pred = numpy.empty((n_queries, n_outputs),
                             dtype=classes_[0].dtype)
        for k, classes_k in enumerate(classes_):
            if weights is None:
                mode, _ = _mode(_y[neigh_ind, k], axis=1)
            else:
                mode, _ = weighted_mode(_y[neigh_ind, k], weights, axis=1)
            mode = numpy.asarray(mode.ravel(), dtype=numpy.intp)
            y_pred[:, k] = classes_k.take(mode)

        if not self.outputs_2d_:
            y_pred = y_pred.ravel()
        return y_pred

    def _predict_proba_pruned(self, X):
        """predict_proba for dtw_search='pruned'.

        Mirrors sklearn.neighbors.KNeighborsClassifier.predict_proba
        (scikit-learn 1.5.2) but sources the neighbors from the exact
        pruned search instead of a precomputed distance matrix.
        """
        neigh_dist, neigh_ind = self._pruned_neighbors_for_predict(X)
        self._check_inf_distance_weights(neigh_dist)

        classes_ = self.classes_
        _y = self._y
        if not self.outputs_2d_:
            _y = self._y.reshape((-1, 1))
            classes_ = [self.classes_]

        n_queries = neigh_dist.shape[0]
        weights = _get_weights(neigh_dist, self.weights)
        if weights is None:
            weights = numpy.ones_like(neigh_ind)
        elif _all_with_any_reduction_axis_1(weights, value=0):
            raise ValueError(
                "All neighbors of some sample is getting zero weights. "
                "Please modify 'weights' to avoid this case if you are "
                "using a user-defined function."
            )

        all_rows = numpy.arange(n_queries)
        probabilities = []
        for k, classes_k in enumerate(classes_):
            pred_labels = _y[:, k][neigh_ind]
            proba_k = numpy.zeros((n_queries, classes_k.size))

            # a simple ':' index doesn't work right
            for i, idx in enumerate(pred_labels.T):  # loop is O(n_neighbors)
                proba_k[all_rows, idx] += weights[:, i]

            # normalize 'votes' into real [0,1] probabilities
            normalizer = proba_k.sum(axis=1)[:, numpy.newaxis]
            proba_k /= normalizer

            probabilities.append(proba_k)

        if not self.outputs_2d_:
            probabilities = probabilities[0]

        return probabilities

    def _more_tags(self):
        return {'allow_nan': True, 'allow_variable_length': True}


class KNeighborsTimeSeriesRegressor(KNeighborsTimeSeriesMixin,
                                    KNeighborsRegressor):
    """Classifier implementing the k-nearest neighbors vote for Time Series.

    Parameters
    ----------
    n_neighbors : int (default: 5)
        Number of nearest neighbors to be considered for the decision.
    weights : str or callable, optional (default: 'uniform')
        Weight function used in prediction. Possible values:

        - 'uniform' : uniform weights. All points in each neighborhood are
          weighted equally.
        - 'distance' : weight points by the inverse of their distance. in this
          case, closer neighbors of a query point
          will have a greater influence than neighbors which are further away.
        - [callable] : a user-defined function which accepts an array of
          distances, and returns an array of the same
          shape containing the weights.
    metric : one of the metrics allowed for :class:`.KNeighborsTimeSeries`
    class (default: 'dtw')
        Metric to be used at the core of the nearest neighbor procedure
    metric_params : dict or None (default: None)
        Dictionnary of metric parameters.
        For metrics that accept parallelization of the cross-distance matrix
        computations, `n_jobs` and `verbose` keys passed in `metric_params`
        are overridden by the `n_jobs` and `verbose` arguments.
        For 'sax' metric, these are hyper-parameters to be passed at the
        creation of the `SymbolicAggregateApproximation` object.

    dtw_search : {'brute', 'pruned'} (default: 'brute')
        Search strategy used when ``metric='dtw'``. See
        :class:`.KNeighborsTimeSeries` for details. With 'pruned',
        predictions are computed from the same exact k nearest neighbors
        as with 'brute' (no full distance matrix is ever built), and a
        ValueError is raised at predict time if `n_neighbors` exceeds
        the number of training samples, or if all selected neighbors of
        a sample lie at an infinite DTW distance while
        ``weights='distance'``.

    n_jobs : int or None, optional (default=None)
        The number of jobs to run in parallel for cross-distance matrix
        computations.
        Ignored if the cross-distance matrix cannot be computed using
        parallelization.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See scikit-learns'
        `Glossary <https://scikit-learn.org/stable/glossary.html#term-n-jobs>`__
        for more details.

    verbose : int, optional (default=0)
        The verbosity level: if non zero, progress messages are printed.
        Above 50, the output is sent to stdout.
        The frequency of the messages increases with the verbosity level.
        If it more than 10, all iterations are reported.
        `Glossary <https://joblib.readthedocs.io/en/latest/parallel.html#parallel-reference-documentation>`__
        for more details.

    Examples
    --------
    >>> clf = KNeighborsTimeSeriesRegressor(n_neighbors=2, metric="dtw")
    >>> clf.fit([[1, 2, 3], [1, 1.2, 3.2], [3, 2, 1]],
    ...         y=[0.1, 0.1, 1.1]).predict([[1, 2.2, 3.5]])
    array([0.1])
    >>> clf = KNeighborsTimeSeriesRegressor(n_neighbors=2,
    ...                                     metric="dtw",
    ...                                     n_jobs=2)
    >>> clf.fit([[1, 2, 3], [1, 1.2, 3.2], [3, 2, 1]],
    ...         y=[0.1, 0.1, 1.1]).predict([[1, 2.2, 3.5]])
    array([0.1])
    >>> clf = KNeighborsTimeSeriesRegressor(n_neighbors=2,
    ...                                     metric="dtw",
    ...                                     metric_params={
    ...                                         "itakura_max_slope": 2.},
    ...                                     n_jobs=2)
    >>> clf.fit([[1, 2, 3], [1, 1.2, 3.2], [3, 2, 1]],
    ...         y=[0.1, 0.1, 1.1]).predict([[1, 2.2, 3.5]])
    array([0.1])
    """  # noqa: E501
    def __init__(self,
                 n_neighbors=5,
                 weights='uniform',
                 metric='dtw',
                 metric_params=None,
                 n_jobs=None,
                 verbose=0,
                 dtw_search='brute'):
        KNeighborsRegressor.__init__(self,
                                     n_neighbors=n_neighbors,
                                     weights=weights,
                                     algorithm='brute')
        self.metric = metric
        self.metric_params = metric_params
        self.n_jobs = n_jobs
        self.verbose = verbose
        self.dtw_search = dtw_search

    def fit(self, X, y):
        """Fit the model using X as training data and y as target values

        Parameters
        ----------
        X : array-like, shape (n_ts, sz, d)
            Training data.
        y : array-like, shape (n_ts, ) or (n_ts, dim_y)
            Target values.

        Returns
        -------
        KNeighborsTimeSeriesRegressor
            The fitted estimator
        """
        self._validate_dtw_search()
        if self.dtw_search == "pruned":
            return self._fit_pruned(X, y)
        if self.metric in TSLEARN_VALID_METRICS:
            self._ts_metric = self.metric
            self.metric = "precomputed"

        X = check_array(X,
                        allow_nd=True,
                        force_all_finite=(self.metric != "precomputed"))
        X = to_time_series_dataset(X)
        X = check_dims(X)
        if self.metric == "precomputed" and hasattr(self, '_ts_metric'):
            self._ts_fit = X
            self._d = X.shape[2]
            self._X_fit = numpy.zeros((self._ts_fit.shape[0],
                                       self._ts_fit.shape[0]))
        else:
            self._X_fit, self._d = to_sklearn_dataset(X, return_dim=True)
        super().fit(self._X_fit, y)
        if hasattr(self, '_ts_metric'):
            self.metric = self._ts_metric
        return self

    def predict(self, X):
        """Predict the target for the provided data

        Parameters
        ----------
        X : array-like, shape (n_ts, sz, d)
            Test samples.

        Returns
        -------
        array, shape = (n_ts, ) or (n_ts, dim_y)
            Array of predicted targets
        """
        if getattr(self, "dtw_search", "brute") == "pruned":
            return self._predict_pruned(X)
        if self.metric in TSLEARN_VALID_METRICS:
            check_is_fitted(self, '_ts_fit')
            X = to_time_series_dataset(X)
            X = check_dims(X, X_fit_dims=self._ts_fit.shape, extend=True,
                           check_n_features_only=True)
            X_ = self._precompute_cross_dist(X)
            pred = super().predict(X_)
            self.metric = self._ts_metric
            return pred
        else:
            check_is_fitted(self, '_X_fit')
            X = check_array(X, allow_nd=True)
            X = to_time_series_dataset(X)
            X_ = to_sklearn_dataset(X)
            X_ = check_dims(X_, X_fit_dims=self._X_fit.shape, extend=False)
            return super().predict(X_)

    def _predict_pruned(self, X):
        """predict for dtw_search='pruned'.

        Mirrors sklearn.neighbors.KNeighborsRegressor.predict
        (scikit-learn 1.5.2) but sources the neighbors from the exact
        pruned search instead of a precomputed distance matrix.
        """
        neigh_dist, neigh_ind = self._pruned_neighbors_for_predict(X)
        self._check_inf_distance_weights(neigh_dist)

        weights = _get_weights(neigh_dist, self.weights)

        _y = self._y
        if _y.ndim == 1:
            _y = _y.reshape((-1, 1))

        if weights is None:
            y_pred = numpy.mean(_y[neigh_ind], axis=1)
        else:
            y_pred = numpy.empty((neigh_dist.shape[0], _y.shape[1]),
                                 dtype=numpy.float64)
            denom = numpy.sum(weights, axis=1)

            for j in range(_y.shape[1]):
                num = numpy.sum(_y[neigh_ind, j] * weights, axis=1)
                y_pred[:, j] = num / denom

        if self._y.ndim == 1:
            y_pred = y_pred.ravel()

        return y_pred

    def _more_tags(self):
        return {'allow_nan': True, 'allow_variable_length': True}
