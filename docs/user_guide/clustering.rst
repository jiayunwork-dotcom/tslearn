.. _clustering:

Time Series Clustering
======================

Clustering is the task of grouping together similar objects.
This task hence heavily relies on the notion of similarity one relies on.

The following Figure illustrates why choosing an adequate similarity function
is key (code to reproduce is available
:ref:`in the Gallery of Examples
<sphx_glr_auto_examples_clustering_plot_kmeans.py>`).

.. figure:: ../_static/img/kmeans.svg
    :width: 100%
    :align: center

    :math:`k`-means clustering with Euclidean distance. Each subfigure represents series from a given cluster and their centroid (in red).

This Figure is the result of a :math:`k`-means clustering that uses Euclidean
distance as a base metric.
One issue with this metric is that it is not invariant to time shifts, while
the dataset at stake clearly holds such invariants.

:math:`k`-means and Dynamic Time Warping
----------------------------------------

To overcome the previously illustrated issue,
distance metrics dedicated to time series, such as
:ref:`Dynamic Time Warping (DTW) <dtw>`, are required.
As can be seen in the Figure below, the use of such metrics produce more
meaningful results.

The :mod:`tslearn.clustering` module in ``tslearn`` offers an
option to use DTW as the core metric in a :math:`k`-means algorithm,
which leads to better clusters and centroids:

.. figure:: ../_static/img/kmeans_dtw.svg
    :width: 100%
    :align: center

    :math:`k`-means clustering with Dynamic Time Warping. Each subfigure represents series from a given cluster and their centroid (in red).

First, clusters gather time series of similar shapes, which is due to the
ability of Dynamic Time Warping (DTW) to deal with time shifts, as explained
above.
Second, cluster centers (aka centroids) are computed as the barycenters
with respect to DTW, hence
they allow to retrieve a sensible average shape whatever the temporal shifts
in the cluster (see :ref:`our dedicated User Guide section <dtw-barycenters>`
for more details on how these barycenters are computed).

In ``tslearn``, clustering a time series dataset with :math:`k`-means and a
dedicated time series metric is as easy as


.. code-block:: python

    from tslearn.clustering import TimeSeriesKMeans

    model = TimeSeriesKMeans(n_clusters=3, metric="dtw",
                             max_iter=10, random_state=seed)
    model.fit(X_train)

where ``X_train`` is the considered unlabelled dataset of time series.
The ``metric`` parameter can also be set to ``"softdtw"`` as an alternative
time series metric (`cf.`
:ref:`our User Guide section on soft-DTW <dtw-softdtw>`).

.. _kmeans-channel-weights:

Learning per-cluster channel weights
------------------------------------

For multivariate time series, different clusters may be best separated by
different channels (sensors): one cluster may be characterized by the shape
of the first channel while another is mostly visible in the second channel.
Setting ``channel_weight_regularization`` to a positive value
:math:`\lambda` (only with ``metric="dtw"``) makes
:class:`tslearn.clustering.TimeSeriesKMeans` learn one channel weight vector
per cluster, jointly with the cluster centers, instead of relying on a
single global rescaling of the channels:

.. math::

    \min_{\{\mu_k, w_k\}_k} \frac{1}{n} \sum_i DTW_{w_{k(i)}}(x_i, \mu_{k(i)})^2
    + \lambda \sum_{k, c} w_{k, c} \log w_{k, c}
    \quad \text{s.t.} \quad \forall k, \sum_c w_{k, c} = d,\ w_{k, c} \geq \varepsilon

where :math:`k(i)` is the cluster assigned to :math:`x_i`,
:math:`DTW_{w}` is the :ref:`channel-weighted DTW <dtw-channel-weights>` and
:math:`\varepsilon` is ``min_channel_weight``. The entropy term keeps the
weights from collapsing onto a single channel: the larger :math:`\lambda`,
the closer the weights stay to uniform.

.. code-block:: python

    from tslearn.clustering import TimeSeriesKMeans

    model = TimeSeriesKMeans(n_clusters=3, metric="dtw",
                             channel_weight_regularization=0.1,
                             random_state=seed)
    model.fit(X_train)
    model.channel_weights_  # shape (n_clusters, d): one weight vector per cluster
    model.objective_        # final value of the regularized objective J
    model.objective_history_  # non-increasing list of accepted J values

Optimization alternates between (i) assigning each series to its closest
center using the current per-cluster weights, (ii) recomputing the optimal
alignment paths, (iii) updating each center with the classic DBA
Majorize-Minimize update, and (iv) updating each cluster's weights by
exactly minimizing the fixed-path surrogate plus the entropy regularizer.
Each of these steps does not increase the true objective :math:`J`, so
``objective_history_`` is non-increasing (up to a numerical tolerance) and
its last entry corresponds to the delivered model. After fitting,
``transform`` and ``predict`` use each cluster's own learned weights when
computing distances. The same weight-learning scheme is available for a
single cluster through
:func:`tslearn.barycenters.dtw_barycenter_averaging` (see
:ref:`dtw-barycenters`).

.. _kernel-kmeans:


Kernel :math:`k`-means and Time Series Kernels
----------------------------------------------

Another option to deal with such time shifts is to rely on the kernel trick.
Indeed, [1]_ introduces a positive semidefinite kernel for time series,
inspired from DTW.
Then, the kernel :math:`k`-means algorithm [2]_, that is equivalent to a
:math:`k`-means
that would operate in the Reproducing Kernel Hilbert Space associated to the
chosen kernel, can be used:

.. figure:: ../_static/img/kernel_kmeans.svg
    :width: 100%
    :align: center

    Kernel :math:`k`-means clustering with Global Alignment Kernel. Each subfigure represents series from a given cluster.

A first significant difference (when compared to :math:`k`-means) is that
cluster centers are never computed
explicitly, hence time series assignments to cluster are the only kind of
information available once the clustering is performed.

Second, one should note that the clusters generated by kernel-:math:`k`-means
are phase dependent (see clusters 2 and 3 that differ in phase rather than in
shape).
This is because Global Alignment Kernel is not invariant to time shifts, as
demonstrated in [3]_ for the closely related soft-DTW [4]_.


.. minigallery:: tslearn.clustering.TimeSeriesKMeans tslearn.clustering.KernelKMeans
    :add-heading: Examples Using Clustering Estimators
    :heading-level: -


.. raw:: html

    <div style="clear: both;" />

References
----------

.. [1] M. Cuturi. "Fast Global Alignment Kernels," ICML 2011.

.. [2] I. S. Dhillon, Y. Guan & B. Kulis.
       "Kernel k-means, Spectral Clustering and Normalized Cuts," KDD 2004.

.. [3] H. Janati, M. Cuturi, A. Gramfort. "Spatio-Temporal Alignments: Optimal
       transport through space and time," AISTATS 2020

.. [4] M. Cuturi, M. Blondel "Soft-DTW: a Differentiable Loss Function for
       Time-Series," ICML 2017.
