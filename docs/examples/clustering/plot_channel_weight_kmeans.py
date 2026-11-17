# -*- coding: utf-8 -*-
"""
Learning per-cluster channel weights with DTW k-means
=====================================================

This example illustrates how :class:`tslearn.clustering.TimeSeriesKMeans`
can learn a different set of channel (sensor) weights for each cluster when
``channel_weight_regularization`` is set.

We build an artificial two-channel dataset (no download needed): in the
first group of series, channel 0 holds a common waveform and channel 1 is
pure noise; in the second group, the roles are reversed. A standard
(unweighted) DTW k-means cannot down-weight the noisy channel, while the
adaptive variant learns, for each cluster, which channel matters.

The example prints the learned per-cluster weights and the evolution of the
true regularized objective during training, and plots the cluster centers.
"""

# License: BSD 3 clause

import numpy
import matplotlib.pyplot as plt

from tslearn.clustering import TimeSeriesKMeans

seed = 0
rng = numpy.random.RandomState(seed)

n_per_group, sz = 20, 40
t = numpy.linspace(0, 4 * numpy.pi, sz)


def make_group(informative_channel):
    series = []
    for _ in range(n_per_group):
        # A shared waveform (two bumps) with random time shifts per series
        shift = rng.randint(-4, 5)
        wave = numpy.sin(t + shift) * numpy.exp(-((t - 5) ** 2) / 20.0)
        wave += 0.5 * numpy.sin(2 * t + shift) * numpy.exp(
            -((t - 9) ** 2) / 8.0
        )
        channels = [2.0 * rng.randn(sz), 2.0 * rng.randn(sz)]
        channels[informative_channel] = wave + 0.1 * rng.randn(sz)
        series.append(numpy.column_stack(channels))
    return series


X = numpy.array(make_group(0) + make_group(1))
y_true = numpy.array([0] * n_per_group + [1] * n_per_group)

km = TimeSeriesKMeans(
    n_clusters=2,
    metric="dtw",
    channel_weight_regularization=0.05,
    max_iter=20,
    n_init=2,
    random_state=seed,
)
km.fit(X)

print("Learned per-cluster channel weights (rows: clusters, "
      "columns: channels):")
print(km.channel_weights_)
print("True objective J along training (accepted states):")
print(numpy.array(km.objective_history_))

# Match clusters to true groups (up to permutation)
pred = km.labels_
acc = max(numpy.mean(pred == y_true), numpy.mean(pred == 1 - y_true))
print("Adjusted group agreement: %.2f" % acc)

plt.figure(figsize=(10, 5))
for k in range(2):
    for c in range(2):
        plt.subplot(2, 2, 2 * k + c + 1)
        plt.plot(X[km.labels_ == k, :, c].T, "k-", alpha=0.2)
        plt.plot(km.cluster_centers_[k, :, c], "r-", linewidth=2)
        plt.title(
            "Cluster %d / channel %d (learned weight: %.2f)"
            % (k, c, km.channel_weights_[k, c])
        )
plt.tight_layout()
plt.show()
