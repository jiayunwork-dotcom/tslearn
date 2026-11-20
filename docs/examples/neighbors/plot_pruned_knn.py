# -*- coding: utf-8 -*-
"""
Exact pruned k-NN search for similar waveforms
===============================================

This example shows how ``dtw_search='pruned'`` speeds up exact
DTW k-nearest-neighbor retrieval and label completion on synthetic
waveforms, without computing the full query-against-training distance
matrix.

We generate a database of noisy sine-like waveforms (two channels,
variable length), retrieve the most similar waveforms to a few queries
with :class:`tslearn.neighbors.KNeighborsTimeSeries`, and then complete
missing labels with
:class:`tslearn.neighbors.KNeighborsTimeSeriesClassifier`. Everything
is *exact*: the neighbors and predictions are identical to those of the
default ``dtw_search='brute'`` (full distance matrix), only cheaper.
"""

# Author: tslearn contributors
# License: BSD 3 clause

import numpy
import matplotlib.pyplot as plt

from tslearn.neighbors import (KNeighborsTimeSeries,
                               KNeighborsTimeSeriesClassifier)

seed = 0
rng = numpy.random.RandomState(seed)


def make_waveform(freq, ampl, offset, length, noise, d=2):
    t = numpy.linspace(0, 2 * numpy.pi, length)
    base = offset + ampl * numpy.sin(freq * t)
    return numpy.stack(
        [base + noise * rng.randn(length) for _ in range(d)], axis=1)


# A database of variable-length waveforms, padded with NaN rows
n_train, max_len = 40, 60
freqs = rng.uniform(0.8, 3.0, size=n_train)
ampls = rng.uniform(0.5, 3.0, size=n_train)
offsets = rng.uniform(-2.0, 2.0, size=n_train)
lengths = rng.randint(40, max_len + 1, size=n_train)
X_train = numpy.full((n_train, max_len, 2), numpy.nan)
for i in range(n_train):
    X_train[i, :lengths[i]] = make_waveform(freqs[i], ampls[i],
                                            offsets[i], lengths[i], 0.1)
# Labels used later for label completion: slow vs fast oscillations
y_train = (freqs > 1.9).astype(int)

# Queries: noisy copies of a few training waveforms (NaN-padded too)
query_ids = [3, 11, 27]
X_query = numpy.full((len(query_ids), max_len, 2), numpy.nan)
for q, i in enumerate(query_ids):
    X_query[q, :lengths[i]] = (X_train[i, :lengths[i]]
                               + 0.05 * rng.randn(lengths[i], 2))

# Exact pruned retrieval: no full distance matrix is computed
knn = KNeighborsTimeSeries(n_neighbors=3, dtw_search="pruned")
knn.fit(X_train)
dist, ind = knn.kneighbors(X_query)
print("Neighbors of the queries (training rows):\n", ind)
print("Work counters:", {k: v for k, v in knn.search_stats_.items()
                         if k.startswith("n_")})

plt.figure(figsize=(9, 3))
for q, row in enumerate(ind):
    ax = plt.subplot(3, 3, 3 * q + 1)
    plt.plot(X_query[q, :, 0], "k-")
    plt.title("query %d" % q)
    plt.xticks([])
    for rank, train_id in enumerate(row):
        plt.subplot(3, 3, 3 * q + 2 + rank, sharey=ax)
        plt.plot(X_train[train_id, :, 0], "r-")
        plt.title("id=%d, d=%.2f" % (train_id, dist[q, rank]))
        plt.xticks([])
plt.tight_layout()

# Label completion: some labels are unknown, k-NN completes them
clf = KNeighborsTimeSeriesClassifier(n_neighbors=3, dtw_search="pruned")
clf.fit(X_train, y_train)
y_query = clf.predict(X_query)
print("Completed labels for the queries:", y_query,
      "(true:", y_train[query_ids], ")")
assert numpy.array_equal(y_query, y_train[query_ids])
plt.show()
