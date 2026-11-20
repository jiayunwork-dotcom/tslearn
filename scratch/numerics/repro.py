import warnings
import numpy as np
from tslearn.metrics import dtw, dtw_path, cdist_dtw
from tslearn.barycenters import dtw_barycenter_averaging
from tslearn.clustering import TimeSeriesKMeans

def case(name, fn):
    print("=" * 20, name)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            r = fn()
        print("RETURNED:", r)
    except Exception as e:
        print("RAISED %s: %s" % (type(e).__name__, e))

case("dtw zero-weight", lambda: dtw([[1e308, 0]], [[-1e308, 1]], channel_weights=[0, 1]))
case("dtw_path zero-weight", lambda: dtw_path([[1e308, 0]], [[-1e308, 1]], channel_weights=[0, 1]))
case("cdist zero-weight", lambda: cdist_dtw([[[1e308, 0.0]]], [[[-1e308, 1.0]]], channel_weights=[0, 1]))

X2 = np.array([[[-1e8, -1e8, -1e8]], [[1e8, 1e8, 1e8]]])
zero13 = np.zeros((1, 3))
def dba_sym():
    bary, w, hist = dtw_barycenter_averaging(
        X2, init_barycenter=zero13, channel_weight_regularization=1.0, max_iter=1)
    return ("bary=", bary.tolist(), "w=", w.tolist(), "w.sum=", float(w.sum()), "hist=", [float(h) for h in hist])
case("DBA symmetric 1e8", dba_sym)

def km_sym():
    km = TimeSeriesKMeans(n_clusters=1, metric="dtw", init=np.zeros((1, 1, 3)), n_init=1,
                          max_iter=1, channel_weight_regularization=1.0)
    km.fit(X2)
    return ("w=", km.channel_weights_.tolist(), "sum=", float(km.channel_weights_.sum()),
            "obj=", float(km.objective_), "hist=", km.objective_history_.tolist())
case("KMeans symmetric 1e8", km_sym)

X3 = np.array([[[0.0, 0.0]], [[2.0, 4.0]]])
def dba_huge_sw():
    return dtw_barycenter_averaging(X3, init_barycenter=np.zeros((1, 2)), max_iter=1,
                                    weights=np.array([1e308, 1e308]))
case("DBA huge sample weights (plain DBA)", dba_huge_sw)

X4 = np.array([[[-1e154]], [[1e154]]])
def dba_154():
    return dtw_barycenter_averaging(
        X4, init_barycenter=np.zeros((1, 1)), channel_weight_regularization=1.0, max_iter=1)
case("DBA 1e154 adaptive", dba_154)
def km_154():
    km = TimeSeriesKMeans(n_clusters=1, metric="dtw", init=np.zeros((1, 1, 1)), n_init=1,
                          max_iter=1, channel_weight_regularization=1.0)
    km.fit(X4)
    return ("center=", None if km.cluster_centers_ is None else km.cluster_centers_.tolist(),
            "obj=", km.objective_)
case("KMeans 1e154 adaptive", km_154)

X5 = np.array([[[1e308, 1e308]], [[1e308, 1e308]]])
def dba_308():
    return dtw_barycenter_averaging(
        X5, init_barycenter=X5[0].copy(), channel_weight_regularization=1.0, max_iter=1)
case("DBA identical 1e308", dba_308)
def km_308():
    km = TimeSeriesKMeans(n_clusters=1, metric="dtw", init=X5[:1].copy(), n_init=1,
                          max_iter=1, channel_weight_regularization=1.0)
    km.fit(X5)
    return ("center=", None if km.cluster_centers_ is None else km.cluster_centers_.tolist(),
            "obj=", km.objective_)
case("KMeans identical 1e308", km_308)
