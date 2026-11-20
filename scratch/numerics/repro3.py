import warnings, traceback
import numpy as np
from tslearn.barycenters import dtw_barycenter_averaging
X3 = np.array([[[0.0, 0.0]], [[2.0, 4.0]]])
# adaptive variant with huge sample weights
try:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = dtw_barycenter_averaging(X3, init_barycenter=np.zeros((1, 2)), max_iter=1,
                                     weights=np.array([1e308, 1e308]),
                                     channel_weight_regularization=1.0)
    print("returned", r)
except Exception as e:
    traceback.print_exc()
