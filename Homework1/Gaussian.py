print(__doc__)

import time as time

import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.datasets import load_digits
from sklearn.preprocessing import scale
from sklearn import metrics
np.random.seed(42)
digits = load_digits()
data = scale(digits.data)
n_samples, n_features = data.shape
n_digits = len(np.unique(digits.target))
labels_true = digits.target
sample_size = 300

# #######################ward######################################################
# Compute clustering
st = time.time()
n_clusters = 27  # number of regions
Gaussian = GaussianMixture(n_components=40)
labels = Gaussian.fit(data).predict(data)
print("Homogeneity: %0.3f" % metrics.homogeneity_score(labels_true, labels))
print("Completeness: %0.3f" % metrics.completeness_score(labels_true, labels))
print("Normalized Mutual Information: %0.3f" % metrics.normalized_mutual_info_score(labels_true, labels))
