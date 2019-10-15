from time import time
import numpy as np
from sklearn.cluster import spectral_clustering
print(__doc__)

from sklearn import metrics
from sklearn.datasets.samples_generator import make_blobs
from time import time
import numpy as np
import matplotlib.pyplot as plt

from sklearn import metrics
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale

# #############################################################################
# Generate sample data
digits = load_digits()
data = scale(digits.data)
n_samples, n_features = data.shape
n_digits = len(np.unique(digits.target))
labels = digits.target

sample_size = 300
labels_true = digits.target
X = data
# #############################################################################
# Compute Affinity Propagation
sc = spectral_clustering(n_clusters=1, n_components=n_digits).fit(X)
cluster_centers_indices = sc.cluster_centers_indices_
labels = sc.labels_

n_clusters_ = len(cluster_centers_indices)

print('Estimated number of clusters: %d' % n_clusters_)
print("Homogeneity: %0.3f" % metrics.homogeneity_score(labels_true, labels))
print("Completeness: %0.3f" % metrics.completeness_score(labels_true, labels))
print("Normalized Mutual Information: %0.3f" % metrics.normalized_mutual_info_score(labels_true, labels))




