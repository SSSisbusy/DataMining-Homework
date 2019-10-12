print(__doc__)

from sklearn.cluster import AffinityPropagation
from sklearn import metrics
from sklearn.datasets.samples_generator import make_blobs
from time import time
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale

# #############################################################################
# Generate sample data
#centers = [[1, 1], [-1, -1], [1, -1]]
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
wh = AgglomerativeClustering().fit(X)
AgglomerativeClustering(affinity='euclidean', compute_full_tree='auto',
                        connectivity=None,
                        linkage='ward', memory=None, n_clusters=2,
                        pooling_func='deprecated')
labels = wh.labels_

print("Homogeneity: %0.3f" % metrics.homogeneity_score(labels_true, labels))
print("Normalized Mutual Information: %0.3f" % metrics.normalized_mutual_info_score(labels_true, labels))
print("Completeness: %0.3f" % metrics.completeness_score(labels_true, labels))
