from sklearn.datasets import fetch_20newsgroups
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer
from sklearn import metrics
from sklearn.cluster import MeanShift, estimate_bandwidth
import numpy as np

print(__doc__)
categories = [
    'alt.atheism',
    'talk.religion.misc',
    'comp.graphics',
    'sci.space',
]

dataset = fetch_20newsgroups(subset='all', categories=categories,
                             shuffle=True, random_state=42)
labels_true = dataset.target
true_k = np.unique(labels_true).shape[0]
# t0 = time()
vectorizer = TfidfVectorizer(max_df=0.5, max_features=10000,
                                 min_df=2, stop_words='english',
                                 use_idf=True)
X = vectorizer.fit_transform(dataset.data)
svd = TruncatedSVD(3)
normalizer = Normalizer(copy=False)
lsa = make_pipeline(svd, normalizer)
X = lsa.fit_transform(X)
# explained_variance = svd.explained_variance_ratio_.sum()

# The following bandwidth can be automatically detected using
bandwidth = estimate_bandwidth(X, quantile=0.3, n_samples=500)

Meanshift = MeanShift(bandwidth=bandwidth, bin_seeding=True).fit(X)
# ms.fit(data)
labels = Meanshift.labels_
# cluster_centers = ms.cluster_centers_

labels_unique = np.unique(labels)
n_clusters_ = len(labels_unique)
print("Homogeneity: %0.3f" % metrics.homogeneity_score(labels_true, labels))
print("Completeness: %0.3f" % metrics.completeness_score(labels_true, labels))
print("Normalized Mutual Information: %0.3f" % metrics.normalized_mutual_info_score(labels_true, labels))
