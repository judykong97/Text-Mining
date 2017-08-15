from __future__ import division, unicode_literals

import re
import math
import numpy as np
import input
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer

def kmeans_cluster(X, num_restarts):
  # Variables for keeping track of the best clustering so far
  best_C = None
  best_num = None
  best_obj = -np.inf
  for i in range(num_restarts):
    # Compute the objective value
    kmeans = KMeans(init = 'k-means++', n_clusters = i + 1)
    kmeans.fit(X)
    C = kmeans.cluster_centers_
    obj = kmeans.score(X)
    print("{} clusters: {}".format(i + 1, obj))
    if obj > best_obj:
      best_C = C
      best_num = i + 1
      best_obj = obj
  return (best_C, best_num)
