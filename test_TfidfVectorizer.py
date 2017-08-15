from __future__ import division, unicode_literals

import re
import math
import input
import optkmeans
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer

import sys
reload(sys)
sys.setdefaultencoding('utf-8')
# corpus = input.read_file("filename_1000.txt", ".txt")

with open("emails.txt") as f:
	lines = f.readlines()
corpus = [x.lower().replace("yeah", "").replace("thanks", "").replace("jet", "").replace("thank you", "").replace("thank", "").replace("hello", "").replace("hi", "") for x in lines]
corpus = [x.replace("just", "").replace("um", "").replace("like", "").replace("don", "").replace("gonna", "").replace("hear", "").replace("alright", "").replace("help", "") for x in corpus]

# with open('NPS.txt') as f:
#     corpus = f.readlines()
# corpus = [re.sub(r'[^a-zA-Z ]+', '', s) for s in corpus]

# TfidfVectorizer(encoding = 'utf-8', lowercase = True, preprocessor = None, tokenizer = None, analyzer = 'word', stop_words = 'english', ngram_range = (1, 1), norm = 'l2', use_idf = True, smooth_idf = True)
vectorizer = TfidfVectorizer(min_df = 1, ngram_range = (1, 3), lowercase = True, norm = 'l2', stop_words = 'english', use_idf = True)
X = vectorizer.fit_transform(corpus)
dict_tfidf = dict(zip(vectorizer.get_feature_names(), vectorizer.idf_)) 

# Run K-Means
num_clusters = 4
kmeans = KMeans(init = 'k-means++', n_clusters = num_clusters, n_init = 10)
kmeans.fit(X)
print(kmeans.score(X))
centers = kmeans.cluster_centers_
for i in range(num_clusters):
	doc = dict(zip(vectorizer.get_feature_names(), centers[i]))
	this_document = sorted(doc.iteritems(), key = lambda (k,v) : (v,k), reverse = True)
	print("Document {} Word Frequencies:".format(i + 1))
	for word, score in this_document[:5]:
	        print("\tWord: {}, TF-IDF: {}".format(word, round(score, 5))) 

# num_restarts = 4
# (best_C, best_num) = optkmeans.kmeans_cluster(X, num_restarts)
# print("The optimal number of clusters is: {}".format(best_num))
# centers = best_C
# num_clusters = best_num

# for i in range(num_clusters):
# 	doc = dict(zip(vectorizer.get_feature_names(), centers[i]))
# 	this_document = sorted(doc.iteritems(), key = lambda (k,v) : (v,k), reverse = True)
# 	print("Document {} Word Frequencies:".format(i + 1))
# 	for word, score in this_document[:5]:
# 	        print("\tWord: {}, TF-IDF: {}".format(word, round(score, 5))) 

