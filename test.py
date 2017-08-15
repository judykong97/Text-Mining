# from __future__ import division, unicode_literals

# import math
# import tfidf
# from sklearn.cluster import KMeans
# from textblob import TextBlob as tb

# # Read in filenames
# with open("filename_1000.txt") as f:
#     files = f.readlines()
# files = [x.strip() for x in files]

# # Read file by file into the bloblist
# bloblist = []
# for (i, filename) in enumerate(files):
# 	file = open("text_1000/" + filename + ".txt")
# 	text = file.read()
# 	bloblist.append(tb(text))

# # Calculate TF-IDF of terms in files
# for i, blob in enumerate(bloblist):
#     print("Top words in document {}".format(i + 1))
#     scores = { word: tfidf.tfidf(word, blob, bloblist) for word in blob.words }
#     sorted_words = sorted(scores.items(), key = lambda x: x[1], reverse = True)
#     for word, score in sorted_words[:10]:
#         print("\tWord: {}, TF-IDF: {}".format(word, round(score, 5)))


from __future__ import division, unicode_literals

import re
import math
import optkmeans
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer

def read_file(title, format):

	# Read in filenames
	with open(title) as f:
	    files = f.readlines()
	files = [x.strip() for x in files]

	# Read file by file into the bloblist
	documents = []
	for (i, filename) in enumerate(files):
		file = open("text_1000/" + filename + format)
		text = file.read()
		documents.append(text)

	return documents

corpus = read_file("filename_1000.txt", ".txt")
# with open('NPS.txt') as f:
#     corpus = f.readlines()
# corpus = [re.sub(r'[^a-zA-Z ]+', '', s) for s in corpus]

# TfidfVectorizer(encoding = 'utf-8', lowercase = True, preprocessor = None, tokenizer = None, analyzer = 'word', stop_words = 'english', ngram_range = (1, 1), norm = 'l2', use_idf = True, smooth_idf = True)
vectorizer = TfidfVectorizer(min_df = 1, ngram_range = (1, 1), lowercase = True, norm = 'l2', stop_words = 'english', use_idf = True)
X = vectorizer.fit_transform(corpus)
dict_tfidf = sorted(dict(zip(vectorizer.get_feature_names(), vectorizer.idf_)).items(), key=lambda (k,v): (v,k))
print(dict_tfidf)
