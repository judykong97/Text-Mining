import input

from collections import defaultdict
from gensim import corpora, models, similarities
from pprint import pprint  # pretty-printer

# documents = ["Human machine interface for lab abc computer applications",
#              "A survey of user opinion of computer system response time",
#              "The EPS user interface management system",
#              "System and human system engineering testing of EPS",
#              "Relation of user perceived response time to error measurement",
#              "The generation of random binary unordered trees",
#              "The intersection graph of paths in trees",
#              "Graph minors IV Widths of trees and well quasi ordering",
#              "Graph minors A survey"]

documents = input.read_file("raw-filename.txt", ".txt")

# remove common words and tokenize
stoplist = set('for a of the and to in'.split())
texts = [[word for word in document.lower().split() if word not in stoplist]
         for document in documents]

# remove words that appear only once
frequency = defaultdict(int)
for text in texts:
	for token in text:
		frequency[token] += 1
texts = [[token for token in text if frequency[token] > 1]
		 for text in texts]
# pprint(texts)

dictionary = corpora.Dictionary(texts)
dictionary.save('/tmp/my_test.dict')
# print(dictionary)
# print(dictionary.token2id)

corpus = [dictionary.doc2bow(text) for text in texts]
corpora.MmCorpus.serialize('/tmp/deerwester.mm', corpus)  # store to disk, for later use
print(corpus)

tfidf = models.TfidfModel(corpus)
vec = [(0, 1), (4, 1)]
# print(tfidf[vec])

index = similarities.SparseMatrixSimilarity(tfidf[corpus], num_features=118)
sims = index[tfidf[vec]]
# print(list(enumerate(sims)))




