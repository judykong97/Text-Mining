import input
import re
import string

import sys
reload(sys)
sys.setdefaultencoding('utf-8')

from collections import defaultdict
import gensim
from gensim import corpora, models
import nltk
from nltk.corpus import stopwords 
from nltk.stem.porter import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer

#######################################
#                Input                #
#######################################
with open("emails.txt") as f:
	lines = f.readlines()
corpus = [x.lower().replace("yeah", "").replace("thanks", "").replace("jet", "").replace("thank you", "").replace("thank", "").replace("hello", "").replace("hi", "").replace("ha", "").replace("you", "") for x in lines]
corpus = [x.replace("just", "").replace("um", "").replace("like", "").replace("don", "").replace("gonna", "").replace("hear", "").replace("alright", "").replace("help", "").replace("im", "").replace("it", "").replace("r", "").replace("t", "") for x in corpus]

documents = [x.strip()for x in corpus]


# input.read_file("filename_1000.txt", ".txt")
# with open('NPS.txt') as f:
#     documents = f.readlines()
# documents = [re.sub(r'[^a-zA-Z ]+', '', s) for s in documents]

#######################################
#            Preprocessing            #
#######################################
# remove stop words
stop = set(stopwords.words('english'))
exclude = set(string.punctuation) 
lemma = WordNetLemmatizer()
def clean(doc):
    stop_free = " ".join([i for i in doc.lower().split() if i not in stop])
    punc_free = ''.join(ch for ch in stop_free if ch not in exclude)
    normalized = " ".join(lemma.lemmatize(word) for word in punc_free.split())
    return normalized
doc_clean = [clean(doc).split() for doc in documents]   

# remove words that appear only once
frequency = defaultdict(int)
for doc in doc_clean:
	for token in doc:
		frequency[token] += 1
doc_clean = [[token for token in doc if frequency[token] > 1]
		 for doc in doc_clean]

#######################################
#     Optimization (tf-idf filter)    #
#######################################
# tf-idf transformation
dictionary = corpora.Dictionary(doc_clean)
dictionary.save('/tmp/my_test.dict')
corpus = [dictionary.doc2bow(doc) for doc in doc_clean]
corpora.MmCorpus.serialize('/tmp/deerwester.mm', corpus)
tfidf = models.TfidfModel(corpus)

# remove low tf-idf valued tokens
low_value = 0.2
low_value_words = []
for bow in corpus:
    low_value_words += [id for id, value in tfidf[bow] if value < low_value]
# dictionary.filter_tokens(bad_ids=low_value_words)
new_corpus = [dictionary.doc2bow(doc) for doc in doc_clean]

#######################################
#                 LDA                 #
#######################################
doc_term_matrix = new_corpus
# Creating the object for LDA model using gensim library
Lda = gensim.models.ldamodel.LdaModel

#######################################
#                Output               #
#######################################
num_topics = 50
# Running and Trainign LDA model on the document term matrix.
ldamodel = Lda(doc_term_matrix, num_topics = num_topics, id2word = dictionary, passes=50)
# print(ldamodel.print_topics(num_topics = num_topics, num_words=3))
for i in range(num_topics):
	print("\tTopic {}: {}".format(i + 1, ldamodel.print_topic(i, topn = 1)))



