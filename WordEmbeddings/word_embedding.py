'''
  Author: Theodore Janson <theodore.janson@mail.mcgill.ca>
  Source Repository: https://github.com/jansont/VaccinationAnalysis
'''
from gensim.models import KeyedVectors
from gensim.models import Word2Vec
from sklearn.cluster import KMeans
from preprocessing import Preprocessor
import numpy as np
import pandas as pd
import utilities

class word_embedding:
    def __init__(self, X):
        #sg ({0, 1}, optional) – Training algorithm: 1 for skip-gram; otherwise CBOW.
        p = Preprocessor()
        tweet_list = [p.tokenise(t) for t in X]
        self.model = Word2Vec(sentences = tweet_list, min_count = 0, workers = 2, sg = 0, window = 5)
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        return np.array([
            np.mean([self.model[w] for w in texts.split() if w in self.model]
                    or [np.zeros(self.dim)], axis=0)
            for texts in X])

class K_means:
    def __init__(self, n_clusters = 8, init='random', n_init=10, max_iter=300, tol=1e-04):
        n_clusters = n_clusters
        self.model = KMeans(n_clusters = n_clusters, init = init, n_init = n_init, max_iter = max_iter, tol=tol)

    def fit_then_predict(self,X):
        y = self.model.fit_predict(X)
        return y

df = utilities.load_processed()
X = df['processed_tweet']
x_vect = word_embedding(X).model
x_vect.save("word2vec.model")
print(x_vect)
 
# km = K_means()
# y = km.fit_then_predict(x_vect)
# print(y)