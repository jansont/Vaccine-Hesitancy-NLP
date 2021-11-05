'''
  Author: Theodore Janson <theodore.janson@mail.mcgill.ca>
  Source Repository: https://github.com/jansont/VaccinationAnalysis
'''
import nltk
import gensim
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import HashingVectorizer
from scipy.sparse import csr_matrix
from gensim.models import TfidfModel
from gensim.corpora import Dictionary
import gensim.corpora as corpora
from nltk.corpus import stopwords
import numpy as np
# nltk.download('stopwords') 


class Vectorizer():
  def __init__(self, corpus, vectorizer_type = 'tf', strip_accents='ascii', stops=stopwords.words("english"),
   nGrams=(1,1), max_df=1.0, min_df=1, max_features=None, n_features=1048576):
    '''
    vectorizer: string (cv, tfidf, hash)
    strip_accents: 'ascii' or None
    ngram_range: tuple (2,2) for bigrams only
    '''
    #Sklearn hashing vectorizer
    if vectorizer_type == 'hash':
      self.model = HashingVectorizer(strip_accents=strip_accents, ngram_range=nGrams,n_features=n_features)
      self.features = self.model.fit_transform(corpus)
      self.inverse = None

    #Sklearn Count vectorizer
    elif vectorizer_type == 'cv':
      self.model = CountVectorizer(strip_accents=strip_accents, ngram_range=nGrams, max_df=max_df, min_df=min_df, max_features=max_features)
      self.features = self.model.fit_transform(corpus)
      self.inverse = self.model.get_feature_names()

    #Sklearn TF-IDF vectorizer
    elif vectorizer_type == 'tf':
      self.model = TfidfVectorizer(strip_accents=strip_accents,ngram_range=nGrams,max_df=max_df, min_df=min_df, max_features=max_features)
      self.features = self.model.fit_transform(corpus)
      self.inverse = self.model.get_feature_names()

    #Gensim Bag of Words vectorizer
    elif vectorizer_type == 'bow':
      tweets_tokenized = [nltk.word_tokenize(tweet) for tweet in corpus]
      self.inverse = corpora.Dictionary(tweets_tokenized)
      self.features = [self.inverse.doc2bow(line) for line in tweets_tokenized]

    #Gensim TF-IDF Vectorizer
    elif vectorizer_type == 'gensim_tf':
      tweets_tokenized = [nltk.word_tokenize(tweet) for tweet in corpus]
      self.inverse = corpora.Dictionary(tweets_tokenized)
      corpus_bow = [self.inverse.doc2bow(line) for line in tweets_tokenized]  # convert corpus to BoW format
      self.features = TfidfModel(corpus=corpus_bow, id2word=self.inverse, dictionary=None)

  def transform(self, X):
    #For gensim models 
    if type(self.vectorizer.features) == list:
      tweets_tokenized = [nltk.word_tokenize(tweet) for tweet in X]
      dct = Dictionary(tweets_tokenized)  # fit dictionary
      corpus = [dct.doc2bow(line) for line in tweets_tokenized]
      return self.vectorizer[corpus]
    #For SKlearn models
    else: 
      return self.vectorizer.transform(X)

  def sparsity(self):
    #sklearn sparcity
    if type(self.features) == csr_matrix:
      density = self.features.todense()
      sparcity = ((density > 0).sum()/density.size)*100
      return sparcity
    #gensim sparsity
    else: 
      #num unique words
      vocab_size = self.inverse.num_nnz
      return vocab_size / (self.inverse.num_docs * self.inverse.num_pos)*100

    

   
