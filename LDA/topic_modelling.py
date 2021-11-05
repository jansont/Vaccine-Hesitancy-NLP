'''
  Author: Theodore Janson <theodore.janson@mail.mcgill.ca>
  Source Repository: https://github.com/jansont/VaccinationAnalysis

  To-do: 
  - Test Gensim LDA
  - Add NMF topic modelling functionality
'''
import numpy as np
import pandas as pd
import gensim
from gensim.corpora import Dictionary
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel
from sklearn.decomposition import LatentDirichletAllocation, TruncatedSVD
from scipy.sparse import csr_matrix
import pyLDAvis
import pyLDAvis.sklearn
from pyLDAvis import gensim_models
import matplotlib.pyplot as plt
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.ERROR)
import warnings
warnings.filterwarnings("ignore",category=DeprecationWarning)

#abstract topic modelling class
class Topic_Model():
  def __init__(self):
    self.vectorizer = None
    self.num_topics = None
    self.model = None
    self.topics = None
  
  def fit(self, X=None):
    return self
  
  def get_topics(self):
    pass

class LDA(Topic_Model):
  def __init__(self, vectorizer, num_topics = 20, multicore = False, workers = 4, a=0.01, b=None):
    super().__init__()
    self.vectorizer = vectorizer
    self.num_topics = num_topics
    #gensim LDA models used with gensim vectorizer
    if type(self.vectorizer.inverse) == gensim.corpora.dictionary.Dictionary:
      if multicore == False:
        #Basic gensim LDA model
        self.model = gensim.models.ldamodel.LdaModel(corpus=self.vectorizer.features, id2word=self.vectorizer.inverse, num_topics=num_topics,
                                                    random_state=100, update_every=1,chunksize=100, passes=10,alpha=a, eta=b,
                                                    per_word_topics=True)
      else: #multicore gesim LDA model (faster)
        self.model = gensim.models.ldamulticore.LdaMulticore(corpus=self.vectorizer.features, num_topics=num_topics,
                                                            id2word=self.vectorizer.inverse, workers=workers, chunksize=2000,                                                          
                                                            passes=1, batch=False, alpha=a, eta=b,
                                                            decay=0.5, offset=1.0, eval_every=10, iterations=50,
                                                            gamma_threshold=0.001, minimum_probability=0.01,
                                                            minimum_phi_value=0.01)
    else: #Sklearn LDA model used with Sklearn vectrizers 
      self.model = LatentDirichletAllocation(n_components=num_topics, learning_method='online', doc_topic_prior = a, topic_word_prior = b,
                                             learning_decay=0.7, learning_offset=10.0, max_iter=10, batch_size=128,
                                             max_doc_update_iter=100)

  def fit(self, X=None):
    #Fit the model to the vectorized data (clustering)
    #Sklearn LDA (Gensim LDA is fitted in constructor, so return self)
    if type(self.vectorizer.features) == csr_matrix: 
      self.model = self.model.fit(self.vectorizer.features)
    return self

  def transform(self, X):
    #transform new data
    if type(self.vectorizer.features) == csr_matrix: 
      return self.model.transform(X)
    else: 
      #TODO: gensim implementation
      pass

  def predict_topic(self, X, transformed = False):
    #X = vectorized data or transformed vectorized data
    #predict topic for new data. 
    #assumes data has been preprocessed and vectorizer
    if transformed == False: 
      X = self.transform(X)
    #requires that LDA be fitted and topics obtained before call 
    topic = self.topics.iloc[np.argmax(X), :].values.tolist()
    return topic, X

  def get_perplexity(self):
    #captures log likelihood (how surprised model is of new data it has not seen before)
    #optimizing for perplexity does not necessary yield the best human iterpretable results
    #Sklearn implementation
    if type(self.vectorizer.features) == csr_matrix: 
      perplexity = self.model.perplexity(self.vectorizer.features)
    #gensim implementation
    else:
      perplexity = self.model.log_perplexity(self.vectorizer.features)
    return perplexity

  def get_likelihood(self):
    if type(self.vectorizer.features) == list:
      log_likelihood = self.model.log_perplexity(self.vectorizer.features)
    else:
      log_likelihood = self.model.score(self.vectorizer.features)
    return log_likelihood   

  def get_coherence(self):
    #Topic Coherence measures score a single topic by measuring the degree of semantic similarity
    # between high scoring words in the topic. Produces human interpretable results. 
    #TODO: SKlearn implementation
      return CoherenceModel(model=self.model, texts=self.vectorizer.features, dictionary=self.vectorizer.inverse, coherence='c_v')


  def display_results(self):
    #Display pyLDAvis Visualizer in Python Notebook 
    if type(self.vectorizer.features) == list: #gensim
      pyLDAvis.enable_notebook()
      self.visualize = pyLDAvis.gensim_models.prepare(self.model, self.vectorizer.corpus, self.vectorizer.inverse)
      return self.visualize
    else: #Sklearn 
      pyLDAvis.enable_notebook()
      panel = pyLDAvis.sklearn.prepare(self.model, self.vectorizer.features, self.vectorizer, mds='tsne')
      return self.visualize

  def get_topics(self, n_words=10):
    if type(self.vectorizer.features) == csr_matrix: #sklearn
    #Return Dataframe of keywords assoaicated with topics 
      keywords = np.array(self.vectorizer.inverse)
      topic_keywords = []
      for topic_weights in self.model.components_:
          top_keyword_locs = (-topic_weights).argsort()[:n_words]
          topic_keywords.append(keywords.take(top_keyword_locs))
      df_topic_keywords = pd.DataFrame(topic_keywords)
      df_topic_keywords.columns = ['Word '+str(i) for i in range(df_topic_keywords.shape[1])]
      df_topic_keywords.index = ['Topic '+str(i) for i in range(df_topic_keywords.shape[0])]
    else: #gensim (TODO: implement gensim)
      df_topic_keywords = None
    self.topics = df_topic_keywords
    return df_topic_keywords



class NMF(Topic_Model):
  def __init__(self):
    super().__init__()

    