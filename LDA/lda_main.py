import pandas as pd 
import numpy as np
import data
from preprocessing import Preprocessor
from vectorizer import Vectorizer
from topic_modelling import LDA
import utilities
from filter import Filter
import pyLDAvis
import pyLDAvis.sklearn
from sklearn.model_selection import GridSearchCV
import warnings
warnings.filterwarnings("ignore",category=DeprecationWarning)
import time
#make sure you're using Python 3.8+ 
from tqdm import tqdm

# df = utilities.load_processed(test=False)
# lda, topics  = utilities.test_lda(df, num_topics = 5, vect = 'cv', a = 1, b = None)
# print(topics)




df = utilities.load_processed(test=False)
lda, topics  = utilities.test_lda(df, num_topics = 20, vect = 'tf', a = 0.01, b = 0.01)
print(topics)

topics.to_csv('/Users/theojanson/Project/COVID_Twitter/topics.csv')





def grid_search():
    df = utilities.load_processed(test=False)
    N = [5,10,20]
    A = [None, 0.01, 1.0]
    B = [None, 0.01, 1.0]

    N_, A_, B_, perp, like = [],[],[],[],[]

    for i in tqdm(range(len(N))):
        for a in A: 
            for b in B: 
                lda, topics  = utilities.test_lda(df, num_topics = N[i], vect = 'tf', a = a, b = b)
                p = lda.get_perplexity()
                l = lda.get_likelihood()
                N_.append(N[i])
                A_.append(a)
                B_.append(b)
                perp.append(p)
                like.append(l)
                

    data = pd.DataFrame({'Topic Count': N_, 'Alpha': A_, 'Beta': B_, 'Perplexity': perp, 'Likelihood':like})
    data.to_csv('/Users/theojanson/Project/grid_search_results.csv')





