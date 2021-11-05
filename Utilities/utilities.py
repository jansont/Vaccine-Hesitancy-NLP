
'''
  Author: Theodore Janson <theodore.janson@mail.mcgill.ca>
  Source Repository: https://github.com/jansont/VaccinationAnalysis
'''
import pandas as pd 
import numpy as np
from sklearn.model_selection import GridSearchCV
from preprocessing import Preprocessor
from filter import Filter
from topic_modelling import LDA
from vectorizer import Vectorizer
import data

#Filter and preprocess the data
def filter_and_process(day = '241', path = data.datapath, process = None):
    processed_path = path+day+'/'+day+'_processed.csv'  #processed data (save file)
    filt = Filter()
    df, t, error_count = filt.create_dataset(day, test = False, process = None)
    #if the texts were not processed as they were filtered
    if process != 'online':
        P = Preprocessor()
        processed_tweets = P.pipeline(df['Tweets'].copy())
        df['processed_tweet'] = processed_tweets
    #save the results in directoy 
    f2 = day+'/'+day+'_processed.csv'
    df.to_csv(path+f2)
    return df, t, error_count

def load_processed(path = data.datapath, test = False, day = '241'):
    #get processed data
    f = day+'/'+day+'_processed.csv'
    if test:
        df = pd.read_csv(path+f, nrows=50000)
        print('Loaded Processed Data')
    else:
        df = pd.read_csv(path+f)
    return df[df['processed_tweet'].notna()]

def test_lda(df, num_topics = 15, a='symmetric',b=None, multicore = False, vect = 'tf'):
    print('\nTesting LDA...')
    #test SKLearn LDA
    #TODO: test gensim LDA
    df['processed_tweet'] = df['processed_tweet'].apply(lambda x: np.str_(x))
    v = Vectorizer(df['processed_tweet'], vectorizer_type= vect, max_df=0.95, min_df=10)
    lda = LDA(v, num_topics = num_topics, a=a, b=b, multicore = multicore).fit()
    topics = lda.get_topics()
    return lda, topics 

def lda_grid_search(df, search=data.LDA_SEARCH_PARAMS):
    #TODO: fix errors (pass scoring to grid search)
    df['processed_tweet'] = df['processed_tweet'].apply(lambda x: np.str_(x))
    v = Vectorizer(df['processed_tweet'], vectorizer_type='cv', max_df=0.95, min_df=10)
    lda = LDA(v)
    model = GridSearchCV(lda.model, param_grid=search)
    model.fit(lda.vectorizer.features)
    best_lda_model = model.best_estimator_
    best_params = model.best_params_
    print(f'Best LDA parameters: {best_params}')
    best_score = model.best_score_
    return best_lda_model, best_params, best_score
