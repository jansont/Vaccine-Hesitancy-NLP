from bertopic import BERTopic
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd 



def init_model():

	CV = CountVectorizer(ngram_range=(1,1),
	                decode_error='replace',
	                )

	TFV = TfidfVectorizer(ngram_range=(1,1),
	                decode_error='replace',
	                )

	model = BERTopic(top_n_words = 10,
	                 n_gram_range = (1,1),
	                 min_topic_size = 10,
	                 nr_topics = 15,
	                 vectorizer_model = CV,
	                 verbose = True,
	                 )
	return model


def merge_dfs(dates: list, path: str):
  errors = []
  files = [i for i in dates]
  tweets = []
  for i in files:
    try: 
      df = pd.read_csv(path+f'Processed_Tweets_{i}.csv', encoding='latin-1')
      tw = df['ProcessedTweet']
      tw = tw.map(lambda t: str(t))
      tw = tw.to_list()
      tweets += tw
    except: 
      errors.append(i)
  return pd.DataFrame({'ProcessedTweet':tweets}), errors


path = '/content/drive/My Drive/Capstone/Processed_Datasets/'
tweets, errors = merge_dfs([i for i in range(262, 269)], path)

tweets = tweets['ProcessedTweet']
docs = tweets.to_list()
del tweets
model = init_model()
topics, probabilities = model.fit_transform(docs)

path = '/content/drive/My Drive/Capstone/TopicModel'
model.save(path)