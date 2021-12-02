
# !pip install  transformers
# !pip  install emoji
# !pip install reverse_geocoder
import numpy as np
import pandas as pd 
import os
import re
import time
import datetime
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.stem import SnowballStemmer
from collections import Counter, defaultdict
from collections import Counter, defaultdict
import transformers
from transformers import BertModel, BertTokenizer
from transformers import AutoConfig
import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder
from psutil import virtual_memory
from sklearn.utils import shuffle
from google.colab import drive
from tqdm import tqdm
import json
import time as time
from ast import literal_eval
import statistics
import matplotlib.pyplot as plt
import math
from time import sleep
import reverse_geocoder
import requests
from tqdm import tqdm
import emoji
from shapely.geometry import mapping, shape
from shapely.prepared import prep
from shapely.geometry import Point


class UnlabelledTweetDataset(Dataset):
  def __init__(self, df, max_length = 64):
    self.tweets = df['Non Lemmatized Preprocess'].to_numpy()
    self.max_length = max_length
    self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

  def __len__(self):
    return len(self.tweets) 

  def __getitem__(self, i):
        
    statement = str(self.tweets[i])

    encoding = self.tokenizer.encode_plus(statement,max_length=self.max_length,padding='max_length',add_special_tokens=True, 
        return_token_type_ids=False,truncation=True,return_attention_mask=True,return_tensors='pt'  
    ) 

    return {'statement_text': statement,
        'input_ids': encoding['input_ids'].flatten(),
        'attention_mask': encoding['attention_mask'].flatten(),
    }
        
class BERTSentimentClassifier(nn.Module):
    def __init__(self, n_classes = 2):
        super(BERTSentimentClassifier, self).__init__()
        self.model = BertModel.from_pretrained('bert-base-uncased')
        self.drop = nn.Dropout(0.1)
        self.output = nn.Linear(self.model.config.hidden_size, n_classes)
        
    def forward(self, input_ids, attention_mask):
        _, pooled_output = self.model(
            input_ids = input_ids,
            attention_mask=attention_mask, return_dict=False
        )

        output = self.drop(pooled_output)
        
        return self.output(output)



class SentimentPredictor():
    def __init__(self, model_path, device, n_classes = 2): 
      self.model = BERTSentimentClassifier(n_classes = 2)
      self.model.load_state_dict(torch.load(model_path))

      self.model = self.model.to(device)
      self.device = device

    def predict(self, dataset, batch_size = 32):
        self.model = self.model.eval()
        
        complete_preds = []
        complete_outputs = []

        data_loader = DataLoader(dataset, batch_size, num_workers = 4)
        
        with torch.no_grad():
            for item in tqdm(data_loader):
                input_ids = item['input_ids'].to(self.device)
                attention_mask = item['attention_mask'].to(self.device)

                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                
                _, preds = torch.max(outputs, dim=1)
                
                complete_preds.append(preds.data.cpu().numpy().tolist())
                complete_outputs.append(outputs.tolist())
        
            complete_preds_flat = [x for y in complete_preds for x in y]
            complete_outputs_flat = [x for y in complete_outputs for x in y]
            
            complete_preds_flat = np.array(complete_preds_flat)
            complete_preds_flat[complete_preds_flat == 0] = -1
            return_items = (complete_preds_flat, complete_outputs_flat)
            
            return return_items

def load_data(file, test = False):
  df = pd.read_csv(file, encoding='latin-1',  engine='python' )
  if test: 
    df = df.iloc[0:5000] 
  if 'processed_tweet' in df.columns:
    df = df.rename(columns={'processed_tweet': 'ProcessedTweet'})
  df = shuffle(df)
  return df

def get_confidence(outputs):
  return [(max(out) - min(out))/2 for out in outputs]

def get_probability(outputs):
  #softmax
  outputs = np.array(outputs)
  softmax = lambda x: np.exp(x) / np.sum(np.exp(x), axis=0)
  return np.array([softmax(y) for y in outputs])

def write_predictions(df, filename, path):
  df.to_csv(path+filename)

def get_device():
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  if (device.type == 'cuda'):
    print('Using cuda GPU')
  return device

data = requests.get("https://raw.githubusercontent.com/datasets/geo-countries/master/data/countries.geojson").json()
countries = {}
for feature in data["features"]:
    geom = feature["geometry"]
    country = feature["properties"]["ADMIN"]
    countries[country] = prep(shape(geom))

def get_country(lon, lat):
    point = Point(lon, lat)
    for country, geom in countries.items():
        if geom.contains(point):
            return country
    return "unknown"

def get_state(lon, lat):
  US_state = []
  if (get_country(lon, lat) == 'United States of America'):
    state_names = ["Alaska", "Alabama", "Arkansas", "Arizona", "California", "Colorado", "Connecticut", "District of Columbia",
                   "Delaware", "Florida","Georgia", "Hawaii", "Iowa", "Idaho", "Illinois", "Indiana", "Kansas", "Kentucky", "Louisiana",
                   "Massachusetts", "Maryland", "Maine", "Michigan", "Minnesota","Missouri", "Mississippi", "Montana", "North Carolina",
                   "North Dakota", "Nebraska", "New Hampshire", "New Jersey", "New Mexico", "Nevada", "New York", "Ohio", "Oklahoma", "Oregon",
                   "Pennsylvania", "Rhode Island", "South Carolina","South Dakota", "Tennessee", "Texas", "Utah", "Virginia", "Vermont", "Washington",
                   "Wisconsin", "West Virginia", "Wyoming"]

    state_abr = ['AK', 'AL', 'AR', 'AZ', 'CA', 'CO', 'CT', 'DC', 'DE', 'FL', 'GA', 'HI', 'IA', 'ID', 'IL', 'IN', 'KS', 'KY', 'LA', 'MA', 'MD', 
              'ME', 'MI', 'MN', 'MO', 'MS', 'MT', 'NC', 'ND', 'NE', 'NH', 'NJ', 'NM', 'NV', 'NY', 'OH', 'OK', 'OR', 'PA', 'RI', 'SC', 'SD',
              'TN', 'TX', 'UT', 'VA', 'VT', 'WA', 'WI', 'WV', 'WY']
    results = reverse_geocoder.search((lat, lon))
    if results[0]['admin1'] in state_names:
      country = (results[0]['admin1'])
      i = state_names.index(country)
      return state_abr[i]
  else: 
      return np.NaN


def format_time(dataset):
  dataset['Datetime'] = dataset['Datetime'].map(lambda d: str(d).split(':', 1)[0][:-3])
  month = [str(d).split(':', 1)[0][-6:-3] for d in dataset['Datetime']]
  dataset['Month'] = month
  return dataset

def get_geo_dataset(df):
  geo_dataset = df[df['Coordinates'].notnull()]
  countries_, US_states = [], []
  for i,row in geo_dataset.iterrows():
    long_lat = row['Coordinates']
    location = long_lat.split('[', 1)[1][:-2]
    location = literal_eval(location)
    latitude, longitude = location[0], location[1]
    c = get_country(latitude, longitude) 
    countries_.append(c)
    state = get_state(latitude, longitude)
    US_states.append(state)
  geo_dataset['Country'] = countries_
  geo_dataset['US State'] = US_states
  return geo_dataset



def predict_and_get_stats(file_number):

  model_file = 'final_model.pth'
  model_path = '/content/drive/My Drive/Capstone/Code/Models/'
  model_path += model_file
  device = get_device()
  model = SentimentPredictor(model_path = model_path, device = device)

  file = f'Processed_Tweets_{file_number}.csv'
  filepath = f'/content/drive/My Drive/Capstone/Processed_Datasets/'
  df = load_data(filepath+file, test = False)


  #PREPROCESS_DATA WITHOUT LEMMATIZATION
  P = Preprocessor()
  processed_tweets = P.pipeline(df['Tweet'].copy())
  df['Non Lemmatized Preprocess'] = processed_tweets
  dataset = UnlabelledTweetDataset(df)

  #PREDICT
  start = time.time()
  predictions, outputs = model.predict(dataset)

  df['Sentiment'] = predictions
  probability = get_probability(outputs)
  df['Positive_Probability'] = probability[:,1]
  df['Negative_Probability'] = probability[:,0]

  print('n/ n/')
  print(f'Process Time: {time.time()-start}')

  #SAVE PREDICTIONS
  df = format_time(df)
  df.to_csv(f'/content/drive/My Drive/Capstone/Predicted_Datasets/Data/Predicted_Tweets_{file_number}.csv')


  
  cols = ['ID', 'Datetime', 'Tweet', 'Coordinates', 'GeoTag', 'ProcessedTweet',
          'Non Lemmatized Preprocess','Sentiment','Positive_Probability','Negative_Probability',
          'Month','Country','US State']
  geo_dataset = get_geo_dataset(df)
  geo_dataset = geo_dataset[cols]
  total_geo_dataset = pd.read_csv('/content/drive/My Drive/Capstone/Predicted_Datasets/GeoData/Predicted_Geo_Tweets.csv', usecols = cols)
  total_geo_dataset = pd.concat([total_geo_dataset, geo_dataset], axis=0)

  total_geo_dataset.to_csv('/content/drive/My Drive/Capstone/Predicted_Datasets/GeoData/Predicted_Geo_Tweets.csv')

  #GET STATS
  sentiment = np.array(df['Sentiment'])
  positive_prob = np.array(df['Positive_Probability'])
  negative_prob = np.array(df['Negative_Probability'])

  #prediction
  sample_size = len(df)
  mean_prediction = np.mean(sentiment)
  variance_prediction = np.var(sentiment)
  std_prediction = np.std(sentiment)
  mode_prediction = statistics.mode(sentiment)

  #positive_probability
  mean_positive_probability = np.mean(positive_prob)
  variance_positive_variance = np.var(positive_prob)
  std_positive_probability = np.std(positive_prob)
  median_positive_probability = np.median(positive_prob)

  #negative_probability
  mean_negative_probabiliy = np.mean(negative_prob)
  variance_negative_variance = np.var(negative_prob)
  std_negative_probability = np.std(negative_prob)
  median_negative_probability = np.median(negative_prob)

  # fig = plt.figure(figsize=(15, 5))
  # fig.add_subplot(121)
  counts_5_bins, _, _ = plt.hist(positive_prob, bins = 5)
  # plt.ylabel('Counts'), plt.xlabel('Probability of positive sentiment'), plt.title('5 Bins')
  # fig.add_subplot(122)
  density_auto_bins, bins_auto, _ = plt.hist(positive_prob, bins = 'auto', density=True)
  # plt.ylabel('Density'), plt.xlabel('Probability of positive sentiment'), plt.title('Sample Sentiment Distribution')
  # plt.show()

  #export into JSON
  json_dict = {
      'sample_size': int(sample_size),
      'mean_prediction': float(mean_prediction),
      'variance_prediction': float(variance_prediction),
      'std_prediction': float(std_prediction),
      'mode_prediction': float(mode_prediction),
      'mean_positive_probability': float(mean_positive_probability),
      'variance_positive_variance': float(variance_positive_variance),
      'std_positive_probability': float(std_positive_probability),
      'median_positive_probability': float(median_positive_probability),
      'mean_negative_probabiliy': float(mean_negative_probabiliy),
      'variance_negative_variance': float(variance_negative_variance),
      'std_negative_probability': float(std_negative_probability),
      'median_negative_probability': float(median_negative_probability),
      '5_level_positive_probability' : [float(x) for x in counts_5_bins],
      'density' : [float(x) for x in density_auto_bins],
      'bin_boundaries' : [float(x) for x in bins_auto]
    }

  json_path = f'/content/drive/My Drive/Capstone/Predicted_Datasets/DailyStats/Day_{file_number}_statistics.json'
  with open(json_path, 'w') as fp:
      json.dump(json_dict, fp)

  return_vals = [sample_size,
                  mean_prediction,
                  variance_prediction,
                  std_prediction,
                  mean_positive_probability, 
                  variance_positive_variance, 
                  std_positive_probability]
  return return_vals


def write_new_data(files):
  sampSize, mean_pred, var_pred, std_pred, mean_pos, var_pos, std_pos = [],[],[],[],[],[],[]

  for file_number in files: 
    errors = []
    try: 
      stats = predict_and_get_stats(file_number)
      sampSize.append(stats[0])
      mean_pred.append(stats[1])
      var_pred.append(stats[2])
      std_pred.append(stats[3])
      mean_pos.append(stats[4])
      var_pos.append(stats[5])
      std_pos.append(stats[6])

      stat_dict = {'Day':files,
                  'SampleSize':sampSize,
                  'MeanSentiment':mean_pred,
                  'VarSentiment':var_pred,
                  'StdSentiment':std_pred,
                  'MeanPositiveProb':mean_pos,
                  'VarPositiveProb':var_pos,
                  'StdPositiveProb':std_pos}

      stats_df = pd.DataFrame(stat_dict)
      cols = ['Day','SampleSize',	'MeanSentiment','VarSentiment','StdSentiment','MeanPositiveProb','VarPositiveProb','StdPositiveProb']
      total_data = pd.read_csv('/content/drive/My Drive/Capstone/Predicted_Datasets/DailyStats.csv', usecols = cols)
      total_data = pd.concat([total_data, stats_df], axis=0)
      total_data.to_csv('/content/drive/My Drive/Capstone/Predicted_Datasets/DailyStats.csv')

    except:
      errors.append(file_number)
  return errors




