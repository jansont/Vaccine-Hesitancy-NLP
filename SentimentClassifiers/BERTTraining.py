# !pip install  transformers
# !pip  install emoji
# !pip install scikit-optimize
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
import transformers
from transformers import BertModel, BertTokenizer, DistilBertTokenizer, RobertaModel, RobertaTokenizer
from transformers import AutoConfig, AutoModel, AdamW, get_linear_schedule_with_warmup
import torch
from torch import nn, optim
from torch.utils.data import Dataset, random_split, DataLoader, RandomSampler, SequentialSampler
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, precision_score, recall_score, f1_score
from sklearn.utils import shuffle
from skopt.space import Real, Categorical
from skopt import gp_minimize
from skopt.utils import use_named_args
from google.colab import drive
from time import sleep
from psutil import virtual_memory
from tqdm import tqdm
import emoji



#split dataset into training, test, and validation
def get_time(elapsed):
    return str(datetime.timedelta(seconds=int(round((elapsed)))))


def split_dataset(df, training_prop = 0.8, test_prop = 0.5):
  training, test_validation = train_test_split(df, 
                                      test_size= ( 1 - training_prop ), 
                                      stratify=df['Score'].values)

  validation, test = train_test_split(test_validation, 
                                   test_size=test_prop, 
                                   stratify=test_validation['Score'].values)
  return training, validation, test

def get_device():
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  if (device.type == 'cuda'):
    print('Using cuda GPU')
  return device


class TweetDataset(Dataset):
  def __init__(self, df, max_length = 64):
    self.tweets = df['ProcessedTweet'].to_numpy()
    self.sentiment = df['Score'].to_numpy() 
    self.max_length = max_length
    self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

  def __len__(self):
    return len(self.tweets)

  def __getitem__(self, i):
        
    statement = str(self.tweets[i])
    label = self.sentiment[i]

    encoding = self.tokenizer.encode_plus(statement,max_length=self.max_length,padding='max_length',add_special_tokens=True, 
        return_token_type_ids=False,truncation=True,return_attention_mask=True,return_tensors='pt'  
    ) 

    return {'statement_text': statement,
        'input_ids': encoding['input_ids'].flatten(),
        'attention_mask': encoding['attention_mask'].flatten(),
        'labels': torch.tensor(label, dtype=torch.long)
    }


class UnlabelledTweetDataset(Dataset):
  def __init__(self, df, max_length = 64):
    self.tweets = df['ProcessedTweet'].to_numpy()
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
    def __init__(self, n_classes = 2, dropout_prob = 0.1):
        super(BERTSentimentClassifier, self).__init__()
        self.model = BertModel.from_pretrained('bert-base-uncased')
        self.drop = nn.Dropout(dropout_prob)
        self.output = nn.Linear(self.model.config.hidden_size, n_classes)
        
    def forward(self, input_ids, attention_mask):
        _, pooled_output = self.model(
            input_ids = input_ids,
            attention_mask=attention_mask, return_dict=False
        )

        output = self.drop(pooled_output)
        
        return self.output(output)

class SentimentClassifierTrainer():

  def __init__(self, model, device, learning_rate, weight_decay, epochs = 8): 
    self.model = model
    self.model = self.model.to(device)
    self.device = device
    self.epochs = epochs
    self.learning_rate = learning_rate 
    self.weight_decay = weight_decay
    self.loss_function = nn.CrossEntropyLoss().to(self.device)

  
  def train_fold(self, dataset, model_save_name, batch_size, Print = True):

      training_set, validation_set, test_set = dataset
      
      start_time = time.time()
      performance = defaultdict(list)
      best_accuracy = 0

      for epoch in range(self.epochs):
          epoch_start_time = time.time()
          print('Epoch ', epoch+1, '/', self.epochs)
          print('-'*50)

          training_output = self.train_model(training_set, batch_size = batch_size)

          train_acc, train_loss, train_precision, train_recall, train_f1 = training_output

          val_acc, val_loss, val_precision, val_recall, val_f1 = self.eval_model(validation_set, batch_size = batch_size)
        
          performance['train_accuracy'].append(train_acc)
          performance['train_loss'].append(train_loss)
          performance['train_precision'].append(train_precision)
          performance['train_recall'].append(train_recall)
          performance['train_f1'].append(train_f1)


          performance['val_accuracy'].append(val_acc)
          performance['val_loss'].append(val_loss)
          performance['val_precision'].append(val_precision)
          performance['val_recall'].append(val_recall)
          performance['val_f1'].append(val_f1)

          if epoch == 10: 
            gpu_info()

          if val_acc > best_accuracy:
              torch.save(self.model.state_dict(), model_save_name)
              best_accuracy = val_acc

          if (epoch+1 == self.epochs) or Print:
            print('Training ->  Loss: ', train_loss, ' | ', ' Accuracy: ', train_acc)
            print('Validation ->  Loss: ', val_loss, ' | ', ' Accuracy: ', val_acc, ' | ', ' Recall: ', val_recall, ' | ', ' Precision: ', val_precision, ' | ', ' F1: ', val_f1)
            print('Epoch Train Time: ', get_time(time.time() - epoch_start_time))
            print('\n')
          
      if Print:
        print('Finished Training.')   
        print('Fold Train Time: ', get_time(time.time() - start_time))
        print('\n')
      return performance


  def train_model(self, dataset, batch_size):

      data_loader = DataLoader(dataset, batch_size, num_workers = 4)

      training_steps = len(data_loader.dataset) * self.epochs

      optimizer = AdamW(self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay, correct_bias=True)

      warmup_steps = int(0.20 * training_steps)
      scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=training_steps)
      
      self.model = self.model.train()
      losses = []
      correct_preds = 0
      complete_preds = []
      complete_labels = []
      
      for _,batch in tqdm(enumerate(data_loader)):
          input_ids = batch['input_ids'].to(self.device)
          attention_mask = batch['attention_mask'].to(self.device)
          labels = batch['labels'].to(device)
          
          outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
              
          _, preds = torch.max(outputs, dim=1)
          loss = self.loss_function(outputs, labels)
          complete_preds.append(preds.data.cpu().numpy().tolist())
          complete_labels.append(labels.data.cpu().numpy().tolist())
          correct_preds += torch.sum(preds == labels)
          losses.append(loss.item())
          
          loss.backward()
          nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
          optimizer.step()
          scheduler.step()
          optimizer.zero_grad()
      
      complete_preds_flat = [x for y in complete_preds for x in y]
      complete_labels_flat = [x for y in complete_labels for x in y]
      acc_score = accuracy_score(y_true=complete_labels_flat, y_pred=complete_preds_flat)

      precision = precision_score(y_true=complete_labels_flat, y_pred=complete_preds_flat)
      recall = recall_score(y_true=complete_labels_flat, y_pred=complete_preds_flat)
      f1 = f1_score(y_true=complete_labels_flat, y_pred=complete_preds_flat)

      print('Testing ->  Loss: ', np.mean(losses), ' | ', ' Accuracy: ', acc_score, ' | ', ' Recall: ', recall, ' | ', ' Precision: ', precision, ' | ', ' F1: ', f1)

      return_items = (acc_score, 
                  np.mean(losses), 
                  precision,
                  recall,
                  f1)
      
      return return_items

  def eval_model(self, dataset, batch_size):
      self.model = self.model.eval()
      
      losses = []
      correct_preds = 0
      complete_preds = []
      complete_labels = []
      complete_outputs = []

      data_loader = DataLoader(dataset, batch_size, num_workers = 4)
      
      with torch.no_grad():
          for item in data_loader:
              input_ids = item['input_ids'].to(self.device)
              attention_mask = item['attention_mask'].to(self.device)
              labels = item['labels'].to(self.device)

              outputs = self.model(input_ids=input_ids, 
                              attention_mask=attention_mask)
                            
              _, preds = torch.max(outputs, dim=1)
              loss = self.loss_function(outputs, labels)
              
              correct_preds += torch.sum(preds == labels)
              complete_preds.append(preds.data.cpu().numpy().tolist())
              complete_labels.append(labels.data.cpu().numpy().tolist())
              complete_outputs.append(outputs.tolist())
              losses.append(loss.item())
          
      accuracy = correct_preds.double() / len(dataset)
      complete_preds_flat = [x for y in complete_preds for x in y]
      complete_labels_flat = [x for y in complete_labels for x in y]
      complete_outputs_flat = [x for y in complete_outputs for x in y]

      acc_score = accuracy_score(y_true=complete_labels_flat, y_pred=complete_preds_flat)

      precision = precision_score(y_true=complete_labels_flat, y_pred=complete_preds_flat)
      recall = recall_score(y_true=complete_labels_flat, y_pred=complete_preds_flat)
      f1 = f1_score(y_true=complete_labels_flat, y_pred=complete_preds_flat)

      return_items = (acc_score, 
                  np.mean(losses), 
                  precision,
                  recall,
                  f1)

      return return_items


  def k_fold_cross_validation(self, n_folds, df, model_save_name, batch_size, Print = True):

    training_performance_history = []
    testing_performance_history = []
    start_time = time.time()

    fold = 0

    train_df, val_df, test_df = split_dataset(df, training_prop = 0.8, test_prop = 0.5)
    test_df = pd.concat([test_df, val_df])

    x_train = train_df['ProcessedTweet']
    y_train = train_df['Score']

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True)

    print(f'Training model with {n_folds} cross validation...')

    for train_index, val_index in skf.split(x_train, y_train):
      print(f'Fold: {fold+1}')
          
      x_ktrain = x_train.iloc[train_index]
      y_ktrain = y_train.iloc[train_index]
      x_kval = x_train.iloc[val_index]
      y_kval = y_train.iloc[val_index]
          
      train = pd.DataFrame(list(zip(x_ktrain, y_ktrain)), columns = ['ProcessedTweet', 'Score'])
      val = pd.DataFrame(list(zip(x_kval, y_kval)), columns = ['ProcessedTweet', 'Score'])

      train_ds = TweetDataset(train)
      val_ds = TweetDataset(val)
      test_ds = TweetDataset(test_df)

      dataset = (training_set, validation_set, test_set)
          
      training_performance = self.train_fold(dataset, model_save_name=model_save_name, batch_size = batch_size, Print = Print)

      training_performance_history.append(training_performance)
     
      acc_score, loss, precision, recall, f1 = self.eval_model(test_ds, batch_size = batch_size)
      testing_performance = defaultdict(list)
      testing_performance['Accuracy'] = acc_score
      testing_performance['Loss'] = loss
      testing_performance['Recall'] = recall
      testing_performance['Precison'] = precision
      testing_performance['F1'] = f1

      testing_performance_history.append(testing_performance)
      
      fold += 1

      print(str(n_folds), 'Fold CV Train Time: ', get_time(time.time() - start_time))
      print(f'Accuracy on last Fold: {acc_score}')

    return training_performance_history, testing_performance_history


  def save_model(self, PATH = '/content/drive/My Drive/Capstone/Code/Models/manual_model_save.pth'):
    torch.save(self.model.state_dict(), PATH)

  def load_model(self, model = BERTSentimentClassifier(2), PATH = '/content/drive/My Drive/Capstone/Code/Models/auto_model_save.pth'):
    model.load_state_dict(torch.load(PATH))
    self.model = model
    self.model = self.model.to(device)


def hyperparameter_search(dataset, num_points = 15):

  search_space = [Real(1e-6, 1e-2, name='learning_rate', prior = 'log-uniform',  transform = 'identity', base = 10), 
                Categorical([8,16,32,64], name='batch_size' , prior = [0.25, 0.35, 0.20, 0.20], transform = 'identity'), 
                Real(1e-4, 1e0, name='weight_decay', prior = 'log-uniform',  transform = 'identity'), 
                Real(0.01, 0.75, name='dropout_prob', prior = 'uniform',  transform = 'identity')]
  
  @use_named_args(dimensions=search_space)
  def evaluate_model(learning_rate, batch_size, weight_decay,dropout_prob): 

    training_histories = []
    testing_histories = []

    batch_size = int(batch_size)
    bert_model = BERTSentimentClassifier(dropout_prob = dropout_prob)
    device = get_device()
    Trainer = SentimentClassifierTrainer(bert_model, device, epochs = 4, learning_rate = learning_rate, weight_decay = weight_decay)
    PATH  = f'/content/drive/My Drive/Capstone/Code/Models/hypertune_bs{batch_size}_lr{learning_rate}_drop{dropout_prob}_wd{weight_decay}.pth'
    train_history, test_history = Trainer.k_fold_cross_validation(n_folds = 2,
                                                                  df = df,
                                                                  model_save_name = PATH,
                                                                  batch_size = batch_size, 
                                                                  Print = False)
    acc = np.mean([test_history[i]['Accuracy'] for i in range(len(test_history))])
    torch.cuda.empty_cache()
    return 1.0 - acc

  results = gp_minimize(evaluate_model, search_space, n_calls = num_points, n_initial_points = 3, x0 = [2e-5, 32, 0.01, 0.1])

  return train_history, test_history, results


def read_results(result):
  best_hyperparameters = result.x
  best_f1 = result.fun
  evaluated_hyperparameters = result.x_iters
  return best_hyperparameters, best_f1, evaluated_hyperparameters


def gpu_info():
  gpu_info = !nvidia-smi
  gpu_info = '\n'.join(gpu_info)
  if gpu_info.find('failed') >= 0:
    print('Not connected to a GPU')
  else:
    print(gpu_info)
  ram_gb = virtual_memory().total / 1e9
  print('Your runtime has {:.1f} gigabytes of available RAM\n'.format(ram_gb))
