import json
import time
import pandas as pd
import numpy as np
from tqdm import tqdm
import Constants
from preprocessor import Preprocessor


def filter_CSV(path, keep_retweets = False):
    '''
    Path: file path of JSON file type
    keep_retweets: Do we want to retain retweets? (Boolean)
    Preprocess: Do we want to preprocess as we go? (Boolean)
    test: Limit filtered tweets to 100 (Boolean)
    
    @Returns
    df: Table with relevant fields for each Tweet (Dataframe)
    process_time: time taken to filter JSON
    error_count: number of JSON Decode Errors
    '''

    df = pd.read_csv(path)
    
    tweets = df['full_text'] #get tweet text field
    if keep_retweets:
        to_remove = to_remove = list(np.where(tweets.isnull())[0]) #remove null tweets
        to_remove = to_remove + list((tweets.index[tweets.str.contains("RT")])) #remove retweets 
        df = df.drop(to_remove)
    
    indices_to_keep = []
    for topic in Constants.TOPICS: 
        #Word is sliced out without need for tokenization
        indices_to_keep = indices_to_keep + list((tweets.index[tweets.str.contains(topic)]))
        
    df = df.iloc[indices_to_keep] #select Tweets
    return df 

  
def check_for_keywords(text):
    return any(word in text for word in Constants.TOPICS)
  
def check_for_retweet(text):
    return 'RT' in text 


def check_for_link(text):
    return 'http' in text 

def filter_JSON(path, keep_links = False, keep_retweets = False, preprocess = False, test = False):
    '''
    @Params
    File: file path name of JSON file type
    keep_retweets: Do we want to retain retweets? (Boolean)
    Preprocess: Do we want to preprocess as we go? (Boolean)
    test: Limit filtered tweets to 10000 (Boolean)
    
    @Returns
    df: Table with relevant fields for each Tweet (Dataframe)
    process_time: time taken to filter JSON
    error_count: number of JSON Decode Errors
    '''

    datetimes, processed_texts, texts, geo, coordinates, tweet_id, sentiment = [], [], [], [], [], [],[]
    start = time.time()
    preprocessor = Preprocessor()
    
    error_count = 0
    
    with open(path, errors='ignore') as f:
        for i, line in tqdm(enumerate(f)):
            try: 
                tweet_data = json.loads(line)
                text = tweet_data['full_text']
                is_retweet = check_for_retweet(text)
                is_link = check_for_link(text)
                if (not is_retweet and not is_link) or keep_retweets or keep_links :
                    text = text.lower()
                    contains_vaccine_keywords = check_for_keywords(text)
                   
                    if contains_vaccine_keywords:
                        datetimes.append(tweet_data['created_at'])    
                        geo.append(tweet_data['geo'])
                        coordinates.append(tweet_data['coordinates'])
                        tweet_id.append(tweet_data['id_str'])
                        texts.append(text)
                    
                        if preprocess:
                            processed_texts.append(preprocessor.pipeline_single_pass(text, Constants.PIPELINE))
                        else:
                            processed_texts.append(None) #preprocess in batches
                    
            #skip and count the JSON decode errors
            except json.JSONDecodeError:
                error_count += 1
            #only grab the first 10k Tweets if testing to speed things up
            if i == 10000 and test == True:
                break
    process_time = time.time() - start
    
    df = pd.DataFrame({'ID': tweet_id, 'Datetime': datetimes, 'Tweet': texts, 'Coordinates': coordinates,
                       'GeoTag': geo, 'ProcessedTweet': processed_texts})
    
    return df, process_time, error_count
