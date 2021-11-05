
#import dependencies
import re
import pandas as pd
import nltk
import string
import numpy as np
import emoji
from nltk.stem import WordNetLemmatizer
from nltk.stem import SnowballStemmer
from nltk.corpus import stopwords
import Constants
import time


class Preprocessor:
	'''
	Text prepocessing class to reduce the size of the corpus vocabulary, and to handle cases
	of non-alphabetic characters
	param lemmatizer: object for word lemmatization
	param stemmer: object for word stemming
	param stop_words: list of stopwords to remove from each tweet 
	param punctuation: list of punctuaation to remove 
	param abbreviation: dictionary of abbreviations and expansions
	'''
	def __init__(self, lemmatizer = WordNetLemmatizer(), stemmer = SnowballStemmer('english'), 
				stop_words = Constants.STOPWORDS, punctuation = string.punctuation,
				abbreviation = Constants.ABBREVIATIONS):

		self.lemmatizer = lemmatizer
		self.punctuation_to_remove = punctuation #list of punctuation to remove
		self.stop_words = stop_words
		self.stemmer = stemmer
		self.abb_to_replace = abbreviation

	def tokenise(self, tweet):
		return nltk.word_tokenize(tweet)

	def detokenise(self, tweet):
		return (' '.join(tweet)) 

	def to_lower(self, tweet):
		'''convert the tweet to lowercase'''
		return tweet.lower()

	def remove_tags(self, tokens):
		'''remove words starting with '@' from Tweet'''
		tokens = [word for word in tokens if word != '']
		tokens = [word for word in tokens if word[0] != '@']
		return tokens

	def remove_numbers(self, tweet):
		'''Remove numerical characters. These don't contain information useful for sentiment classification'''
		return ''.join(c for c in tweet if not c.isdigit())

	def remove_punctuation(self, tweet):
		'''Remove punctuation from tweet'''
		return ''.join(c for c in tweet if c not in self.punctuation_to_remove)

	def lemmatize(self, tokens):
		'''Lemmatize each word contained in the tweet'''
		return [self.lemmatizer.lemmatize(word) for word in tokens]

	def stem(self, tokens):
		'''Reduce each word contained in the tweet to its stem'''
		return [self.stemmer.stem(word) for word in tokens]

	def handle_emojis(self, tweet):
		'''Tweets contain emojis which are very useful in conveying sentiment. Convert these to text'''
		tweet = emoji.demojize(tweet)
		tweet = re.sub(r'_',' ', tweet)
		return tweet

	def remove_stopwords(self, tokens):
		'''Remove stopwords which don't contain much information'''
		return [word for word in tokens if word not in self.stop_words]

	def replace_abbreviations(self, tokens):
		return [self.abb_to_replace.get(word,word) for word in tokens]

	def remove_urls(self, tokens):
		tokens = [word for word in tokens if word != ' ']
		return [word for word in tokens if 'http' not in word]

	def pipeline(self, series, function_names = Constants.PIPELINE):

		start = time.time()

		pipe_dict = {'to_lower': self.to_lower, 'handle_emojis': self.handle_emojis, 'remove_numbers': self.remove_numbers, 
					'remove_punctuation': self.remove_punctuation, 'tokenise': self.tokenise, 'remove_tags': self.remove_tags, 
					'remove_urls': self.remove_urls, 'remove_stopwords': self.remove_stopwords, 'replace_abbreviations': self.replace_abbreviations,
					'lemmatize': self.lemmatize, 'stem': self.stem, 'detokenise': self.detokenise}
		pipes = [pipe_dict[name] for name in function_names]
		processed =  []
		for i, tweet in enumerate(series): 
			for fn in pipes: 
				tweet = fn(tweet)
			processed.append(tweet)

		process_time = time.time() - start

		return processed, process_time

	def pipeline_single_pass(self, tweet, function_names = Constants.PIPELINE):

		pipe_dict = {'to_lower': self.to_lower,
					'handle_emojis': self.handle_emojis,
					'remove_numbers': self.remove_numbers, 
					'remove_punctuation': self.remove_punctuation,
					'tokenise': self.tokenise,
					'remove_tags': self.remove_tags, 
					'remove_urls': self.remove_urls,
					'remove_stopwords': self.remove_stopwords,
					'replace_abbreviations': self.replace_abbreviations,
					'lemmatize': self.lemmatize,
					'stem': self.stem,
					'detokenise': self.detokenise}

		pipes = [pipe_dict[name] for name in function_names]
		for fn in pipes: 
			tweet = fn(tweet)
		return tweet



