
from filter import filter_JSON
from preprocessor import Preprocessor

'''
Read me: 
Dependencies: 
- tdqm: pip install tdqm
- pandas: pip install pandas
- nltk: pip install nltk
- emojii: pip install emoji

Data directory structure
- Read JSON from  .. / Hydrated_Datasets / corona_tweets_{day}.jsonl
- Write CSV to .. / Processed_Datasets / Processed_Tweets_{day}.csv

Constants
- day: int value of IEEE dataset day
- path: path to where your hydrated andd processed directories exist
- preprocess_while_filtering: preprocess as we filter, rather than after


Run this file: 
- python ./main.py

'''

day = '260'
path =  '/Users/theojanson/Project/Capstone/Data/'
preprocess_while_filtering = True

def main():

	hydrated_dataset_folder = 'Hydrated_Datasets/'
	json_file_path = path + hydrated_dataset_folder + f'corona_tweets_{day}.jsonl'

	print('Filtering dataset...')
	if preprocess_while_filtering: 
		print('Preprocessing as well...')

	df, process_time, error_count = filter_JSON(path = json_file_path,
												keep_retweets = False,
												preprocess = preprocess_while_filtering,
												test = False)

	print(f'Filtering time: {process_time}')
	print(f'Number of JSON Decode Errors: {error_count}')


	if not preprocess_while_filtering: 

		print('Preprocessing dataset...')

		P = Preprocessor()
		processed_tweets, process_time = P.pipeline(df['Tweet'].copy())

		df['ProcessedTweet'] = processed_tweets


	processed_dataset_folder = 'Processed_Datasets/'
	processed_file_path = path + processed_dataset_folder + f'Processed_Tweets_{day}.csv'

	df.to_csv(processed_file_path)

	print(f'Preprocessing time: {process_time}')
	print(f'File written to: {processed_file_path}')

	print('Done')

	print(f'{len(df)} tweets retained after filtering.')

if __name__ == "__main__":
	main()
