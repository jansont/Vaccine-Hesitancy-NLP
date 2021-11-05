import pandas as pd

path = '/Users/theojanson/Project/Capstone/Data/ID_datasets/'
print('Input file number')
number = input()
file = f'corona_tweets_{number}.csv'


try: 
	df = pd.read_csv(path+file, names = ['Tweet', 'Score'])
except: 
	print(f'File corona_tweets_{number}.csv not found.')

if len(df.columns) > 1:
	df = df.drop(df.columns[1], axis = 1)
	df.to_csv(path+file, index = False, header = False)


print(f'Dataset {file} ready for Hydrator.')