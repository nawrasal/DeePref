import pandas as pd
import numpy as np
from argparse import ArgumentParser
###########################################
parser: ArgumentParser = ArgumentParser()
parser.add_argument('--EC', type=int, required=False, help="edge_capacity")
parser.add_argument('--NO_edges', type=int, default=3, help="")
parser.add_argument('--algo', type=str, required=False, default="DRQN", help="DQN")
parser.add_argument('--RNN', type=str, default="LSTM", help="LSTM, GRU")
parser.add_argument('--approach', type=str, default="edge", help="edge, cloud")
parser.add_argument('--body', type=str, default="None", help="None, LinearBody")
parser.add_argument('--eviction', type=str, default="LRU", help="LRU, FIFO") 
parser.add_argument('--exploration', type=str, default= "epsilon", help="epsilon, softmax")
parser.add_argument('--run_id', type=int, required=False , help="runs")
parser.add_argument('--edge_id', type=int, required=True, help="0,1,2")
parser.add_argument('--sample_size', type=int, required=False, default=1000,help="10000,1000")
parser.add_argument('--dataset', type=str, default="ml-100k", help="")
parser.add_argument('--way', type=str, default="sample", help="sample, dummy, full")
parser.add_argument('--rnn_memory', type=str, default="sequential", help="sequential, random")
parser.add_argument('--mode', type=str, required=False, default="test", help="train, test")

args = parser.parse_args()
def get_dataset():
	if args.dataset == "ml-1m":
		# movies = pd.read_csv('datasets/ml-1m/movies.dat', sep='::', names=['movie_id', 'title', 'genre']
		# 	,encoding='latin-1', engine= 'python')

		# movies_dataset = pd.DataFrame(movies, columns= ['movie_id'])
		if args.NO_edges == 1:
			df = pd.read_csv("datasets/ml-1m/df_k_3.csv", encoding='latin-1', low_memory=False)
		elif args.NO_edges == 3:
			df = pd.read_csv("datasets/ml-1m/df_k_3.csv", encoding='latin-1', low_memory=False)
		elif args.NO_edges == 7:
			df = pd.read_csv("datasets/ml-1m/df_k_7.csv", encoding='latin-1', low_memory=False) #ignore group any df would work

	elif args.dataset == "ml-100k":
		if args.NO_edges == 3:
			df = pd.read_csv("datasets/ml-100k/df_k_3.csv", encoding='latin-1', low_memory=False)

	#sorting requests chronologically
	df = df.sort_values(by=['timestamp'], ascending=True)
	df = df.reset_index(drop=True)
	df['date_time'] = pd.to_datetime(df['timestamp'], unit='s') #don't normalize (dt.normalize())

	deleted_requests = int(df.index.size % args.NO_edges)
	if deleted_requests != 0:
		df = df[:-deleted_requests]

	# #one-hot encoding
	# df['movie_id'] = df['movie_id'].astype('category').cat.codes
	# #df['movie_id'] = pd.factorize(df['movie_id'])[0] + 1 #start with 1 as index for movies
	# #print(df)
	# data= pd.get_dummies(df.movie_id)
	#print(data)

	# if args.way == "dummy":
	# 	example= [0,2,4,3,1,0,4,4,3,2]
	# 	data_columns = [0,1,2,3,4,5,6,7,8,9]
	# 	df = pd.DataFrame(example, columns=['movie_id'])
	# 	data = pd.get_dummies(df.movie_id, columns=data_columns, prefix='movie_id')

	#add size to df
	np.random.seed(0) #fix seed (generate same numbers)
	movies_size = np.random.uniform(low=4, high=10, size=len(df.movie_id.unique()))
	#create a dic
	keys=df.movie_id.unique().tolist()
	size_dict= dict(zip(keys, movies_size))
	#print(size_dict)
	df['movie_size'] = df['movie_id'].map(size_dict)
	#print(df['movie_id'].value_counts(normalize=False))
	#add popularity
	df['movie_popularity'] = df.groupby(['movie_id'])['movie_id'].transform('count')

	#create a chunk containing both edge#0 and edge#1 for cloud approach
	cloud_df = df[df['group'] !=2] #df containing edge#0 and edge#1
	#cloud_data = data.iloc[list(cloud_df.index),:]

	if args.way == "sample" and args.approach == "cloud":
		train_size = args.sample_size
		test_size = int(args.sample_size/5)
	
		#construct a new df (mainly, to minimize the action space)
		train_df_ = cloud_df.iloc[:train_size, :] #1000 for training, 2000 for testing
		#train_data_ = cloud_data.iloc[:train_size, :]
		test_df_ = cloud_df.iloc[train_size:train_size+test_size, :]
		#test_data_ = cloud_data.iloc[train_size:train_size+test_size, :]
		transfer_df_ = df[df['group'] == 2]
		transfer_df_ = transfer_df_.iloc[:train_size,:]
		merge_frames = [train_df_, test_df_, transfer_df_]
		df = pd.concat(merge_frames)
		df = df.reset_index(drop=True)
		#data= pd.get_dummies(df.movie_id)
		#now build train and test df for cloud approach
		train_df = df.iloc[:train_size, :]
		#train_data = data.iloc[:train_size, :]
		#test_data = data.iloc[train_size:train_size+test_size, :]

		#reset index
		train_df = train_df.reset_index(drop=True)
		#train_data = train_data.reset_index(drop=True)
		#TEST ON ONE EDGE ONLY
		#test_df = df.iloc[train_size:train_size+test_size, :]
		test_df = df[df['group'] == 1]
		test_df = test_df.iloc[:test_size, :]
		test_df = test_df.reset_index(drop=True)
		#test_data = test_data.reset_index(drop=True)

		return train_df, test_df, df


	elif args.way == "sample" and args.approach == "edge":
		# #construct a new df (mainly, to minimize the action space)
		# train_df_ = cloud_df.iloc[:round(len(cloud_df.index)*0.8), :]
		# train_data_ = cloud_data.iloc[:round(len(cloud_df.index)*0.8), :]
		# test_df_ = cloud_df.iloc[round(len(cloud_df.index)*0.8):, :]
		# test_data_ = cloud_data.iloc[round(len(cloud_df.index)*0.8):, :]
		# #this is used to define the action space in params
		# transfer_df_ = df[df['group'] == 2]
		# transfer_df_ = transfer_df_.iloc[:round(len(transfer_df_.index)*0.8),:]
		# merge_frames = [train_df_, test_df_, transfer_df_]
		# df = pd.concat(merge_frames)
		# df = df.reset_index(drop=True)
		# data_columns= [i for i in range(0, len(df.movie_id.unique()))]
		# data= pd.get_dummies(df.movie_id, columns=data_columns, prefix='movie_id')
		# #now build train and test df for cloud approach
		# train_df = df.iloc[:round(len(df.index)*0.8), :] #1000 for training, 2000 for testing
		# train_data = data.iloc[:round(len(df.index)*0.8), :]
		# test_df = df.iloc[round(len(df.index)*0.8):, :]
		# test_data = data.iloc[round(len(df.index)*0.8):, :]

		return None, None, df
	else:
		raise NotImplementedError


	#JUST FOR TESTING (TOY EXAMPLE)
	# train_df = train_df.iloc[:10,:]
	# train_data = train_data.iloc[:10,:]
	#print(train_df)


def slice_df(edge_id):
	_, _, df= get_dataset()
	edge_df = df[df['group'] == edge_id]
	edge_df.reset_index(drop=True)
	# edge_data = data.iloc[list(edge_df.index),:]
	# edge_data = edge_data.reset_index(drop=True)
	#edge_data= pd.get_dummies(df.movie_id)	

	#Only 10% of the entire dataset
	if args.way == "sample":
		train_size = args.sample_size
		test_size = int(args.sample_size/5)
	
		# df = df.iloc[:round(len(df.index)*0.1), :]
		# data = data.iloc[:round(len(data.index)*0.1), :]
		train_df_ = edge_df.iloc[:train_size, :] #1000 for training, 200 for testing
		#train_data_ = edge_data.iloc[:train_size, :]
		test_df_ = edge_df.iloc[train_size:train_size+test_size, :]
		#print("train_df_", train_df_)
		#print("test_df_", test_df_)
		#test_data_ = edge_data.iloc[train_size:train_size+test_size, :]
		#transfer_df_ = df[df['group'] == 2]
		#transfer_df_ = transfer_df_.iloc[:train_size,:]
		merge_frames = [train_df_, test_df_] #transfer_df_
		df = pd.concat(merge_frames)
		df = df.reset_index(drop=True)
		#data= pd.get_dummies(df.movie_id)
		#now build train and test df for edge approach
		train_df = df.iloc[:train_size, :]
		#train_data = data.iloc[:train_size, :]
		test_df = df.iloc[train_size:train_size+test_size, :]
		#test_data = data.iloc[train_size:train_size+test_size, :]

		#reset index
		train_df = train_df.reset_index(drop=True)
		#train_data = train_data.reset_index(drop=True)
		test_df = test_df.reset_index(drop=True)
		# print(train_df)
		# print(test_df)
		# print(df)
		#test_data = test_data.reset_index(drop=True)

	##
	else: #full: split 80/20
		# train_df = edge_df.iloc[:round(len(edge_df.index)*0.8), :]
		# train_data = edge_data.iloc[:round(len(edge_data.index)*0.8), :]
		# test_df = edge_df.iloc[round(len(edge_df.index)*0.8):, :]
		# test_data = edge_data.iloc[round(len(edge_df.index)*0.8):, :]
		raise NotImplementedError


	#JUST FOR TESTING (TOY EXAMPLE)
	# train_df = train_df.iloc[:10,:]
	# train_data = train_data.iloc[:10,:]
	#print(train_df)
	#print(train_data)
	#print(train_data.apply(lambda row: row[row == 1].index, axis=1))
	return train_df, test_df, df

def prepare_data_k_means():
	if args.dataset == "ml-1m":
		users= pd.read_csv('datasets/ml-1m/users.dat', sep='::', names=['user_id', 'gender', 'age', 'occupation', 'zip_code']
			,encoding='latin-1', engine= 'python', dtype='str')

		ratings = pd.read_csv('datasets/ml-1m/ratings.dat', sep='::', names=['user_id', 'movie_id', 'rating', 'timestamp']
			,encoding='latin-1', engine= 'python', dtype='str')

		# movies = pd.read_csv('datasets/ml-1m/movies.dat', sep='::', names=['movie_id', 'title', 'genre']
		# 	,encoding='latin-1', engine= 'python')

	elif args.dataset == "ml-100k":
		users= pd.read_csv('datasets/ml-100k/u.user', sep='|', names=['user_id', 'age', 'gender', 'occupation', 'zip_code']
			,encoding='latin-1', engine= 'python', dtype='str')
		ratings = pd.read_csv('datasets/ml-100k/u.data', sep='\t', names=['user_id', 'movie_id', 'rating', 'timestamp']
			,encoding='latin-1', engine= 'python', dtype='str')


	us_census = pd.read_csv('datasets/us_census.csv', sep=',', encoding='latin-1', names=["zip_code", "state", "latitude", "longitude", "population"])

	#Merge to include zip codes in ratings dataframe
	df_ = ratings.merge(users[['user_id', 'zip_code']], on='user_id', how='left')
	df = df_.merge(us_census[['zip_code', 'latitude', 'longitude']], on='zip_code', how='left')
	#X = df[['user_id', 'zip_code', 'longitude', 'latitude']]
	X = df[['longitude', 'latitude']]
	#X['zip_code'] = X['zip_code'].apply(str)
	#print(X['zip_code'].dtype)
	X = X.iloc[:2000,:]
	df = df.iloc[:2000, :]
	df= df.dropna()
	df=df.reset_index(drop=True)
	X= X.dropna()
	X = X.reset_index(drop=True)
	print(X)
	print(df)
	return X, df


