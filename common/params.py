import pandas as pd
import numpy as np
from argparse import ArgumentParser
from collections import namedtuple
from common.prepare_dataset import get_dataset, slice_df
import torch
import torch.nn as nn
#from common.models import *
#from common.common import *
##########################################
#prepare args
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
##########################################
###PARAMS
if args.mode == "train" and args.algo == "DQN" or args.mode == "train" and args.algo=="DRQN":
	num_episodes = 10000
else:
	num_episodes = 1
#learning params
GAMMA = 0.99
#exponential decay for epsilon-greedy
EXPLORATION_STEPS = 0 # == batch_size for DQN
#EPS_START = 1
EPS_END = 0.05
#ExponentialScheduler
if args.sample_size == 10000:
	EPS_DECAY = 1500 #decay every time_step #150 for 1000 samples
elif args.sample_size == 1000:
	EPS_DECAY = 150
else:
	EPS_DECAY = 10

step_size= 1
Temperature = 1
###
# if args.approach == "cloud":
# 	train_df, train_data, test_df, test_data, df, data= get_dataset() #for initializing edge storage to top-n and latency normalization
# else:
# 	train_df, train_data, test_df, test_data, df, data = slice_df(args.edge_id)

if args.approach == "cloud":
	train_df, test_df, df = get_dataset() #for initializing edge storage to top-n and latency normalization

elif args.approach == "edge":
	train_df, test_df, df = slice_df(args.edge_id)


#tokenize for both DQN and DRQN
movies= df.movie_id.unique().tolist()
tokens= [i for i in range(len(movies))]
movie_2_idx = dict(zip(movies,tokens))
idx_2_movie = dict(zip(tokens, movies))
movies_list = list(movie_2_idx.values())
n_actions = len(movies_list)
print("n_actions", n_actions)
assert n_actions == len(df.movie_id.unique()) , "n_actions mismatch" # +1 for no prefetchings
###for models
one_hot_size = n_actions

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device", device)
print("len(movie_2_idx.values())", len(movie_2_idx.values()))

#LSTM for both LSTM classifier and DRQN
hidden_size=512
num_layers=1
num_directions = 1
body_size = 512
POLICY_UPDATE= 1
TARGET_UPDATE = 100
BATCH_SIZE = 10
dropout = 0 #for LSTM classifier

if args.algo == "DRQN":
	if args.rnn_memory == "sequential":
		BATCH_SIZE = 1
		SEQUENCE_LENGTH = train_df.index.size #to instantiate ReplayMemory for DRQN
	elif args.rnn_memory == "random":
		BATCH_SIZE = 10
		if args.sample_size >= 1000:
			SEQUENCE_LENGTH = 500
		else:
			SEQUENCE_LENGTH = 10
	print("SEQUENCE_LENGTH", SEQUENCE_LENGTH)
	input_size = one_hot_size+ len(movie_2_idx.values())
	input_shape = (input_size,)
	print("input_shape", input_shape)
	#TBPTT params
	k1= 500
	k2= 500

	# # #cloud approach
	# if args.approach == "cloud":
	# 	memory= RecurrentReplayMemory(10000)
	# 	policy_net=DRQN(input_shape, n_actions).to(device)
	# 	target_net=DRQN(input_shape, n_actions).to(device)
	# 	target_net.load_state_dict(policy_net.state_dict())
	# 	target_net.eval()
	# 	episode_buffer = []
	# 	time_step = 0


elif args.algo == "DQN":
	SEQUENCE_LENGTH = 1 #unused parameter

	user_history_len = 100 #last user_requests for DQN
	input_size = user_history_len*one_hot_size+ len(movie_2_idx.values())
	input_shape = (input_size,)
	#cloud approach
	# if args.approach == "cloud":
	# 	memory = ReplayMemory(10000)
	# 	policy_net = DQN(input_shape, n_actions).to(device)
	# 	target_net = DQN(input_shape, n_actions).to(device)
	# 	target_net.load_state_dict(policy_net.state_dict())
	# 	target_net.eval()
	# 	time_step = 0
elif args.algo == "LSTM_classifier":
	if args.mode == "train":
		BATCH_SIZE = 10
	else:
		BATCH_SIZE = 1
		
	SEQUENCE_LENGTH = 100
	EPOCHS = 200
	num_layers=1
	hidden_size=512
	body_size = 512

	lr= 0.001
	params = {'batch_size': BATCH_SIZE,
			'shuffle': False,
			'num_workers': 0, 'pin_memory': True}  # Setting > 0 causes problems with CUDA
	input_size = one_hot_size
	input_shape = (input_size, )
	print("input_size", input_size)

	#prepare dataset
	df = df[['movie_id']]
	df['idx'] = df['movie_id'].apply(lambda x: movie_2_idx[x])
	data = pd.get_dummies(df.idx)
	#print(train_data)
	#split train_data into training and validation
	train_size = int(0.8 * args.sample_size)
	val_size = int(0.2 * args.sample_size) + train_size
	#split df and one hot data
	train_df = df.iloc[:train_size, :]
	train_df_one_hot = data.iloc[:train_size ,:]
	val_df = df.iloc[train_size: val_size, :]
	val_df.reset_index()
	val_df_one_hot = data.iloc[train_size:val_size, :]
	val_df_one_hot.reset_index()
	test_df = df.iloc[train_size+val_size:, :]
	test_df.reset_index()
	test_df_one_hot = data.iloc[train_size+val_size:, :]
	test_df_one_hot.reset_index()

else: #others
	SEQUENCE_LENGTH = 1 #unused parameter

	user_history_len = 100 #last user_requests for DQN
	input_size = user_history_len*one_hot_size+ len(movie_2_idx.values())
	input_shape = (input_size,)


