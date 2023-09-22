import numpy as np
import pandas as pd
import random
import math
import numpy as np
import pandas as pd
import sys
from lru import LRU
#from collargs.ECtions import Counter
from datetime import datetime, timedelta
from argparse import ArgumentParser
from common.params import *
from common.common import *
import multiprocessing 
########################################
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
parser.add_argument('--rnn_memory', type=str, default="random", help="sequential, random")
parser.add_argument('--mode', type=str, required=False, default="test", help="train, test")
args = parser.parse_args()
#########################################
class OTHERS():
	def __init__(self, agents):
		super(OTHERS, self).__init__()
		self.agents = agents
		self.popularity_df = pd.DataFrame()
		self.previous_request = None #for LRU_prefetch

	def select_action(self, user_request_row, ID):
		#figure out the most common movie_id over the last 24 hours (time_window = all movies processed over the last 24 hours)
		self.popularity_df = self.popularity_df.append(user_request_row, ignore_index=True)

		if args.algo == "popularity_recent":
			time_window = self.popularity_df[self.popularity_df['date_time'] >= (self.popularity_df.iloc[-1]['date_time']- timedelta(hours=24))]
			#get most popular items (items with the same value_counts)
			top_k_popular_items = time_window['movie_id'].value_counts().nlargest(n=args.EC, keep='all').index.values
			#most_popular_items = time_window[time_window.groupby('movie_id')['movie_id'].transform('size') == max_count]
			#pick a random item from the list of most popular items
			#most_popular_items_set = list(set(most_popular_items['movie_id'].values.tolist()))
			#most_popular_items_set = most_popular_items['movie_id'].values.tolist()
			#convert to tokens
			most_popular_items_set_idx = [movie_2_idx[item] for item in top_k_popular_items]
			#filter all popular items not in edge storage
			best_items =[item for item in most_popular_items_set_idx if item not in self.agents[ID].edge_storage_LRU.keys()]
			#pick prefetch item from best_items
			if best_items != []:
				action = np.random.choice(best_items)
			else:
				#select_action = np.random.choice(most_popular_items_set)
				action = None

		elif args.algo == "popularity_all":
			#get most popular items (items with the same value_counts)
			top_k_popular_items = self.popularity_df['movie_id'].value_counts().nlargest(n=args.EC).index.values
			#most_popular_items = self.popularity_df[self.popularity_df.groupby('movie_id')['movie_id'].transform('size') == max_count]
			#pick a random item from the list of most popular items
			#most_popular_items_set = list(set(most_popular_items['movie_id'].values.tolist()))
			#most_popular_items_set = most_popular_items['movie_id'].values.tolist()
			#convert to tokens
			most_popular_items_set_idx = [movie_2_idx[item] for item in top_k_popular_items]
			# print(most_popular_items_set_idx)
			# print(self.agents[1].edge_storage_LRU.keys())
			# print(ID)
			#filter all popular items not in edge storage

			best_items =[item for item in most_popular_items_set_idx if item not in self.agents[ID].edge_storage_LRU.keys()]
			#pick prefetch item from best_items
			if best_items != []:
				action = np.random.choice(best_items)
			else:
				#select_action = np.random.choice(most_popular_items_set)
				action = None
			#action = np.random.choice(most_popular_items_set)


		elif args.algo == "LRU_prefetch":
			assert args.eviction == "LRU", "LRU eviction must be in use"
			if user_request_row.index.values == 0:
				self.previous_request = int(user_request_row.movie_id.values)
				action = None
			else:
				action = movie_2_idx[self.previous_request]
				self.previous_request = int(user_request_row.movie_id.values)

		elif args.algo == "FIFO_prefetch":
			assert args.eviction == "FIFO", "FIFO eviction must be in use"
			if user_request_row.index.values == 0:
				self.previous_request = int(user_request_row.movie_id.values)
				action = None
			else:
				action = movie_2_idx[self.previous_request]
				self.previous_request = int(user_request_row.movie_id.values)

		elif args.algo == "top_k_popularity":
			#assert args.approach == "cloud", "approach must equal to cloud"
			#change edge items
			most_popular_movies= df['movie_id'].value_counts()[:args.EC].index.tolist()			
			most_popular_movies_idx = [movie_2_idx[i] for i in most_popular_movies]
			self.agents[ID].edge_items = np.array(most_popular_movies_idx)
			self.agents[ID].update_LRU_storage()
			self.agents[ID].sent_prefetches = args.EC
			action = None#take no action in other words

		#print(cloud_df)

		elif args.algo == "top_k_size":
			#assert args.approach == "cloud", "approach must equal to cloud"
			top_k_size_movies_ = df[['movie_id', 'movie_size']].drop_duplicates().nlargest(args.EC, 'movie_size')
			top_k_size_movies = top_k_size_movies_['movie_id'].values.tolist()
			top_k_size_movies_idx = [movie_2_idx[i] for i in top_k_size_movies]
			self.agents[ID].edge_items = np.array(top_k_size_movies_idx)
			self.agents[ID].update_LRU_storage()
			self.agents[ID].sent_prefetches = args.EC
			action = None

		elif args.algo == "Belady_prefetch":
			assert args.eviction == "Belady", "Belady eviction must be in use"
			if user_request_row.index.values == 0:
				self.previous_request = int(user_request_row.movie_id.values)
				action = None
			else:
				self.previous_request = int(user_request_row.movie_id.values)
				action = movie_2_idx[self.previous_request]

		return action


	def reset_all(self):
		#reset agent variables and model variables
		#reset agent variables
		for agent in self.agents:
			if agent != None:
				agent.reset_env()

	def step(self, user_request_row):
		user_request = int(user_request_row.movie_id.values)
		ID = int(user_request_row.group)

		#select action
		#edge_items_np = self.init_items

		action = self.select_action(user_request_row, ID)

		assert set(self.agents[ID].edge_items.tolist()) == set(self.agents[ID].edge_storage_LRU.keys()),"state should be reflected on LRU"
		#Take action and Observe reward and next state
		reward, obs = self.agents[ID].interact(action, user_request_row)
		#agent_step
		self.agents[ID].step_()
		return
