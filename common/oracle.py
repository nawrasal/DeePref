import numpy as np
import pandas as pd
import random
import math
import sys
from argparse import ArgumentParser
#####
from common.params import *
#from common.models import *
from common.common import *
############################################
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
parser.add_argument('--rnn_memory', type=str, default="random", help="sequential, random")
parser.add_argument('--mode', type=str, required=False, default="test", help="train, test")

args = parser.parse_args()
##########################################        
 
####################################################
class ORACLE():
	def __init__(self):
		super(ORACLE, self).__init__()
		self.ID = "ORACLE"
		self.num_episodes = 1
		self.metrics_list= [[[],[],[],[],[],[], []] for _ in range(self.num_episodes)]
	def run_oracle(self, df):
		#reset env
		self.reset_env()
		for i in range(0, df.index.size, step_size):
			user_request_row = df[i:i+1]
			user_request = int(user_request_row.movie_id.values)
			#check at t-1 for cache hits
			if user_request in self.edge_storage_LRU.values():
				self.cache_hits+=1 #increase hits
				reward = 2
			else:
				#assume we prefetched request at t-1
				#evict an item LRU (oracle decouples eviction from prefetching)
				evicted_item = self.edge_storage_LRU.peek_last_item()[1]
				del self.edge_storage_LRU[evicted_item]
				#replace evicted item with prefetch_item --> user_request
				self.edge_storage_LRU[user_request]= user_request
				self.sent_prefetches +=1
				self.used_prefetches+=1
				assert len(self.edge_storage_LRU.keys()) == args.EC, "edge storage mismatch"
				#calcuate latency 
				latency = self.calc_metrics(user_request_row)
				reward = - latency

			self.metrics_list[0][5].append(reward)
			if i == df.index.size - step_size:
				#print
				self.end_of_episode(0)

		return
