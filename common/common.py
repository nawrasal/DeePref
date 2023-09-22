import numpy as np
import pandas as pd
import random
import math
import sys
#from collargs.ECtions import Counter
from datetime import datetime, timedelta
from argparse import ArgumentParser
#####
from collections import namedtuple
from itertools import count
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from common.params import *
from common.models import *
from common.utilities import *
import time
import copy
###########
#from torch.profiler import schedule, profile, record_function, ProfilerActivity
##########

#import importlib
#importlib.import_module(common.prepare_dataset)
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
parser.add_argument('--rnn_memory', type=str, default="sequential", help="sequential, random")
parser.add_argument('--mode', type=str, required=False, default="test", help="train, test")
args = parser.parse_args()
##########################################        
class Prefetch_Env():
	def __init__(self, ID):
		self.metrics_tuple = namedtuple('episode_metrics', ['accuracy', 'coverage', 'sent_prefetches', 'timeliness',
															 'last_edge_items', 'reward_list', 'latency'])

		self.ID = ID
		#LinearScheduler init
		self.EPS_START = 1 #decay epsilon after every episode through self.reset_env
		self.current = self.EPS_START
		self.inc = (EPS_END - self.EPS_START) / float(args.sample_size)

	def reset_env(self):
		np.random.seed(0)
		#prefetching metrics
		self.cache_hits=0.0
		self.cache_misses=0.0
		self.accuracy = 0.0
		self.coverage= 0.0
		self.aggressiveness = 0.0
		self.no_prefetching_flag= False
		self.from_edge_flag = False
		self.done = False
		self.exploit_count = 0
		self.explore_count = 0
		self.time_step = 0
		self.metrics = self.metrics_tuple([],[],[],[],[],[],[])
		self.reward_list = []
		self.item_latency_list = []

		#LinearScheduler init
		self.EPS_START = self.EPS_START*0.9997		#decay epsilon exponentially
		self.current = self.EPS_START
		self.inc = (EPS_END - self.EPS_START) / float(args.sample_size)


		#edge items init
		#if args.mode == "train": #initial edge items distribution (rho0) to be most popular movies on the dataset
		# most_popular_movies= df['movie_id'].value_counts()[:args.EC].index.tolist()
		# most_popular_movies_idx = [movie_2_idx[i] for i in most_popular_movies]
		# self.edge_items = np.array(most_popular_movies_idx)
		# self.init_items = np.copy(self.edge_items)

		#else: 
		self.edge_items= np.random.choice(movies_list, size= args.EC, replace= False)
		self.init_items = np.copy(self.edge_items)

		#LRU
		self.edge_storage_LRU= LRU(len(self.edge_items))
		for i in range(0, len(self.edge_items)):
			self.edge_storage_LRU[self.edge_items[i]]= self.edge_items[i]
		assert set(self.edge_storage_LRU.values()) == set(self.edge_items.tolist()), "init state and LRU storage mismatch"
		
		#initialize indicator vector
		self.edge_items_indicator_vector = np.zeros([len(list(movie_2_idx.values()))], dtype=int)
		for item in self.edge_items: #any edge would work
			self.edge_items_indicator_vector[item] = 1

		self.sent_prefetches= 0.0
		#self.used_prefetches = 0.0
		# self.prefetches_keys = self.edge_items.tolist() #movie_id
		# self.prefetches_values = [0 for _ in range(len(self.prefetches_keys))] #binary indicator dictates whether the sent item is used or not
		self.prefetches_dict = {} #dict(zip(self.prefetches_keys, self.prefetches_values))
		self.timeliness = dict(zip(self.edge_items.tolist(), [0 for _ in range(len(self.edge_items))]))
		self.timeliness_evicted = {} #dict to keep timeliness of evicted items

	def update_LRU_storage(self): #for testing phase only
		#LRU
		self.edge_storage_LRU= LRU(len(self.edge_items))
		for i in range(0, len(self.edge_items)):
			self.edge_storage_LRU[self.edge_items[i]]= self.edge_items[i]
		assert set(self.edge_storage_LRU.values()) == set(self.edge_items.tolist()), "init state and LRU storage mismatch"


	def calc_normalized_latency(self, user_request_row):
		#this function is used for reward calculation
		#calcutate latency
		gig_to_bit = 1024*1024*1024*8
		RTT = 0.1 #seconds
		window_size = 65536 * 8
		throughput = window_size/RTT
		movie_size = int(user_request_row.movie_size.values)* gig_to_bit #*1024*1024*1024*8 #GB --> bit
		latency= movie_size/throughput
		#normalize latency
		#latency = 1/(1 + math.exp(-latency))
		min_latency = int(df.movie_size.min())*gig_to_bit/throughput
		max_latency = int(df.movie_size.max())*gig_to_bit/throughput

		normalized_latency = np.round((latency - min_latency)/(max_latency - min_latency), 1) #[0,1]
		#normalized_latency = np.round(1/(1 + math.exp(-normalized_latency)), 1)
		if normalized_latency == 0:
			normalized_latency = 0.1
		# if normalized_latency == 1:
		# 	normalized_latency = 0.9
		return normalized_latency


	def calc_latency(self, user_request_row):
		if self.from_edge_flag:
			#send item from edge to end user
			RTT = 0.01 #seconds

		else:
			#send item from cloud to end user
			RTT = 0.1

		gig_to_bit = 1024*1024*1024*8
		window_size = 65536 * 8
		throughput = window_size/RTT
		movie_size = int(user_request_row.movie_size.values)* gig_to_bit #*1024*1024*1024*8 #GB --> bit
		item_latency= movie_size/throughput

		return item_latency

	def interact(self, action, user_request_row):
		user_request = int(user_request_row.movie_id.values)
		user_request_idx = movie_2_idx[user_request]


		#First, check if we chose an action that's already in edge (take action & move to the next state)
		if action in self.edge_items or action is None:
			self.no_prefetching_flag= True
			#print("NO PREFETCHING")

		else:
			#evict
			#print("prefetching action", action)
			self.no_prefetching_flag = False
			self.sent_prefetches +=1

			if args.eviction == "LRU":
				#send the prefetch action from cloud to edge, and evict an item based on LRU
				evicted_item = self.edge_storage_LRU.peek_last_item()[1]
				#update storage
				del self.edge_storage_LRU[evicted_item]
				#replace evicted item with prefetch_item --> action
				self.edge_storage_LRU[action]= action
				#move to the next state
				evicted_item_index = np.where(self.edge_items==evicted_item)[0]
				self.edge_items[evicted_item_index] = action
				assert len(self.edge_storage_LRU.keys()) == args.EC, "edge storage mismatch"
				assert set(self.edge_items.tolist()) == set(self.edge_storage_LRU.keys()),"next state should be reflected on LRU"
				#update timeliness (reset) upon eviction
				self.timeliness[action]=0
				self.timeliness_evicted[evicted_item] = self.timeliness[evicted_item]
				#delete key from self.timeliness
				del self.timeliness[evicted_item]
				assert set(self.timeliness.keys()) == set(self.edge_storage_LRU.keys()), "timeliness dict has the same items as LRU dict"

			elif args.eviction == "Belady":
				movies_at_storage = []
				for i in self.edge_items:
					movie_id = idx_2_movie[i]
					movies_at_storage.append(movie_id)
				#get proper df
				if args.mode == "train" or args.mode == "transfer": 
					df_ = train_df[self.time_step:]
				elif args.mode == "test":
					df_ = test_df[self.time_step:]

				#find the index of last occurence of each item
				index_distance = []
				#print(df_.iloc[167])
				for i in movies_at_storage:
					distance = max(df_.index[df_["movie_id"]== i].tolist(), default=float("inf")) #default inf if list is empty
					index_distance.append(distance)

				#print(df_)
				print(index_distance)
				item_to_evict_idx = index_distance.index(max(index_distance))
				item_to_evict = movies_at_storage[item_to_evict_idx]
				evicted_item = movie_2_idx[item_to_evict]
				del self.edge_storage_LRU[evicted_item]
				self.edge_storage_LRU[action]= action
				evicted_item_index = np.where(self.edge_items==evicted_item)[0]
				self.edge_items[evicted_item_index] = action
				assert len(self.edge_storage_LRU.keys()) == args.EC, "edge storage mismatch"
				assert set(self.edge_items.tolist()) == set(self.edge_storage_LRU.keys()),"next state should be reflected on LRU"
				#update timeliness (reset) upon eviction
				self.timeliness[action]=0
				self.timeliness_evicted[evicted_item] = self.timeliness[evicted_item]
				#delete key from self.timeliness
				del self.timeliness[evicted_item]
				assert set(self.timeliness.keys()) == set(self.edge_storage_LRU.keys()), "timeliness dict has the same items as LRU dict"


			elif args.eviction == "FIFO":
				edge_items_list = self.edge_items.tolist()
				evicted_item = edge_items_list.pop(0)
				edge_items_list.append(action)
				self.edge_items = np.array(edge_items_list, dtype=int)
				self.update_LRU_storage()
				#update timeliness (reset) upon eviction
				self.timeliness[action]=0
				self.timeliness_evicted[evicted_item] = self.timeliness[evicted_item]
				#delete key from self.timeliness
				del self.timeliness[evicted_item]
				assert set(self.timeliness.keys()) == set(self.edge_storage_LRU.keys()), "timeliness dict has the same items as LRU dict"				


			else:
				print("error in cache eviction strategy")

		#Observe reward
		if user_request_idx in self.edge_items: #if  user_request_idx == action
			#cache hit
			self.cache_hits+=1 #increase hits
			#calc latency
			self.from_edge_flag = True
			item_latency = self.calc_latency(user_request_row)


			#update used prefetches
			#if user_request_idx in self.edge_items:
			#if user_request_idx in self.edge_items:
			if user_request_idx not in self.init_items or self.time_step > args.EC:
				try:
					self.prefetches_dict[user_request_idx] += 1
				except KeyError:
					self.prefetches_dict[user_request_idx] = 1

			#update timeliness (reset) upon cache hits
			self.timeliness[user_request_idx] = 0
			#calc reward
			if self.no_prefetching_flag==True:
				reward= 2.0
			else:
				if user_request_idx == action:
					#add prefetching cost reward = - prefetching cost
					latency = self.calc_normalized_latency(user_request_row)
					reward= 2.0 -latency
					#self.edge_storage_LRU[user_request]   #make that item a MRU (Most Recently Used)
				else:
					reward = 2.0

		else:
			# print("cache miss")
			# print("action", action)
			# print("user_request", user_request)
			# print("edge_items", self.edge_items)
			self.cache_misses+=1
			#calc latency
			self.from_edge_flag = False
			item_latency = self.calc_latency(user_request_row)
			#update timeliness
			for i in self.timeliness.keys():
				self.timeliness[i]+=1
			#calc reward
			if self.no_prefetching_flag == True:
				#cache miss
				reward= -1.0
			else:
				#add prefetching cost reward = -1 - prefetching cost
				latency= self.calc_normalized_latency(user_request_row)
				reward= -1.0 - latency
		#print(self.timeliness)

		#add item latency to metrics
		self.item_latency_list.append(item_latency)
		#obs
		self.edge_items_indicator_vector = np.zeros([len(list(movie_2_idx.values()))], dtype=int)
		for item in self.edge_items:
			self.edge_items_indicator_vector[item] = 1

		self.reward_list.append(reward)

		return reward, self.edge_items_indicator_vector

	#for cloud approaches
	def check_miss(self, user_request_idx):
		if user_request_idx in self.edge_storage_LRU.keys():
			return False
		else:
			return True

	##########***Plotting and Printing***###############   
	def end_of_episode(self):
		assert self.done == True, "episode isn't done yet"
		#accuracy
		try:
			used_prefetches = sum(self.prefetches_dict.values())
			#print(self.prefetches_dict)

			self.accuracy= used_prefetches/self.sent_prefetches				
		except ZeroDivisionError:
			print("ZeroDivisionError")
			print("prefetches_dict", self.prefetches_dict)
		#coverage
		self.coverage= self.cache_hits/(self.cache_misses+self.cache_hits)
		#timeliness
		timeliness_list = list(self.timeliness.values()) + list(self.timeliness_evicted.values())
		timeliness_np = np.array(timeliness_list)
		timeliness_mean = np.mean(timeliness_np)
		timeliness_std = np.std(timeliness_np)
		self.timeliness_metrics = [timeliness_mean, timeliness_std]
		self.aggressiveness = self.sent_prefetches/self.time_step
		print("ID", self.ID, "accuracy", self.accuracy, "coverage", self.coverage, "sent prefeteches", self.sent_prefetches,
			 "timeliness", self.timeliness_metrics, "aggressiveness", self.aggressiveness)

		self.metrics.accuracy.append(self.accuracy) #accuracy
		self.metrics.coverage.append(self.coverage) #coverage
		self.metrics.sent_prefetches.append(self.sent_prefetches) #sent_prefetches
		#timeliness
		self.metrics.timeliness.append(self.timeliness_metrics[0]) #mean
		self.metrics.timeliness.append(self.timeliness_metrics[1]) #std
		self.metrics.last_edge_items.append(self.edge_storage_LRU.keys()) #last_edge_items

		#reward_list
		if args.mode == "train": #to calculate average episodic reward
			self.reward_list = np.array(self.reward_list)
			reward_list_mean = np.mean(self.reward_list)
			reward_list_std = np.std(self.reward_list)
			self.metrics.reward_list.append(reward_list_mean.item(0))
			self.metrics.reward_list.append(reward_list_std.item(0))

		elif args.mode == "test" or args.mode == "transfer": #to calculate commulative reward
			self.reward_list = np.array(self.reward_list)
			#print(self.reward_list)
			#print(self.reward_list.item(0))
			self.metrics.reward_list.append(self.reward_list)


		#item latency
		self.metrics.latency.append(np.mean(self.item_latency_list).item(0))
		self.metrics.latency.append(np.std(self.item_latency_list).item(0))
		print("used_prefetches", used_prefetches)
		print("sent_prefetches", self.sent_prefetches)
		print("cache hits", self.cache_hits)
		print("cache misses", self.cache_misses)

		# if args.algo == "DRQN" or args.algo == "DQN":
		# 	self.loss = np.array(self.loss)
		# 	loss_mean = np.mean(self.loss)
		# 	loss_std = np.std(self.loss)
		# 	self.metrics.loss.append(loss_mean.item(0))
		# 	self.metrics.loss.append(loss_std.item(0))
		# if i_episode != 0:
		# 	if np.mean(np.asarray(self.metrics_list[i_episode][5])) > np.mean(np.asarray(self.metrics_list[i_episode-1][5])):
		# 		#save training model after every episode
		# 		if args.algo == "DQN":
		# 			path = "./DQN_models/" + str(args) + ".csv"
		# 		elif args.algo == "DRQN":
		# 			path = "./DRQN_models/" + str(args) + ".csv"
		# 		torch.save(self.policy_net.state_dict(), path)


		#print(self.metrics_list)
		return self.metrics


	def step_(self):
		self.time_step += step_size
###############################################