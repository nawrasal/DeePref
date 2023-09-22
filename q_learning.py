import random
import math
import numpy as np
import pandas as pd
import sys
from lru import LRU
#from collargs.ECtions import Counter
from datetime import datetime, timedelta
from argparse import ArgumentParser
#####
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple
from itertools import count
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
import common.cloud_net
from common.params import *
from common.models import *
#from common.common import *
from common.edge import * 
#from common.cloud import CLOUD 
from common.oracle import ORACLE
from common.others import *
from common.prepare_dataset import *
from common.utilities import *
from scipy.interpolate import make_interp_spline, BSpline
import os
import glob
from pathlib import Path
from codecarbon import OfflineEmissionsTracker
import multiprocessing
import time
###########
#from torch.profiler import schedule, profile, record_function, ProfilerActivity
##########

#from torchsummary import summary
#import importlib
#importlib.import_module(common.prepare_dataset)
############################################
##########################################
#prepare args
parser: ArgumentParser = ArgumentParser()
parser.add_argument('--EC', type=int, required=True, help="edge_capacity")
parser.add_argument('--NO_edges', type=int, default=3, help="We use 2 edge nodes for testing and 1 edge node for transfer learning.")
parser.add_argument('--algo', type=str, required=True, default="DRQN", help="DQN")
parser.add_argument('--RNN', type=str, default="LSTM", help="LSTM, GRU")
parser.add_argument('--approach', type=str, default="edge", help="edge, cloud")
parser.add_argument('--body', type=str, default="None", help="None, LinearBody")
parser.add_argument('--eviction', type=str, default="LRU", help="LRU, FIFO") 
parser.add_argument('--exploration', type=str, default= "epsilon", help="epsilon, softmax")
parser.add_argument('--run_id', type=int, required=True , help="runs")
parser.add_argument('--edge_id', type=int, required=True, help="0,1,2")
parser.add_argument('--sample_size', type=int, required=False, default=1000,help="10000,1000")
parser.add_argument('--dataset', type=str, default="ml-100k", help="")
parser.add_argument('--way', type=str, default="sample", help="sample, dummy, full")
parser.add_argument('--rnn_memory', type=str, default="sequential", help="sequential, random")
parser.add_argument('--mode', type=str, required=True, default="test", help="train, test")

args = parser.parse_args()
##########################################  
#for federated learning     
# def calc_average_model():
# 	name = str(args)[:-35] #strip mode and after
# 	print("name", name)
# 	models_list = []
# 	for path in Path(path).rglob(name +'*.csv'):
# 		model_params = torch.load(path)
# 		model_params = models_list.append(model_params)

# 	#for model in models_list:
# 	for key in models_list[0]: #access first model
# 		models_list[0][key] = sum([models_list[i][key] for i in range(len(models_list))])/len(models_list)

# 	return models_list[0]


##########***MAIN***#########################
def run():

	# def trace_handler(p):
	# 	output = p.key_averages().table(sort_by="self_cuda_time_total", row_limit=10)
	# 	print(output)
	# 	p.export_chrome_trace("./trace_" + str(p.step_num) + ".json")

	#get dir
	file_name = str(args)[10:]

	if str(device) == "cuda":
		#tracking CO emissions
		tracker = OfflineEmissionsTracker(country_iso_code="USA") #for tracking carbon emissions
		tracker.start()
	#####initilaize edges
	#global last_edge_items
	if args.approach == "edge":
		#get dataframe
		if args.mode == "train":
			edge_df, _, _ = slice_df(args.edge_id)
		elif args.mode == "test":
			_, edge_df, _ = slice_df(args.edge_id)
		elif args.mode == "transfer":
			edge_df, _, _ = slice_df(2)

		# if args.mode == "transfer":
		# 	agent_0 = Prefetch_Env(ID= 2)
		# else:
		agent_0 = Prefetch_Env(ID= args.edge_id)
		results_0 = np.empty(shape=(num_episodes, 7), dtype=object)
		#create a list of agents where only one agent is active (edge approoach)
		if args.mode != "transfer":
			agents = [None for i in range(args.edge_id)]
		else:
			agents = [None for i in range(args.NO_edges-1)]

		agents.append(agent_0)
		if args.algo == "DQN" or args.algo == "DRQN":
			edge= EDGE(agents)
			if args.mode == "test" or args.mode == "transfer":
				#load training model
				model_path = find_model()
				edge.policy_net.load_state_dict(torch.load(model_path, map_location= device))

		else: #OTHERS
			edge = OTHERS(agents)

		if args.mode == "train":
			file_path_0 = file_path_1 = "./results/train_results/" + file_name
		elif args.mode == "test":
			file_path_0 = file_path_1 = "./results/eval/test_results/" + file_name
		elif args.mode == "transfer":
			file_path_0 = file_path_1 = "./results/eval/transfer_results/" + file_name
		np.save(file_path_0, results_0, allow_pickle=True)
		del results_0

	elif args.approach == "cloud":
		if args.mode == "train":
			edge_df, _, _ = get_dataset()
		elif args.mode == "test":
			_, edge_df, _ = get_dataset()
		elif args.mode == "transfer":
			edge_df, _, _  = slice_df(2)

		#we need two edge instances to run for training
		if args.mode == "train":
			agent_0 = Prefetch_Env(ID= 0)
			agent_1 = Prefetch_Env(ID= 1)
			results_0 = np.empty(shape=(num_episodes, 7), dtype=object)
			results_1 = np.empty(shape=(num_episodes, 7), dtype=object)
			file_path_0 = "./results/train_results/" + file_name + "Cloud_ID_0"
			file_path_1 = "./results/train_results/" + file_name + "Cloud_ID_1"


		elif args.mode == "test":
			# we ae testing on agent#1
			agent_0 = agent_1 = Prefetch_Env(ID= 1)
			results_0 = results_1 = np.empty(shape=(num_episodes, 7), dtype=object)
			file_path_0 = file_path_1 = "./results/eval/test_results/" + file_name + "Cloud_ID_1"

		elif args.mode == "transfer":
			agent_0 = agent_1= Prefetch_Env(ID= 2)
			results_0 = results_1 = np.empty(shape=(num_episodes, 7), dtype=object)
			file_path_0 = file_path_1 = "./results/eval/transfer_results/" + file_name + "Cloud_ID_2"


		if args.algo == "DQN" or args.algo == "DRQN":
			edge= EDGE([agent_0, agent_1])
			if args.mode == "transfer":
				edge= EDGE([None, None, agent_1])

			if args.mode == "test" or args.mode == "transfer":
				#load training model
				model_path = find_model()
				edge.policy_net.load_state_dict(torch.load(model_path, map_location= device))
		else: #OTHERS
			edge = OTHERS([None, agent_1, None])



		np.save(file_path_0, results_0, allow_pickle=True)
		np.save(file_path_1, results_1, allow_pickle=True)
		del results_0
		del results_1

		#training
		# with profile(
		# activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
		# schedule=torch.profiler.schedule(
		# wait=1,
		# warmup=1,
		# active=10),
		# on_trace_ready=trace_handler
		# ) as p:

	for i_episode in range(num_episodes):
		print("***********************************")
		print("episode %i" % i_episode)

		# if args.algo == "DQN" or args.algo == "DRQN":
		# 	if torch.device == "GPU":
		# 		#tracking time across GPUs
		# 		start = torch.cuda.Event(enable_timing=True)
		# 		end = torch.cuda.Event(enable_timing=True)
		# 		start.record()

		# 	#CPU timing
		#	t0 = time.time()

		#reset env
		edge.reset_all()

		#run

		print("#time_steps ", edge_df.index.size)
		print(edge_df)
		for i in  range(0, edge_df.index.size,step_size):
			#print("time_step %i" % i, "total time_steps %i" % train_df.index.size)
			#take a step
			user_request_row = edge_df[i:i+1]
			ID = int(user_request_row.group)
			edge.step(user_request_row)

			#save results after every episode
			if i == edge_df.where(edge_df["group"]== 0).last_valid_index() or i == edge_df.where(edge_df["group"]== 1).last_valid_index() or i == edge_df.where(edge_df["group"]== 2).last_valid_index() : #ID=0a
				edge.agents[ID].done = True
				metrics_tuple = edge.agents[ID].end_of_episode()
				if ID == 0 or ID == 2:
					save_episode(metrics_tuple, i_episode, file_path_0)
				else:
					save_episode(metrics_tuple, i_episode, file_path_1)
					
			# else:
			# 	if i == edge_df.where(edge_df["group"]==0).last_valid_index(): #ID=0a
			# 		edge.agents[0].done = True
			# 		metrics_tuple = edge.agents[0].end_of_episode()
			# 		save_episode(metrics_tuple, i_episode, file_path_0)
			# 	elif i == edge_df.where(edge_df["group"]==1).last_valid_index(): #ID=1
			# 		edge.agents[1].done = True
			# 		metrics_tuple = edge.agents[1].end_of_episode()
			# 		save_episode(metrics_tuple, i_episode, file_path_1)


		if args.algo == "DQN" or args.algo == "DRQN":
			# Update the target network, copying all weights and biases in DQN
			if i_episode % TARGET_UPDATE == 0:
				edge.target_net.load_state_dict(edge.policy_net.state_dict())
				edge.target_net.eval()

			# #end time profiling.
			# if torch.device == "GPU":
			# 	end.record()
			# 	torch.cuda.synchronize()
			# 	print("elapsed time on GPU for one episode", start.elapsed_time(end))

			# t1 = time.time()
			# print("episode started at CPU time", time.ctime(t0))
			# print("episode finished at CPU time", time.ctime(t1))

			#CHECK POINT
			if args.mode == "train":
				if i_episode % 100 == 0: 
					#save training model after every episode
					if args.algo == "DQN":
						path = "./DQN_models/" + file_name + ".csv"
					elif args.algo == "DRQN":
						path = "./DRQN_models/" + file_name + ".csv"
					torch.save(edge.target_net.state_dict(), path)

			#profiler step
			#p.step()

			# #save results after every episode
			# train_results = np.asarray(train_edge.metrics_list, dtype=object)
			# np.save("./train_results/" + file_name, train_results, allow_pickle=True)


	if str(device) == "cuda":
		emissions: float = tracker.stop()
		print(f"Emissions: {emissions} kg")

def main():
	run()

if __name__ == '__main__':
	main()
print('Complete')
############################################
