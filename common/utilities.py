import numpy as np
import math
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from argparse import ArgumentParser
import glob

########################################
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

########################################


def LinearScheduler(current, inc, EPS_END, steps=1):
	eps_threshold = current
	current = max(current + inc * steps, EPS_END)
	return eps_threshold, current

def ExponentialScheduler(time_step, EPS_START, EPS_END, EPS_DECAY):
	#decay eps exponentially
	eps_threshold = EPS_END + (EPS_START - EPS_END) * \
				math.exp(-1. * time_step / EPS_DECAY)
	return eps_threshold

def MSE(x):
	return 0.5 * x.pow(2)

def nn_MSE():
	return nn.MSELoss()


def nn_CrossEntropyLoss():
	return nn.CrossEntropyLoss()

def huber(x):
	cond = (x.abs() < 1.0).float().detach()
	return 0.5 * x.pow(2) * cond + (x.abs() - 0.5) * (1.0 - cond)

def to_onehot(idx, movies_list):
	one_hot_np = np.zeros(len(movies_list))
	one_hot_np[idx] = 1
	return one_hot_np


def save_episode(metrics_tuple, i_episode, file_path):
	metrics_np = np.array(metrics_tuple, dtype=object)
	results = np.load(file_path + ".npy", allow_pickle=True)
	results[i_episode]= metrics_np
	np.save(file_path, results, allow_pickle= True)
	del results
	del metrics_np


def find_model():
	if args.algo == "LSTM_classifier":
		file_name = str(args)[10:]
	else:
		file_name = str(args)
	print(file_name)
	if args.mode == "test":
		file_name = file_name.replace(", mode='test'", ", mode='train'")

	elif args.mode == "transfer":
		file_name = file_name.replace(", mode='transfer'", ", mode='train'")
		#file_name = file_name.replace(", edge_id=2", ", edge_id=0")
	print(file_name)
	#find model
	if args.algo == "DQN" or args.algo == "DRQN" or args.algo == "LSTM_classifier":
		if args.algo == "DQN":
			path = "./DQN_models/" + file_name + ".csv"
			files = glob.glob("./DQN_models/*.csv")

		elif args.algo == "DRQN":
			path = "./DRQN_models/" + file_name + ".csv"
			files = glob.glob("./DRQN_models/*.csv")

		elif args.algo == "LSTM_classifier":
			path = "./LSTM_classifier_models/" + file_name + ".csv"
			files = glob.glob("./LSTM_classifier_models/*.csv")

		if path in files:
			print("*** MODEL FOUND ***" )
			return path
		else:
			print("*** MODEL NOT FOUND ***")
			return None