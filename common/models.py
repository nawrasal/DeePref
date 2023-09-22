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
#####
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple
from itertools import count
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T
from torch.autograd import Variable
from collections import OrderedDict
from common.params import *
from scipy.interpolate import make_interp_spline, BSpline
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
parser.add_argument('--rnn_memory', type=str, default="random", help="sequential, random")
parser.add_argument('--mode', type=str, required=False, default="test", help="train, test")

args = parser.parse_args()

##########################################
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
idx = 0 #for tracking bw and fw paths DRQN

##########################################  
class ReplayMemory:
	def __init__(self, capacity):
		self.capacity = capacity
		self.memory = []

	def push(self, transition):
		self.memory.append(transition)
		if len(self.memory) > self.capacity:
			del self.memory[0]

	def sample(self, batch_size):
		return random.sample(self.memory, batch_size)

	def __len__(self):
		return len(self.memory)
###############################################
class RecurrentReplayMemory:
	def __init__(self, capacity, sequence_length= SEQUENCE_LENGTH):
		self.capacity = capacity
		self.memory = []
		self.seq_length=sequence_length

	def push(self, episode):
		if len(self.memory) + 1 >= self.capacity:
			self.memory[0:(1+len(self.memory))-self.capacity] = []
		self.memory.append(episode)

	def sample(self, batch_size):
		sampled_episodes = random.sample(self.memory, batch_size)
		sampled_traces = []
		if args.rnn_memory == "random":
			#random sampling
			for episode in sampled_episodes:
				point = np.random.randint(0, len(episode)+1 - self.seq_length)
				sampled_traces.append(episode[point:point+self.seq_length])
			sampled_traces = np.array(sampled_traces)

		elif args.rnn_memory == "sequential":
			#sequential sampling
			for episode in sampled_episodes:
				sampled_traces.append(episode[-self.seq_length:]) #seq_length = total timesteps in episode (the entire episoode)
			sampled_traces = np.array(sampled_traces)
		#print(sampled_traces)
		#print(sampled_traces[0])
		#print(type(sampled_traces[0]))
		#print("sample_traces", sampled_traces.shape)
		return sampled_traces

	def __len__(self):
		return len(self.memory)


##########****LinearBody****###################
class LinearBody(nn.Module):
	def __init__(self, input_shape, num_actions):
		super(LinearBody, self).__init__()
		
		self.input_shape = input_shape
		self.num_actions = num_actions

		self.fc1 = nn.Linear(input_shape[0], body_size)
	def forward(self, x):
		x = F.relu(self.fc1(x))
		return x

	def feature_size(self):
		return self.fc1(torch.zeros(1, *self.input_shape)).view(1, -1).size(1) #size(0)
###############################################
###########****DQN*****########################
class DQN(nn.Module):
	def __init__(self, input_shape, n_actions):
		super(DQN, self).__init__()
		# Inputs to hidden layer linear transformation
		self.input_shape = input_shape
		self.n_actions= n_actions
		self.fc1 = nn.Linear(self.input_shape[0], 512)
		self.fc2 = nn.Linear(512, 512)
		self.fc3 = nn.Linear(512, self.n_actions)

	# Called with either one element to determine next action, or a batch
	# during optimization. Returns tensor([[left0exp,right0exp]...]).
	def forward(self, x):
		x = F.relu(self.fc1(x))
		x = F.relu(self.fc2(x))
		x = self.fc3(x)
		return x
###############################################
class DRQN(nn.Module):
	def __init__(self, input_shape, n_actions, hidden_size= hidden_size, num_layers= num_layers, num_directions= num_directions):
		super(DRQN, self).__init__()
		#init
		self.input_shape= input_shape
		#self.sequence_length = SEQUENCE_LENGTH
		self.hidden_size = hidden_size
		self.n_actions = n_actions
		self.num_layers = num_layers
		self.num_directions = num_directions
		if args.body== "LinearBody":
			self.body = LinearBody(input_shape, n_actions) #feed input to linear layer before LSTM
			self.in_features= self.body.feature_size()
		else:    
			self.in_features = input_shape[0] #(num_feats, )
		# Inputs to hidden layer linear transformation

		if args.RNN == "GRU":
			self.gru = nn.GRU(input_size= self.in_features, hidden_size= hidden_size, num_layers= num_layers, batch_first= True)
		elif args.RNN == "LSTM":
			self.lstm = nn.LSTM(input_size= self.in_features, hidden_size= hidden_size, num_layers= num_layers, batch_first= True)

		#self.fc1 = nn.Linear(hidden_size, 1000)
		self.fc2 = nn.Linear(hidden_size, self.n_actions)

	# Called with either one element to determine next action, or a batch
	# during optimization.
	# def forward(self, x, hidden= None):
	# 	batch_size= x.size(0)
	# 	sequence_length= x.size(1)
	# 	#x = x.view((-1,)+self.input_shape)
	# 	if args.body == "LinearBody":
	# 		x= self.body(x).view(batch_size, sequence_length, -1)
		
	# 	hidden = self.init_hidden(batch_size) if hidden is None else hidden
	# 	if args.RNN == "GRU":
	# 		out, new_hidden = self.gru(x, hidden)
	# 	elif args.RNN == "LSTM":
	# 		out, new_hidden = self.lstm(x, hidden)

	# 	#x = F.relu(self.fc1(x))
	# 	x = self.fc2(out)
	# 	return x, new_hidden

	#TBPTT
	def forward(self, x, hidden=None, policy_net_optim=True):
		torch.autograd.set_detect_anomaly(True)
		global idx
		batch_size= x.size(0)
		sequence_length= x.size(1)
		#x = x.view((-1,)+self.input_shape)
		if args.body == "LinearBody":
			x= self.body(x).view(batch_size, sequence_length, -1)
		hidden = self.init_hidden(batch_size) if hidden is None else hidden #for target_net
		if args.RNN == "GRU":
			self.gru.flatten_parameters()
			out, new_hidden = self.gru(x, hidden)
		elif args.RNN == "LSTM":
			self.lstm.flatten_parameters()
			out, new_hidden = self.lstm(x, hidden)

		#x = F.relu(self.fc1(x))
		x = self.fc2(out)
		if policy_net_optim:
			def get_pr(idx_val):
				def pr(*args):
					print("doing backward {}".format(idx_val))
				return pr
			# print("x leaf", x.is_leaf)
			# print(new_hidden[0].is_leaf)
			# print(new_hidden[1].is_leaf)
			# print(x.grad)
			# print(new_hidden[0].grad)
			# print(new_hidden[1].grad)
			# print(new_hidden[0].requires_grad)
			# print(new_hidden[1].requires_grad)
			#x.requires_grad = True
			#print(x.requires_grad)
			#new_hidden[0].requires_grad = True
			#new_hidden[1].requires_grad = True
			x.register_hook(get_pr(idx))
			if args.RNN == "LSTM":
				new_hidden[0].register_hook(get_pr(idx)) #hidden cell
				new_hidden[1].register_hook(get_pr(idx)) #control cell

			elif args.RNN == "GRU":
				new_hidden.register_hook(get_pr(idx))
			print("doing fw {}".format(idx))
			idx += 1
		return x, new_hidden

	def init_hidden(self, batch_size):
		if args.RNN == "GRU":
			return torch.zeros(self.num_layers*self.num_directions, batch_size, self.hidden_size, device=device, dtype=torch.float)

		else: #LSTM
			return (torch.zeros(self.num_layers*self.num_directions, batch_size, self.hidden_size, device=device, dtype=torch.float),
				torch.zeros(self.num_layers*self.num_directions, batch_size, self.hidden_size, device=device, dtype=torch.float))
###############################################
class LSTM_classifier(nn.Module):
	def __init__(self, input_shape, n_actions, hidden_size= hidden_size, num_layers= num_layers, num_directions= num_directions, dropout= dropout):
		super(LSTM_classifier, self).__init__()
		self.input_shape= input_shape
		#self.sequence_length = SEQUENCE_LENGTH
		self.hidden_size = hidden_size
		self.n_actions = n_actions
		self.num_layers = num_layers
		self.num_directions = num_directions
		if args.body== "LinearBody":
			self.body = LinearBody(input_shape, n_actions) #feed input to linear layer before LSTM
			self.in_features= self.body.feature_size()

		elif args.body == "Embedding":
			self.body = nn.Embedding(input_shape[0], body_size) # sparse= True
			self.in_features = self.body(torch.zeros(1, *self.input_shape, dtype=torch.long)).view(1, -1).size(1)

		else:    
			self.in_features = input_shape[0] #(num_feats, )
		# Inputs to hidden layer linear transformation
		if args.RNN == "GRU":
			self.gru = nn.GRU(input_size= self.in_features, hidden_size= hidden_size, num_layers= num_layers, batch_first= True)
		elif args.RNN == "LSTM":
			self.lstm = nn.LSTM(input_size= self.in_features, hidden_size= hidden_size, num_layers= num_layers, batch_first= True, dropout= dropout)

		self.fc_1 = nn.Linear(hidden_size, 512)
		self.fc = nn.Linear(hidden_size, self.n_actions)
		self.relu = nn.ReLU()
		self.dropout = nn.Dropout(dropout)

	def forward(self, x, hidden= None, policy_net_optim=True):
		batch_size= x.size(0)
		sequence_length= x.size(1)
		if args.body == "LinearBody":
			x= self.body(x).view(batch_size, sequence_length, -1)

		elif args.body == "Embedding":
			x= self.body(x.long()).view(batch_size, sequence_length, -1)

		hidden = self.init_hidden(batch_size) if hidden is None else hidden #for target_net

		if args.RNN == "GRU":
			self.gru.flatten_parameters()
			out, new_hidden = self.gru(x, hidden)
		elif args.RNN == "LSTM":
			self.lstm.flatten_parameters()
			out, new_hidden = self.lstm(x, hidden)


		#h_0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size)).to(device) #hidden state
		#c_0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size)).to(device) #internal state
		# Propagate input through LSTM
		#hn = hn.view(-1, self.hidden_size) #reshaping the data for Dense layer next

		out = out[:, -1, :]

		# out = self.dropout(out)
		# out = self.relu(out)
		# out = self.fc_1(out) #first Dense
		# out = self.dropout(out)
		# out = self.relu(out) #relu

		out = self.fc(out) #Final Output
		return out, new_hidden

		# torch.autograd.set_detect_anomaly(True)
		# global idx
		# batch_size= x.size(0)
		# sequence_length= x.size(1)
		# #x = x.view((-1,)+self.input_shape)
		# if args.body == "LinearBody":
		# 	x= self.body(x).view(batch_size, sequence_length, -1)
		# hidden = self.init_hidden(batch_size) if hidden is None else hidden #for target_net
		# if args.RNN == "GRU":
		# 	self.gru.flatten_parameters()
		# 	out, new_hidden = self.gru(x, hidden)
		# elif args.RNN == "LSTM":
		# 	self.lstm.flatten_parameters()
		# 	out, new_hidden = self.lstm(x, hidden)

		# x = self.fc1(out)
		# x = nn.Relu(x)
		# x = self.fc2()
		# if policy_net_optim:
		# 	def get_pr(idx_val):
		# 		def pr(*args):
		# 			print("doing backward {}".format(idx_val))
		# 		return pr
		# 	# print("x leaf", x.is_leaf)
		# 	# print(new_hidden[0].is_leaf)
		# 	# print(new_hidden[1].is_leaf)
		# 	# print(x.grad)
		# 	# print(new_hidden[0].grad)
		# 	# print(new_hidden[1].grad)
		# 	# print(new_hidden[0].requires_grad)
		# 	# print(new_hidden[1].requires_grad)
		# 	#x.requires_grad = True
		# 	#print(x.requires_grad)
		# 	#new_hidden[0].requires_grad = True
		# 	#new_hidden[1].requires_grad = True
		# 	x.register_hook(get_pr(idx))
		# 	if args.RNN == "LSTM":
		# 		new_hidden[0].register_hook(get_pr(idx)) #hidden cell
		# 		new_hidden[1].register_hook(get_pr(idx)) #control cell

		# 	elif args.RNN == "GRU":
		# 		new_hidden.register_hook(get_pr(idx))
		# 	print("doing fw {}".format(idx))
		# 	idx += 1
		#return x, new_hidden

	def init_hidden(self, batch_size):
		if args.RNN == "GRU":
			return torch.zeros(self.num_layers*self.num_directions, batch_size, self.hidden_size, device=device, dtype=torch.float)

		else: #LSTM
			return (torch.zeros(self.num_layers*self.num_directions, batch_size, self.hidden_size, device=device, dtype=torch.float),
				torch.zeros(self.num_layers*self.num_directions, batch_size, self.hidden_size, device=device, dtype=torch.float))



###############################################