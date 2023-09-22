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
from common.common import *
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

##########*****EDGE****########################
class EDGE():
	def __init__(self, agents, n_actions= n_actions, input_shape= input_shape, batch_size= BATCH_SIZE, device= device, ReplayMemory=ReplayMemory):
		super(EDGE, self).__init__()
		self.agents = agents
		self.batch_size = batch_size
		self.input_shape = input_shape
		self.device = device
		self.n_actions = n_actions
		self.movies_list = movies_list
		self.model_step=0
		#self.init_items= np.random.choice(movies_list, size= args.EC, replace= False)

		#DRQN
		self.sequence_length = SEQUENCE_LENGTH
	
		#DRQN
		if args.algo == "DRQN":
			self.memory= RecurrentReplayMemory(100000)
			self.policy_net= DRQN(self.input_shape, self.n_actions).to(self.device)
			self.target_net= DRQN(self.input_shape, self.n_actions).to(self.device)
			self.target_net.load_state_dict(self.policy_net.state_dict())
			self.target_net.eval()
			#TBPTT
			if args.RNN == "GRU":
				self.init_state = torch.zeros(num_layers*num_directions, self.batch_size, hidden_size, device=device, dtype=torch.float)
				self.states = [(None, self.init_state)]
			elif args.RNN == "LSTM":
				self.init_state = (torch.zeros(num_layers*num_directions, self.batch_size, hidden_size, device=device, dtype=torch.float, requires_grad= True),
				torch.zeros(num_layers*num_directions, self.batch_size, hidden_size, device=device, dtype=torch.float, requires_grad= True))
				self.states = [((None, None), self.init_state)]

			self.k1= k1
			self.k2 = k2
			self.retain_graph = self.k1 <= self.k2

		elif args.algo == "DQN":
			self.memory = ReplayMemory(10000)
			self.policy_net = DQN(self.input_shape, self.n_actions).to(self.device)
			self.target_net = DQN(self.input_shape, self.n_actions).to(self.device)
			self.target_net.load_state_dict(self.policy_net.state_dict())
			self.target_net.eval()

		#self.z = torch.zeros(input_shape	)

	def reset_hidden(self):
		assert args.algo == "DRQN", "Error in reseting hidden"
		self.seq = [np.zeros(self.input_shape) for j in range(self.sequence_length)]

	def reset_all(self):
		#reset agent variables and model variables
		#reset agent variables
		for agent in self.agents:
			if agent != None:
				agent.reset_env()
		#reset model variables
		self.policy_net.eval()
		if args.algo == "DQN":
			self.user_history = np.zeros([user_history_len*one_hot_size])
		elif args.algo == "DRQN":
			self.episode_buffer = []
			self.reset_hidden()

	def select_action(self, state_np, ID):
		#epsilson-greedy
		if args.exploration == "epsilon":
			if self.agents[ID].time_step < EXPLORATION_STEPS:
				eps_threshold = self.agents[ID].EPS_START
			else:
				#eps_threshold = ExponentialScheduler(self.time_step, self.EPS_START, EPS_END, EPS_DECAY)
				eps_threshold, self.agents[ID].current = LinearScheduler(self.agents[ID].current, self.agents[ID].inc, EPS_END)


			sample = random.random()
			# print("EPS_START", self.EPS_START)
			# print("eps_threshold", eps_threshold)
			# print("sample", sample)
			if sample > eps_threshold:
				#print("EXPLOITED")
				self.agents[ID].exploit_count+=1
				with torch.no_grad():
					if args.algo == "DRQN":
						#print(state)
						X = torch.tensor([self.seq], device=self.device, dtype=torch.float)
						qval, _ = self.policy_net(X, policy_net_optim=False)
						action_val = qval[:, -1, :] #select last element of seq
						action = action_val.max(1)[1].view(1,1)
						return action.item()

					elif args.algo == "DQN":
						X = torch.tensor([state_np], device=device, dtype=torch.float)
						qval = self.policy_net(X)
						action = qval.max(1)[1].view(1,1)
						# top_k_actions = qval.topk(args.EC)[1][0]
						# for i in range(args.EC):
						# 	action = top_k_actions[i]
						# 	if action.item() in self.edge_storage_LRU:
						# 		continue
						# 	else:
						# 		break
						return action.item()
			else:
				self.agents[ID].explore_count+=1
				if args.algo == "DRQN":
					selected_action = np.random.choice([i for i in self.movies_list if i not in self.agents[ID].edge_storage_LRU.keys()])
					action= torch.tensor([[selected_action]], device=self.device, dtype=torch.long)
					return action.item()
				elif args.algo == "DQN":
					selected_action = np.random.choice([i for i in self.movies_list if i not in self.agents[ID].edge_storage_LRU.keys()])
					action= torch.tensor([[selected_action]], device=self.device, dtype=torch.long)
					return action.item()

		elif args.exploration == "softmax":
			with torch.no_grad():
				if args.algo == "DRQN":
					#print(state)
					X = torch.tensor([self.seq], device=self.device, dtype=torch.float)

					qval, _ = self.policy_net(X, policy_net_optim=False)
					action_val = qval[:, -1, :] #select last element of seq
					softmax_probs = F.softmax(action_val/Temperature, dim=1)
					action = softmax_probs.max(1)[1].view(1,1)
					return action.item()

				elif args.algo == "DQN":
					X = torch.tensor([state_np], device=device, dtype=torch.float)
					qval = self.policy_net(X)
					softmax_probs= F.softmax(qval/Temperature, dim=1)
					action = softmax_probs.max(1)[1].view(1,1)
					# top_k_actions = softmax_probs.topk(args.EC)[1][0]
					# for i in range(args.EC):
					# 	action = top_k_actions[i]
					# 	if action.item() in self.edge_storage_LRU:
					# 		continue
					# 	else:
					# 		break
					#print(action.item())
					return action.item()
		else:
			print("error in exploration strategy")

	def evaluate_action(self, state_np, ID):
		#no exploration
		with torch.no_grad():
			if args.algo == "DRQN":
				#assuming we have access to the hidden layer when evaluating (the hidden layer isn't reset)
				#print(state)
				X = torch.tensor([self.seq], device=self.device, dtype=torch.float)
				qval, _ = self.policy_net(X, policy_net_optim=False)
				action = qval[:, -1, :] #select last element of seq
				action = action.max(1)[1].view(1,1)

				# sample = random.random()
				# eps_threshold = 0.1
				# if sample > eps_threshold:
				# 	print("exploited")
				# 	return action.item()
				# else:
				# 	selected_action = np.random.choice([i for i in self.movies_list if i not in self.agents[ID].edge_storage_LRU.keys()])
				# 	print("selected_action", selected_action)
				# 	action= torch.tensor([[selected_action]], device=self.device, dtype=torch.long)					

				# top_k_actions = qval[:, -1, :].topk(args.EC)[1][0]
				# for i in range(args.EC):
				# 	action = top_k_actions[i]
				# 	if action.item() in self.agents[ID].edge_storage_LRU:
				# 		continue
				# 	else:
				# 		break
				return action.item()

			elif args.algo == "DQN":
				X = torch.tensor([state_np], device=device, dtype=torch.float)
				qval = self.policy_net(X)
				action = qval.max(1)[1].view(1,1)
				# top_k_actions = qval.topk(args.EC)[1][0]
				# for i in range(args.EC):
				# 	action = top_k_actions[i]
				# 	if action.item() in self.agents[ID].edge_storage_LRU:
				# 		continue
				# 	else:
				# 		break
				return action.item()

	def prepare_minibatch_DQN(self):
		transitions = self.memory.sample(self.batch_size)
		state_batch, action_batch, reward_batch, next_state_batch = zip(*transitions)
		shape = (-1, )+self.input_shape
		state_batch = torch.tensor(state_batch, device=self.device, dtype=torch.float).view(shape)
		action_batch = torch.tensor(action_batch, device=self.device, dtype=torch.long).squeeze().view(-1, 1)
		reward_batch = torch.tensor(reward_batch, device=self.device, dtype=torch.float).squeeze().view(-1, 1)

		non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, next_state_batch)), device=self.device, dtype=torch.bool)

		try:
			non_final_next_states = torch.tensor([s for s in next_state_batch if s is not None], device=self.device, dtype=torch.float).view(shape)
			empty_next_state_values = False
		except:
			non_final_next_states = None
			empty_next_state_values = True

		return state_batch, action_batch, reward_batch, non_final_next_states, non_final_mask, empty_next_state_values

	def prepare_minibatch_DRQN(self):
		episodes = self.memory.sample(self.batch_size)
		#state_batch, action_batch, reward_batch, next_state_batch = zip(*transitions)
		episodes_T = episodes.T
		#print(episodes_T.shape)
		state_batch= episodes_T[0,:,:]
		action_batch = episodes_T[1,:,:].astype(np.int_)
		reward_batch = episodes_T[2,:,:].astype(np.float)
		next_state_batch = episodes_T[3,:,:]
		#transpose back to original shape
		state_batch = state_batch.T
		state_batch = state_batch.tolist()
		next_state_batch = next_state_batch.T
		action_batch = action_batch.T
		reward_batch = reward_batch.T

		#ensures we have the right shapes
		shape = (self.batch_size, self.sequence_length)+self.input_shape
		state_batch = torch.tensor(state_batch, device=self.device, dtype=torch.float).view(shape)
		action_batch = torch.tensor(action_batch, device=self.device, dtype=torch.long).view(self.batch_size, self.sequence_length, -1)
		reward_batch = torch.tensor(reward_batch, device=self.device, dtype=torch.float).view(self.batch_size, self.sequence_length)
		# #get set of next states for end of each sequence
		# next_state_batch = tuple([next_state_batch[i] for i in range(len(next_state_batch)) if (i+1)%(self.sequence_length)==0])

		non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, next_state_batch)), device=self.device, dtype=torch.bool)

		try:
			non_final_next_states = torch.tensor([s for s in next_state_batch if s is not None], device=self.device, dtype=torch.float)
			#non_final_next_states = torch.cat([state_batch[non_final_mask, 1:, :], non_final_next_states], dim=1)
			empty_next_state_values = False
		except:
			print("empty next states")
			print(next_state_batch.shape)
			empty_next_state_values = True
		# print("state.shape", state_batch.shape)
		# print("reward.shape", reward_batch.shape)
		# print("action shape", action_batch.shape)
		#print("non_final.shape", non_final_next_states.shape)
		return state_batch, action_batch, reward_batch, non_final_next_states, non_final_mask, empty_next_state_values 

	def optimize_DQN_model(self):
		state_batch, action_batch, reward_batch, non_final_next_states, non_final_mask, empty_next_state_values = self.prepare_minibatch_DQN()

		#estimate
		current_q_values = self.policy_net(state_batch).gather(1, action_batch)

		#target
		#target
		with torch.no_grad():
			max_next_q_values = torch.zeros(self.batch_size, device=self.device, dtype=torch.float).unsqueeze(dim=1)
			if not empty_next_state_values:
				max_next_action = self.target_net(non_final_next_states).max(dim=1)[1].view(-1, 1)
			max_next_q_values[non_final_mask] = self.target_net(non_final_next_states).gather(1, max_next_action)

			expected_q_values = reward_batch + GAMMA*max_next_q_values

		# diff = (expected_q_values - current_q_values)
		# loss = MSE(diff)
		# loss = loss.mean()
		#print("expected_q_values", expected_q_values.shape)
		#print("current_q_values", current_q_values.shape)
		loss_fn = nn_MSE()
		loss = loss_fn(current_q_values, expected_q_values)
		#loss_np = loss.clone().detach().cpu().numpy()
		#self.loss.append(loss_np)
		#print(loss)

		optimizer = optim.Adam(self.policy_net.parameters(), lr=0.0001) #Adam, RMSprop

		#self.z = list(self.policy_net.parameters())[0].clone()

		# Optimize the model
		optimizer.zero_grad()
		loss.backward()
		#plot gradient flow before clipping
		#self.plot_grad_flow()
		#gradient clipping
		for param in self.policy_net.parameters():
			param.grad.data.clamp_(-1, 1) #gradient clipping
		#nn.utils.clip_grad_norm_(self.policy_net.parameters(), 10)
		optimizer.step()
		
	def optimize_DRQN_model(self):
		state_batch, action_batch, reward_batch, non_final_next_states, non_final_mask, empty_next_state_values  = self.prepare_minibatch_DRQN()


		#TBPTT
		if args.RNN == "GRU":
			state = self.states[-1][1].clone().detach()
			state.requires_grad = True

		elif args.RNN == "LSTM":
			state_h = self.states[-1][1][0].detach()
			state_c = self.states[-1][1][1].detach()
			#state[0].clone().detach() #hidden_cell
			#state[1].clone().detach() #control_cell
			#print("state_h leaf", state_h.is_leaf)
			#print("state_c leaf", state_c.is_leaf)
			#state[0].retain_grad()
			#state[1].retain_grad()
			state_h.requires_grad = True
			state_c.requires_grad = True
			state = (state_h, state_c)
		#estimate
		current_q_values, new_state = self.policy_net(state_batch, state)
		current_q_values = current_q_values.gather(2, action_batch).squeeze()
		self.states.append((state, new_state))
		#target
		with torch.no_grad():
			max_next_q_values = torch.zeros((self.batch_size, self.sequence_length), device=self.device, dtype=torch.float)
			if not empty_next_state_values:
				max_next, _ = self.target_net(non_final_next_states, policy_net_optim=False)
			max_next_q_values[non_final_mask] = max_next.max(dim=2)[0]

			expected_q_values = reward_batch + GAMMA*max_next_q_values
			expected_q_values = expected_q_values.squeeze()

		#diff = (expected_q_values - current_q_values)
		# loss = self.MSE(diff)
		# loss = loss.mean()
		while len(self.states) > self.k2:
			# Delete stuff that is too old
			del self.states[0]

		if (self.model_step+1)%self.k1 == 0:
			loss_fn = nn_MSE()
			loss = loss_fn(current_q_values, expected_q_values)
			loss_np = loss.clone().detach().cpu().numpy()
			self.loss.append(loss_np)
			optimizer = optim.Adam(self.policy_net.parameters(), lr=0.0001) #Adam, RMSprop
			optimizer.zero_grad()
			start = time.time()
			#print("self.retain_graph", self.retain_graph)
			loss.backward(retain_graph=self.retain_graph)
			for i in range(self.k2-1):
				# if we get all the way back to the "init_state", stop
				if args.RNN == "LSTM":
					if self.states[-i-2][0] == (None, None):
						break
					curr_grad_h = self.states[-i-1][0][0].grad
					curr_grad_c = self.states[-i-1][0][1].grad
					#we shouldn't call backprop step twice for each cell, need to fix
					self.states[-i-2][1][0].backward(curr_grad_h, retain_graph=self.retain_graph)
					self.states[-i-2][1][1].backward(curr_grad_c, retain_graph=self.retain_graph)

				elif args.RNN == "GRU":
					if self.states[-i-2][0] is None:
						break
					curr_grad = self.states[-i-1][0].grad
					self.states[-i-2][1].backward(curr_grad, retain_graph=self.retain_graph)
			print("bw time: {}".format(time.time()-start))
			#gradient clipping
			for param in self.policy_net.parameters():
				param.grad.data.clamp_(-5, 5) #gradient clipping
			#nn.utils.clip_grad_norm_(self.policy_net.parameters(), 10)

			optimizer.step()

		# #estimate
		# current_q_values, _ = self.policy_net(state_batch)
		# current_q_values = current_q_values.gather(2, action_batch).squeeze().view(-1,1)
		# #target
		# with torch.no_grad():
		# 	max_next_q_values = torch.zeros((self.batch_size, self.sequence_length), device=self.device, dtype=torch.float)
		# 	if not empty_next_state_values:
		# 		max_next, _ = self.target_net(non_final_next_states)
		# 	max_next_q_values[non_final_mask] = max_next.max(dim=2)[0]

		# 	expected_q_values = reward_batch + GAMMA*max_next_q_values
		# 	expected_q_values = expected_q_values.view(-1,1)

		# diff = (expected_q_values - current_q_values)
		# # #loss = huber(diff)
		# loss = MSE(diff).view(1, -1)
		# #mask first half of losses
		# split = self.sequence_length // 2
		# mask = torch.zeros([self.sequence_length, self.batch_size], device=self.device, dtype=torch.float)
		# mask[split:] = 1.0
		# mask = mask.view(1, -1)
		# # #print(mask.shape)
		# # #print(loss.shape)
		# loss *= mask
		# loss = loss.mean()

		# loss_np = loss.clone().detach().cpu().numpy()
		# self.loss.append(loss_np)

		# optimizer = optim.Adam(self.policy_net.parameters(), lr=0.001) #Adam, RMSprop

		# # Optimize the model
		# optimizer.zero_grad()
		# loss.backward()
		# #gradient clipping
		# for param in self.policy_net.parameters():
		# 	param.grad.data.clamp_(-5, 5) #gradient clipping
		# #nn.utils.clip_grad_norm_(self.policy_net.parameters(), 10)

		# optimizer.step()



	def step(self, user_request_row):
		#print("********************************")
		user_request = int(user_request_row.movie_id.values)
		ID = int(user_request_row.group)
		user_request_idx = movie_2_idx[user_request]
		one_hot_request = to_onehot(user_request_idx, movies_list)
		#print(one_hot_request)

		# if args.approach == "cloud":
		# 	global policy_net
		# 	global target_net
		# 	global memory
		# 	global user_history

			#print(list(policy_net.parameters()))
			#check if it's a miss (we only process cache misses here)
			# miss = self.check_miss(user_request_idx) #assuming transmission happens here
			# if miss == False:
			# 	return 	#memory, policy_net, target_net

		if args.algo == "DQN":
			self.user_history = np.delete(self.user_history, [i for i in range(one_hot_size)]) #index 0,1,2 ... one_hot_size
			self.user_history = np.append(self.user_history, one_hot_request)

			#prepare state representation
			#state = previous observations + binary indicator_vector for edge_items
			state_np = np.append(self.user_history, self.agents[ID].edge_items_indicator_vector)
		elif args.algo == "DRQN":
			state_np = np.append(one_hot_request, self.agents[ID].edge_items_indicator_vector).astype(np.float)

			#update observation seq
			self.seq.pop(0)
			self.seq.append(state_np)

		if args.mode== "train":
			action = self.select_action(state_np, ID)
		else: #test/transfer mode
			action = self.evaluate_action(state_np, ID)
		assert set(self.agents[ID].edge_items.tolist()) == set(self.agents[ID].edge_storage_LRU.keys()),"state should be reflected on LRU"
		#Take action and Observe reward and next state
		reward, obs = self.agents[ID].interact(action, user_request_row)

		# #check if end
		# if i == self.edge_df.index.size - step_size:
		#   next_state_np = None
			
		if args.algo == "DQN":
			next_state_np = state_np[:-len(movie_2_idx.values())]
			next_state_np = np.append(next_state_np, obs).astype(np.float)

		elif args.algo == "DRQN":
			next_state_np = np.append(one_hot_request, obs).astype(np.float)

		#train and optimize at every time-step
		if args.mode == "train":
			#store transition
			#action_one_hot = to_onehot(action)
			transition = (state_np, action, reward, next_state_np)
			if args.algo == "DQN":
				self.memory.push(transition)
			elif args.algo == "DRQN":
				self.episode_buffer.append(transition)			
			if (self.agents[ID].time_step+1) > EXPLORATION_STEPS:
				if (self.agents[ID].time_step+1) % POLICY_UPDATE == 0:
					#train and optimize a local model
					self.policy_net.train()
					# Perform one step of the optimization (on the target network)
					if len(self.memory) >= self.batch_size:
						if args.algo == "DQN":
							self.optimize_DQN_model()
						elif args.algo == "DRQN":
							self.optimize_DRQN_model()

					self.model_step +=1

		#agent_step
		self.agents[ID].step_()
		return



