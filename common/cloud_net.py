from argparse import ArgumentParser
from collections import namedtuple
from common.models import *
from common.params import *
import torch
import torch.nn as nn
#from common.models import *
#from common.common import *
##########################################
#prepare args
parser: ArgumentParser = ArgumentParser()
parser.add_argument('--EC', type=int, required=True, help="edge_capacity")
parser.add_argument('--NO_edges', type=int, default=3, help="")
parser.add_argument('--algo', type=str, required=False, default="DRQN", help="DQN")
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
parser.add_argument('--mode', type=str, required=False, default="train", help="train, test")

args = parser.parse_args()

def init_cloud():
	#global variables for cloud
	global memory
	global policy_net
	global target_net
	global episode_buffer
	global seq
	global user_history
	global states

	if args.algo == "DRQN":
		memory = RecurrentReplayMemory(100000)
		policy_net=DRQN(input_shape, n_actions).to(device)
		target_net=DRQN(input_shape, n_actions).to(device)
		target_net.load_state_dict(policy_net.state_dict())
		target_net.eval()
		episode_buffer = []
		seq = [np.zeros(input_shape) for j in range(SEQUENCE_LENGTH)]
		#TBPTT
		if args.RNN == "GRU":
			init_state = torch.zeros(num_layers*num_directions, BATCH_SIZE, hidden_size, device=device, dtype=torch.float)
			states = [(None, init_state)]
		elif args.RNN == "LSTM":
			init_state = (torch.zeros(num_layers*num_directions, BATCH_SIZE, hidden_size, device=device, dtype=torch.float, requires_grad= True),
			torch.zeros(num_layers*num_directions, BATCH_SIZE, hidden_size, device=device, dtype=torch.float, requires_grad= True))
			states = [((None, None), init_state)]


	elif args.algo == "DQN":
		#global memory
		memory = ReplayMemory(10000)
		#global policy_net
		policy_net = DQN(input_shape, n_actions).to(device)
		#global target_net
		target_net = DQN(input_shape, n_actions).to(device)
		target_net.load_state_dict(policy_net.state_dict())
		target_net.eval()
		user_history = np.zeros([user_history_len*one_hot_size])


#def load_cloud():
#	#global variables for cloud
#	global memory
#	global policy_net
#	global target_net
#	global episode_buffer
#	global seq
#	global user_history
#
#	if args.algo == "DQN":
#		#global memory
#		memory = ReplayMemory(10000)
#		name= str(args)
#		path = "./DQN_models/" + str(args) + ".csv"
#		#global policy_net
#		policy_net = DQN(input_shape, n_actions).to(device)
#		policy_net.load_state_dict(torch.load(path, map_location= device))
#		#global target_net
#		target_net = DQN(input_shape, n_actions).to(device)
#		target_net.load_state_dict(policy_net.state_dict())
#		target_net.eval()
#		user_history = np.zeros([user_history_len*one_hot_size])


# def reset_cloud():
# 	print("RESETTING CLOUD")
# 	if args.algo == "DRQN":
# 		episode_buffer = []
# 		seq = [np.zeros(input_shape) for j in range(SEQUENCE_LENGTH)]
# 	elif args.algo == "DQN":
# 		user_history = np.zeros([user_history_len*one_hot_size])
	
