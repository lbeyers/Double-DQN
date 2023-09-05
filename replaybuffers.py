import numpy as np
import tensorflow as tf
from tensorflow import keras
import sonnet as snt

#for super agent: smac_doubledqn
#for decentralised: dec_smac_doubledqn
class ReplayBuffer():
	def __init__(self, max_size, input_dims,action_space_size):
		self.mem_size = max_size
		self.action_space_size = action_space_size
		self.mem_cntr = 0
		self.state_memory = np.zeros((self.mem_size,*input_dims), dtype=np.float32)
		self.new_state_memory = np.zeros((self.mem_size, *input_dims), 
					dtype = np.float32)
		self.action_memory = np.zeros(self.mem_size, dtype=np.int32)
		self.reward_memory = np.zeros(self.mem_size, dtype=np.float32)
		self.terminal_memory = np.zeros(self.mem_size, dtype = np.int32)
		self.avail_action_memory = np.zeros((self.mem_size,action_space_size),dtype=bool)
		
	def store_transition(self,state,action,reward,state_,done,avail_acts):
		index = self.mem_cntr % self.mem_size
		self.state_memory[index] = state
		self.new_state_memory[index] = state_
		self.reward_memory[index] = reward
		self.action_memory[index] = action
		self.terminal_memory[index] = int(done)
		self.avail_action_memory[index] = avail_acts
		self.mem_cntr += 1
		
	def sample_buffer(self, batch_size):
		max_mem = min(self.mem_cntr, self.mem_size)
		batch = np.random.choice(max_mem, batch_size, replace = False)
		
		states = self.state_memory[batch]
		states_ = self.new_state_memory[batch]
		rewards = self.reward_memory[batch]
		actions = self.action_memory[batch]
		terminal = self.terminal_memory[batch]
		avail_acts = self.avail_action_memory[batch]
		
		return states, actions, rewards, states_, terminal, avail_acts
	
#for shared weights and vdn
class SharingReplayBuffer(ReplayBuffer):
	def __init__(self, max_size, input_dims,action_space_size, n_agents):
		self.mem_size = max_size
		self.n_agents = n_agents
		self.action_space_size = action_space_size
		self.mem_cntr = 0
		self.state_memory = np.zeros((self.mem_size,n_agents,*input_dims), dtype=np.float32)
		self.new_state_memory = np.zeros((self.mem_size, n_agents, *input_dims), 
					dtype = np.float32)
		self.action_memory = np.zeros((self.mem_size,n_agents), dtype=np.int32)
		self.reward_memory = np.zeros(self.mem_size, dtype=np.float32)
		self.terminal_memory = np.zeros(self.mem_size, dtype = np.int32)
		self.avail_action_memory = np.zeros((self.mem_size,n_agents,action_space_size),dtype=bool)
		