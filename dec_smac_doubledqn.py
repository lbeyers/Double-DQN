import numpy as np
import tensorflow as tf
from tensorflow import keras
import sonnet as snt

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
		
	def store_transition(self,state,action,reward,state_, done, avail_act_oh):
		index = self.mem_cntr % self.mem_size
		self.state_memory[index] = state
		self.new_state_memory[index] = state_
		self.reward_memory[index] = reward
		self.action_memory[index] = action
		self.terminal_memory[index] = int(done)

		#build one-hot of available actions for storage (already accessed this way)
		#avail_act_onehot = np.zeros(self.action_space_size, dtype=bool)
		#avail_act_onehot[avail_acts] = 1
		self.avail_action_memory[index] = avail_act_oh

		self.mem_cntr += 1
		
	def sample_buffer(self, batch_size):
		max_mem = min(self.mem_cntr, self.mem_size)
		batch = np.random.choice(max_mem, batch_size, replace = False)

		states = self.state_memory[batch]
		states_ = self.new_state_memory[batch]
		rewards = self.reward_memory[batch]
		actions = self.action_memory[batch]
		terminal = self.terminal_memory[batch]

		# convert from one-hot to list of arrays of avail acts
		avail_acts_oh = self.avail_action_memory[batch]
		#avail_acts = [np.where(x)[0] for x in avail_acts]
		
		return states, actions, rewards, states_, terminal, avail_acts_oh
		
def build_dqn(n_actions, fc1_dims, fc2_dims):
	model = snt.Sequential([
		snt.Linear(fc1_dims),
		tf.nn.relu,
		snt.Linear(fc2_dims),
		tf.nn.relu,
		snt.Linear(n_actions)
	])
	return model
		
class Agent():
	def __init__(self, lr, gamma, n_actions, epsilon, batch_size,
			input_dims, epsilon_dec=1e-4, epsilon_end=0.01,
			mem_size=1e6, fname='smac_doubledqn_model.h5'):
		self.action_space = [i for i in range(n_actions)]
		self.gamma = gamma
		self.epsilon = epsilon
		self.eps_dec = epsilon_dec
		self.eps_min = epsilon_end
		self.batch_size = batch_size
		self.model_file = fname
		self.memory = ReplayBuffer(mem_size, input_dims, n_actions)
		self.q_online = build_dqn(n_actions,256,256)
		self.q_target = build_dqn(n_actions,256,256)
		self.optimizer = snt.optimizers.RMSProp(learning_rate=lr)

	# update target network every now and again
	def update_target_network(self):

		online_variables = self.q_online.variables
		target_variables = self.q_target.variables
		for source, dest in zip(online_variables, target_variables):
			dest.assign(source)
		
	def store_transition(self, state, action, reward, new_state, done, avail_acts):
		self.memory.store_transition(state, action, reward, new_state, done, avail_acts)
		
	def choose_action(self, observation,avail_acts_oh):
		avail_acts = np.nonzero(avail_acts_oh)[0]
		if np.random.random() < self.epsilon:
			action = np.random.choice(avail_acts)
		else:
			# https://stackoverflow.com/questions/64520917/valueerror-error-when-checking-input-expected-dense-1-input-to-have-shape-8
			state = np.reshape(np.array(observation), (1,len(observation)))
			actions = self.q_online.__call__(state).numpy()[0]
			
			subset_idx = np.argmax(actions[avail_acts])
			action = avail_acts[subset_idx]
			
		return action
		
	def learn(self):
		if self.memory.mem_cntr < self.batch_size:
			return {}
		states, actions, rewards, states_, dones, avail_acts_oh = \
			self.memory.sample_buffer(self.batch_size)
			
		logs = self.train_step(states, actions, rewards, states_, dones, avail_acts_oh)
	
		self.epsilon = self.epsilon - self.eps_dec if self.epsilon > \
			self.eps_min else self.eps_min
		
		return logs

	@tf.function
	def train_step(self, states, actions, rewards, states_, dones, avail_acts_oh):
		dones = tf.cast(dones, dtype="float32")

		# for getting the values
		q_target_next = self.q_target(states_)
		# for ranking the actions
		q_online_next = self.q_online(states_)

		# get target actions using onehot availability with online values
		q_next = tf.where(avail_acts_oh, q_online_next,-1e10)
		target_actions = tf.math.argmax(q_next,axis=1)

		# grab static values for winning actions
		q_target = tf.gather(q_target_next,target_actions,batch_dims=1)

		target = rewards + self.gamma * (1 - dones) * q_target
		# prevents GradientTape from watching these later
		target = tf.stop_gradient(target)

		with tf.GradientTape() as tape:

			q_online = self.q_online(states)
			q_taken = tf.gather(q_online,actions,batch_dims=1)

			# mse
			error = (target-q_taken)**2
			loss = tf.reduce_mean(error)

		# sonnet
		variables = self.q_online.trainable_variables
		gradients = tape.gradient(loss, variables)
		self.optimizer.apply(gradients, variables)

		return {"loss": loss, "mean q": tf.reduce_mean(q_taken)}
			
