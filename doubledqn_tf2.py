import numpy as np
import tensorflow as tf
from tensorflow import keras
import sonnet as snt
from tensorflow.keras.saving import load_model

class ReplayBuffer():
	def __init__(self, max_size, input_dims):
		self.mem_size = max_size
		self.mem_cntr = 0
		
		self.state_memory = np.zeros((self.mem_size,*input_dims), dtype=np.float32)
		self.new_state_memory = np.zeros((self.mem_size, *input_dims), 
					dtype = np.float32)
		self.action_memory = np.zeros(self.mem_size, dtype=np.int32)
		self.reward_memory = np.zeros(self.mem_size, dtype=np.float32)
		self.terminal_memory = np.zeros(self.mem_size, dtype = np.int32)
		
	def store_transition(self,state,action,reward,state_, done):
		index = self.mem_cntr % self.mem_size
		self.state_memory[index] = state
		self.new_state_memory[index] = state_
		self.reward_memory[index] = reward
		self.action_memory[index] = action
		self.terminal_memory[index] = 1 - int(done)
		self.mem_cntr += 1
		
	def sample_buffer(self, batch_size):
		max_mem = min(self.mem_cntr, self.mem_size)
		batch = np.random.choice(max_mem, batch_size, replace = False)
		
		states = self.state_memory[batch]
		states_ = self.new_state_memory[batch]
		rewards = self.reward_memory[batch]
		actions = self.action_memory[batch]
		terminal = self.terminal_memory[batch]
		
		return states, actions, rewards, states_, terminal
		
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
			mem_size=1e6, fname='doubledqn_model.h5'):
		self.action_space = [i for i in range(n_actions)]
		self.gamma = gamma
		self.epsilon = epsilon
		self.eps_dec = epsilon_dec
		self.eps_min = epsilon_end
		self.batch_size = batch_size
		self.model_file = fname
		self.memory = ReplayBuffer(mem_size, input_dims)
		self.q_online = build_dqn(n_actions,256,256)
		self.q_target = build_dqn(n_actions,256,256)
		self.optimizer = snt.optimizers.RMSProp(learning_rate=lr)

	# update target network every now and again
	def update_target_network(self):
		# https://github.com/deepmind/sonnet/issues?page=1&q=is%3Aissue+is%3Aopen
		tf.saved_model.save(self.q_online,self.model_file)
		self.q_target = tf.saved_model.load(self.model_file)

		# online_variables = self.q_online.variables
		# target_variables = self.q_target.variables
		# for source, dest in zip(online_variables, target_variables):
		# 	dest.assign(source)
		
	def store_transition(self, state, action, reward, new_state, done):
		self.memory.store_transition(state, action, reward, new_state, done)
		
	def choose_action(self, observation):
		if np.random.random() < self.epsilon:
			action = np.random.choice(self.action_space)
		else:
			# https://stackoverflow.com/questions/64520917/valueerror-error-when-checking-input-expected-dense-1-input-to-have-shape-8
			state = np.reshape(np.array(observation), (1,len(observation)))
			actions = self.q_online.__call__(state)
			
			action = np.argmax(actions)
			
		return action
		
	def learn(self):
		if self.memory.mem_cntr < self.batch_size:
			return {}
		states, actions, rewards, states_, dones = \
			self.memory.sample_buffer(self.batch_size)
			
		self.train_step(states, actions, rewards, states_, dones)
		logs = self.train_step(states, actions, rewards, states_, dones)
	
		self.epsilon = self.epsilon - self.eps_dec if self.epsilon > \
			self.eps_min else self.eps_min
		
		return logs

	@tf.function
	def train_step(self, states, actions, rewards, states_, dones):
		dones = tf.cast(dones, dtype="float32")

		# for getting the values
		q_target_next = self.q_target(states_)
		# for ranking the actions
		q_online_next = self.q_online(states_)
		
		# choose action using online values
		target_actions = tf.math.argmax(q_online_next, axis=1)

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
			
	def save_model(self):
		self.q_online.save(self.model_file)
	
	def load_model(self):
		self.q_online = load_model(self.model_file)
