import numpy as np
import tensorflow as tf
from tensorflow import keras
import sonnet as snt

class ReplayBuffer():
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
		
	def store_transition(self,state,action,reward,state_, done, avail_act_oh):
		index = self.mem_cntr % self.mem_size
		self.state_memory[index] = state
		self.new_state_memory[index] = state_
		self.reward_memory[index] = reward
		self.action_memory[index] = action
		self.terminal_memory[index] = int(done)
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
		avail_acts_oh = self.avail_action_memory[batch]
		
		return states, actions, rewards, states_, terminal, avail_acts_oh
		
# todo change for RNN later?
def build_dqn(n_actions, fc1_dims, fc2_dims):
	model = snt.Sequential([
		snt.Linear(fc1_dims),
		tf.nn.relu,
		snt.Linear(fc2_dims),
		tf.nn.relu,
		snt.Linear(n_actions)
	])
	return model
		
class Cohort():
	def __init__(self, lr, gamma, n_actions, epsilon, batch_size,
			input_dims, n_agents, epsilon_dec=1e-4, epsilon_end=0.01,
			mem_size=1e6, fname='smac_doubledqn_model.h5'):
		self.action_space = [i for i in range(n_actions)]
		self.gamma = gamma
		self.epsilon = epsilon
		self.eps_dec = epsilon_dec
		self.eps_min = epsilon_end
		self.batch_size = batch_size
		self.n_agents = n_agents
		self.model_file = fname
		self.memory = ReplayBuffer(mem_size, input_dims, n_actions, n_agents)
		self.q_online = build_dqn(n_actions,256,256)
		self.q_target = build_dqn(n_actions,256,256)
		self.optimizer = snt.optimizers.RMSProp(learning_rate=lr)

	# update target network every now and again
	def update_target_network(self):
		online_variables = self.q_online.variables
		target_variables = self.q_target.variables
		for source, dest in zip(online_variables, target_variables):
			dest.assign(source)
		
	def store_transition(self, states, actions, reward, new_states, done, avail_acts_ls):
		self.memory.store_transition(states, actions, reward, new_states, done, avail_acts_ls)

	#def state_concat_agentid(states_block):

		#return state_agent_id
		
	def choose_action(self, observation,avail_acts_oh, agent_id):
		avail_acts = np.nonzero(avail_acts_oh)[0]
		if np.random.random() < self.epsilon:
			action = np.random.choice(avail_acts)
		else:
			# https://stackoverflow.com/questions/64520917/valueerror-error-when-checking-input-expected-dense-1-input-to-have-shape-8
			state = np.reshape(np.array(observation), (1,len(observation)))

			# concat with one-hot vector to condition Q-network
			oh_agentid = np.zeros(self.n_agents,dtype=np.float32)
			oh_agentid[agent_id] = 1
			oh_agentid = np.reshape(oh_agentid,(1,self.n_agents))
			agentid_state = np.concatenate((oh_agentid,state),axis=1,dtype=np.float32)

			actions = self.q_online.__call__(agentid_state).numpy()[0]
			
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
	def train_step(self, states_ls, actions_ls, rewards, states__ls, dones, avail_acts_oh_ls):
		dones = tf.cast(dones, dtype="float32")
		#rewards=rewards

		with tf.GradientTape() as tape:
			qvals = []
			target_qvals = []
			for agent_id in range(self.n_agents):
				states = states_ls[:,agent_id,:]
				actions = actions_ls[:,agent_id]
				states_ = states__ls[:,agent_id,:]
				avail_acts_oh = avail_acts_oh_ls[:,agent_id,:]

				# concatenate one-hot for agent id
				oh_agentid = tf.one_hot([agent_id]*self.batch_size,self.n_agents)

				agentid_states = tf.concat([oh_agentid,states],1)
				agentid_states_ = tf.concat([oh_agentid,states_],1)
				
				# for ranking the actions
				q_online_next = self.q_online(agentid_states_)

				# get target actions using onehot availability with online values
				q_next = tf.where(avail_acts_oh, q_online_next,-1e10)
				target_actions = tf.math.argmax(q_next,axis=1)

				# grab static values for winning actions
				q_target_next = self.q_target(agentid_states_)
				q_target = tf.gather(q_target_next,target_actions,batch_dims=1)

				q_online = self.q_online(agentid_states)
				q_taken = tf.gather(q_online,actions,batch_dims=1)

				qvals.append(q_taken)
				target_qvals.append(q_target)

			q_tot = tf.reduce_sum(qvals,axis=0)
			q_tot_target = tf.reduce_sum(target_qvals,axis=0)

			target = rewards + self.gamma * (1 - dones) * q_tot_target

			# prevents GradientTape from watching these later
			target = tf.stop_gradient(target)

			# mse
			error = (target-q_tot)**2
			loss = tf.reduce_mean(error)

		# sonnet
		variables = self.q_online.trainable_variables
		gradients = tape.gradient(loss, variables)
		self.optimizer.apply(gradients, variables)

		return {"loss": loss, "mean q": tf.reduce_mean(q_taken)}
			
