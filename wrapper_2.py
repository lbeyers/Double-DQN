# https://semaphoreci.com/community/tutorials/building-and-testing-an-api-wrapper-in-python

# for an env wrapper, we need:
# - n_actions: int
# - the ability to: step
#   - obs, reward, done
# - the ability to: reset env
#   - obs?
# - the shape of the observations
import itertools
import numpy as np
from smac.env import StarCraft2Env
#def 

# SMAC 3m

class GymWrapper():
    def __init__(self, mapname, seed):
        self.smac_env = StarCraft2Env(map_name=mapname, seed=seed)
        self.n_actions = self.smac_env.get_env_info()["n_actions"]
        self.n_agents = self.smac_env.get_env_info()["n_agents"]

        # get length of a flattened observation & hope it stays
        # the same for a game
        self.smac_env.reset()
        obs_list = self.smac_env.get_obs()
        super_obs = np.array(obs_list).flatten()
        self.obs_len = super_obs.shape

        # create all combinations with cartesian product
        self.superact_arr = list(itertools.product(range(self.n_actions),repeat = self.n_agents))


    def get_legal_multiact(self):

        avail_acts = []
        for agent_id in range(self.n_agents):
            avail_actions = self.smac_env.get_avail_agent_actions(agent_id)
            avail_acts.append(avail_actions)
        
        #claude code
        superact_legls = []
        for superact in self.superact_arr:
            legals_sum = 0
            for i in range(self.n_agents):
                legals_sum += avail_acts[i][superact[i]]
            if legals_sum != self.n_agents:
                superact_legls.append(0) # illegal action
            else:
                superact_legls.append(1) # legal action
        
        return superact_legls
    
    def wrap_up(self):
        self.smac_env.close()

    def step(self, action):
        # central writes down orders from tuple
        action_arr = self.superact_arr[action]
        reward, terminated, _ = self.smac_env.step(list(action_arr))

        obs_list = self.smac_env.get_obs()
        super_obs = np.array(obs_list).flatten()

        placeholder=False

        available_actions = self.get_legal_multiact()

        return super_obs, reward, terminated, placeholder, available_actions

    def reset(self):
        #self.smac_env.seed(seed)
        self.smac_env.reset()

        obs_list = self.smac_env.get_obs()
        super_obs = np.array(obs_list).flatten()

        available_actions = self.get_legal_multiact()

        return super_obs, available_actions

    def action_to_multiact(self,action):
        multiact = []
        for agent_id in range(self.n_agents):
            action
        return multiact
