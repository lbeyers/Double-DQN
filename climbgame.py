"""Implementation of the Climb game as in Contrasting Centralized and Decentralized Critics in
Multi-Agent Reinforcement Learning based on code from OG-MARL"""
import numpy as np
from gymnasium.spaces import Discrete, Box
from og_marl.environments.base import BaseEnvironment    

class ClimbGame(BaseEnvironment):

    def __init__(
        self
    ):
        self._environment = None
        self.state = [0,0]
        self.possible_agents = [f"agent_{n}" for n in range(2)]

        self._num_agents = 2
        self._num_actions = 3
        self._obs_dim = 1
        self.rewards = [[11,-30,0],[-30,7,6],[0,0,5]]

        self.action_spaces = {agent: Discrete(self._num_actions) for agent in self.possible_agents}
        self.observation_spaces = {agent: Box(-np.inf, np.inf, (self._obs_dim,)) for agent in self.possible_agents}

        self.info_spec = {
            "state": np.zeros((2,), "float32"),
            "legals": {agent: np.zeros((self._num_actions,), "int64") for agent in self.possible_agents}
        }

    def reset(self):
        """Resets the env."""

        # Reset the environment
        self.state = [np.random.randint(0,4),np.random.randint(0,4)]
        self._done = False

        # Get observation from env
        observations = self.state
        observations = {agent: observations[i] for i, agent in enumerate(self.possible_agents)}

        legal_actions = self._get_legal_actions()
        legals = {agent: legal_actions[i] for i, agent in enumerate(self.possible_agents)}
        

        info = {
            "legals": legals,
            "state": self.state
        }

        return observations, info

    def step(self, actions):
        """Step in env."""

        # Convert dict of actions to list for SMAC
        smac_actions = []
        for agent in self.possible_agents:
            act = actions[agent]
            if act==1:
                self.state[agent] -=1
                self.state[agent] = int(self.state%3)
            elif act==2:
                self.state[agent] +=1
                self.state[agent] = int(self.state%3)

        # Step the SMAC environment
        done = self.state[0]==0 & self.state[1] ==0
        reward = self.rewards[self.state]

        # Get the next observations
        observations = self.state
        observations = {agent: observations[i] for i, agent in enumerate(self.possible_agents)}

        legal_actions = self._get_legal_actions()
        legals = {agent: legal_actions[i] for i, agent in enumerate(self.possible_agents)}

        env_state = self.state
        # Convert team reward to agent-wise rewards
        rewards = {agent: np.array(reward, "float32") for agent in self.possible_agents}

        terminals = {agent: np.array(done) for agent in self.possible_agents}
        truncations = {agent: False for agent in self.possible_agents}

        info = {
            "legals": legals,
            "state": env_state
        }

        return observations, rewards, terminals, truncations, info

    def _get_legal_actions(self):
        """Get legal actions from the environment."""
        return [0,1,2]

    def get_stats(self):
        """Return extra stats to be logged."""
        return None

