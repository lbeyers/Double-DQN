from agentclasses import VDNCohort
import numpy as np
import tensorflow as tfs
import wandb
from absl import flags, app
import random
from smac.env import StarCraft2Env
import copy
from evals import *

FLAGS = flags.FLAGS
flags.DEFINE_float("lr", 3e-4, "Learning Rate")
flags.DEFINE_float("eps_dec", 1e-5, "Decay rate of epsilon")
flags.DEFINE_integer("seed", 42, "Random seed")
flags.DEFINE_float("gamma", 0.99, "Gamma value for update")
flags.DEFINE_integer("targ_update", 500, "Number of steps before copying network weights")
flags.DEFINE_integer("buffer_size",200000,"Size of memory")
flags.DEFINE_integer("train_period",1,"Number of steps to take in between training")


def main(_):

    # hyperparams & fixed variables
    eps_end = 0.05
    max_timesteps = 1000000
    target_update_period = FLAGS.targ_update

    # start the wandb logger
    run = wandb.init(reinit=True,project="SMAC")
    
    random.seed(FLAGS.seed)
    np.random.seed(FLAGS.seed)

    # make StarCraft environment
    env = StarCraft2Env(map_name="3m", seed=FLAGS.seed)

    # get number of agents and actions in the env
    env_info = env.get_env_info()
    n_agents = env_info["n_agents"]
    n_actions = env_info["n_actions"]   # per agent
    obs_shape = env_info["obs_shape"]   # per agent

    cohort = VDNCohort(gamma=FLAGS.gamma, epsilon=1.0,lr=FLAGS.lr, \
        input_dims=[obs_shape], n_agents=n_agents, \
        n_actions=n_actions,mem_size=FLAGS.buffer_size,batch_size=32, \
        epsilon_end=eps_end, epsilon_dec=FLAGS.eps_dec)

    # storage & tracking variables - these change in the loop
    scores = []
    eps_history = []
    wins = []
    timesteps = 0
    i = 0 
    tardy = False

    while not tardy:
        # storage & tracking variables - these change in the loop
        done=False
        score=0
        game_length = 0
        won = 0

        # todo
        env.reset()
        obs_list = env.get_obs()
        
        while not (done or tardy):
            timesteps+=1
            actions = []
            for agent_id in range(n_agents):
                avail_actions = env.get_avail_agent_actions(agent_id)
                action = cohort.choose_action(obs_list[agent_id],avail_actions,agent_id)
                actions.append(action)
            
            # reward and terminated are shared values
            reward, terminated, info = env.step(actions)
            try:
                won = int(info['battle_won'])
            except:
                won = 1

            obs_list_ = env.get_obs()
            score += reward
            done = terminated 

            # store all transitions and learn

            avail_acts_ls = np.zeros((n_agents,n_actions))
            for agent_id in range(n_agents):
                avail_actions = env.get_avail_agent_actions(agent_id)   #already one-hot
                avail_acts_ls[agent_id] = avail_actions



            cohort.store_transition(np.array(obs_list), actions, reward, \
            np.array(obs_list_), done, avail_acts_ls)

            if timesteps%FLAGS.train_period==0:
                train_logs = cohort.learn()

            logs = {
                    'epsilon' : cohort.epsilon,
                    **train_logs
                }

            if timesteps % target_update_period ==0:
                cohort.update_target_network()
            logs["steps"] = timesteps
            wandb.log(logs)

            obs_list = obs_list_

            
            if timesteps>=max_timesteps:
                tardy=True

            game_length +=1
            
            
        eps_history.append(cohort.epsilon)
        scores.append(score)
        wins.append(won)
        
        avg_score = np.mean(scores[-100:])

        logs = {'game_length': game_length,
            'episode': i ,
            'score' : score,
            'won' : won,
            'average_score' : avg_score,
            'steps' : timesteps,
            'epsilon' : cohort.epsilon}
        wandb.log(logs)

        i+=1
        if i % 25 == 0:
            print(f"episode {i}")

            #store the eps
            current_eps = copy.deepcopy(cohort.epsilon)

            #evaluation
            logs = perform_shared_eval(cohort,env)
            wandb.log(logs)

            # restore the eps
            cohort.epsilon = copy.deepcopy(current_eps)


if __name__ == "__main__":
    app.run(main)
	


