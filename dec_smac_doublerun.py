from dec_smac_doubledqn import Cohort
import numpy as np
import tensorflow as tfs
import wandb
from absl import flags, app
import random
from smac.env import StarCraft2Env
import copy

FLAGS = flags.FLAGS
flags.DEFINE_float("lr", 3e-4, "Learning Rate")
flags.DEFINE_float("eps_dec", 1e-5, "Decay rate of epsilon")
flags.DEFINE_integer("seed", 42, "Random seed")
flags.DEFINE_float("gamma", 0.99, "Gamma value for update")
flags.DEFINE_integer("targ_update", 500, "Number of steps before copying network weights")
flags.DEFINE_integer("buffer_size",200000,"Size of memory")


def perform_eval(agent_list, env):
    # storage & tracking variables - these change in the loop
    done=False
    score=0
    game_length = 0
    won = 0

    # todo
    env.reset()
    obs_list = env.get_obs()
    
    while not (done):

        actions = []
        for agent_id,agent in enumerate(agent_list):
            avail_actions = env.get_avail_agent_actions(agent_id)
            action = agent.choose_action(obs_list[agent_id],avail_actions)
            actions.append(action)
        
        # reward and terminated are shared values
        reward, terminated, info = env.step(actions)
        try:
            won = int(info['battle_won'])
        except:
            won = 1
        obs_list = env.get_obs()
        score += reward
        done = terminated

        game_length +=1
        
    logs = {
        'eval_length' : game_length,
        'eval_score' : score,
        'eval_won' : won
    }
    wandb.log(logs)


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

    # create n agents
    agent_list = []
    for agent_id in range(n_agents):
        # todo fix initialising of agents
        agent = Cohort(gamma=FLAGS.gamma, epsilon=1.0,lr=FLAGS.lr, \
            input_dims=[obs_shape], \
            n_actions=n_actions,mem_size=FLAGS.buffer_size,batch_size=32, \
            epsilon_end=eps_end, epsilon_dec=FLAGS.eps_dec)
        agent_list.append(agent)

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
            for agent_id,agent in enumerate(agent_list):
                avail_actions = env.get_avail_agent_actions(agent_id)
                action = agent.choose_action(obs_list[agent_id],avail_actions)
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
            logs = {}
            for agent_id,agent in enumerate(agent_list):
                
                avail_actions = env.get_avail_agent_actions(agent_id)
                agent.store_transition(obs_list[agent_id], actions[agent_id], reward, \
                obs_list_[agent_id], done, avail_actions)

                train_logs = agent.learn()

                new_train_logs = {}
                for key, value in train_logs.items():
                    new_train_logs[f"agent_{agent_id}_{key}"] = value

                logs.update({
                        f'agent_{agent_id}_epsilon' : agent.epsilon,
                        **new_train_logs
                    })

                if timesteps % target_update_period ==0:
                    agent.update_target_network()
            logs["steps"] = timesteps
            wandb.log(logs)

            obs_list = obs_list_

            
            if timesteps>=max_timesteps:
                tardy=True

            game_length +=1
            
            
        eps_history.append(agent.epsilon)
        scores.append(score)
        wins.append(won)
        
        avg_score = np.mean(scores[-100:])

        logs = {'game_length': game_length,
            'episode': i ,
            'score' : score,
            'won' : won,
            'average_score' : avg_score,
            'steps' : timesteps,
            'epsilon' : agent.epsilon}
        wandb.log(logs)

        i+=1
        if i % 25 == 0:
            print(f"episode {i}")

            # store epsilons
            eps_storage = [copy.deepcopy(agent.epsilon) for agent in agent_list]
            for a_id, agent in enumerate(agent_list):
                agent.epsilon=0

            perform_eval(agent_list,env)

            # restore epses
            for a_id, agent in enumerate(agent_list):
                agent.epsilon=copy.deepcopy(eps_storage[a_id])




if __name__ == "__main__":
    app.run(main)
	


