from agentclasses import SuperAgent
from gymwrapper import GymWrapper
import numpy as np
from smac.env import StarCraft2Env
import tensorflow as tf
import wandb
from absl import flags, app
import random
import copy

FLAGS = flags.FLAGS
flags.DEFINE_float("lr", 3e-4, "Learning Rate")
flags.DEFINE_float("eps_dec", 1e-5, "Decay rate of epsilon")
flags.DEFINE_integer("seed", 42, "Random seed")
flags.DEFINE_float("gamma", 0.99, "Gamma value for update")
flags.DEFINE_integer("targ_update", 500, "Number of steps before copying network weights")
flags.DEFINE_integer("buffer_size",200000,"Size of memory")

def perform_eval(agent, env):
    done=False
    score=0
    observation, available_actions = env.reset()
    agent.epsilon=0

    game_length = 0
    
    while not (done):
        
        action = agent.choose_action(observation,available_actions)
        
        won, observation, reward, terminated, truncated, available_actions = env.step(action)
        score += reward

        done = terminated or truncated

        game_length +=1
        
    logs = {
        'eval_length' : game_length,
        'eval_score' : score,
        'eval_won' : won
    }
    wandb.log(logs)


def main(_):
    env = GymWrapper("3m", FLAGS.seed)
    eps_end = 0.05
    random.seed(FLAGS.seed)
    np.random.seed(FLAGS.seed)

    doublerun = wandb.init(reinit=True,project="SMAC")

    agent = SuperAgent(gamma=FLAGS.gamma, epsilon=1.0,lr=FLAGS.lr, \
            input_dims=env.obs_len, \
            n_actions=env.n_actions**env.n_agents,mem_size=FLAGS.buffer_size,batch_size=32, \
            epsilon_end=eps_end, epsilon_dec=FLAGS.eps_dec)
    
    scores = []
    eps_history = []
    timesteps = 0
    max_timesteps = 1000000
    i = 0   #number of episodes
    tardy = False
    target_update_period = FLAGS.targ_update

    while not tardy:
        done=False
        score=0
        observation, available_actions = env.reset()

        game_length = 0
        
        while not (done or tardy):
            
            action = agent.choose_action(observation,available_actions)
            
            won, observation_, reward, terminated, truncated, available_actions = env.step(action)
            score += reward

            done = terminated or truncated
            
            agent.store_transition(observation, action, reward, \
                observation_, done, available_actions)
            observation = observation_
            
            train_logs = agent.learn()

            timesteps+=1
            if timesteps>=max_timesteps:
                tardy=True

            game_length +=1

            # update target network every n steps
            if timesteps % target_update_period ==0:
                agent.update_target_network()
                logs = {
                    'steps' : timesteps,
                    'epsilon' : agent.epsilon,
                    **train_logs
                }
                wandb.log(logs)
            
        eps_history.append(agent.epsilon)
        scores.append(score)
        
        avg_score = np.mean(scores[-100:])

        logs = {'game_length': game_length, 'episode': i ,'score' : score,
            'average_score' : avg_score,
            'steps' : timesteps,
            'won' : won,
            'epsilon' : agent.epsilon}
        wandb.log(logs)
        i+=1

        if i % 25 == 0:
            print(f"episode {i}")

            #store the eps
            current_eps = copy.deepcopy(agent.epsilon)

            #evaluation
            perform_eval(agent,env)

            # restore the eps
            agent.epsilon = copy.deepcopy(current_eps)

    env.wrap_up()


if __name__ == "__main__":
    app.run(main)
	


