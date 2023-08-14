from dqn_tf2 import Agent
import numpy as np
import gymnasium as gym
import tensorflow as tf
import wandb
from absl import flags, app
import random

FLAGS = flags.FLAGS
flags.DEFINE_float("lr", 0.01, "Learning Rate")
flags.DEFINE_float("eps_dec", 0.01, "Decay rate of epsilon")
flags.DEFINE_integer("seed", 42, "Random seed")
flags.DEFINE_float("gamma", 0.99, "Gamma value for update")


def main(_):
    env = gym.make('LunarLander-v2')
    eps_end = 0.01
    random.seed(FLAGS.seed)
    np.random.seed(FLAGS.seed)

    #config = {"lr": FLAGS.lr,
    #    "eps_dec": FLAGS.eps_dec,
    #    "gamma":FLAGS.gamma,
    #    "seed": FLAGS.seed}
    #config = FLAGS
    run = wandb.init(reinit=True)#config=config)

    agent = Agent(gamma=FLAGS.gamma, epsilon=1.0,lr=FLAGS.lr, \
            input_dims=env.observation_space.shape, \
            n_actions=env.action_space.n,mem_size=100000,batch_size=64, \
            epsilon_end=eps_end, epsilon_dec=FLAGS.eps_dec)
    scores = []
    eps_history = []
    timesteps = 0
    max_timesteps = 500000
    i = 0   #number of episodes
    tardy = False

    while not tardy:
        done=False
        score=0
        episode_seed = random.randint(0,1000)
        observation, info = env.reset(seed=episode_seed)

        game_length = 0
        procrastinating = False
        
        # to store, convert to unique number
        # obs_for_storage = hash(tuple(observation[0].flatten()))
        while not (done or tardy or procrastinating):
            action = agent.choose_action(observation)
            
            observation_, reward, done, truncated, info = env.step(action)
            score += reward
            #obs__for_storage = hash(tuple(observation_[0].flatten()))
            agent.store_transition(observation, action, reward, \
                observation_, done)
            observation = observation_
            # obs_for_storage = obs__for_storage
            agent.learn()

            timesteps+=1
            if timesteps>=max_timesteps:
                tardy=True

            # the lunar lander tries to stay up high - wasted computation
            game_length +=1
            if game_length >= 1000:
                procrastinating = True
            
        eps_history.append(agent.epsilon)
        scores.append(score)
        
        avg_score = np.mean(scores[-100:])

        logs = {'game_length': game_length, 'episode': i ,'score' : score,
            'average_score' : avg_score,
            'steps' : timesteps,
            'epsilon' : agent.epsilon}
        wandb.log(logs)
        print(i)
        i+=1


if __name__ == "__main__":
    app.run(main)
	


