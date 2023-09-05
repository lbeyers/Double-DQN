# for superagent
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
    return logs

#for decentralised
def perform_dec_eval(agent_list, env):
    # storage & tracking variables - these change in the loop
    done=False
    score=0
    game_length = 0
    won = 0

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
    return logs

#for shared weight and vdn
def perform_shared_eval(agent, env):
    # storage & tracking variables - these change in the loop
    done=False
    score=0
    game_length = 0
    won = 0

    env.reset()
    obs_list = env.get_obs()
    
    while not (done):

        actions = []
        for agent_id in range(agent.n_agents):
            avail_actions = env.get_avail_agent_actions(agent_id)
            action = agent.choose_action(obs_list[agent_id],avail_actions,agent_id) #only diff
            actions.append(action)
        
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
    return logs




