def run_online_adaptation(agent, envs, episodes=20):
    rewards=[]
    for env in envs:
        for _ in range(episodes):
            r=agent.train_episode(env); rewards.append(r[0] if isinstance(r,tuple) else r)
    return rewards
