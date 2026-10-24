import os, json, yaml, numpy as np
import torch
from workloads.generator import generate_workload
from environment.cpu_scheduler_env import CPUSchedulerEnv
from rl_agents.q_learning import QLearningAgent
from rl_agents.dqn import DQNAgent
from rl_agents.ddqn import DDQNAgent
from rl_agents.d3qn import D3QNAgent
from rl_agents.ppo import PPOAgent

def make_agent(name,n_state,n_actions):
    return {"q_learning":QLearningAgent(n_actions),"dqn":DQNAgent(n_state,n_actions),
            "ddqn":DDQNAgent(n_state,n_actions),"d3qn":D3QNAgent(n_state,n_actions),
            "ppo":PPOAgent(n_state,n_actions)}[name]

def save_agent(agent, name):
    os.makedirs("models", exist_ok=True)

    if name == "q_learning":
        np.save(
            "models/q_learning_q.npy",
            dict(agent.q),
            allow_pickle=True
        )

    elif name in ["dqn", "ddqn", "d3qn"]:
        torch.save(
            {
                "net": agent.net.state_dict(),
                "target": agent.target.state_dict()
            },
            f"models/{name}.pt"
        )

    elif name == "ppo":
        torch.save(
        {
            "model": agent.model.state_dict()
        },
        "models/ppo.pt"
    )

    print(f"Saved trained {name} agent.")

def train(config_path="experiments/configs/default.yaml", agent_name=None):
    cfg=yaml.safe_load(open(config_path))
    if agent_name is not None:
        cfg["agent"] = agent_name
    np.random.seed(cfg["seed"])
    p=generate_workload(cfg["workload_n"],cfg["workload_type"],cfg["seed"])
    env=CPUSchedulerEnv(p,cfg["quanta"])
    agent=make_agent(cfg["agent"],8,len(cfg["quanta"]))
    rewards=[]
    for ep in range(cfg["episodes"]):
        result=agent.train_episode(env)
        rewards.append(result[0] if isinstance(result,tuple) else result)
        if (ep+1)%50==0: print(f"Episode {ep+1}/{cfg['episodes']} reward={rewards[-1]:.3f}")
    os.makedirs("results",exist_ok=True)
    training_file = f"results/training_{cfg['agent']}.json"
    with open(training_file, "w") as f:
        json.dump({"config": cfg, "rewards": rewards}, f)
        print(f"Saved training results to {training_file}")
        save_agent(agent, cfg["agent"])
        return agent,rewards

if __name__=="__main__":
    train()
