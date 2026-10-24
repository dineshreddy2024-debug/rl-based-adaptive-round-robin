from workloads.generator import generate_workload
from environment.cpu_scheduler_env import CPUSchedulerEnv
from rl_agents.q_learning import QLearningAgent
def test_agent():
    e=CPUSchedulerEnv(generate_workload(5,"mixed",1))
    a=QLearningAgent(e.action_space.n)
    r=a.train_episode(e)
    assert isinstance(r,float)
