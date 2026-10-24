from workloads.generator import generate_workload
from environment.cpu_scheduler_env import CPUSchedulerEnv
def test_env():
    env=CPUSchedulerEnv(generate_workload(5,"mixed",1))
    s,_=env.reset()
    assert s.shape==(8,)
    assert env.action_space.n==7
