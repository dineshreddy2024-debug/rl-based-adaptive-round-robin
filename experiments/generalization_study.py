from workloads.generator import generate_workload
from environment.cpu_scheduler_env import CPUSchedulerEnv

def workloads_for_generalization(types,n=20):
    return {t:CPUSchedulerEnv(generate_workload(n,t,42)) for t in types}
