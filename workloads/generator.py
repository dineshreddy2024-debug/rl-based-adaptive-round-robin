import random
from environment.process import Process

WORKLOAD_TYPES = [
    "short_cpu","long_cpu","mixed","interactive","batch","bursty",
    "high_arrival","low_arrival","priority_mixed","heavy_tail","uniform","stress"
]

def generate_workload(n=20, kind="mixed", seed=42):
    rng=random.Random(seed); out=[]; t=0
    for i in range(n):
        if kind=="short_cpu": burst=rng.randint(1,4)
        elif kind=="long_cpu": burst=rng.randint(8,20)
        elif kind=="interactive": burst=rng.choice([1,2,3,4,6])
        elif kind=="batch": burst=rng.randint(8,25)
        elif kind=="bursty": burst=rng.choice([1,2,3,15,20])
        elif kind=="high_arrival": burst=rng.randint(1,10); t=max(0,t-rng.randint(0,2))
        elif kind=="low_arrival": burst=rng.randint(2,12); t+=rng.randint(1,4)
        elif kind=="heavy_tail": burst=max(1,int(rng.expovariate(1/6)))
        elif kind=="uniform": burst=rng.randint(1,10)
        elif kind=="stress": burst=rng.randint(1,25)
        else: burst=rng.randint(1,12)
        if kind=="mixed": t += rng.randint(0,2)
        else: t += rng.randint(0,1)
        priority=rng.randint(1,5)
        out.append(Process(i,t,burst,priority))
    return out
