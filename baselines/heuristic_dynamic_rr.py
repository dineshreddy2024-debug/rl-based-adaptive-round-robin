import numpy as np
from scheduler.round_robin import round_robin
from metrics.scheduling_metrics import calculate_metrics

def heuristic_quantum(processes):
    bursts=[p.burst_time for p in processes]
    return max(1,int(np.median(bursts))) if bursts else 1

def run_heuristic_rr(processes):
    q=heuristic_quantum(processes)
    ps,_,cs=round_robin(processes,q)
    return q,calculate_metrics(ps,cs)
