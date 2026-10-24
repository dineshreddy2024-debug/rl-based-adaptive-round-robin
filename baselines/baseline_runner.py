from .fixed_quantum_rr import run_fixed_rr
from .heuristic_dynamic_rr import run_heuristic_rr
from scheduler.fcfs import fcfs
from scheduler.sjf import sjf
from scheduler.priority_scheduler import priority_scheduling
from metrics.scheduling_metrics import calculate_metrics

def run_all(processes, quanta=(1,2,4,6,8,10,12)):
    result={"fixed_rr":run_fixed_rr(processes,quanta)}
    q,m=run_heuristic_rr(processes); result["heuristic_rr"]={q:m}
    for name,fn in [("fcfs",fcfs),("sjf",sjf),("priority",priority_scheduling)]:
        ps,_,cs=fn(processes); result[name]=calculate_metrics(ps,cs)
    return result
