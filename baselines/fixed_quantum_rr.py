from scheduler.round_robin import round_robin
from metrics.scheduling_metrics import calculate_metrics

def run_fixed_rr(processes, quanta=(1,2,4,6,8,10,12)):
    out={}
    for q in quanta:
        ps,_,cs=round_robin(processes,q)
        out[q]=calculate_metrics(ps,cs)
    return out
