import numpy as np
def jains_index(values):
    x=np.asarray(values,dtype=float)
    return float((x.sum()**2)/(len(x)*(x*x).sum())) if len(x) and (x*x).sum()>0 else 1.0
def fairness_metrics(processes):
    waits=[(p.completion_time-p.arrival_time-p.burst_time) for p in processes if p.completion_time is not None]
    return {"jain_waiting_fairness":jains_index(waits), "max_waiting_time":max(waits,default=0),
            "starvation_count":sum(w>2*max(1,np.mean(waits)) for w in waits) if waits else 0}
