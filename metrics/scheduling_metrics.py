import numpy as np

def calculate_metrics(processes, context_switches=0):
    wt=[]; tat=[]; rt=[]
    for p in processes:
        if p.completion_time is None: continue
        tat.append(p.completion_time-p.arrival_time)
        wt.append(tat[-1]-p.burst_time)
        rt.append((p.first_start or p.arrival_time)-p.arrival_time)
    finish=max((p.completion_time or 0 for p in processes), default=0)
    busy=sum(p.burst_time for p in processes)
    return {
        "waiting_time": float(np.mean(wt)) if wt else 0,
        "turnaround_time": float(np.mean(tat)) if tat else 0,
        "response_time": float(np.mean(rt)) if rt else 0,
        "context_switches": int(context_switches),
        "cpu_utilization": 100*busy/max(1,finish),
        "throughput": len(processes)/max(1,finish)
    }
