from workloads.generator import generate_workload
from scheduler.fcfs import fcfs
from metrics.scheduling_metrics import calculate_metrics
def test_metrics():
    p,_,cs=fcfs(generate_workload(5,"mixed",1))
    m=calculate_metrics(p,cs)
    assert m["waiting_time"]>=0
