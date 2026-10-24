from workloads.generator import generate_workload
from scheduler.round_robin import round_robin
from scheduler.fcfs import fcfs
def test_schedulers():
    p=generate_workload(5,"mixed",1)
    for fn in [fcfs]:
        ps,_,_=fn(p); assert all(x.completion_time is not None for x in ps)
    ps,_,_=round_robin(p,4); assert all(x.completion_time is not None for x in ps)
