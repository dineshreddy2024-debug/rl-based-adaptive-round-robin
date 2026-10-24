def fcfs(processes):
    ps = sorted([p.clone() for p in processes], key=lambda x:(x.arrival_time,x.pid))
    t=0; gantt=[]; cs=0; last=None
    for p in ps:
        t=max(t,p.arrival_time)
        if p.first_start is None: p.first_start=t
        if last is not None and last != p.pid: cs += 1
        gantt.append((p.pid,t,t+p.remaining_time))
        t += p.remaining_time; p.remaining_time=0; p.completion_time=t; last=p.pid
    return ps,gantt,cs
