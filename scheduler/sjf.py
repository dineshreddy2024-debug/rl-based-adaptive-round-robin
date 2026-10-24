def sjf(processes):
    ps=[p.clone() for p in processes]; t=0; done=[]; gantt=[]; cs=0; last=None
    while len(done)<len(ps):
        avail=[p for p in ps if p.arrival_time<=t and p not in done]
        if not avail:
            t=min(p.arrival_time for p in ps if p not in done); continue
        p=min(avail,key=lambda x:(x.burst_time,x.arrival_time,x.pid))
        p.first_start=t
        if last is not None and last != p.pid: cs+=1
        gantt.append((p.pid,t,t+p.burst_time)); t+=p.burst_time
        p.remaining_time=0; p.completion_time=t; done.append(p); last=p.pid
    return ps,gantt,cs
