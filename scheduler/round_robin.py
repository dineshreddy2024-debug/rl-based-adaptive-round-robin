from collections import deque

def round_robin(processes, quantum):
    ps = [p.clone() for p in processes]
    ready, t, cs = deque(), 0, 0
    gantt, last = [], None
    remaining = ps[:]
    while any(p.remaining_time > 0 for p in ps):
        for p in ps:
            if p.arrival_time <= t and p.remaining_time > 0 and p not in ready:
                ready.append(p)
        if not ready:
            t = min(p.arrival_time for p in ps if p.remaining_time > 0)
            continue
        p = ready.popleft()
        if p.first_start is None: p.first_start = t
        if last is not None and last != p.pid: cs += 1
        run = min(quantum, p.remaining_time)
        gantt.append((p.pid, t, t+run))
        t += run; p.remaining_time -= run; last = p.pid
        for x in ps:
            if x.arrival_time <= t and x.remaining_time > 0 and x not in ready and x.pid != p.pid:
                ready.append(x)
        if p.remaining_time > 0: ready.append(p)
        else: p.completion_time = t
    return ps, gantt, cs
