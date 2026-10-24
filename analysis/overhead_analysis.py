import time
def inference_time(agent,state,repeat=1000):
    t=time.perf_counter() if False else None
    start=time.perf_counter()
    for _ in range(repeat): agent.act(state,training=False)
    return (time.perf_counter()-start)/repeat
