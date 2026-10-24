import time
def benchmark(fn,*args,repeat=20):
    start=time.perf_counter()
    for _ in range(repeat): fn(*args)
    return {"avg_seconds":(time.perf_counter()-start)/repeat}
