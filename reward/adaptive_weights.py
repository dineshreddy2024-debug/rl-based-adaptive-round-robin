def adaptive_weights(workload_size, avg_burst):
    # Simple configurable policy; replace with learned meta-controller later.
    if avg_burst < 4:
        return {"waiting_weight":1.2,"response_weight":0.8,"cs_weight":0.05}
    if workload_size > 20:
        return {"waiting_weight":0.8,"response_weight":0.4,"cs_weight":0.15}
    return {"waiting_weight":1.0,"response_weight":0.5,"cs_weight":0.1}
