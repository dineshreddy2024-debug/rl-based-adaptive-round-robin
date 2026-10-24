import numpy as np
def adaptive_summary(quanta):
    q=np.asarray(quanta,dtype=float)
    return {"quantum_change_frequency":float(np.mean(q[1:]!=q[:-1])) if len(q)>1 else 0,
            "quantum_variance":float(np.var(q)) if len(q) else 0,
            "quantum_mean":float(np.mean(q)) if len(q) else 0,
            "quantum_std":float(np.std(q)) if len(q) else 0}
