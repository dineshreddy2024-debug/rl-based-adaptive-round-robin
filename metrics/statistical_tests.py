import numpy as np
from scipy import stats
def compare(a,b):
    a=np.asarray(a); b=np.asarray(b)
    out={}
    out["t_test"]=stats.ttest_ind(a,b,equal_var=False)._asdict()
    out["wilcoxon"]=stats.wilcoxon(a[:min(len(a),len(b))],b[:min(len(a),len(b))])._asdict() if len(a)>1 and len(b)>1 else {}
    out["cohens_d"]=float((a.mean()-b.mean())/np.sqrt((a.var()+b.var())/2)) if len(a)>1 and len(b)>1 else 0
    return out
