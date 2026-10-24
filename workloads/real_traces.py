import pandas as pd
from environment.process import Process

def load_trace_csv(path, arrival_col="arrival_time", burst_col="burst_time", priority_col=None):
    df=pd.read_csv(path)
    out=[]
    for i,row in df.iterrows():
        priority=int(row[priority_col]) if priority_col and priority_col in df else 1
        out.append(Process(int(i),int(row[arrival_col]),max(1,int(row[burst_col])),priority))
    return out
