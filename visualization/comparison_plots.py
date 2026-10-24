import matplotlib.pyplot as plt
def plot_comparison(data,path="results/comparison.png"):
    names=list(data); vals=[data[k] for k in names]
    plt.figure(figsize=(9,5)); plt.bar(names,vals); plt.ylabel("Value"); plt.xticks(rotation=30); plt.tight_layout(); plt.savefig(path); plt.close()
