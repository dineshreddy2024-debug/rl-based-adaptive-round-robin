import matplotlib.pyplot as plt
def plot_quanta(quanta,path="results/quantum.png"):
    plt.figure(figsize=(9,5)); plt.plot(quanta,marker="o"); plt.xlabel("Decision"); plt.ylabel("Quantum"); plt.tight_layout(); plt.savefig(path); plt.close()
