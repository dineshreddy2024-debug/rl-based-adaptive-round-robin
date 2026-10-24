import matplotlib.pyplot as plt
def plot_rewards(rewards,path="results/convergence.png"):
    plt.figure(figsize=(9,5)); plt.plot(rewards); plt.xlabel("Episode"); plt.ylabel("Reward"); plt.tight_layout(); plt.savefig(path); plt.close()
