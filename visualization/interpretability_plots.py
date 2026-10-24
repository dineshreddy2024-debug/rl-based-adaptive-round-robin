import matplotlib.pyplot as plt
def plot_policy(matrix,path="results/policy.png"):
    plt.figure(figsize=(8,5)); plt.imshow(matrix,aspect="auto"); plt.colorbar(); plt.xlabel("Action"); plt.ylabel("State sample"); plt.tight_layout(); plt.savefig(path); plt.close()
