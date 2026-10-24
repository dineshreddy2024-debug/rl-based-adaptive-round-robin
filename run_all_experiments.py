import os
from experiments.run_experiment import train
from visualization.convergence_plots import plot_rewards

if __name__=="__main__":
    agent,rewards=train("experiments/configs/default.yaml")
    os.makedirs("results",exist_ok=True)
    plot_rewards(rewards)
    print("Done. See results/training.json and results/convergence.png")
