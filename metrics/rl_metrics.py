import numpy as np
def rl_summary(rewards, losses=None):
    return {"cumulative_reward":float(np.sum(rewards)), "mean_reward":float(np.mean(rewards)) if rewards else 0,
            "final_loss":float(losses[-1]) if losses else None}
