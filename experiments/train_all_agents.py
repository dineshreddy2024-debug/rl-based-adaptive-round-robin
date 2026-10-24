from run_experiment import train


agents = [
    "q_learning",
    "dqn",
    "ddqn",
    "d3qn",
    "ppo"
]

for agent_name in agents:

    print("\n" + "=" * 60)
    print(f"TRAINING {agent_name.upper()}")
    print("=" * 60)

    train(agent_name=agent_name)

print("\n" + "=" * 60)
print("ALL 5 RL AGENTS TRAINED")
print("=" * 60)