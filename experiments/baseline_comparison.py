import os
import json
import copy
import csv
import numpy as np
import torch

from workloads.generator import generate_workload

from scheduler.fcfs import fcfs
from scheduler.sjf import sjf
from scheduler.round_robin import round_robin

from baselines.heuristic_dynamic_rr import run_heuristic_rr

from metrics.scheduling_metrics import calculate_metrics

from environment.cpu_scheduler_env import CPUSchedulerEnv

from rl_agents.q_learning import QLearningAgent
from rl_agents.dqn import DQNAgent
from rl_agents.ddqn import DDQNAgent
from rl_agents.d3qn import D3QNAgent
from rl_agents.ppo import PPOAgent


# ============================================================
# TRADITIONAL / BASELINE SCHEDULERS
# ============================================================

def run_fcfs(workload):
    processes, gantt, context_switches = fcfs(
        copy.deepcopy(workload)
    )

    return calculate_metrics(processes, context_switches)


def run_sjf(workload):
    processes, gantt, context_switches = sjf(
        copy.deepcopy(workload)
    )

    return calculate_metrics(processes, context_switches)


def run_fixed_rr(workload, quantum=4):
    processes, gantt, context_switches = round_robin(
        copy.deepcopy(workload),
        quantum
    )

    return calculate_metrics(processes, context_switches)


def run_dynamic_rr(workload):
    quantum, metrics = run_heuristic_rr(
        copy.deepcopy(workload)
    )

    metrics["quantum"] = quantum

    return metrics


# ============================================================
# LOAD TRAINED RL AGENTS
# ============================================================

def load_q_learning(quanta):
    agent = QLearningAgent(
        n_actions=len(quanta)
    )

    q_file = "models/q_learning_q.npy"

    if not os.path.exists(q_file):
        raise FileNotFoundError(
            f"Missing model file: {q_file}"
        )

    loaded_q = np.load(
        q_file,
        allow_pickle=True
    ).item()

    agent.q.update(loaded_q)

    return agent


def load_dqn(quanta):
    agent = DQNAgent(
        n_state=8,
        n_actions=len(quanta)
    )

    model_file = "models/dqn.pt"

    if not os.path.exists(model_file):
        raise FileNotFoundError(
            f"Missing model file: {model_file}"
        )

    checkpoint = torch.load(
        model_file,
        map_location=agent.device
    )

    agent.net.load_state_dict(checkpoint["net"])
    agent.target.load_state_dict(checkpoint["target"])

    return agent


def load_ddqn(quanta):
    agent = DDQNAgent(
        n_state=8,
        n_actions=len(quanta)
    )

    model_file = "models/ddqn.pt"

    if not os.path.exists(model_file):
        raise FileNotFoundError(
            f"Missing model file: {model_file}"
        )

    checkpoint = torch.load(
        model_file,
        map_location=agent.device
    )

    agent.net.load_state_dict(checkpoint["net"])
    agent.target.load_state_dict(checkpoint["target"])

    return agent


def load_d3qn(quanta):
    agent = D3QNAgent(
        n_state=8,
        n_actions=len(quanta)
    )

    model_file = "models/d3qn.pt"

    if not os.path.exists(model_file):
        raise FileNotFoundError(
            f"Missing model file: {model_file}"
        )

    checkpoint = torch.load(
        model_file,
        map_location=agent.device
    )

    agent.net.load_state_dict(checkpoint["net"])
    agent.target.load_state_dict(checkpoint["target"])

    return agent


def load_ppo(quanta):
    agent = PPOAgent(
        n_state=8,
        n_actions=len(quanta)
    )

    model_file = "models/ppo.pt"

    if not os.path.exists(model_file):
        raise FileNotFoundError(
            f"Missing model file: {model_file}"
        )

    checkpoint = torch.load(
        model_file,
        map_location=agent.device
    )

    agent.model.load_state_dict(checkpoint["model"])

    return agent


# ============================================================
# RUN ONE RL AGENT
# ============================================================

def run_rl_agent(agent, workload, quanta):
    """
    Evaluate a trained RL agent on the given workload.

    training=False is important because we want
    evaluation without random exploration.
    """

    env = CPUSchedulerEnv(
        copy.deepcopy(workload),
        action_quanta=quanta
    )

    state, info = env.reset()

    done = False
    truncated = False

    while not done and not truncated:

        action = agent.act(
            state,
            training=False
        )

        state, reward, done, truncated, info = env.step(
            action
        )

    return info


# ============================================================
# MAIN COMPARISON
# ============================================================

def run_baseline_comparison(
    num_processes=20,
    workload_type="mixed",
    seed=42,
    fixed_quantum=4
):

    # --------------------------------------------------------
    # RL ACTION QUANTA
    # --------------------------------------------------------

    quanta = (1, 2, 4, 6, 8, 10, 12)

    # --------------------------------------------------------
    # Generate ONE workload
    # --------------------------------------------------------

    workload = generate_workload(
        n=num_processes,
        kind=workload_type,
        seed=seed
    )

    print("\n" + "=" * 80)
    print("SCHEDULER COMPARISON")
    print("=" * 80)

    print("\nWorkload:")
    print("-" * 80)

    for p in workload:
        print(
            f"P{p.pid}: "
            f"arrival={p.arrival_time}, "
            f"burst={p.burst_time}, "
            f"priority={p.priority}"
        )

    # --------------------------------------------------------
    # Traditional schedulers
    # --------------------------------------------------------

    print("\nRunning FCFS...")
    fcfs_metrics = run_fcfs(workload)

    print("Running SJF...")
    sjf_metrics = run_sjf(workload)

    print(
        f"Running Fixed RR (quantum={fixed_quantum})..."
    )
    fixed_rr_metrics = run_fixed_rr(
        workload,
        fixed_quantum
    )

    print("Running Dynamic RR...")
    dynamic_rr_metrics = run_dynamic_rr(workload)

    # --------------------------------------------------------
    # RL schedulers
    # --------------------------------------------------------

    print("Loading Q-Learning model...")
    q_learning_agent = load_q_learning(quanta)

    print("Running Q-Learning RR...")
    q_learning_metrics = run_rl_agent(
        q_learning_agent,
        workload,
        quanta
    )

    print("Loading DQN model...")
    dqn_agent = load_dqn(quanta)

    print("Running DQN RR...")
    dqn_metrics = run_rl_agent(
        dqn_agent,
        workload,
        quanta
    )

    print("Loading DDQN model...")
    ddqn_agent = load_ddqn(quanta)

    print("Running DDQN RR...")
    ddqn_metrics = run_rl_agent(
        ddqn_agent,
        workload,
        quanta
    )

    print("Loading D3QN model...")
    d3qn_agent = load_d3qn(quanta)

    print("Running D3QN RR...")
    d3qn_metrics = run_rl_agent(
        d3qn_agent,
        workload,
        quanta
    )

    print("Loading PPO model...")
    ppo_agent = load_ppo(quanta)

    print("Running PPO RR...")
    ppo_metrics = run_rl_agent(
        ppo_agent,
        workload,
        quanta
    )

    # --------------------------------------------------------
    # Store results
    # --------------------------------------------------------

    results = {

        "experiment": "scheduler_comparison",

        "workload": {
            "num_processes": num_processes,
            "type": workload_type,
            "seed": seed
        },

        "schedulers": {

            "FCFS": fcfs_metrics,

            "SJF": sjf_metrics,

            "Fixed_RR": fixed_rr_metrics,

            "Dynamic_RR": dynamic_rr_metrics,

            "Q_Learning_RR": q_learning_metrics,

            "DQN_RR": dqn_metrics,

            "DDQN_RR": ddqn_metrics,

            "D3QN_RR": d3qn_metrics,

            "PPO_RR": ppo_metrics
        }
    }

    # --------------------------------------------------------
    # Print results
    # --------------------------------------------------------

    print("\n" + "=" * 100)
    print("FINAL RESULTS")
    print("=" * 100)

    print(
        f"{'Scheduler':<20}"
        f"{'Waiting':>12}"
        f"{'Turnaround':>14}"
        f"{'Response':>12}"
        f"{'CS':>8}"
        f"{'CPU %':>10}"
        f"{'Throughput':>14}"
    )

    print("-" * 100)

    for name, metrics in results["schedulers"].items():

        print(
            f"{name:<20}"
            f"{metrics['waiting_time']:>12.2f}"
            f"{metrics['turnaround_time']:>14.2f}"
            f"{metrics['response_time']:>12.2f}"
            f"{metrics['context_switches']:>8}"
            f"{metrics['cpu_utilization']:>10.2f}"
            f"{metrics['throughput']:>14.4f}"
        )

    # --------------------------------------------------------
    # Save JSON
    # --------------------------------------------------------

    os.makedirs("results", exist_ok=True)

    output_file = "results/scheduler_comparison.json"

    with open(output_file, "w") as f:
        json.dump(
            results,
            f,
            indent=4
        )

    # --------------------------------------------------------
    # Save CSV
    # --------------------------------------------------------

    csv_file = "results/scheduler_comparison.csv"

    csv_fields = [
        "scheduler",
        "waiting_time",
        "turnaround_time",
        "response_time",
        "context_switches",
        "cpu_utilization",
        "throughput"
    ]

    with open(
        csv_file,
        "w",
        newline=""
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=csv_fields
        )

        writer.writeheader()

        for name, metrics in results["schedulers"].items():

            writer.writerow({

                "scheduler": name,

                "waiting_time":
                    metrics["waiting_time"],

                "turnaround_time":
                    metrics["turnaround_time"],

                "response_time":
                    metrics["response_time"],

                "context_switches":
                    metrics["context_switches"],

                "cpu_utilization":
                    metrics["cpu_utilization"],

                "throughput":
                    metrics["throughput"]
            })

    print("\nResults saved to:")
    print(output_file)
    print(csv_file)

    return results


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    run_baseline_comparison(
        num_processes=20,
        workload_type="mixed",
        seed=42,
        fixed_quantum=4
    )