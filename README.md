
# RL-Based Adaptive Round Robin Scheduler

## 1. What this implementation does

The project models CPU scheduling as a Gymnasium environment. At each Round Robin
decision point, an RL agent observes workload/scheduler state and chooses a quantum
from:

`[1, 2, 4, 6, 8, 10, 12]`

The reward combines waiting time, response time, context switches and CPU utilization.
Baselines include FCFS, SJF, Priority, fixed-quantum RR and a median-based heuristic RR.

Implemented agents:
- Tabular Q-learning
- DQN
- DDQN-compatible extension
- Dueling DQN extension
- PPO

## 2. Windows installation

Open Command Prompt or PowerShell:

```text
cd C:\Users\dines\.gemini\antigravity\scratch\rl-arr
py -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

If `py` is unavailable, use `python` instead.

## 3. Run tests

```text
pytest
```

## 4. Run the first experiment

```text
python run_all_experiments.py
```

Outputs:
- `results/training.json`
- `results/convergence.png`

## 5. Change the RL algorithm

Edit `experiments/configs/default.yaml`:

```yaml
agent: q_learning
```

or:

```yaml
agent: dqn
agent: ddqn
agent: d3qn
agent: ppo
```

Then run:

```text
python run_all_experiments.py
```

## 6. Run from the project root

Do NOT run package files such as `environment/cpu_scheduler_env.py` directly.
Run from the root so imports work:

```text
C:\Users\dines\.gemini\antigravity\scratch\rl-arr> python run_all_experiments.py
```

## 7. Real trace CSV

`workloads/real_traces.py` expects a CSV with:

```text
arrival_time,burst_time,priority
0,8,2
1,3,1
2,10,3
```

Then:

```python
from workloads.real_traces import load_trace_csv
processes = load_trace_csv("my_trace.csv")
```

For Google/Alibaba traces, first map the trace's timestamp/runtime fields to the
three scheduler fields. Do not assume their raw schemas are identical.

## 8. Research workflow

Recommended experiment sequence:

1. Validate all schedulers on synthetic workloads.
2. Train Q-learning as the simplest RL baseline.
3. Train DQN.
4. Train DDQN.
5. Train D3QN.
6. Train PPO.
7. Compare all against fixed RR and heuristic dynamic RR.
8. Evaluate on unseen workload distributions.
9. Evaluate online workload changes.
10. Run ablations for reward terms and state features.
11. Measure inference overhead.
12. Run statistical tests over multiple random seeds.

## 9. Important research improvement

This repository is a runnable baseline implementation, not a final publication-quality
RL scheduler. For a serious paper, extend the environment to make arrivals, queue
membership, process-level waiting times, fairness, and context-switch cost explicit
at every decision step. Also use multiple independent seeds and report mean ± 95% CI.

## 10. Expected result

After training you should see a convergence plot and a JSON file containing episode
rewards. To produce a paper table, add an evaluation script that runs the trained
agent for many seeds and compares:

- Average Waiting Time
- Average Turnaround Time
- Average Response Time
- Context Switches
- CPU Utilization
- Throughput
- Jain Fairness Index
- Quantum change frequency
- Quantum variance
- RL inference overhead
"# rl-arr" 
=======
# rl-based-adaptive-round-robin

