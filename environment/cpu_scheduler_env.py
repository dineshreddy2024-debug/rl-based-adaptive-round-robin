import numpy as np
import gymnasium as gym
from gymnasium import spaces
from .process import Process
from reward.reward_function import MultiObjectiveReward

class CPUSchedulerEnv(gym.Env):
    """
    One episode = scheduling a generated workload.
    At every decision point the agent selects a quantum from action_quanta.
    A selected quantum is applied to the next process in RR order.
    """
    metadata = {"render_modes": []}

    def __init__(self, processes, action_quanta=(1,2,4,6,8,10,12), reward_config=None):
        super().__init__()
        self.original = [p.clone() for p in processes]
        self.action_quanta = np.asarray(action_quanta, dtype=np.int64)
        self.action_space = spaces.Discrete(len(self.action_quanta))
        # [queue_len, avg_remaining, avg_burst, avg_wait, arrival_rate,
        #  utilization_so_far, last_quantum, completed_ratio]
        self.observation_space = spaces.Box(0, np.inf, shape=(8,), dtype=np.float32)
        self.rewarder = MultiObjectiveReward(**(reward_config or {}))
        self.reset()

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.procs = [p.clone() for p in self.original]
        self.time = 0
        self.completed = []
        self.ready = []
        self.last_quantum = float(self.action_quanta[0])
        self.context_switches = 0
        self.busy_time = 0
        self.last_pid = None
        self.total_wait_before = 0
        return self._obs(), {}

    def _arrivals(self):
        for p in self.procs:
            if p.arrival_time <= self.time and p.remaining_time > 0 and p not in self.ready:
                self.ready.append(p)

    def _obs(self):
        self._arrivals()
        if self.ready:
            rem = np.mean([p.remaining_time for p in self.ready])
            burst = np.mean([p.burst_time for p in self.ready])
            waits = np.mean([max(0, self.time-p.arrival_time) for p in self.ready])
        else:
            rem = burst = waits = 0.0
        total_burst = sum(p.burst_time for p in self.procs) or 1
        util = self.busy_time / max(1, self.time)
        return np.array([
            len(self.ready), rem, burst, waits,
            len(self.procs)/max(1, self.time+1),
            util, self.last_quantum,
            len(self.completed)/len(self.procs)
        ], dtype=np.float32)

    def step(self, action):
        self._arrivals()
        if not self.ready:
            future = [p.arrival_time for p in self.procs if p.remaining_time > 0 and p.arrival_time > self.time]
            if future:
                self.time = min(future)
                self._arrivals()
            else:
                return self._obs(), 0.0, True, False, self._info()

        p = self.ready.pop(0)
        q = int(self.action_quanta[int(action)])
        if p.first_start is None:
            p.first_start = self.time
        if self.last_pid is not None and self.last_pid != p.pid:
            self.context_switches += 1

        run = min(q, p.remaining_time)
        self.time += run
        self.busy_time += run
        p.remaining_time -= run
        self.last_pid = p.pid

        self._arrivals()
        if p.remaining_time > 0:
            self.ready.append(p)
        else:
            p.completion_time = self.time
            self.completed.append(p)

        self.last_quantum = q
        done = len(self.completed) == len(self.procs)
        reward = self.rewarder.compute(self._info(), q)
        return self._obs(), reward, done, False, self._info()

    def _info(self):
        if not self.procs:
            return {}
        completed = self.completed
        tat = [p.completion_time-p.arrival_time for p in completed if p.completion_time is not None]
        wt = [x-p.burst_time for x,p in zip(tat, completed)]
        rt = [p.first_start-p.arrival_time for p in completed if p.first_start is not None]
        #total_burst = sum(p.burst_time for p in self.procs)
        return {
            "time": self.time,
            "completed": len(completed),
            "n": len(self.procs),
            "waiting_time": float(np.mean(wt)) if wt else 0.0,
            "turnaround_time": float(np.mean(tat)) if tat else 0.0,
            "response_time": float(np.mean(rt)) if rt else 0.0,
            "context_switches": self.context_switches,
            "cpu_utilization": (100.0 * self.busy_time / max(1, self.time)),
            "throughput": len(completed) / max(1, self.time),
        }
