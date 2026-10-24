from dataclasses import dataclass, field

@dataclass
class Process:
    pid: int
    arrival_time: int
    burst_time: int
    priority: int = 1
    remaining_time: int = field(init=False)
    first_start: int | None = None
    completion_time: int | None = None

    def __post_init__(self):
        self.remaining_time = self.burst_time

    def clone(self):
        return Process(self.pid, self.arrival_time, self.burst_time, self.priority)
