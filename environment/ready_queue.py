from collections import deque

class ReadyQueue:
    def __init__(self):
        self.q = deque()

    def add(self, process):
        if process not in self.q:
            self.q.append(process)

    def pop(self):
        return self.q.popleft()

    def __len__(self):
        return len(self.q)

    def is_empty(self):
        return not self.q

    def items(self):
        return list(self.q)
