import numpy as np
from collections import deque


class ReplayBuffer:
    def __init__(self, max_size: int = 1024000):
        self.buffer = deque(maxlen=max_size)
        self.max_size = max_size

    def add(self, experience: tuple):
        self.buffer.append(experience)

    def sample(self, batch_size: int) -> None | list[tuple]:
        if len(self.buffer) < batch_size:
            return None
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        return zip(*[self.buffer[i] for i in indices])

    def __len__(self):
        return len(self.buffer)
