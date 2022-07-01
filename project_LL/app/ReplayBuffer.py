from collections import deque
import numpy as np


class ReplayBuffer:

    def __init__(self, max_len, batch_size):
        self._memory = deque(maxlen=max_len)
        self._batch_size = batch_size

    def save_new_experience(self, state, action, reward, next_state, done):
        self._memory.append((state, action, reward, next_state, done))

    def get_sample(self):
        batch = np.random.choice(len(self._memory), self._batch_size)
        states = []
        actions = []
        rewards = []
        next_states = []
        done_s = []
        for i in batch:
            states.append(self._memory[i][0])
            actions.append(self._memory[i][1])
            rewards.append(self._memory[i][2])
            next_states.append(self._memory[i][3])
            done_s.append(self._memory[i][4])

        return np.array(states), np.array(actions), np.array(rewards), np.array(next_states), np.array(done_s)

    def __len__(self):
        return len(self._memory)
