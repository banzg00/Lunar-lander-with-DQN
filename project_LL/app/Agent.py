import numpy as np
from keras import Sequential
from keras.layers import Dense
from keras.activations import relu, linear
from keras.optimizers import Adam
from ReplayBuffer import ReplayBuffer
import random

from keras.saving.save import load_model

UPDATE_TARGET_EVERY = 5
TAU = 0.001


class Agent:

    def __init__(self, alpha, gamma, epsilon, min_epsilon, epsilon_dec, buffer_len, batch_size, update_every):
        self.lr = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.min_epsilon = min_epsilon
        self.epsilon_dec = epsilon_dec
        self.batch_size = batch_size
        self.memory = ReplayBuffer(buffer_len, batch_size)
        self.model = self._create_model()

        self.target_model = self._create_model()
        self.target_model.set_weights(self.model.weights)

        self.update_every = update_every
        self.update_cnt = 0

        self.update_target_model_cnt = 0

    def step(self, state, action, reward, next_state, done):
        self._update_replay_memory(state, action, reward, next_state, done)
        self.update_cnt += 1
        # Learn every UPDATE_EVERY time steps.
        if self.update_cnt > 0 and self.update_cnt % self.update_every == 0:
            if len(self.memory) > self.batch_size:
                self.learn(done)

    def choose_action(self, state):
        state = np.reshape(state, (1, 8))
        rand = random.random()
        if rand <= self.epsilon:
            action = np.random.choice(4)
        else:  # greedy
            actions = self.model.predict(state)
            action = np.argmax(actions)

        return action

    def learn(self, terminal_state):
        states, actions, rewards, next_states, done_s = self.memory.get_sample()

        current_qs_list = self.model.predict(states)
        next_qs_list = self.target_model.predict(next_states)

        y = []  # outputs
        for index, (state, action, reward, next_state, done) in enumerate(
                zip(states, actions, rewards, next_states, done_s)):
            max_next_q = np.max(next_qs_list[index])
            new_q = reward + self.gamma * max_next_q * (1 - done)

            current_qs = current_qs_list[index]
            current_qs[action] = new_q
            y.append(current_qs)

        self.model.fit(states, np.array(y), verbose=0, shuffle=False, batch_size=self.batch_size)

        # if terminal_state:
        #     self.update_target_model_cnt += 1
        #
        # if self.update_target_model_cnt >= UPDATE_TARGET_EVERY:
        #     self.target_model.set_weights(self.model.weights)
        #     self.update_target_model_cnt = 0

        # θ_target = τ*θ_local + (1 - τ)*θ_target
        t_local = np.array(self.model.weights, dtype=object)
        t_target = np.array(self.target_model.weights, dtype=object)
        self.target_model.set_weights(TAU * t_local + (1 - TAU) * t_target)

        if self.epsilon > self.min_epsilon:
            self.epsilon *= self.epsilon_dec

    def save_model(self, file_path):
        self.model.save(file_path)

    def load_model(self, file_path):
        self.model = load_model(file_path)
        self.target_model.set_weights(self.model.weights)

    def _update_replay_memory(self, state, action, reward, next_state, done):
        self.memory.save_new_experience(state, action, reward, next_state, done)

    def _create_model(self):
        model = Sequential()
        model.add(Dense(64, input_dim=8, activation=relu))
        model.add(Dense(64, activation=relu))
        model.add(Dense(4, activation=linear))
        model.compile(loss="mse", optimizer=Adam(learning_rate=self.lr))
        return model
