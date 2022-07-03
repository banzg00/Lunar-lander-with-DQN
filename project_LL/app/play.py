import gym
from Agent import Agent
import numpy as np
import time

BUFFER_SIZE = 100_000
BATCH_SIZE = 64
GAMMA = 0.99
ALPHA = 0.0005
UPDATE_EVERY = 5
EPISODES = 10
epsilon = 0.001
EPSILON_END = 0.01
EPSILON_DECREMENT = 0.995


def play():
    env = gym.make('LunarLander-v2')
    agent = Agent(ALPHA, GAMMA, epsilon, EPSILON_END, EPSILON_DECREMENT, BUFFER_SIZE, BATCH_SIZE, UPDATE_EVERY)
    agent.load_model("./models/model3/model_1230_episodes.h5")
    scores = []
    np.random.seed(0)
    for ep in range(1, EPISODES + 1):
        done = False
        score = 0
        state = env.reset()
        start = time.time()
        while not done:
            action = agent.choose_action(state)
            next_state, reward, done, _ = env.step(action)
            env.render()
            score += reward
            state = next_state

        scores.append(score)
        end = time.time()
        print(f"****************\n\nEpisode {ep}   -   {str(end - start)}   -   Score: ({score})\n\n***************")
    print(f"****************\n\nAverage score: {np.mean(scores[-10:])}\n\n***************")


if __name__ == '__main__':
    play()
