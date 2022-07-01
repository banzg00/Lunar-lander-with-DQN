import gym
from Agent import Agent
import numpy as np

BUFFER_SIZE = 10_000
BATCH_SIZE = 64
GAMMA = 0.99
ALPHA = 0.0005
UPDATE_EVERY = 5
EPISODES = 2000
epsilon = 1
EPSILON_END = 0.01
EPSILON_DECREMENT = 0.996


def main():
    env = gym.make('LunarLander-v2')
    agent = Agent(ALPHA, GAMMA, epsilon, EPSILON_END, EPSILON_DECREMENT, BUFFER_SIZE, BATCH_SIZE, UPDATE_EVERY)
    scores = []
    np.random.seed(0)
    for ep in range(1, EPISODES + 1):
        score = 0
        state = env.reset()
        done = False
        if ep % 500 == 0:
            agent.save_model("./models/model_" + str(ep) + "_episodes.h5")
        while not done:
            action = agent.choose_action(state)
            next_state, reward, done, _ = env.step(action)
            agent.step(state, action, reward, next_state, done)
            env.render()
            score += reward
            state = next_state

        scores.append(score)
        if ep % 100 == 0:
            print(f"Episode: {ep}  -  Average score: {np.mean(scores[-100:])}")

    # save model


if __name__ == '__main__':
    main()
