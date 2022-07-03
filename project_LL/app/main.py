import gym
from Agent import Agent
import numpy as np
import time

BUFFER_SIZE = 100_000
BATCH_SIZE = 64
GAMMA = 0.99
ALPHA = 0.0005
UPDATE_EVERY = 5
EPISODES = 1300
epsilon = 0.01
EPSILON_END = 0.01
EPSILON_DECREMENT = 0.995
MAX_TIME = 1000


def write_score(score):
    with open("./scores/scores.txt", "a") as f:
        f.write(str(score) + "\n")


def main():
    env = gym.make('LunarLander-v2')
    agent = Agent(ALPHA, GAMMA, epsilon, EPSILON_END, EPSILON_DECREMENT, BUFFER_SIZE, BATCH_SIZE, UPDATE_EVERY)
    agent.load_model("./models/model3/model_1230_episodes.h5")
    scores = []
    np.random.seed(0)
    for ep in range(1231, EPISODES + 1):
        score = 0
        state = env.reset()
        start = time.time()
        for i in range(MAX_TIME):
            action = agent.choose_action(state)
            next_state, reward, done, _ = env.step(action)
            agent.step(state, action, reward, next_state, done)
            env.render()
            score += reward
            state = next_state
            if done:
                break

        scores.append(score)
        end = time.time()
        print(f"****************\n\nEpisode {ep}   -   {str(end-start)} - ({score})\n\n***************")
        if ep % 10 == 0:
            print(f"Episode: {ep}  -  Average score: {np.mean(scores[-10:])}")
            write_score(np.mean(scores[-10:]))
            agent.save_model("./models/model3/model_" + str(ep) + "_episodes.h5")


if __name__ == '__main__':
    start = time.time()
    main()
    end = time.time()
    time_in_hours = (end - start) / 3600
    print("Time: " + str(time_in_hours) + "h")
