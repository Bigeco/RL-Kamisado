import gymnasium as gym
import numpy as np

from train import train_dqn_agent

def sample_play():
    env = gym.make('Kamisado-v0', render_mode="human")
    obs, info = env.reset()
    env.step(np.array([3, 13]))
    env.step(np.array([1, 9]))
    env.step(np.array([3, 0]))
    env.step(np.array([1, 4]))
    env.step(np.array([3, 0]))
    next_state, reward, _, done, info = env.step(np.array([1, 15]))
    print("Changed board: \n", info['board'])
    print("Done")

if __name__ == "__main__":
    train_dqn_agent()
    # sample_play()