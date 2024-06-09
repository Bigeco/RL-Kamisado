import random

import gymnasium as gym
import numpy as np

import gym_kamisado
from gym_kamisado.agents.ai_agents import DQNAgent

# from collections import deque



def train_dqn_agent(episodes=100, batch_size=32, learning_rate=0.01, discount_factor=0.99, epsilon_start=1.0, epsilon_min=0.01, epsilon_decay=0.995):
    env = gym.make('Kamisado-v0', render_mode="rgb_array")
    state_size = 8 * 8 + 1  # env.observation_space.shape[0]
    action_size = 22  # 
    dqn_agent = DQNAgent(state_size, action_size)
    episodes = 10

    for e in range(episodes):
        state, info = env.reset()
        done = False

        while not done:
            # tower - 1~8 값 가능 / target - 0~21 값 가능
            state = np.reshape(state, (1, len(state)))
            action = dqn_agent.act(state)

            tower = env.get_current_tower()    
            target = action     # dqn_agnet가 이거 값을 리턴해야함.
            print(f"Current player: {info['current_player']}, Action: {action}")
            print(f"Board state:\n{info['board']}")

            next_state, reward, _, done, info = env.step(np.array([tower, target]))
            dqn_agent.remember(state, action, reward, next_state, done)
            state = next_state

            if len(dqn_agent.memory) > batch_size:
                dqn_agent.replay(batch_size)

            print(f"episode: {e+1}/{episodes}, score: {reward}, epsilon: {dqn_agent.epsilon}")

    dqn_agent.save_model()  


if __name__ == "__main__":
    train_dqn_agent()
    # pass

