import random
import numpy as np
# from collections import deque

import gymnasium as gym
import gym_kamisado

from gym_kamisado.agents.ai_agents import DQNAgent

def train_dqn_agent(episodes=100, batch_size=32, learning_rate=0.01, discount_factor=0.99, epsilon_start=1.0, epsilon_min=0.01, epsilon_decay=0.995):
    env = gym.make('Kamisado-v0', render_mode="rgb_array")
    state_size = (8, 8)  # board shape
    action_size = 8 * 8  # 64개의 위치 가능

    dqn_agent = DQNAgent(25, 25)

    # state = env.reset()
    # print(state) 
    """
    [print result]
    array([  7,  6,  5,  4,  3,  2,  1,  0, 
             8,  8,  8,  8,  8,  8,  8,  8, 
             8,  8,  8,  8,  8,  8,  8,  8,
             8,  8,  8,  8,  8,  8,  8,  8,  
             8,  8,  8,  8,  8,  8,  8,  8, 
             8,  8,  8,  8,  8,  8,  8,  8,
             8,  8,  8,  8,  8,  8,  8,  8,
            16, 15, 14, 13, 12, 11, 10,  9,  0]  
    -> (1d array) board and current_player
    {'current_player': 0, 
    'board': array([[-1, -2, -3, -4, -5, -6, -7, -8],
       [ 0,  0,  0,  0,  0,  0,  0,  0],
       [ 0,  0,  0,  0,  0,  0,  0,  0],
       [ 0,  0,  0,  0,  0,  0,  0,  0],
       [ 0,  0,  0,  0,  0,  0,  0,  0],
       [ 0,  0,  0,  0,  0,  0,  0,  0],
       [ 0,  0,  0,  0,  0,  0,  0,  0],
       [ 8,  7,  6,  5,  4,  3,  2,  1]], dtype=int64)}
    -> (dictionary) 'current_player' and 'board'
    """

    episodes = 1

    for e in range(episodes):
        state, info = env.reset()
        done = False

        while not done:
            action = dqn_agent.act(state)
            tower = info['current_player'] * 8 + (action // 8)
            print(f"Current player: {info['current_player']}, Action: {action}, Tower: {tower}")
            print(f"Board state:\n{info['board']}")
            valid_moves = env.unwrapped.possible_moves(tower)
            print(f"Valid moves for tower {tower}: {valid_moves}")

            if valid_moves:
                next_pos = random.choice(valid_moves)
                print(f"Chosen next position: {next_pos}")
                try:
                    target_action = np.array([tower, np.where((env.unwrapped.relative_actions == (next_pos - env.unwrapped.get_tower_coords(tower))).all(axis=1))[0][0]])
                except ValueError as ve:
                    print(f"Error in getting tower coordinates: {ve}")
                    break
                next_state, reward, done, _, info = env.step(target_action)
                dqn_agent.remember(state, action, reward, next_state, done)
                state = next_state

            if len(dqn_agent.memory) > batch_size:
                dqn_agent.replay(batch_size)
                # break
            done = True
            print(f"episode: {e+1}/{episodes}, score: {reward}, epsilon: {dqn_agent.epsilon}")

    