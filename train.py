import random
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

import gymnasium as gym
import gym_kamisado
from gym_kamisado.agents.ai_agents import DQNAgent, QLearningAgent, SARSAAgent

# from collections import deque

def print_cum_rewards_graph(cum_rewards):
    sns.lineplot(data=cum_rewards)
    plt.plot(np.arange(len(cum_rewards)), cum_rewards, 'ro')
    plt.show()

def train_dqn_agent(params):
    cum_rewards = []
    total_reward = 0
    episodes = params['episodes']
    batch_size = params['batch_size']

    env = gym.make('Kamisado-v0', render_mode="rgb_array")
    state_size = 8 * 8 + 1  # env.observation_space.shape[0]
    action_size = 22  # 
    dqn_agent = DQNAgent(state_size, action_size)

    for e in range(episodes):
        episode_reward = 0
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

            episode_reward += reward
            
            
        total_reward += episode_reward
        cum_rewards.append(total_reward)
        dqn_agent.save_model()

    dqn_agent.save_model('./gym_kamisado/agents/model/')

    print_cum_rewards_graph(cum_rewards)


def train_qlearning_agent(episodes=1000, batch_size=32, learning_rate=0.01, discount_factor=0.99, epsilon_start=1.0, epsilon_min=0.01, epsilon_decay=0.995):
    env = gym.make('Kamisado-v0', render_mode="rgb_array")
    state_size = 8 * 8 + 1
    action_size = 22
    qlearning_agent = QLearningAgent(state_size, action_size)
    
    for e in range(episodes):
        state = env.reset()
        done = False
        total_reward = 0

        while not done:
            state = state[0]  # Convert environment state to scalar value
            action = qlearning_agent.select_action(state)
            tower = env.get_current_tower()
            target = action  # Use scalar value directly

            next_state, reward, done, _, info = env.step(np.array([tower, target]))  # Pass tower and target as tuple
            qlearning_agent.learn(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward

        # Decay epsilon
        qlearning_agent.epsilon = max(epsilon_min, qlearning_agent.epsilon * epsilon_decay)

        qlearning_agent.save('gym_kamisado/agents/model/kamisado_sarsa_model.weights.npy')
        qlearning_agent.save_model('gym_kamisado/agents/model/')

        print(f"Episode: {e + 1}, Total Reward: {total_reward}, Epsilon: {qlearning_agent.epsilon}")

    env.close()
    qlearning_agent.save_model('./gym_kamisado/agents/model/')

def train_sarsa_agent(episodes=100):
    cum_rewards = []

    env = gym.make('Kamisado-v0', render_mode='rgb_array')
    state_size = 8 * 8 + 1
    action_size = 22
    sarsa_agent = SARSAAgent(state_size, action_size)
        
    for e in range(episodes):
        state, info = env.reset()
        state = np.reshape(state, (1, len(state)))[0]
        action = sarsa_agent.select_action(state)
        done = False

        while not done:
            tower = env.get_current_tower()
            target = action
            print(f"Current player: {info['current_player']}, Action: {action}")
            print(f"Board state:\n{info['board']}")

            next_state, reward, _, done, info = env.step(np.array([tower, target]))
            next_state = np.reshape(next_state, (1, -1))[0]

            next_action = sarsa_agent.select_action(next_state)
            sarsa_agent.learn(state, action, reward, next_state, next_action, done)

            state, action = next_state, next_action

            print(f"episode: {e+1}/{episodes}, score: {reward}")
            cum_rewards.append(np.sum(cum_rewards) + reward)
        
        sarsa_agent.save('gym_kamisado/agents/model/' + 'kamisado_sarsa_model.weights.npy')
    sarsa_agent.save_model('./gym_kamisado/agents/model/')
    print_cum_rewards_graph(cum_rewards)

if __name__ == "__main__":
    CONFIG={
        'episodes': 100,
        'batch_size': 32,
        'learning_rate': 0.01,
        'discount_factor': 0.99,
        'epsilon_start': 1.0,
        'epsilon_min': 0.01,
        'epsilon_decay': 0.995
    }
    # train_dqn_agent(CONFIG)
    train_qlearning_agent()
    # train_sarsa_agent()

