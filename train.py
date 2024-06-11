import random

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

import gym_kamisado
from gym_kamisado.agents.ai_agents import DQNAgent, QLearningAgent, SARSAAgent

# from collections import deque

def print_cum_rewards_graph(cum_rewards, agent_name):
    plt.title(f'{agent_name}: Average cumulative sum of rewards')
    sns.lineplot(data=cum_rewards)
    plt.plot(np.arange(len(cum_rewards)), cum_rewards, 'ro')
    plt.show()

def train_dqn_agent(params):
    cum_rewards = []
    mean_cum_rewards = []
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
        mean_cum_rewards.append(total_reward/(e+1))
        dqn_agent.save_model('./gym_kamisado/agents/model/')
    env.close()
    # print_cum_rewards_graph(mean_cum_rewards, "DQN")
    return mean_cum_rewards

def train_qlearning_agent(params):
    cum_rewards = []
    mean_cum_rewards = [] # ddd
    total_reward = 0 # ddd
    episodes = params['episodes'] # ddd
    gamma = params['gamma']
    lr = params['learning_rate']
    epsilon_min = 0.01
    epsilon_decay = 0.995

    env = gym.make('Kamisado-v0', render_mode="rgb_array")
    state_size = 8 * 8 + 1
    action_size = 22
    qlearning_agent = QLearningAgent(state_size, action_size, gamma, lr)
    
    for e in range(episodes):
        episode_reward = 0
        state, info = env.reset()
        done = False

        while not done:
            # state = state  # Convert environment state to scalar value
            action = qlearning_agent.select_action(state)
            tower = env.get_current_tower()
            # target = action  # Use scalar value directly

            next_state, reward, _, done, info = env.step(np.array([tower, action]))  # Pass tower and target as tuple 
            qlearning_agent.learn(list(state), action, reward, next_state)
            state = next_state

            episode_reward += reward # ddd
            print(f"Episode: {e + 1}, Reward: {reward}, Epsilon: {qlearning_agent.epsilon}")

        # Decay epsilon
        qlearning_agent.epsilon = max(epsilon_min, qlearning_agent.epsilon * epsilon_decay)

        qlearning_agent.save('gym_kamisado/agents/model/kamisado_sarsa_model.weights.npy')
        qlearning_agent.save_model('gym_kamisado/agents/model/')


        total_reward += episode_reward # ddd
        cum_rewards.append(total_reward) # ddd
        mean_cum_rewards.append(total_reward/(e+1)) # ddd

    env.close()
    qlearning_agent.save_model('./gym_kamisado/agents/model/')
    # print_cum_rewards_graph(mean_cum_rewards, "Q-Learning")
    return mean_cum_rewards

def train_sarsa_agent(episodes=100, learning_rate=0.001, gamma=0.95, epsilon_decay=0.995):
    cum_rewards = []
    mean_cum_rewards = []
    total_reward = 0

    env = gym.make('Kamisado-v0', render_mode='rgb_array')
    state_size = 8 * 8 + 1
    action_size = 22
    sarsa_agent = SARSAAgent(state_size, action_size, learning_rate=learning_rate, gamma=gamma, epsilon_decay=epsilon_decay)
        
    for e in range(episodes):
        episode_reward = 0
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

            episode_reward += reward

        total_reward += episode_reward
        cum_rewards.append(total_reward)
        mean_cum_rewards.append(total_reward/(e+1))
        sarsa_agent.save('gym_kamisado/agents/model/' + 'kamisado_sarsa_model.weights.npy')

    sarsa_agent.save_model('./gym_kamisado/agents/model/')
    # print_cum_rewards_graph(mean_cum_rewards, "SARSA")
    env.close()
    return mean_cum_rewards



def print_cum_rewards_graphs2(lr, lst_gamma, lst_e_decay, lst_cum_rewards, model):
    if model == "SARSA":
        try:
            lst_line = ['r--', 'bs', 'g^', 'ro']
            plt.title(f'SARSA: Average cumulative sum of rewards')
            # sns.lineplot(data=cum_rewards)
            t = np.arange(len(lst_cum_rewards[0]))
            i = 0
            for g in lst_gamma:
                for ed in lst_e_decay:
                    plt.plot(t, lst_cum_rewards[i], lst_line[i], label=f'lr={lr}, gamma={g}, e_decay={ed}')
                    i += 1

            plt.legend(prop={'size': 6})
            plt.show()
        except:
            return
        
    if model == "QLearning":

        lst_line = ['r--', 'bs', 'g^', 'ro']
        plt.title(f'Q-Learning: Average cumulative sum of rewards')
        # sns.lineplot(data=cum_rewards)
        t = np.arange(len(lst_cum_rewards[0]))
        for i, g in enumerate(lst_gamma):
            plt.plot(t, lst_cum_rewards[i], lst_line[i], label=f'lr={lr}, gamma={g}')
        plt.legend(prop={'size': 6})
        plt.show()



def grid_search_QLearning():
    params = {
        'learning_rate': [0.001, 0.01, 0.1, 0.3, 0.4, 0.5, 0.6, 0.7],
        'gamma': [0.95, 0.9, 0.85, 0.8]
    }
    lst_mean_cum_rewards = []

    for i, lr in enumerate(params['learning_rate']):
        for g in params['gamma']:
            new_params = {'episodes': 60,
                          'learning_rate': lr,
                          'gamma': g}
            mean_cum_rewards = train_qlearning_agent(new_params)
            lst_mean_cum_rewards.append(mean_cum_rewards)
        print_cum_rewards_graphs2(lr, params['gamma'], None, lst_mean_cum_rewards[i*4:i*4+4], model='QLearning')

def grid_search_sarsa():
    params = {
        'learning_rate': [0.001, 0.01, 0.1, 0.3, 0.4, 0.5, 0.6, 0.7],
        'gamma': [0.95, 0.9],
        'epsilon_decay': [0.995, 0.99]
    }
    lst_mean_cum_rewards = []

    for i, lr in enumerate(params['learning_rate']):
        for g in params['gamma']:
            for ed in params['epsilon_decay']:
                mean_cum_rewards = train_sarsa_agent(60, lr, g, ed)
                lst_mean_cum_rewards.append(mean_cum_rewards)
        print_cum_rewards_graphs2(lr, params['gamma'], params['epsilon_decay'], lst_mean_cum_rewards[i*4:i*4+4], model="SARSA")

if __name__ == "__main__":
    CONFIG={
        'episodes': 60,
        'batch_size': 32,
        'learning_rate': 0.01,
        'discount_factor': 0.99,
        'epsilon_start': 1.0,
        'epsilon_min': 0.01,
        'epsilon_decay': 0.995,
        'gamma': 0.95
    }
    # train_dqn_agent(CONFIG)
    # train_qlearning_agent(CONFIG)

    grid_search_sarsa()
    # grid_search_QLearning()


