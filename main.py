import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np

from gym_kamisado.agents.ai_agents import DQNAgent, QLearningAgent, SARSAAgent
from train import train_dqn_agent, train_qlearning_agent, train_sarsa_agent

state_size = 8 * 8 + 1  # env.observation_space.shape[0]
action_size = 22  # 

dqn_agent = DQNAgent(state_size, action_size)
dqn_agent.load(name='gym_kamisado/agents/model/kamisado_DQN_weight.keras')

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

    env.close()

def play_dqn():
    env = gym.make('Kamisado-v0', render_mode="human")
    done = False
    obs, info = env.reset()

    while not done:
        obs = np.reshape(obs, (1, len(obs)))
        action = dqn_agent.act(obs)
        tower = env.get_current_tower()  
        target = action
        next_state, reward, _, done, info = env.step(np.array([tower, target]))
        dqn_agent.remember(obs, action, reward, next_state, done)
        obs = next_state
        print("Reward: ", reward)
    
    env.close()


def play_qlearning():
    env = gym.make('Kamisado-v0', render_mode='human')
    done = False
    obs, info = env.reset()

    qlearning_agent = QLearningAgent(state_size, action_size)
    qlearning_agent.load('gym_kamisado/agents/model/kamisado_QL_weight.npy')


    while not done:
        obs = np.reshape(obs, (1, len(obs)))
        action = qlearning_agent.select_action(obs)
        tower = env.get_current_tower()
        target = action

        next_state, reward, _, done, info = env.step(np.array([tower, target]))
        obs = next_state

        print("Reward: ", reward)
    
    env.close()


def play_sarsa():
    env = gym.make('Kamisado-v0', render_mode="human")
    done = False
    obs, info = env.reset()

    sarsa_agent = SARSAAgent(state_size, action_size)
    sarsa_agent.load(name='gym_kamisado/agents/model/kamisado_SARSA_weight.npy')

    while not done:
        obs = np.reshape(obs, (1, len(obs)))
        action = sarsa_agent.act(obs)
        tower = env.get_current_tower()  
        target = action
        next_state, reward, _, done, info = env.step(np.array([tower, target]))
        sarsa_agent.remember(obs, action, reward, next_state, done)
        obs = next_state
        print("Reward: ", reward)
    
    env.close()

'''
if __name__ == "__main__":
    agent_choice = input("Select agent to play ('dqn', 'qlearning', 'sarsa'): ").lower()
    
    if agent_choice == 'dqn':
        play_dqn()
    elif agent_choice == 'qlearning':
        play_qlearning()
    elif agent_choice == 'sarsa':
        play_sarsa()
    else:
        print("Invalid choice. Please select 'dqn', 'qlearning', or 'sarsa'.")
    pass'''

sarsa_agent = SARSAAgent(state_size, action_size)
sarsa_agent.load(name='gym_kamisado/agents/model/kamisado_SARSA_weight.npy')

qlearning_agent = QLearningAgent(state_size, action_size)
qlearning_agent.load('gym_kamisado/agents/model/kamisado_QL_weight.npy')


# dqn, q-learning, sarsa 
def play_and_plot(episodes):
    dqn_params = {
        'episodes': episodes,
        'batch_size': 32,
        'learning_rate': 0.001
    }
    dqn_mean_cum_rewards = train_dqn_agent(dqn_params)
    qlearning_params = {
        'episodes': episodes,
        'gamma': 0.9,
        'learning_rate': 0.3
    }
    qlearning_mean_cum_rewards = train_qlearning_agent(qlearning_params)
    sarsa_mean_cum_rewards = train_sarsa_agent(episodes=episodes, 
                                               learning_rate=0.4, 
                                               gamma=0.95, 
                                               epsilon_decay=0.995)
    
    plt.title('Comparison of Reinforcement Learning Algorithms')
    lst_line = ['r--', 'bs', 'g^']
    t = np.arange(60)
    plt.plot(t, dqn_mean_cum_rewards, 
             lst_line[0], 
             label=f"DQN: lr={dqn_params['learning_rate']}, batch size={dqn_params['batch_size']}")
    plt.plot(t, qlearning_mean_cum_rewards,
             lst_line[1],
             label=f"QLearning: lr={qlearning_params['learning_rate']}, gamma={qlearning_params['gamma']}")
    plt.plot(t, sarsa_mean_cum_rewards,
             lst_line[2],
             label=f'SARSA: lr=0.4, gamma=0.95, e_decay=0.995')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.legend()
    plt.show()
    
    return dqn_mean_cum_rewards, qlearning_mean_cum_rewards, sarsa_mean_cum_rewards

if __name__ == "__main__":
    dqn_mean_cum_rewards, qlearning_mean_cum_rewards, sarsa_mean_cum_rewards = play_and_plot(60)
