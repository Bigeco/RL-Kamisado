import gymnasium as gym
import numpy as np

from gym_kamisado.agents.ai_agents import DQNAgent, SARSAAgent

state_size = 8 * 8 + 1  # env.observation_space.shape[0]
action_size = 22  # 
dqn_agent = DQNAgent(state_size, action_size)
dqn_agent.load(name='gym_kamisado/agents/model/kamisado_DQN_weight.h5')

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

if __name__ == "__main__":
    play_dqn()
    #sample_play()
    
    