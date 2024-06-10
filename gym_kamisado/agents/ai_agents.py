import os
import random
from collections import deque

import numpy as np
from keras import layers, models, optimizers
from tensorflow import keras


class BaseAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
           
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        return self.select_action(state)
    
    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = reward + self.gamma * np.amax(self.model.predict(next_state)[0])
            target_f = self.model.predict(state)
            target_f[0][action] = target
            state = np.expand_dims(state, axis=0)
            target_f = np.expand_dims(target_f, axis=0)
            self.model.fit(state, target_f, epochs=1, verbose=0)

    def load(self, name):
        raise NotImplementedError

    def save(self, name):
        raise NotImplementedError


class DQNAgent(BaseAgent):
    def __init__(self, state_size, action_size):
        super().__init__(state_size, action_size)
        self.weight_backup = "kamisado_DQN_weight.h5"
        self.epsilon = 1.0
        self.exploration_min = 0.01
        self.exploration_decay = 0.995
        self.model = self._build_model()

    def _build_model(self): 
        model = models.Sequential()
        model.add(layers.Dense(128, activation='relu', input_dim=self.state_size))
        model.add(layers.Dense( 64, activation='relu'))
        model.add(layers.Dense(self.action_size))
        model.compile(loss='mse', 
                      optimizer=optimizers.Adam(learning_rate=self.learning_rate))
        return model

    def save_model(self):
        self.model.save('gym_kamisado/agents/model/' + self.weight_backup)

    def act(self, state):
        value_function = self.model.predict(state)
        if np.random.rand() > self.epsilon:
            action = np.argmax(value_function)
        else:
            action = np.random.choice(self.action_size)
        return action

    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            return
        
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                next_state = np.reshape(next_state, (1, -1))
                target = reward + self.gamma * np.amax(self.model.predict(next_state)[0])
            target_f = self.model.predict(state)
            target_f[0][action] = target
            state = np.expand_dims(state, axis=0)
            state = np.reshape(state, (1, -1))
            target_f = np.expand_dims(target_f, axis=0)
            target_f = np.reshape(target_f, (1, -1))
            self.model.fit(state, target_f, epochs=1, verbose=0)
        if self.epsilon > self.exploration_min:
            self.epsilon *= self.exploration_decay

    def load(self, name):
        self.model.load_weights(name)
        with open('gym_kamisado/agents/model/dqn_epsilon_log.txt', 'r') as file:
            lines = file.readlines()
            last_line = lines[-1].strip()
            self.epsilon = float(last_line)

    def save(self, name):
        self.model.save_weights(name)
        with open('gym_kamisado/agents/model/dqn_epsilon_log.txt', 'a') as file:
            file.write(str(self.epsilon) + '\n')


class QLearningAgent(BaseAgent):
    def __init__(self, state_size, action_size):
        super().__init__(state_size, action_size)
        self.q_table = np.zeros((state_size, action_size))
        self.weight_backup = "kamisado_QL_weight.npy"
        
    def select_action(self, state):
        return np.argmax(self.q_table[state])

    def learn(self, state, action, reward, next_state, done):
        best_next_action = np.argmax(self.q_table[next_state])
        target = reward + self.gamma * self.q_table[next_state][best_next_action] * (not done)
        self.q_table[state][action] += self.learning_rate * (target - self.q_table[state][action])

    def load(self, name):
        self.q_table = np.load(name)
        with open('gym_kamisado/agents/model/qlearning_epsilon_log.txt', 'r') as file:
            lines = file.readlines()
            last_line = lines[-1].strip()
            self.epsilon = float(last_line)

    def save(self, name):
        np.save(name, self.q_table)
        with open('gym_kamisado/agents/model/qlearning_epsilon_log.txt', 'a') as file:
            file.write(str(self.epsilon) + '\n')
    
    def save_model(self, path):
        model = models.Sequential()
        model.add(layers.Dense(128, input_dim=self.state_size, activation='relu'))
        model.add(layers.Dense(64, activation='relu'))
        model.add(layers.Dense(self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer=optimizers.Adam(learning_rate=self.learning_rate))

        # Transfer Q-table values to the model weights
        weights = model.get_weights()
        for i in range(self.state_size):
            for j in range(self.action_size):
                weights[0][i][j] = self.q_table[i][j]
        model.set_weights(weights)
        
        model.save(os.path.join(path, 'q_learning.keras'))


class SARSAAgent(BaseAgent):
    def __init__(self, state_size, action_size):
        super().__init__(state_size, action_size)
        self.q_table = np.zeros((state_size, action_size))
        #self.q_table_file = "sarsa_q_table.npy"
        self.weight_backup = "kamisado_SARSA_weight.h5"

    def select_action(self, state):
        if np.random.rand() <= self.epsilon:
            return np.random.choice(self.action_size)
        return np.argmax(self.q_table[state])

    def learn(self, state, action, reward, next_state, next_action, done):
        target = reward + self.gamma * self.q_table[next_state][next_action] * (not done)
        self.q_table[state][action] += self.learning_rate * (target - self.q_table[state][action])

        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
            
    def load(self, name):
        self.q_table = np.load(name)
        with open('gym_kamisado/agents/model/sarsa_epsilon_log.txt', 'r') as file:
            self.epsilon = float(file.read())

    def save(self, name):
        np.save(name, self.q_table)
        with open('gym_kamisado/agents/model/sarsa_epsilon_log.txt', 'a') as file:
            file.write(str(self.epsilon) + '\n')
