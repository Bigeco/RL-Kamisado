import random
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

#define Network Model(with 3 fully-connected layer)

class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

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
            self.learn(state, action, reward, next_state, done)

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def load(self, name):
        raise NotImplementedError

    def save(self, name):
        raise NotImplementedError

class DQNAgent(BaseAgent):
    def __init__(self, state_size, action_size):
        super().__init__(state_size, action_size)
        self.model = DQN(state_size, action_size)
        self.target_model = DQN(state_size, action_size)
        self.update_target_model()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def select_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0)
        act_values = self.model(state)
        return torch.argmax(act_values[0]).item()

    def learn(self, state, action, reward, next_state, done):
        target = reward
        if not done:
            next_state = torch.FloatTensor(next_state).unsqueeze(0)
            target = (reward + self.gamma *
                      torch.max(self.target_model(next_state)[0]).item())
        target_f = self.model(torch.FloatTensor(state).unsqueeze(0)).detach().numpy()
        target_f[0][action] = target
        target_f = torch.FloatTensor(target_f)
        self.optimizer.zero_grad()
        outputs = self.model(torch.FloatTensor(state).unsqueeze(0))
        loss = nn.MSELoss()(outputs, target_f)
        loss.backward()
        self.optimizer.step()

    def load(self, name):
        self.model.load_state_dict(torch.load(name))

    def save(self, name):
        torch.save(self.model.state_dict(), name)

class QLearningAgent(BaseAgent):
    def __init__(self, state_size, action_size):
        super().__init__(state_size, action_size)
        self.q_table = np.zeros((state_size, action_size))

    def select_action(self, state):
        return np.argmax(self.q_table[state])

    def learn(self, state, action, reward, next_state, done):
        best_next_action = np.argmax(self.q_table[next_state])
        target = reward + self.gamma * self.q_table[next_state][best_next_action] * (not done)
        self.q_table[state][action] += self.learning_rate * (target - self.q_table[state][action])

    def load(self, name):
        self.q_table = np.load(name)

    def save(self, name):
        np.save(name, self.q_table)

class SARSAAgent(BaseAgent):
    def __init__(self, state_size, action_size):
        super().__init__(state_size, action_size)
        self.q_table = np.zeros((state_size, action_size))

    def select_action(self, state):
        return np.argmax(self.q_table[state])

    def learn(self, state, action, reward, next_state, next_action, done):
        target = reward + self.gamma * self.q_table[next_state][next_action] * (not done)
        self.q_table[state][action] += self.learning_rate * (target - self.q_table[state][action])

    def load(self, name):
        self.q_table = np.load(name)

    def save(self, name):
        np.save(name, self.q_table)
