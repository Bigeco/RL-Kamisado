import numpy as np
import gym
from gym import spaces

class Kamisado(gym.Env):
    metadata = {'render.modes': ['human']}

    def parse_action(self, action: np.ndarray) -> tuple[bool, int, np.ndarray]:
        tower = int(action[0] + 1)
        target = self.relative_actions[int(action[1])]

        if self.current_tower is not None:
            tower = self.current_tower

        valid_actions = self.valid_targets(tower)
        return (valid_actions == target).all(1).any(), tower, target
    
    def move_tower(self, tower: int, target: np.ndarray):
        board = self.board
        coords = self.get_tower_coords(tower)
        board[coords[0], coords[1]] = 0
        new_coords = coords + target
        board[new_coords[0], new_coords[1]] = tower
        self.board = board

    def valid_targets(self, tower: int) -> np.ndarray:
        return self.relative_actions[self.target_mask(tower)]
    
    def step(self, action):
        done = False
        reward = 0

        tower = action // 8
        target_idx = action % 8
        target = self.relative_actions[target_idx]

        valid_actions = self.valid_targets(tower)
        valid = (valid_actions == target).all(1).any()

        if valid:
            self.move_tower(tower, target)

            done = self.is_won()
            if done:
                reward = 1
        else:
            reward = -1

        next_state = self.board
        return next_state, reward, done, {}    