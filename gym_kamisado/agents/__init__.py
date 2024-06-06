"""
  Copyright (c) Tibor Völcker <tibor.voelcker@hotmail.de>
  Created on 14.07.2023
"""
from abc import ABC, abstractmethod

from gym_kamisado.envs import KamisadoEnv


class Model(ABC):
    """An agent to play Kamisado.

    Props:
        player: The side the agent plays for.
        game: The game the agent plays.
    """

    def __init__(self, env: KamisadoEnv):
        """Initialize the class."""
        self.env = env

    @abstractmethod
    def predict(self, obs, **kwargs):
        """Choose an action based on a observation."""
        pass
