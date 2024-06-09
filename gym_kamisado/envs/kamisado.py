"""
  Copyright (c) Tibor Völcker <tibor.voelcker@hotmail.de>
  Created on 09.08.2023
"""
import gymnasium as gym
import numpy as np
import pygame
from gymnasium import spaces


ORANGE = (255, 153, 0)
BLUE   = (0, 0, 255)
PURPLE = (153, 0, 204)
PINK   = (255, 0, 255)
YELLOW = (255, 255, 0)
RED    = (255, 0, 0)
GREEN  = (0, 255, 0)
BROWN  = (102, 51, 0)

COLORS = [ORANGE, BLUE, PURPLE, PINK, YELLOW, RED, GREEN, BROWN]


def draw_tower(canvas, tower, coords, radius):
    player = (0, 0, 0) if tower > 0 else (255, 255, 255)
    tower = COLORS[abs(tower) - 1]
    points = np.array([[np.cos(a), np.sin(a)] for a in np.arange(0, 2 * np.pi, np.pi / 3)])
    points = points * radius + coords
    pygame.draw.polygon(canvas, player, points)
    pygame.draw.circle(canvas, tower, coords, radius * 0.6)


class GameState:
    def __init__(self):
        self._board = np.array(
            [
                [-1, -2, -3, -4, -5, -6, -7, -8],
                [0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0],
                [8, 7, 6, 5, 4, 3, 2, 1],
            ],
            dtype=np.int64,
        )

        self.board_colors = np.array(
            [
                [1, 2, 3, 4, 5, 6, 7, 8],
                [6, 1, 4, 7, 2, 5, 8, 3],
                [7, 4, 1, 6, 3, 8, 5, 2],
                [4, 3, 2, 1, 8, 7, 6, 5],
                [5, 6, 7, 8, 1, 2, 3, 4],
                [2, 5, 8, 3, 6, 1, 4, 7],
                [3, 8, 5, 2, 7, 4, 1, 6],
                [8, 7, 6, 5, 4, 3, 2, 1],
            ]
        )
        self.current_tower = None
        # black corresponds to 0, white corresponds to 1
        self.current_player = 0

        self.relative_actions = np.concatenate(
            (
                # not moving
                np.array([[0, 0]]),
                # going left
                np.dstack((np.arange(-1, -8, -1), np.arange(-1, -8, -1)))[0],
                # going up
                np.dstack((np.arange(-1, -8, -1), np.zeros(7, dtype=int)))[0],
                # going right
                np.dstack((np.arange(-1, -8, -1), np.arange(1, 8)))[0],
            )
        )
        # self.relative_actions.setflags(write=False)


    @property
    def board(self):
        """
        Process:
            코드에서 self.board 를 접근하게 되면 해당 코드가 실행된다.
            black 플레이어(0)인 경우 뒤집지 않고
            white 플레이어(1)인 경우 board에 -1를 곱하고 180도 회전한다.
            이는 항상 플레이하는 사람 시점으로 보드 상태를 바꾸는 것이다.
            즉, 현재 플레이어의 출발선이 항상 아래쪽에 있으며
            플레이어의 타워는 양수, 상대의 타워는 음수가 되도록 하는 것이다.
        Return:
            _board 리턴
        """
        if self.current_player == 1:
            return np.flip(self._board * -1)
        return self._board.copy()


    @board.setter
    def board(self, new_board: np.ndarray):
        """
        Process:
            self.board = ? 처럼 값을 할당하게 되면 해당 코드가 실행된다.
            _board를 new_board 로 바꿔준다. 
        """
        if self.current_player == 1:
            self._board = np.flip(new_board * -1)
        else:
            self._board = new_board.copy()


    def get_tower_coords(self, tower: int) -> np.ndarray:
        """
        Parameters:
            tower (int): 1~8 그리고 -1 ~ -8 숫자가 올 수 있다.
        
        Process:
            _board에서 해당 tower가 밟고 있는 좌표를 찾는다.
        
        Returns:
            1d array: tower의 좌표
        """
        y, x = np.where(self.board == tower)
        return np.array([y[0], x[0]], dtype=np.int64)


    def tower_is_blocked(self, tower: int) -> bool:
        """
        Parameters:
            tower (int): 1~8 그리고 -1 ~ -8 숫자가 올 수 있다.
        
        Process:
            해당 tower 좌표에 대해서 
            북서, 북, 북동 방향 또는 남동, 남, 남서 방향에 (1칸)
            타워가 모두 막혔는지 확인한다.
        
        Returns:
            True or False: 모두 막혔으면 True, 그 외 False
        """
        tower_coords = self.get_tower_coords(tower)
        test = np.array([[-1, -1], [-1, 0], [-1, 1]])
        if tower < 0:
            test *= -1
        test = test + tower_coords
        test = test[((test >= 0) & (test <= 7)).all(1)]
        return (self.board[tuple(test.T)] != 0).all()


    def target_mask(self, tower: int) -> np.ndarray:
        """
        Parameters:
            tower (int): 1~8 그리고 -1 ~ -8 숫자가 올 수 있다.
        
        Process:
            [step0] 북서, 북, 북동 방향 (1칸~7칸) 으로 움직이는 
            상대적인 좌표 relative_actions에 현재 tower 좌표를 더해서 
            절대적인 action의 좌표 22개를 획득한다. 
            [step1] board 좌표를 넘어서는 좌표에 대해서는 False로 바꾼다.
            [step2] 넘어서지 않는 board 좌표에 대해서 이미 상대 타워나
            내 타워가 있는 경우 False로 바꾼다.
            [step3] 첫 번째 action의 경우 [1:] 를 통해서 제외시킨다.
            이 step3에서는 타워 하나가 어딘가에 놓여져 있을 경우 
            그 방향에 대해서 뒤쪽으로는 다 갈 수 없다. 
            따라서 갈 수 없는 좌표를 False로 바꾼다.
            예를 들어, 북서 방향으로 2칸 쯤에 타워 하나가 있다면 
            북서 방향으로 3칸 부터는 갈 수 없는 것이다. 
        
        Returns:
            1d array: 각 action에 대한 True 와 False (갈 수 있는지 여부)
        """
        if self.tower_is_blocked(tower):
            return np.append(True, np.zeros(21, dtype=bool))

        tower_coords = self.get_tower_coords(tower)
        # [step0]
        if tower > 0:
            abs_actions = self.relative_actions + tower_coords
        else:
            abs_actions = self.relative_actions * -1 + tower_coords

        # [step1] set indexes outside of board invalid
        mask = ((abs_actions >= 0) & (abs_actions <= 7)).all(1)
        # [step2] set indexes with tower on them invalid
        mask[mask] = self.board[tuple(abs_actions[mask].T)] == 0
        # [step3] set all indexes after invalid indexes also invalid
        paths = mask[1:].reshape(3, 7)
        mask = np.invert(np.invert(paths).cumsum(1, dtype=bool))

        return np.append(False, mask.flatten())
    

    def valid_targets(self, tower: int) -> np.ndarray:
        """
        Parameters:
            tower (int): 1~8 그리고 -1 ~ -8 숫자가 올 수 있다.

        Returns:
            2d array: 가능한 action 리스트
                e.g. array([[-1, -1], [-2, -2], ...)
        """
        return self.relative_actions[self.target_mask(tower)]
    

    def parse_action(self, action: np.ndarray) -> tuple[bool, int, np.ndarray]:
        """
        Parameters:
            action (int, int): The action to be parsed.
                첫 번째 인덱스 값은 tower이며
                두 번째 인덱스 값은 target action 좌표이다.
                tower - 1~8 값 가능
                target - 0~22 값 가능
        
        Progress:
            이 메소드에서는 [3, 13] 을 입력으로 받으면
            tower 4 를 상대적인 좌표 리스트에서 13 인덱스에 해당하는
            좌표로 옮길 수 있는지 그 여부와 
            tower, target 을 리턴한다.

        Returns:
            bool: Wether the action is valid.
                주어진 target action에 대해서 가능한 action인지 
                그 여부를 True or False로 리턴
            int: The tower to move.
                움직일 타워 정의
            np.ndarray: The relative target to move to.
                [-1, 1] 과 같이 상대적인 action 좌표 1개 
        """
        tower = int(action[0] + 1)
        target = self.relative_actions[int(action[1])]

        if self.current_tower is not None:
            tower = self.current_tower

        # check if tower can move to the provided target
        # 처음에는 tower=None 이므로 모든 action이 valid_actions에 들어간다.
        valid_actions = self.valid_targets(tower) 

        print("Action:", action)
        print("Valid actions: \n", valid_actions)
        print("Target:", target)
        return (valid_actions == target).all(1).any(), tower, target


    def move_tower(self, tower: int, target: np.ndarray):
        """
        Parameter:
            tower (int): 이동할 타워 번호 1 ~ 8
            target (1d array, 2 elements): 이동할 상대적 위치 좌표
        
        Progress:
            tower을 target 위치에 놓는다.
        """
        board = self.board
        coords = self.get_tower_coords(tower)
        board[*coords] = 0
        board[*coords + target] = tower
        self.board = board


    def color_at_coords(self, coords: list[int] | np.ndarray):
        """
        Parameters:
            coords (int, int): 좌표
        
        Returns:
            int: 해당 좌표에 해당하는 board color를 반환함.
        """
        return self.board_colors[coords[0], coords[1]]


    @property
    def is_won(self) -> bool:
        """Check whether the game is won by reaching the goal line.

        This function requires the `self.current_player` to already be pointed
        to the next player.
        """
        return any(self.board[7] < 0)


    @property
    def is_deadlocked(self) -> bool:
        """Check whether the game is deadlocked.

        This function requires the `self.current_tower` to already be pointed
        to the first tower that might be part of the deadlock.
        """
        if self.current_tower is None:
            return False
        pointer = self.current_tower
        while self.tower_is_blocked(pointer):
            pointer_coords = self.get_tower_coords(pointer)
            color = self.color_at_coords(pointer_coords)
            pointer = -color if pointer > 0 else color

            # check if loop is complete
            if pointer == self.current_tower:
                return True
        return False


class KamisadoEnv(gym.Env):
    INVALID_ACTION_REWARD = -1000
    WINNING_REWARD = 50
    LOOSING_REWARD = -50
    ACTION_REWARD = 0

    metadata = {"render_modes": ["human", "rgb_array", "ansi"], "render_fps": 1}

    def __init__(self, render_mode=None, size=5):
        self.GAMESTATE = GameState()
        self.window_size = 512  # The size of the PyGame window
        self.observation_space = spaces.MultiDiscrete([17] * 64 + [9])
        self.action_space = spaces.MultiDiscrete([8, 22])

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        """
        If human-rendering is used, `self.window` will be a reference
        to the window that we draw to. `self.clock` will be a clock that is used
        to ensure that the environment is rendered at the correct framerate in
        human-mode. They will remain `None` until human-mode is used for the
        first time.
        """
        self.window = None
        self.clock = None
        self.font = None

    def _get_obs(self):
        return np.append(self.GAMESTATE.board.flatten() + 8, self.GAMESTATE.current_tower if self.GAMESTATE.current_tower else 0) # arr, values, axis=None

    def _get_info(self):
        return {"current_player": self.GAMESTATE.current_player, "board": self.GAMESTATE.board}

    def reset(self, seed=None, options=None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)

        if self.render_mode == "human":
            self._render_frame()

        return self._get_obs(), self._get_info()

    def step(self, action: np.ndarray):
        valid, tower, target = self.GAMESTATE.parse_action(action)
        if not valid:
            return self._get_obs(), self.INVALID_ACTION_REWARD, True, False, self._get_info()

        # move tower
        self.GAMESTATE.move_tower(tower, target)
        print("Changed board: \n", self.GAMESTATE._board)

        # set next tower and player
        self.GAMESTATE.current_tower = self.GAMESTATE.color_at_coords(self.GAMESTATE.get_tower_coords(tower))
        self.GAMESTATE.current_player = 1 if self.GAMESTATE.current_player == 0 else 0

        if self.render_mode == "human":
            self._render_frame()

        # check if game was won
        if self.GAMESTATE.is_won:
            return (
                self._get_obs(),
                self.WINNING_REWARD,
                False,
                True,
                self._get_info(),
            )

        # check if deadlocked
        if self.GAMESTATE.is_deadlocked:
            return (
                self._get_obs(),
                self.LOOSING_REWARD,
                False,
                True,
                self._get_info(),
            )

        return (
            self._get_obs(),
            self.ACTION_REWARD,
            False,
            False,
            self._get_info(),
        )

    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()

    def _render_frame(self):
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((self.window_size, self.window_size))
            self.font = pygame.font.SysFont(None, 72)
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.window_size, self.window_size))

        # draw board
        SQUARE_SIZE = self.window_size / 8
        for y, x in np.ndindex(self.GAMESTATE._board.shape):
            color = self.GAMESTATE.board_colors[y, x]
            pygame.draw.rect(
                canvas,
                COLORS[color - 1],
                (x * SQUARE_SIZE, y * SQUARE_SIZE, SQUARE_SIZE, SQUARE_SIZE),
            )

        # draw towers
        for y, x in np.ndindex(self.GAMESTATE._board.shape):
            tower = self.GAMESTATE._board[y, x]
            if tower != 0:
                draw_tower(
                    canvas,
                    tower,
                    [(x + 0.5) * SQUARE_SIZE, (y + 0.5) * SQUARE_SIZE],
                    SQUARE_SIZE * 0.4,
                )

        # draw winner
        winner = None
        if self.GAMESTATE.is_won:
            # attention! current_player is already set to the next player
            # this is required for the `self.is_deadlocked` property
            winner = "White" if self.GAMESTATE.current_player == 0 else "Black"
        elif self.GAMESTATE.is_deadlocked:
            # attention! current_player is already set to the next player
            # this is required for the `self.is_deadlocked` property
            winner = "Black" if self.GAMESTATE.current_player == 0 else "White"

        if winner:
            img = self.font.render(f"{winner} wins", True, GREEN)
            rect = img.get_rect()
            canvas.blit(
                img, (self.window_size / 2 - rect[2] / 2, self.window_size / 2 - rect[3] / 2)
            )

        if self.render_mode == "human":
            # The following line copies our drawings from `canvas` to the visible window
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()

            # We need to ensure that human-rendering occurs at the predefined framerate.
            # The following line will automatically add a delay to keep the framerate stable.
            self.clock.tick(self.metadata["render_fps"])
        else:  # rgb_array
            return np.transpose(np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2))

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
