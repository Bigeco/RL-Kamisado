class KamisadoState:
    def __init__(self):
        # 8x8 보드의 초기 상태를 정의합니다. 각 칸은 해당 색상을 나타냅니다.
        self.board = [
            ["orange", "blue", "purple", "pink", "yellow", "red", "green", "brown"],
            ["red", "orange", "pink", "yellow", "brown", "purple", "blue", "green"],
            ["green", "yellow", "orange", "red", "pink", "blue", "brown", "purple"],
            ["purple", "pink", "red", "orange", "green", "brown", "yellow", "blue"],
            ["blue", "brown", "yellow", "green", "orange", "red", "purple", "pink"],
            ["pink", "purple", "brown", "blue", "red", "yellow", "green", "orange"],
            ["orange", "blue", "purple", "pink", "yellow", "red", "green", "brown"],
            ["red", "orange", "pink", "yellow", "brown", "purple", "blue", "green"]
        ]

        # 각 타워의 초기 위치를 정의합니다. 초기 위치는 (행, 열)로 표현됩니다.
        self.towers = {
            "white": [(0, 0), (0, 1), (0, 2), (0, 3), (0, 4), (0, 5), (0, 6), (0, 7)],
            "black": [(7, 0), (7, 1), (7, 2), (7, 3), (7, 4), (7, 5), (7, 6), (7, 7)]
        }

        # 현재 플레이어를 정의합니다. 'white' 또는 'black'
        self.current_player = "white"

    def get_possible_moves(self, tower):
        # 특정 타워에 대한 가능한 이동을 반환하는 함수입니다.
        # 이 예제에서는 단순히 상하좌우로 한 칸 이동하는 것을 고려합니다.
        row, col = tower
        possible_moves = []

        # 상하좌우 이동
        if row > 0: possible_moves.append((row-1, col))
        if row < 7: possible_moves.append((row+1, col))
        if col > 0: possible_moves.append((row, col-1))
        if col < 7: possible_moves.append((row, col+1))

        return possible_moves

    def move_tower(self, player, tower_index, new_position):
        # 타워를 새로운 위치로 이동시키는 함수입니다.
        self.towers[player][tower_index] = new_position

        # 타워가 멈춘 칸의 색상을 반환하여 다음 플레이어의 타워를 결정하는 데 사용됩니다.
        row, col = new_position
        return self.board[row][col]

    def switch_player(self):
        # 현재 플레이어를 전환하는 함수입니다.
        self.current_player = "black" if self.current_player == "white" else "white"

    def get_state(self):
        # 현재 보드 상태를 반환하는 함수입니다.
        return {
            "board": self.board,
            "towers": self.towers,
            "current_player": self.current_player
        }

# 예제 사용
kamisado = KamisadoState()
print(kamisado.get_state())
possible_moves = kamisado.get_possible_moves(kamisado.towers["white"][0])
print("Possible moves for white tower 0:", possible_moves)
new_color = kamisado.move_tower("white", 0, (1, 0))
print("New color at position (1, 0):", new_color)
kamisado.switch_player()
print("Current player after switch:", kamisado.current_player)
