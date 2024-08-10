import random
from common import *
from typing import List, Tuple, Optional, Union

class Tetromino:
    MASSIVE_WEIGHT = 0.135  # the chance of a piece whose name ends with 'massive' compared with other pieces
    RNG_THRESHOLD: List[float] = []
    __POOL: List['Tetromino'] = []

    @classmethod
    def create_pool(cls) -> None:
        if len(cls.__POOL) != 0:
            return
        # regular Tetris
        if GAME_TYPE == 'regular':
            cls.__POOL.extend([
                Tetromino([[1, 0], [2, 0], [0, 1], [1, 1]], 4, -1, 1.0, 0.0, 'S', 0, 2),
                Tetromino([[0, 0], [1, 0], [1, 1], [2, 1]], 4, -1, 1.0, 1.0, 'Z', 0, 2),
                Tetromino([[0, 1], [1, 1], [2, 1], [3, 1]], 3, -1, 1.5, 1.5, 'I', 0, 2),
                Tetromino([[1, 0], [0, 1], [1, 1], [2, 1]], 4, -1, 1.0, 1.0, 'T', 0, 4),
                Tetromino([[0, 0], [0, 1], [1, 1], [2, 1]], 4, -1, 0.5, 0.5, 'J', 0, 4),
                Tetromino([[2, 0], [0, 1], [1, 1], [2, 1]], 4, -1, 1.5, 0.5, 'L', 0, 4),
                Tetromino([[1, 0], [1, 1], [2, 1], [2, 0]], 3, -1, 1.5, 0.5, 'O', 0, 1)
            ])
        # mini Tetris
        elif GAME_TYPE == 'mini':
            cls.__POOL.extend([
                Tetromino([[1, 0], [1, 1]], 0, -1, 1.0, 0.0, 'i', 0, 2),
                Tetromino([[0, 0], [1, 1]], 0, -1, 0.5, 0.5, '/', 0, 2),
                Tetromino([[0, 0], [1, 0], [1, 1]], 0, -1, 0.5, 0.5, 'l', 0, 4)
            ])
        # extra Tetris
        elif GAME_TYPE == 'extra':
            cls.__POOL.extend([
                Tetromino([[1, 1]], 4, -1, 1.0, 1.0, '._extra', 0, 1),
                Tetromino([[0, 0], [1, 0]], 4, -1, 0.0, 0.0, 'i.extra', 0, 2),
                Tetromino([[0, 0], [1, 0], [2, 0]], 4, -1, 1.0, 0.0, '1.extra', 0, 2),
                Tetromino([[0, 0], [0, 1], [1, 1], [2, 1], [2, 0]], 4, -1, 1.0, 1.0, 'C.extra', 0, 4),
                Tetromino([[0, 0], [0, 1], [1, 1], [2, 1], [3, 1]], 4, -1, 1.0, 1.0, 'J.extra', 0, 4),
                Tetromino([[3, 0], [0, 1], [1, 1], [2, 1], [3, 1]], 4, -1, 2.0, 1.0, 'L.extra', 0, 4),
                Tetromino([[0, 0], [1, 0], [1, 1], [2, 1], [3, 1]], 4, -1, 0.5, 0.5, 'Z.extra', 0, 4),
                Tetromino([[0, 1], [1, 1], [1, 0], [2, 0], [3, 0]], 4, -1, 1.5, 0.5, 'S.extra', 0, 4),
                Tetromino([[0, 0], [1, 0], [2, 0], [1, 1], [1, 2]], 3, -1, 1.0, 1.0, 'T.extra', 0, 4),
                Tetromino([[1, 0], [1, 1], [1, 2], [2, 2], [3, 2], [3, 1], [3, 0], [2, 0]],
                          3, -1, 2.0, 1.0, 'O.massive', 0, 1),
                Tetromino([[0, 0], [0, 1], [0, 2], [1, 1], [2, 1], [2, 0], [2, 2]],
                          3, -1, 1, 1, 'H.massive', 0, 2),
                Tetromino([[1, 0], [1, 1], [1, 2], [2, 2], [2, 1], [3, 2], [3, 1], [3, 0], [2, 0]],
                          3, -1, 2.0, 1.0, 'Donut.massive', 0, 1),
                Tetromino([[1, 0], [0, 1], [1, 1], [2, 1], [1, 2], [3, 1]],
                          3, -1, 1.0, 1.0, 'Sword.massive', 0, 4),
            ])

        for tet in cls.__POOL:
            if 'massive' in tet.type_str:
                cls.RNG_THRESHOLD.append(cls.MASSIVE_WEIGHT)
            else:
                cls.RNG_THRESHOLD.append(1.0)

        rng_sum = sum(cls.RNG_THRESHOLD)
        cls.RNG_THRESHOLD = [x / rng_sum for x in cls.RNG_THRESHOLD]

        for i in range(1, len(cls.RNG_THRESHOLD)):
            cls.RNG_THRESHOLD[i] += cls.RNG_THRESHOLD[i - 1]

    @classmethod
    def pool_size(cls) -> int:
        return len(cls.__POOL)

    @classmethod
    def type_str_to_num(cls, type_str_arg: str) -> Optional[int]:
        for count, tetromino in enumerate(cls.__POOL, start=1):
            if type_str_arg == tetromino.type_str:
                return count
        print(f"type_str:{type_str_arg} not found")
        return None

    @classmethod
    def num_to_type_str(cls, num: int) -> str:
        # num start from 1, because 0 is reserved for empty
        return cls.__POOL[num - 1].type_str

    def __init__(self, tet: List[List[int]], start_x: int, start_y: int, rot_x: float, rot_y: float,
                 type_str_arg: str, rot: int, rot_max: int):
        self.tet = [list(sq) for sq in tet]  # make sure this is copy
        self.center_x = start_x
        self.center_y = start_y
        self.rot_x = rot_x
        self.rot_y = rot_y
        self.type_str = type_str_arg
        self.rot = rot
        self.rot_max = rot_max

    @classmethod
    def new_tetromino(cls, type_str_arg: str) -> Optional['Tetromino']:
        for tet in cls.__POOL:
            if tet.type_str == type_str_arg:
                return tet.copy()
        print("type_str is not found")
        return None

    @classmethod
    def new_tetromino_num(cls, type_num: int) -> 'Tetromino':
        return Tetromino.__POOL[type_num].copy()

    @classmethod
    def new_tetromino_fl(cls, rng_fl: Optional[float] = None) -> Optional['Tetromino']:
        if rng_fl is None:
            rng_fl = random.random()

        for i, threshold in enumerate(cls.RNG_THRESHOLD):
            if rng_fl < threshold:
                return cls.__POOL[i].copy()

        print('ERROR: rng_fl must be between 0 and 1')
        return None

    @classmethod
    def random_type_str(cls, rng_fl: Optional[float] = None) -> str:
        return cls.new_tetromino_fl(rng_fl).type_str

    def copy(self) -> 'Tetromino':
        return Tetromino(self.tet, self.center_x, self.center_y, self.rot_x, self.rot_y, self.type_str, self.rot,
                         self.rot_max)

    # turn +1 rotate counterclockwise
    def move(self, mov: Tuple[int, int, int]) -> 'Tetromino':
        (right, down, turn) = mov
        if self.type_str in ['S', 'Z'] and turn != 0:
            # for S and Z pieces, it will rotate back if they have been rotated
            turn = -1 if self.rot == 1 else 1

        if turn != 0:
            for sq in self.tet:
                a, b = sq
                x, y = self.rot_x, self.rot_y
                sq[0] = round(turn * (b - y) + x)
                sq[1] = round(-turn * (a - x) + y)

        self.center_x += right
        self.center_y += down
        self.rot = (self.rot + turn) % self.rot_max

        return self

    def to_str(self) -> str:
        displaced = self.get_displaced()
        s = " ".join(f"[{sq[0]}, {sq[1]}]" for sq in displaced)
        s += f" centerXY: {self.center_x}, {self.center_y} "
        s += f"type: {self.type_str}"
        return s

    def to_num(self) -> Optional[int]:
        return self.type_str_to_num(self.type_str)

    def get_displaced(self) -> List[List[int]]:
        return [[sq[0] + self.center_x, sq[1] + self.center_y] for sq in self.tet]

    def to_main_grid(self) -> List[List[int]]:
        disp = self.get_displaced()
        grid = [[0] * GAME_BOARD_WIDTH for _ in range(GAME_BOARD_HEIGHT)]
        for sq in disp:
            if 0 <= sq[1] < GAME_BOARD_HEIGHT and 0 <= sq[0] < GAME_BOARD_WIDTH:
                grid[sq[1]][sq[0]] = self.to_num()
        return grid

    def to_above_grid(self) -> List[int]:
        disp = self.get_displaced()
        above_grid = [0] * GAME_BOARD_WIDTH
        for sq in disp:
            if sq[1] == -1 and 0 <= sq[0] < GAME_BOARD_WIDTH:
                above_grid[sq[0]] = self.to_num()
        return above_grid

    def check_above_grid(self) -> bool:
        return any(sq[1] < 0 for sq in self.get_displaced())

    @classmethod
    def to_small_window(cls, type_str: Optional[str]) -> List[List[int]]:
        small = [[0, 0, 0, 0] for _ in range(3)]
        if type_str is None:
            return small  # if hold is None
        tetro = cls.new_tetromino(type_str)
        for sq in tetro.tet:
            a, b = sq
            small[b][a] = cls.type_str_to_num(type_str)
        return small

Tetromino.create_pool()