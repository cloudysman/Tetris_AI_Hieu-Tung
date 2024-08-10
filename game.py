import pygame
from lib import helper
from tetromino import Tetromino
import random
import numpy as np
from gui import Gui
import time
from common import *
from typing import List, Tuple, Optional

INITIAL_EX_WIGHT = 0.0
SPIN_SHIFT_FOR_NON_T = [(1, 0, 0), (-1, 0, 0),
                        (0, 1, 0), (0, -1, 0),
                        (1, 1, 0), (-1, 1, 0),
                        (1, -1, 0), (-1, -1, 0),
                        (0, 2, 0), (1, 2, 0), (-1, 2, 0)]

SPIN_SHIFT_FOR_T = [(1, 0, 0), (-1, 0, 0),
                    (0, 1, 0), (0, -1, 0),
                    (1, 1, 0), (-1, 1, 0),
                    (1, -1, 0), (-1, -1, 0),
                    (0, 2, 0), (1, 2, 0), (-1, 2, 0)]  # disable triple t-spin

ACTIONS = [
    "left", "right", "down", "turn left", "turn right", "drop"
]

IDLE_MAX = 9999

class Gamestate:
    def __init__(self, grid: Optional[List[List[int]]] = None, seed: Optional[int] = None, rd: Optional[random.Random] = None, height: int = 0):
        if seed is None:
            self.seed = random.randint(0, round(9e9))
        else:
            self.seed = seed
        self.rand_count = 0

        if rd is None:
            self.rd = random.Random(seed)
        else:
            self.rd = rd

        if grid is None:
            self.grid = self.initial_grid(height)
        else:
            self.grid = [list(row) for row in grid]

        self.tetromino = Tetromino.new_tetromino_fl(self.get_random().random())
        self.hold_type: Optional[str] = None
        self.next: List[str] = []
        for _ in range(5):
            self.next.append(Tetromino.random_type_str(self.get_random().random()))
        self.next_next = Tetromino.random_type_str(self.get_random().random())
        self.n_lines = [0, 0, 0, 0]
        self.t_spins = [0, 0, 0, 0]
        self.game_status = "playing"
        self.is_hold_last = False
        self.ex_weight = INITIAL_EX_WIGHT
        self.score = 0
        self.lines = 0
        self.pieces = 0
        self.idle = 0

    def start(self) -> None:
        self.tetromino = Tetromino.new_tetromino(self.next[0])
        self.next[:-1] = self.next[1:]
        self.next[-1] = self.next_next
        self.next_next = Tetromino.random_type_str(self.get_random().random())

    def initial_grid(self, height: int = 0) -> List[List[int]]:
        grid = [[0] * GAME_BOARD_WIDTH for _ in range(GAME_BOARD_HEIGHT)]

        if height == 0:
            return grid

        for i in range(GAME_BOARD_HEIGHT - height, GAME_BOARD_HEIGHT):
            for j in range(GAME_BOARD_WIDTH):
                grid[i][j] = self.get_random().randint(0, Tetromino.pool_size())
            grid[i][self.get_random().randint(0, GAME_BOARD_WIDTH - 1)] = 0

        return grid

    def get_random_grid(self) -> List[List[int]]:
        grid = [[0] * GAME_BOARD_WIDTH for _ in range(GAME_BOARD_HEIGHT)]

        height = self.get_random().randint(0, min(15, GAME_BOARD_HEIGHT - 2))

        for i in range(GAME_BOARD_HEIGHT - height, GAME_BOARD_HEIGHT):
            for j in range(GAME_BOARD_WIDTH):
                grid[i][j] = self.get_random().randint(0, Tetromino.pool_size())
            grid[i][self.get_random().randint(0, GAME_BOARD_WIDTH - 1)] = 0

        all_brick = True
        for j in range(GAME_BOARD_WIDTH):
            grid[GAME_BOARD_HEIGHT - height - 1][j] = self.get_random().randint(0, 1)
            if grid[GAME_BOARD_HEIGHT - height - 1][j] == 0:
                all_brick = False
        if all_brick:
            grid[GAME_BOARD_HEIGHT - height - 1][self.get_random().randint(0, GAME_BOARD_WIDTH - 1)] = 0

        return grid

    @staticmethod
    def random_gamestate(seed: Optional[int] = None) -> 'Gamestate':
        if seed is None:
            large_int = 999999999
            seed = random.randint(0, large_int)
        gamestate = Gamestate(seed=seed)
        gamestate.grid = gamestate.get_random_grid()
        return gamestate

    def copy(self) -> 'Gamestate':
        state_copy = Gamestate(self.grid, rd=self.rd)

        state_copy.seed = self.seed
        state_copy.tetromino = self.tetromino.copy()
        state_copy.hold_type = self.hold_type
        state_copy.next = list(self.next)
        state_copy.next_next = self.next_next
        state_copy.n_lines = list(self.n_lines)
        state_copy.t_spins = list(self.t_spins)
        state_copy.game_status = self.game_status
        state_copy.is_hold_last = self.is_hold_last
        state_copy.ex_weight = self.ex_weight
        state_copy.score = self.score
        state_copy.lines = self.lines
        state_copy.pieces = self.pieces
        state_copy.rand_count = self.rand_count
        state_copy.idle = self.idle

        return state_copy

    def copy_value(self, state_original: 'Gamestate') -> None:
        self.grid = [row[:] for row in state_original.grid]

        self.seed = state_original.seed
        self.tetromino = state_original.tetromino.copy()
        self.hold_type = state_original.hold_type
        self.next = list(state_original.next)
        self.next_next = state_original.next_next
        self.n_lines = list(state_original.n_lines)
        self.t_spins = list(state_original.t_spins)
        self.game_status = state_original.game_status
        self.is_hold_last = state_original.is_hold_last
        self.ex_weight = state_original.ex_weight
        self.score = state_original.score
        self.lines = state_original.lines
        self.pieces = state_original.pieces
        self.rand_count = state_original.rand_count
        self.idle = state_original.idle

    def put_tet_to_grid(self, tetro: Optional[Tetromino] = None) -> List[List[int]]:
        grid_copy = helper.copy_2d(self.grid)
        if tetro is None:
            tetro = self.tetromino

        disp = tetro.get_displaced()
        for sq in disp:
            x, y = sq
            if 0 <= x < GAME_BOARD_WIDTH and 0 <= y < GAME_BOARD_HEIGHT:
                grid_copy[y][x] = tetro.to_num()
        return grid_copy

    def check_collision(self, tetro: Optional[Tetromino] = None) -> bool:
        if tetro is None:
            tetro = self.tetromino

        disp = tetro.get_displaced()
        for sq in disp:
            x, y = sq
            if x < 0 or x >= GAME_BOARD_WIDTH or y >= GAME_BOARD_HEIGHT:
                return True
            if y >= 0 and self.grid[y][x] != 0:
                return True
        return False

    def check_t_spin(self) -> bool:
        if self.tetromino.type_str != "T":
            return False
        check_mov = [(0, -1, 0), (1, 0, 0), (-1, 0, 0)]
        return all(self.check_collision(self.tetromino.copy().move(mov)) for mov in check_mov)

    def check_completed_lines(self, above_grid: Optional[List[int]] = None) -> int:
        completed_lines = 0
        for row_num, row in enumerate(self.grid):
            if all(sq != 0 for sq in row):
                self.remove_line(row_num, above_grid=above_grid)
                completed_lines += 1
        return completed_lines

    def remove_line(self, row_num: int, above_grid: Optional[List[int]] = None) -> None:
        self.grid[1:row_num + 1] = self.grid[:row_num]
        self.grid[0] = above_grid[:] if above_grid is not None else [0] * GAME_BOARD_WIDTH

    def check_clear_board(self) -> bool:
        return all(all(block == 0 for block in row) for row in self.grid)

    def update_score(self, lines: int, is_t_spin: bool, is_clear: bool) -> float:
        if is_t_spin:
            score_lines = [0, 2, 4, 5][min(lines, 3)]
            self.t_spins[lines] += 1
        else:
            score_lines = lines

        add_score = (score_lines + 1) * score_lines / 2 * 10

        if T_SPIN_MARK and is_t_spin:
            self.score = int(self.score) + add_score + 0.1
            add_score += 0.1
        else:
            self.score += add_score
        self.lines += lines

        if lines != 0:
            self.n_lines[lines - 1] += 1
        self.pieces += 1
        return add_score

    def get_score_text(self) -> str:
        return f"score:  {int(self.score)}\n" \
               f"lines:  {int(self.lines)}\n" \
               f"pieces: {self.pieces}\n" \
               f"n_lines: {' '.join(map(str, self.n_lines))}\n" \
               f"t_spins: {' '.join(map(str, self.t_spins))}\n"

    def get_info_text(self) -> str:
        return f"seed: {self.seed}"

    def soft_drop(self) -> int:
        tetro = self.tetromino
        down = 0
        while not self.check_collision(tetro.move((0, 1, 0))):
            down += 1
        tetro.move((0, -1, 0))
        return down

    def hard_drop(self) -> Tuple[float, bool]:
        self.soft_drop()
        return self.process_down_collision()

    def process_down_collision(self) -> Tuple[float, bool]:
        is_t_spin = self.check_t_spin()
        is_above_grid = self.tetromino.check_above_grid()
        above_grid = self.tetromino.to_above_grid()
        self.freeze()
        completed_lines = self.check_completed_lines(above_grid=above_grid)
        is_clear = self.check_clear_board()
        add_score = self.update_score(completed_lines, is_t_spin, is_clear)
        if self.check_collision() or (is_above_grid and completed_lines == 0):
            self.game_status = "gameover"
            done = True
        else:
            done = False
        return add_score, done

    def process_turn(self) -> bool:
        if not self.check_collision():
            return True

        spin_moves = SPIN_SHIFT_FOR_T if self.tetromino.type_str.lower() == 't' else SPIN_SHIFT_FOR_NON_T
        for mov in spin_moves:
            shifted = self.tetromino.copy().move(mov)
            if not self.check_collision(shifted):
                self.tetromino = shifted
                return True
        return False

    def process_left_right(self) -> bool:
        return not self.check_collision()

    @classmethod
    def cls_put_tet_to_grid(cls, grid: List[List[int]], tetro: Tetromino) -> Tuple[List[List[int]], bool]:
        grid_copy = helper.copy_2d(grid)
        collide = False
        for x, y in tetro.get_displaced():
            if grid_copy[y][x] != 0:
                collide = True
            grid_copy[y][x] = tetro.to_num()
        return grid_copy, collide

    def hold(self) -> bool:
        if self.is_hold_last:
            return False

        new_hold_type = self.tetromino.type_str
        if self.hold_type is None:
            self.tetromino = Tetromino.new_tetromino(self.next[0])
            self.next[:-1] = self.next[1:]
            self.next[-1] = self.next_next
            self.next_next = Tetromino.random_type_str(self.get_random().random())
        else:
            self.tetromino = Tetromino.new_tetromino(self.hold_type)

        self.hold_type = new_hold_type
        self.is_hold_last = True
        self.pieces += 1

        if self.check_collision():
            self.game_status = "gameover"
        return True

    def freeze(self) -> None:
        self.grid = self.put_tet_to_grid()
        self.tetromino = Tetromino.new_tetromino(self.next[0])
        self.next[:-1] = self.next[1:]
        self.next[-1] = self.next_next
        self.next_next = Tetromino.random_type_str(self.get_random().random())
        self.is_hold_last = False

    def get_random(self) -> random.Random:
        return self.rd

    def check_up_collision(self) -> bool:
        self.tetromino.move((0, -1, 0))
        collision = self.check_collision()
        self.tetromino.move((0, 1, 0))
        return collision

    def get_turn_expansion(self) -> Tuple[List['Gamestate'], List[List[str]]]:
        state_turn = self.copy()
        states_turn = [state_turn.copy()]
        moves_turn = [[]]
        for i in range(1, self.tetromino.rot_max):
            state_turn.tetromino.move((0, 0, 1))
            success = state_turn.process_turn()
            if not success:
                break
            state = state_turn.copy()
            states_turn.append(state)
            moves_turn.append(["turn left"] * i)

        return states_turn, moves_turn

    def get_left_right_expansion(self, moves_turn: List[str]) -> Tuple[List['Gamestate'], List[List[str]]]:
        states_lr = [self.copy()]
        moves_lr = [moves_turn]

        # move left
        state_copy = self.copy()
        left = 0
        while True:
            state_copy.tetromino.move((-1, 0, 0))
            if state_copy.check_collision():
                break
            left += 1
            moves = moves_turn + ["left"] * left
            states_lr.append(state_copy.copy())
            moves_lr.append(moves)

        # move right
        state_copy = self.copy()
        right = 0
        while True:
            state_copy.tetromino.move((1, 0, 0))
            if state_copy.check_collision():
                break
            right += 1
            moves = moves_turn + ["right"] * right
            states_lr.append(state_copy.copy())
            moves_lr.append(moves)

        # soft drop
        for s, m in list(zip(states_lr, moves_lr)):
            s.soft_drop()
            m.append("soft")

        return states_lr, moves_lr

    def get_tuck_spin_expansion(self, moves_lr: List[str]) -> Tuple[List['Gamestate'], List[List[str]]]:
        states_ts = [self.copy()]
        moves_ts = [moves_lr]

        # move left
        state_copy = self.copy()
        left = 0
        while True:
            state_copy.tetromino.move((-1, 0, 0))
            if state_copy.check_collision() or not state_copy.check_up_collision():
                break
            left += 1
            moves = moves_lr + ["left"] * left
            states_ts.append(state_copy.copy())
            moves_ts.append(moves)

        # move right
        state_copy = self.copy()
        right = 0
        while True:
            state_copy.tetromino.move((1, 0, 0))
            if state_copy.check_collision() or not state_copy.check_up_collision():
                break
            right += 1
            moves = moves_lr + ["right"] * right
            states_ts.append(state_copy.copy())
            moves_ts.append(moves)

        if self.tetromino.rot_max == 1:
            return states_ts, moves_ts

        more_states_ts = []
        more_moves_ts = []
        for i in range(len(states_ts)):
            state_copy = states_ts[i].copy()
            state_copy.tetromino.move((0, 0, 1))
            if state_copy.process_turn() and state_copy.check_up_collision():
                more_states_ts.append(state_copy)
                more_moves_ts.append(moves_ts[i] + ["turn left"])

                if self.tetromino.rot_max > 2:
                    state_copy = state_copy.copy()
                    state_copy.tetromino.move((0, 0, 1))
                    if state_copy.process_turn() and state_copy.check_up_collision():
                        more_states_ts.append(state_copy)
                        more_moves_ts.append(moves_ts[i] + ["turn left"] * 2)

            if self.tetromino.rot_max == 2:
                continue

            state_copy = states_ts[i].copy()
            state_copy.tetromino.move((0, 0, -1))
            if state_copy.process_turn() and state_copy.check_up_collision():
                more_states_ts.append(state_copy)
                more_moves_ts.append(moves_ts[i] + ["turn right"])

                if self.tetromino.rot_max > 2:
                    state_copy = state_copy.copy()
                    state_copy.tetromino.move((0, 0, -1))
                    if state_copy.process_turn() and state_copy.check_up_collision():
                        more_states_ts.append(state_copy)
                        more_moves_ts.append(moves_ts[i] + ["turn right"] * 2)

        return states_ts + more_states_ts, moves_ts + more_moves_ts

    def get_height_sum(self) -> int:
        return sum(self.get_heights())

    def get_hole_depth(self) -> List[int]:
        depth = [0] * GAME_BOARD_WIDTH
        for j in range(GAME_BOARD_WIDTH):
            has_found_brick = False
            highest_brick = 0
            for i in range(GAME_BOARD_HEIGHT):
                if not has_found_brick:
                    if self.grid[i][j] > 0:
                        has_found_brick = True
                        highest_brick = i
                elif self.grid[i][j] == 0:
                    depth[j] = i - highest_brick
                    break
        return depth

    def get_heights(self) -> List[int]:
        return [GAME_BOARD_HEIGHT - next((i for i, val in enumerate(column) if val > 0), GAME_BOARD_HEIGHT)
                for column in zip(*self.grid)]

class Game:
    # def __init__(self, gui: Optional[Gui] = None, seed: Optional[int] = None, height: int = 0):
    #     self.gui = gui
    #     self.seed = seed
    #     self.current_state = Gamestate(seed=seed, height=height)
    #     self.all_possible_states: List[Gamestate] = []
    #     self.height = height
    def __init__(self, gui: Optional[Gui] = None, seed: Optional[int] = None, height: int = 0, headless: bool = False):
            self.headless = headless
            self.gui = None if headless else gui
            self.seed = seed
            self.current_state = Gamestate(seed=seed, height=height)
            self.all_possible_states: List[Gamestate] = []
            self.height = height



    def act(self, action: str) -> Tuple[List[np.ndarray], float, bool, bool]:
        if self.current_state.game_status == "gameover":
            return self.get_state_ac(), 0, True, False

        success = False
        done = False
        add_score = 0
        action = action.lower()

        copy_state = self.current_state.copy()

        if action == "left":
            copy_state.tetromino.move((-1, 0, 0))
            success = copy_state.process_left_right()
        elif action == "right":
            copy_state.tetromino.move((1, 0, 0))
            success = copy_state.process_left_right()
        elif action == "turn left":
            copy_state.tetromino.move((0, 0, 1))
            success = copy_state.process_turn()
        elif action == "turn right":
            copy_state.tetromino.move((0, 0, -1))
            success = copy_state.process_turn()
        elif action == "down":
            copy_state.tetromino.move((0, 1, 0))
            if copy_state.check_collision():
                copy_state.tetromino.move((0, -1, 0))
                add_score, done = copy_state.process_down_collision()
            success = True  # move down will take effect no matter what
        elif action == "soft":
            # not a real move for human players
            copy_state.soft_drop()
            success = True
        elif action == "drop":
            add_score, done = copy_state.hard_drop()
            success = True
        elif action == "hold":
            success = copy_state.hold()
        else:
            print(f"{action} action is not found. Please check.")

        if success:
            self.current_state = copy_state

        if action in ["down", "drop", "hold"]:
            self.current_state.idle = 0
        elif self.current_state.idle >= IDLE_MAX:
            self.current_state.idle = 0
            self.current_state.tetromino.move((0, 1, 0))
            if self.current_state.check_collision():
                self.current_state.tetromino.move((0, -1, 0))
                add_score, done = self.current_state.process_down_collision()
            success = True  # move down will take effect no matter what
        else:
            self.current_state.idle += 1

        return self.get_state_ac(), add_score, done, success

    # def render(self) -> None:
    #     if self.gui is not None:
    #         self.update_gui()
    #         self.gui.redraw()
    #         for event in pygame.event.get():
    #             if event.type == pygame.QUIT:
    #                 pass
    def render(self) -> None:
        if self.headless or self.gui is None:
            return
        self.update_gui()
        self.gui.redraw()
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pass

    def restart(self, height: Optional[int] = None) -> None:
        self.current_state = Gamestate(seed=self.seed, height=height if height is not None else self.height)
        self.current_state.start()

    def update_gui(self, gamestate: Optional[Gamestate] = None, is_display_current: bool = True) -> None:
        if self.headless or self.gui is None:
            return
        if self.gui is None:
            return
        if gamestate is None:
            gamestate = self.current_state

        if is_display_current:
            above_grid = gamestate.tetromino.to_above_grid()
            main_grid = helper.copy_2d(gamestate.put_tet_to_grid())
        else:
            above_grid = [0] * GAME_BOARD_WIDTH
            main_grid = helper.copy_2d(gamestate.grid)

        hold_grid = Tetromino.to_small_window(gamestate.hold_type)
        next_grids = [Tetromino.to_small_window(n) for n in gamestate.next]
        self.gui.update_grids_color((main_grid, hold_grid, next_grids), above_grid)

        self.gui.set_score_text(gamestate.get_score_text())
        self.gui.set_info_text(gamestate.get_info_text())

    def run(self) -> None:
        if self.headless:
            print("Cannot run in headless mode")
            return
        is_run = True
        while is_run:
            if self.gui is not None:
                self.update_gui()
                self.gui.redraw()

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    is_run = False

                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_a:
                        self.act("left")
                    if event.key == pygame.K_d:
                        self.act("right")
                    if event.key == pygame.K_s:
                        self.act("down")
                    if event.key == pygame.K_j:
                        self.act("turn left")
                    if event.key == pygame.K_k:
                        self.act("turn right")
                    if event.key == pygame.K_SPACE:
                        self.act("drop")
                    if event.key == pygame.K_q:
                        self.act("hold")
                    if event.key == pygame.K_r:
                        self.current_state = Gamestate(seed=self.seed)
                        self.current_state.start()
                    if event.key == pygame.K_1:
                        self.display_all_possible_state()
                    if event.key == pygame.K_i:
                        self.info_print()
                    if event.key == pygame.K_2:
                        # changing current tetromino
                        pool_size = Tetromino.pool_size()
                        num = self.current_state.tetromino.to_num()
                        # remember the return num has already been increased by 1, leaving room for 0
                        if num >= pool_size:
                            num = num - pool_size
                        self.current_state.tetromino = Tetromino.new_tetromino_num(num)

    def info_print(self) -> None:
        print(self.current_state.score)

    def reset(self, height: Optional[int] = None) -> List[np.ndarray]:
        self.restart(height)
        return self.get_state_ac()

    def get_state_ac(self) -> List[np.ndarray]:
        return [self.get_main_grid_np_ac(), self.get_hold_next_np_ac()]

    def get_main_grid_np_ac(self) -> np.ndarray:
        tet_to_grid = self.current_state.tetromino.to_main_grid()
        buffer = []
        for i in range(len(self.current_state.grid)):
            for j in range(len(self.current_state.grid[i])):
                buffer.append([self.current_state.grid[i][j]])
                buffer.append([tet_to_grid[i][j]])

        buffer = np.reshape(np.array(buffer), [1, GAME_BOARD_HEIGHT, GAME_BOARD_WIDTH, 2])
        buffer = (buffer > 0) * 2 - 1
        return buffer

    def get_hold_next_np_ac(self) -> np.ndarray:
        buffer = Tetromino.to_small_window(self.current_state.hold_type)
        for tetro_type in self.current_state.next:
            buffer.extend(Tetromino.to_small_window(tetro_type))
        buffer = np.reshape(np.array(buffer), [1, 18, 4, 1])
        buffer = (buffer > 0) * 2 - 1
        return buffer

    def step(self, action: Optional[str] = None, chosen: Optional[int] = None) -> Tuple[Tuple[np.ndarray, np.ndarray], float, bool, dict]:
        if action is not None:
            state, reward, done, success = self.act(action)
        elif chosen is not None:
            self.current_state = self.all_possible_states[chosen]
            state = self.get_state_dqn_conv2d(self.current_state)
            reward = 0  # Hoặc tính toán reward dựa trên trạng thái mới
            done = self.is_done()
            success = True
        else:
            raise ValueError('Cần cung cấp action hoặc chosen')
    
        info = {'success': success}
        
        return state, reward, done, info

    def is_done(self) -> bool:
        return self.current_state.game_status == 'gameover'

    @staticmethod
    def get_state_dqn_conv2d(gamestate: Gamestate) -> Tuple[np.ndarray, np.ndarray]:
        return Game.get_main_grid_np_dqn(gamestate), Game.get_hold_next_np_dqn(gamestate)

    @staticmethod
    def get_main_grid_np_dqn(gamestate: Gamestate) -> np.ndarray:
        buffer = [[gamestate.grid[i][j]] for i in range(len(gamestate.grid)) for j in range(len(gamestate.grid[i]))]
        buffer = np.reshape(np.array(buffer), [1, GAME_BOARD_HEIGHT, GAME_BOARD_WIDTH, 1])
        return buffer > 0

    @staticmethod
    def get_hole_np_dqn(gamestate: Gamestate) -> np.ndarray:
        buffer = gamestate.get_hole_depth() + gamestate.get_heights()
        return np.reshape(np.array(buffer), [1, GAME_BOARD_WIDTH * 2, 1])

    @staticmethod
    def get_hold_next_np_dqn(gamestate: Gamestate) -> np.ndarray:
        if STATE_INPUT == 'short':
            buffer1 = [sum(gamestate.get_heights()), sum(gamestate.get_hole_depth())]
        else:
            buffer1 = gamestate.get_heights() + gamestate.get_hole_depth()

        hold_num = 1
        current_num = 1
        next_num = 4
        pool_size = Tetromino.pool_size()
        buffer2 = [0] * (pool_size * (hold_num + current_num + next_num) + hold_num)

        if hold_num == 1:
            if gamestate.is_hold_last:
                buffer2[0] = 1
            if gamestate.hold_type is not None:
                tetro_type_num = Tetromino.type_str_to_num(gamestate.hold_type) - 1
                buffer2[tetro_type_num + hold_num] = 1

        tetro_type_num = Tetromino.type_str_to_num(gamestate.tetromino.type_str) - 1
        buffer2[hold_num + hold_num * pool_size + tetro_type_num] = 1

        for i in range(next_num):
            tetro_type_num = Tetromino.type_str_to_num(gamestate.next[i]) - 1
            buffer2[hold_num + (i + hold_num + current_num) * pool_size + tetro_type_num] = 1

        return np.reshape(np.array(buffer1 + buffer2, dtype=np.int8), [1, -1])

    def get_all_possible_gamestates(self, gamestate: Optional[Gamestate] = None) -> Tuple[List[Gamestate], List[List[str]], List[float], List[bool], bool, bool]:
        if gamestate is None:
            gamestate_original = self.current_state.copy()
        else:
            gamestate_original = gamestate.copy()

        states_lr_all = []
        moves_lr_all = []
        ss, ms = gamestate_original.get_turn_expansion()
        for s, m in zip(ss, ms):
            s_lr, m_lr = s.get_left_right_expansion(m)
            states_lr_all.extend(s_lr)
            moves_lr_all.extend(m_lr)

        gamestates = []
        moves = []
        for s, m in zip(states_lr_all, moves_lr_all):
            s_ts, m_ts = s.get_tuck_spin_expansion(m)
            gamestates.extend(s_ts)
            moves.extend(m_ts)

        add_scores = []
        dones = []

        for s, m in zip(gamestates, moves):
            add_score, done = s.hard_drop()
            m.append("drop")
            add_scores.append(add_score)
            dones.append(done)

        is_include_hold = False
        is_new_hold = False
        if gamestate_original.hold_type != gamestate_original.tetromino.type_str and \
                not gamestate_original.is_hold_last:
            is_include_hold = True
            if gamestate_original.hold_type is None:
                is_new_hold = True
            gamestate_original.hold()
            gamestates.append(gamestate_original.copy())
            moves.append(["hold"])
            add_scores.append(0)
            dones.append(gamestate_original.game_status == "gameover")

        self.all_possible_states = gamestates

        return gamestates, moves, add_scores, dones, is_include_hold, is_new_hold

    def get_all_possible_states_conv2d(self) -> Tuple[List[np.ndarray], np.ndarray, List[bool], bool, bool, List[List[str]]]:
        gamestates, moves, add_scores, dones, is_include_hold, is_new_hold = self.get_all_possible_gamestates(
            self.current_state)

        mains = []
        hold_next = []
        for gamestate in gamestates:
            in1, in2 = Game.get_state_dqn_conv2d(gamestate)
            mains.append(in1)
            hold_next.append(in2)

        return [np.concatenate(mains), np.concatenate(hold_next)], np.array([add_scores]).reshape(
            [len(add_scores), 1]), dones, is_include_hold, is_new_hold, moves

    def display_all_possible_state(self) -> None:
        if self.headless or self.gui is None:
            return
        if self.gui is None:
            return

        states, moves, _, _, _, _ = self.get_all_possible_gamestates()
        for s, m in zip(states, moves):
            self.update_gui(s, is_display_current=False)
            self.gui.set_info_text(helper.text_list_flatten(m))
            self.gui.redraw()
            time.sleep(0.1)

if __name__ == "__main__":
    #game = Game(gui=Gui(), seed=None)
    game = Game(seed=None, height=None, headless=True)
    game.restart()
    game.run()