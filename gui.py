import pygame
from lib import color
from common import *
from typing import List, Tuple, Optional

class RectC:
    def __init__(self, rect: pygame.Rect, c: Tuple[int, int, int]):
        self.rect = rect
        self.color = c

MAIN_GRID_BEGIN_CORNER = (330, 19)
ABOVE_GRID_BEGIN_CORNER = (330, -15)
HOLD_GRID_BEGIN_CORNER = (210, 19)
NEXT_GRID_BEGIN_CORNER = (669, 19)
NEXT_GRID_Y_DIFF = 80

SCORE_BOARD_RECTC = RectC(pygame.Rect(669, 459, 200, 200), color.LIGHT_GRAY)
INFO_BOARD_RECTC = RectC(pygame.Rect(110, 459, 200, 200), color.LIGHT_GRAY)

WIN_SIZE = (1000, 700)

pygame.init()
#pygame.display.set_mode((1,1))
FONT_SIZE = 16
FONT = pygame.font.SysFont('couriernew', FONT_SIZE)

class Gui:
    def __init__(self, is_display: bool = True, delay: int = 50):
        self.is_display = is_display
        self.delay = delay

        if not self.is_display:
            return

        # setup window
        pygame.init()
        self.win = pygame.display.set_mode(WIN_SIZE)
        pygame.display.set_caption('tetris_ai')

        # setup grids
        self.main_grid_rectc = Gui.__create_grid_rectc__(MAIN_GRID_BEGIN_CORNER, 32, 2, GAME_BOARD_HEIGHT,
                                                         GAME_BOARD_WIDTH)
        self.above_grid_rectc = Gui.__create_grid_rectc__(ABOVE_GRID_BEGIN_CORNER, 32, 2, 1, GAME_BOARD_WIDTH)
        self.hold_grid_rectc = Gui.__create_grid_rectc__(HOLD_GRID_BEGIN_CORNER, 25, 1, 3, 4)
        self.next_grid_rectc: List[List[List[RectC]]] = []
        for i in range(5):
            begin = list(NEXT_GRID_BEGIN_CORNER)
            begin[1] = begin[1] + i * NEXT_GRID_Y_DIFF
            begin = tuple(begin)
            self.next_grid_rectc.append(Gui.__create_grid_rectc__(begin, 25, 1, 3, 4))

        # setup text panels
        self.__score_board_text__ = "score board \ntesting"
        self.__info_board_text__ = "info board \ntesting"

        # Initialize font
        pygame.font.init()
        self.FONT = pygame.font.SysFont('couriernew', FONT_SIZE)

    @staticmethod
    def __create_grid_rectc__(begin_corner: Tuple[int, int], size: int, gap: int, height: int, width: int) -> List[List[RectC]]:
        return [[RectC(pygame.Rect(begin_corner[0] + j * size + gap,
                                   begin_corner[1] + i * size + gap,
                                   size - gap * 2, size - gap * 2),
                       color.cmap[0])
                 for j in range(width)]
                for i in range(height)]

    def redraw(self) -> None:
        if not self.is_display:
            return

        try:
            pygame.time.delay(self.delay)
        except KeyboardInterrupt:
            pass

        self.win.fill(color.DARK_GRAY)
        self.__paint_grids__()
        self.__paint_panels__()

        pygame.display.update()

    def __paint_grids__(self) -> None:
        for grid in [self.main_grid_rectc, self.hold_grid_rectc, *self.next_grid_rectc, self.above_grid_rectc]:
            for row in grid:
                for rc in row:
                    pygame.draw.rect(self.win, rc.color, rc.rect)

    def update_grids_color(self, grids_int: Tuple[List[List[int]], List[List[int]], List[List[List[int]]]], above_grid: Optional[List[int]] = None) -> None:
        main_int, hold_int, next_int = grids_int

        for r, row in enumerate(self.main_grid_rectc):
            for c, rc in enumerate(row):
                rc.color = color.cmap[main_int[r][c]]

        for r, row in enumerate(self.hold_grid_rectc):
            for c, rc in enumerate(row):
                rc.color = color.cmap[hold_int[r][c]]

        for g, grid in enumerate(self.next_grid_rectc):
            for r, row in enumerate(grid):
                for c, rc in enumerate(row):
                    rc.color = color.cmap[next_int[g][r][c]]

        if above_grid is not None:
            for c, rc in enumerate(self.above_grid_rectc[0]):
                rc.color = color.DARK_GRAY if above_grid[c] == 0 else color.cmap[above_grid[c]]

    def __paint_panels__(self) -> None:
        # score text
        pygame.draw.rect(self.win, SCORE_BOARD_RECTC.color, SCORE_BOARD_RECTC.rect)
        for i, line in enumerate(self.__score_board_text__.split('\n')):
            text = FONT.render(line, True, color.BLACK)
            self.win.blit(text, (671, 461 + i * FONT_SIZE))

        # info text
        pygame.draw.rect(self.win, INFO_BOARD_RECTC.color, INFO_BOARD_RECTC.rect)
        for i, line in enumerate(self.__info_board_text__.split('\n')):
            text = FONT.render(line, True, color.BLACK)
            self.win.blit(text, (112, 461 + i * FONT_SIZE))

    def set_score_text(self, score_text: str) -> None:
        self.__score_board_text__ = score_text

    def set_info_text(self, info_text: str) -> None:
        self.__info_board_text__ = info_text

if __name__ == "__main__":
    gui = Gui()

    test_main_grid = [
        [i % 10 for i in range(10)] for _ in range(20)
    ]

    test_hold_grid = [
        [i % 4 for i in range(4)] for _ in range(3)
    ]

    test_next_grid = [
        [[i % 4 for i in range(4)] for _ in range(3)] for _ in range(5)
    ]

    gui.update_grids_color((test_main_grid, test_hold_grid, test_next_grid))

    is_run = True

    while is_run:
        gui.redraw()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                is_run = False