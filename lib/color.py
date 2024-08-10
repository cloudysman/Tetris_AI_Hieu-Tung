from typing import Tuple, Dict, List

ColorType = Tuple[int, int, int]

RED: ColorType = (255, 0, 0)
GREEN: ColorType = (0, 255, 0)
BLUE: ColorType = (0, 0, 255)
YELLOW: ColorType = (255, 255, 0)
MAGENTA: ColorType = (255, 0, 255)
CYAN: ColorType = (0, 255, 255)
WHITE: ColorType = (255, 255, 255)
BLACK: ColorType = (0, 0, 0)
GRAY: ColorType = (128, 128, 128)
LIGHT_GRAY: ColorType = (192, 192, 192)
DARK_GRAY: ColorType = (64, 64, 64)
PINK: ColorType = (255, 175, 175)
ORANGE: ColorType = (255, 200, 0)
PURPLE: ColorType = (102, 0, 153)
PINK: ColorType = (255, 105, 180)  # Note: PINK is defined twice, the latter will override the former

colors: List[ColorType] = [GRAY, RED, GREEN, CYAN, PURPLE, PINK, ORANGE, YELLOW, BLUE, (0, 237, 165), (22, 172, 237)]
cmap: Dict[int, ColorType] = {i: colors[i] if i < len(colors) else ORANGE for i in range(17)}