from typing import Literal

USE_GUI = False  
GAME_BOARD_WIDTH: int = 10
GAME_BOARD_HEIGHT: int = 20

STATE_INPUT: Literal['short', 'long'] = 'short'
T_SPIN_MARK: bool = True
OUTER_MAX: int = 200
CPU_MAX: int = 4 # num of cpu used to collect samples = min(multiprocessing.cpu_count(), CPU_MAX)

#   1.  choose what kind of Tetris you'd like to play.
#       If 'extra', it's custom Tetris. Go to tetromino.py and search "elif GAME_TYPE == 'extra'" to edit pieces.
GAME_TYPE: Literal['extra', 'regular', 'mini'] = 'extra'
# GAME_TYPE = 'regular'

#   2.  folder name to store dataset and model. './anything_you_like/'
FOLDER_NAME: str = '/home/tung-archlinux/Documents/Tetris_AI_Hieu-Tung/trained_models'
# FOLDER_NAME = './tetris_regular/'

#   3.  if > 0, then model {FOLDER_NAME}/whole_model/outer_{OUT_START} will be loaded to continue training or watch it play
#       if 0, then create a brand new model.
OUT_START: int= 0

#   4.  choose the mode
MODE: Literal['human_player', 'ai_player_training', 'ai_player_watching'] = 'ai_player_training'
# MODE = 'human_player'
#MODE = 'ai_player_watching'

#   5.  run tetris_ai.py