from diamond_board import Diamond
from triangle_board import Triangle

BOARD = Triangle
BOARD_SIZE = 5
EMPTY_CELLS = [5]

# BOARD = Diamond
# BOARD_SIZE = 4
# EMPTY_CELLS = [7]

EPISODES = 600
USE_NN = False
HIDDEN_LAYER_NODES = [64]  # In addition there will be an input layer matching the board and an output (1 node)
LEARN_RATE_ACTOR = 0.1
LEARN_RATE_CRITIC = 0.0007  # 0.01 or more for tabular helps, but this have worked every time so far
ELIGIBILITY_DECAY_ACTOR = 0.9
ELIGIBILITY_DECAY_CRITIC = 0.9  # 0.9 for tabular
DISCOUNT_FACTOR_ACTOR = 0.9
DISCOUNT_FACTOR_CRITIC = 0.9  # 0.9 for tabular
INITIAL_EPSILON = 1
EPSILON_DECAY_RATE = 0.98  # 0.99 for tabular, 0.98 for nn
EPISODES_BEFORE_VISUALIZATION = EPISODES
FRAME_DELAY = 300
DETAILED_PLOT = False
