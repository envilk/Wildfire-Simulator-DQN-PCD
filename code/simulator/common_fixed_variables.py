import torch
import random
import numpy

# COMMON VARIABLES

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SYSTEM_RANDOM = random.SystemRandom()  # ... Not available on all systems ... (Python official doc)

print(' ------------ COMMON_FIXED_VARIABLES.PY ------------ ')
print(' --------------------- OPTIONS: -------------------- ')
print('1. Train DQN ')
print('2. DQN automatic evaluation ')
print('3. PCD automatic evaluation ')
print('4. Random selection automatic evaluation (future RNN/LSTM/etc) ')
print('5. DQN interface evaluation ')
print('6. Random selection interface evaluation (future RNN/LSTM/etc)')

keep_looping = True
option = -1
while keep_looping:
    option = int(input('SELECT AN OPTION (introduce an int number): '))
    keep_looping = not (0 < option < 7) # if in range, throws false
    if keep_looping:
        print('WRONG OPTION')
print(' ---- LAUNCHING OPTION:', option, '...')

INFERENCE = False  # False = training | True = inference (eval) | (IMPORTANT, Inference must be true for using predictor)
INTERFACE = ''  # Type down "Interface" or "Automatic"
HORIZON_EVAL = False  # True = Horizon predictive (INFERENCE and RNN must be True) | False = RNN (TODO RNN not implemented)
RNN = False  # False = DRL is automatically evaluated | True = predictive algorithm is automatically evaluated

if option == 1:
    INFERENCE = False
elif option == 2:
    INFERENCE = True
    INTERFACE = 'Automatic'
    HORIZON_EVAL = False
    RNN = False
elif option == 3:
    INFERENCE = True
    INTERFACE = 'Automatic'
    HORIZON_EVAL = True
elif option == 4:
    INFERENCE = True
    INTERFACE = 'Automatic'
    HORIZON_EVAL = False
    RNN = True
elif option == 5:
    INFERENCE = True
    INTERFACE = 'Interface'
    RNN = False
elif option == 6:
    INFERENCE = True
    INTERFACE = 'Interface'
    RNN = True

if HORIZON_EVAL:
    DEVICE = torch.device("cpu")

# model ...

BATCH_SIZE = 90
N_ACTIONS = 4
WIDTH = 60  # in python [height, width] for grid, in js [width, heigh]
HEIGHT = 60

# model | forest params

DENSITY_PROB = 1  # Tree density (Float number in the interval [0, 1])
BURNING_RATE = 1
MU = 0.9  # Wind velocity (Float number in the interval [0, 1])
FIRE_SPREAD_SPEED = 2
FUEL_UPPER_LIMIT = 10

NUM_AGENTS = 3
FUEL_BOTTOM_LIMIT = 7
UAV_OBSERVATION_RADIUS = 8
side = ((UAV_OBSERVATION_RADIUS * 2) + 1)  # allows to build a square side
N_OBSERVATIONS = side * side  # calculates the observation square
VEGETATION_COLORS = ["#414141", "#9eff89", "#85e370", "#72d05c", "#62c14c", "#459f30", "#389023", "#2f831b",
                     "#236f11", "#1c630b", "#175808", "#124b05"]  # index is remaining fuel when IT ISN'T burning
FIRE_COLORS = ["#414141", "#d8d675", "#eae740", "#fefa01", "#fed401", "#feaa01", "#fe7001", "#fe5501",
               "#fe3e01", "#fe2f01", "#fe2301", "#fe0101"]  # index is remaining fuel when IT IS burning
# SMOKE_COLORS = ["#c8c8c8", "#c2c2c2", "#bbbbbb", "#b4b4b4", "#b1b1b1", "#ababab", "#a2a2a2", "#9b9b9b",
#                 "#949494", "#8f8f8f", "#8a8a8a", "#808080"]  # index is smoke density
SMOKE_COLORS = ["#ababab"]  # index is smoke density
BLACK_AND_WHITE_COLORS = ["#ffffff", "#e6e6e6", "#c9c9c9", "#b1b1b1", "#a1a1a1", "#818181", "#636363",
                          "#474747", "#303030", "#1a1a1a", "#000000"]
COLORS_LEN = len(VEGETATION_COLORS)

ACTIVATE_SMOKE = False
ACTIVATE_WIND = False
FIXED_WIND = False
# To avoid throwing "KeyError: 'Layer'" when prob or flag burning maps are shown,
# so UAV won't get its "Layer" attribute in the "portrayal_method(obj)", NUM_AGENTS must
# be set to 0.
PROBABILITY_MAP = False
BURNING_FLAGS_MAP = False
NOISE = False

# SECUIRITY DISTANCE TO OTHER UAVS

SECURITY_DISTANCE = 10

# NOISE PARAMS

OMEGA = 0
LOW = 0.1
MEDIUM = 0.5
HIGH = 1
if NOISE:
    OMEGA = HIGH

# DQN

GAMMA = 0.5
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 1000
TAU = 0.005
LR = 1e-4


# functions

# function that normalize fuel values to fit them with vegetation and fire colors
def normalize_fuel_values(fuel, limit):
    if fuel > limit:
        fuel = limit
    return max(0, round((fuel / limit) * COLORS_LEN - 1))


def euclidean_distance(x1, y1, x2, y2):
    a = numpy.array((x1, y1))
    b = numpy.array((x2, y2))
    dist = numpy.linalg.norm(a - b)
    return dist
