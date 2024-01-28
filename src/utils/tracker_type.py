from enum import Enum

class TrackerType(Enum):
    GREEDY = 1
    HUNGARIAN = 2
    HUNGARIAN_KALMAN = 3