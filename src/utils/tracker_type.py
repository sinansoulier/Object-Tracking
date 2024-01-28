from enum import Enum

class TrackerType(Enum):
    """
    An enum representing the type of tracker to use.
    """
    GREEDY = 1
    HUNGARIAN = 2
    HUNGARIAN_KALMAN = 3
    NN_HUNGARIAN_KALMAN = 4