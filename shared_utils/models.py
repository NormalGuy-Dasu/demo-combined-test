from enum import Enum

class OrientationStatus(Enum):
    LIKELY_CORRECT = "LIKELY_CORRECT"
    LIKELY_ROTATED = "LIKELY_ROTATED"
    UNCERTAIN = "UNCERTAIN"