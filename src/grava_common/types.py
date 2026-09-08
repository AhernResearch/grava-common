from enum import Enum
from dataclasses import dataclass
from typing import Optional


class ObstacleType(str, Enum):
    CAR = "CAR"
    TRUCK = "TRUCK"
    BUS = "BUS"
    VAN = "VAN"
    PEDESTRIAN = "PEDESTRIAN"
    BICYCLE = "BICYCLE"
    TRAFFIC_CONE = "TRAFFIC_CONE"
    BARRIER = "BARRIER"
    UNKNOWN = "UNKNOWN"

    def __str__(self):
        return self.value


class RecordType(Enum):
    NAVSIM = "navsim"
    INTERNAL = "internal"


@dataclass
class EgoDecisionResult:
    lateral: Optional[str] = None
    longitudinal: Optional[str] = None
