from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union


@dataclass
class EgoState:
    speed_mps: float
    acceleration: float
    ego_curr_navi: str
    future_trajectory: List[List[float]]


@dataclass
class ObstacleGT:
    id: Union[int, str]
    sub_type: str
    bbox_3d: List[float]
    velocity: Optional[Dict[str, float]] = None
    speed_mps: Optional[float] = None
    distance: Optional[float] = None


@dataclass
class FrameGT:
    case_id: str
    timestamp: str
    record_type: str
    ego: EgoState
    obstacles: List[ObstacleGT] = field(default_factory=list)
    obstacle_alias: Dict[str, Union[int, str]] = field(default_factory=dict)
