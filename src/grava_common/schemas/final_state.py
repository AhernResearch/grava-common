from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class StageInfo:
    status: str = ""
    model: str = ""
    timestamp: str = ""


@dataclass
class FinalStateMeta:
    stages: Dict[str, StageInfo] = field(default_factory=dict)


@dataclass
class SceneContext:
    case_id: str = ""
    timestamp: str = ""
    scene_context: Dict[str, str] = field(default_factory=dict)
    ego_state: Dict[str, Any] = field(default_factory=dict)
    future_trajectory: Any = None


@dataclass
class ObjectDecisionEntry:
    think: str = ""
    chain: str = ""
    answer: str = ""


@dataclass
class EgoDecisionCot:
    think: str = ""
    chain: str = ""
    answer: str = ""


@dataclass
class TrajectoryStage:
    planning_trajectory: Any = None
    chain: str = ""
    think: str = ""
