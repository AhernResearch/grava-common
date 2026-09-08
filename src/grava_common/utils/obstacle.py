from dataclasses import asdict, is_dataclass
import re
import math
from typing import Any, Dict, List, Mapping, Optional, Union

OBSTACLE_ALIAS_PATTERN = re.compile(r'^[A-Z]$')


def _to_dict(obj: Any) -> Optional[Dict[str, Any]]:
    if obj is None:
        return None
    if isinstance(obj, Mapping):
        return dict(obj)
    if is_dataclass(obj) and not isinstance(obj, type):
        return asdict(obj)
    try:
        data = vars(obj)
    except TypeError:
        return None
    if isinstance(data, dict):
        return dict(data)
    return None


def _get_field(obj: Any, field: str) -> Any:
    if isinstance(obj, Mapping):
        return obj.get(field)
    return getattr(obj, field, None)


def compute_obstacle_speed(obstacle: Mapping[str, Any]) -> Optional[float]:
    """Compute speed scalar from velocity dict/list. Source: nous-agent frame_entity_info.py + nous-data-qa adapters/sdrive.py"""
    if obstacle is None:
        return None
    speed = _get_field(obstacle, "speed_mps")
    if isinstance(speed, (int, float)):
        return float(speed)
    velocity = _get_field(obstacle, "velocity")
    if isinstance(velocity, (int, float)):
        return abs(float(velocity))
    velocity_dict = _to_dict(velocity)
    if velocity_dict is not None:
        vx = float(velocity_dict.get("x", 0.0))
        vy = float(velocity_dict.get("y", 0.0))
        vz = float(velocity_dict.get("z", 0.0))
        return math.sqrt(vx * vx + vy * vy + vz * vz)
    if isinstance(velocity, (list, tuple)):
        components = [float(v) for v in velocity[:3]]
        while len(components) < 3:
            components.append(0.0)
        vx, vy, vz = components
        return math.sqrt(vx * vx + vy * vy + vz * vz)
    return None


def resolve_obstacle_alias(frame_or_alias_map: Any, entity_id: Union[str, int]) -> Optional[str]:
    """Resolve obstacle alias to canonical ID. Source: nous-agent frame_entity_info.py _resolve_obstacle_id"""
    try:
        sid = str(entity_id)
    except Exception:
        return None
    alias = frame_or_alias_map
    if not isinstance(frame_or_alias_map, Mapping):
        obstacle_alias = getattr(frame_or_alias_map, 'obstacle_alias', None)
        if obstacle_alias is not None:
            alias = obstacle_alias
    alias_dict = _to_dict(alias)
    if alias_dict is not None:
        key = sid.upper()
        if key in alias_dict:
            return str(alias_dict[key])
    return sid


def match_obstacle(frame_or_obstacles: Any, entity_id: Union[str, int], obstacle_alias: Optional[Dict] = None) -> Optional[Dict[str, Any]]:
    """Match obstacle by alias or ID. Source: nous-agent frame_entity_info.py _match_obstacle"""
    if isinstance(frame_or_obstacles, list):
        obstacles = frame_or_obstacles
        alias_map = obstacle_alias or {}
    else:
        obstacles = getattr(frame_or_obstacles, 'obstacles', None) or []
        alias_map = obstacle_alias if obstacle_alias is not None else (getattr(frame_or_obstacles, 'obstacle_alias', None) or {})
    target = resolve_obstacle_alias(alias_map, entity_id)
    if target is None:
        return None
    for o in obstacles:
        obstacle_dict = _to_dict(o)
        oid = obstacle_dict.get("id") if obstacle_dict is not None else _get_field(o, "id")
        if oid is not None and str(oid) == target:
            return obstacle_dict or {"id": oid}
    return None
