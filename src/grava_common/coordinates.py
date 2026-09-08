"""Shared ego-frame coordinate definitions and conversions.

This module centralizes NOUS/NUPLAN coordinate protocol so all projects
(nous-agent, nous-train, grava-sim-engine) use the same conversion logic.
"""

from __future__ import annotations

import math
from enum import Enum
from typing import Tuple

import numpy as np


class EgoFrame(str, Enum):
    """Ego-centric coordinate frame identifiers.

    NOUS:
        - x: lateral (positive = right)
        - y: longitudinal (positive = forward)
        - heading = 0 means facing +Y

    NUPLAN:
        - x: longitudinal (positive = forward)
        - y: lateral (positive = left)
        - heading = 0 means facing +X
    """

    NOUS = "nous"
    NUPLAN = "nuplan"


class CoordinateConverter:
    """Convert points/vectors/headings between NOUS and NUPLAN ego frames."""

    @staticmethod
    def nous_to_nuplan(x_nous: float, y_nous: float) -> Tuple[float, float]:
        """Convert NOUS [x_right, y_forward] -> NUPLAN [x_forward, y_left]."""
        return y_nous, -x_nous

    @staticmethod
    def nuplan_to_nous(x_nuplan: float, y_nuplan: float) -> Tuple[float, float]:
        """Convert NUPLAN [x_forward, y_left] -> NOUS [x_right, y_forward]."""
        return -y_nuplan, x_nuplan

    @staticmethod
    def nous_to_nuplan_batch(points: np.ndarray) -> np.ndarray:
        """Convert batched NOUS points of shape (N, 2) to NUPLAN."""
        arr = np.asarray(points, dtype=np.float64)
        if arr.ndim != 2 or arr.shape[1] != 2:
            raise ValueError(f"Expected shape (N, 2), got {arr.shape}")
        return np.column_stack([arr[:, 1], -arr[:, 0]])

    @staticmethod
    def nuplan_to_nous_batch(points: np.ndarray) -> np.ndarray:
        """Convert batched NUPLAN points of shape (N, 2) to NOUS."""
        arr = np.asarray(points, dtype=np.float64)
        if arr.ndim != 2 or arr.shape[1] != 2:
            raise ValueError(f"Expected shape (N, 2), got {arr.shape}")
        return np.column_stack([-arr[:, 1], arr[:, 0]])

    @staticmethod
    def heading_nous_to_nuplan(heading_nous: float) -> float:
        """Convert heading from NOUS to NUPLAN frame."""
        return heading_nous + math.pi / 2

    @staticmethod
    def heading_nuplan_to_nous(heading_nuplan: float) -> float:
        """Convert heading from NUPLAN to NOUS frame."""
        return heading_nuplan - math.pi / 2
