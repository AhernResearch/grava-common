"""
Open-loop evaluation metrics for trajectory prediction.

Implements ADE (Average Displacement Error) and FDE (Final Displacement Error)
for open-loop evaluation of CoHC trajectory outputs.

Usage:
    from src.eval.metrics_openloop import compute_ade, compute_fde, compute_openloop_metrics

    ade = compute_ade(pred_trajectory, gt_trajectory)  # in meters
    fde = compute_fde(pred_trajectory, gt_trajectory)  # in meters
    metrics = compute_openloop_metrics(predictions, ground_truths)
"""

import os
import numpy as np
from typing import List, Dict, Tuple, Optional, Union
from dataclasses import dataclass
import re
from pathlib import Path

from grava_common.trajectory.codec import TrajectoryCodec, PolarDeltaCodec, BezierCodec

_TRAJ_CODEC = TrajectoryCodec()
_BEZIER_CODEC = BezierCodec()


def _load_polar_codec() -> PolarDeltaCodec:
    """Load PolarDeltaCodec, auto-detecting adaptive edges config.

    Priority:
      1. POLAR_EDGES_CONFIG env var (explicit path)
      2. configs/adaptive_edges.json (convention default)
      3. Default handcrafted edges (fallback)
    """
    config_path = os.environ.get("POLAR_EDGES_CONFIG")
    if config_path and Path(config_path).exists():
        return PolarDeltaCodec.from_config(config_path)

    default_config = Path(__file__).resolve().parents[2] / "configs" / "adaptive_edges.json"
    if default_config.exists():
        return PolarDeltaCodec.from_config(str(default_config))

    return PolarDeltaCodec()


_POLAR_CODEC = _load_polar_codec()


def _align_trajectories(pred: np.ndarray, gt: np.ndarray):
    """Truncate pred and gt to the same length (the shorter one)."""
    n = min(len(pred), len(gt))
    return pred[:n], gt[:n]


@dataclass
class TrajectoryMetrics:
    """Container for trajectory evaluation metrics."""

    # Core metrics
    ade: float  # Average Displacement Error (meters)
    fde: float  # Final Displacement Error (meters)

    # Optional metrics
    miss_rate: float = 0.0  # Binary: 1 if FDE > threshold
    ahe: float = 0.0  # Average Heading Error (radians)
    fhe: float = 0.0  # Final Heading Error (radians)

    # Multi-horizon metrics
    ade_3s: Optional[float] = None
    ade_5s: Optional[float] = None
    fde_3s: Optional[float] = None
    fde_5s: Optional[float] = None

    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary for serialization."""
        result = {
            'ade': self.ade,
            'fde': self.fde,
            'miss_rate': self.miss_rate,
            'ahe': self.ahe,
            'fhe': self.fhe
        }
        if self.ade_3s is not None:
            result['ade_3s'] = self.ade_3s
        if self.ade_5s is not None:
            result['ade_5s'] = self.ade_5s
        if self.fde_3s is not None:
            result['fde_3s'] = self.fde_3s
        if self.fde_5s is not None:
            result['fde_5s'] = self.fde_5s
        return result


def compute_ade(
    pred_trajectory: np.ndarray,
    gt_trajectory: np.ndarray,
    return_per_step: bool = False
) -> Union[float, Tuple[float, np.ndarray]]:
    """
    Compute Average Displacement Error (ADE).

    ADE is the average L2 distance between predicted and ground truth
    trajectory points across all timesteps.

    Formula:
        ADE = (1/T) * sum_{t=1}^{T} ||pred_t - gt_t||_2

    Args:
        pred_trajectory: Predicted trajectory, shape (T, 2) or (T, 3)
                        where columns are [x, y, (heading)]
        gt_trajectory: Ground truth trajectory, shape (T, 2) or (T, 3)
        return_per_step: If True, also return per-step distances

    Returns:
        ade: Average displacement error in meters
        distances (optional): Per-step L2 distances, shape (T,)

    Example:
        >>> pred = np.array([[0, 0], [1, 1], [2, 2]])  # 3 timesteps
        >>> gt = np.array([[0, 0], [1.1, 0.9], [2.2, 1.8]])
        >>> ade = compute_ade(pred, gt)
        >>> print(f"ADE: {ade:.3f}m")
    """
    assert pred_trajectory.shape == gt_trajectory.shape, \
        f"Shape mismatch: {pred_trajectory.shape} vs {gt_trajectory.shape}"
    assert pred_trajectory.ndim >= 2, \
        f"Expected 2D array, got {pred_trajectory.ndim}D"

    # Use only x, y coordinates (ignore heading if present)
    pred_xy = pred_trajectory[:, :2]
    gt_xy = gt_trajectory[:, :2]

    # Compute L2 distances for each timestep
    distances = np.linalg.norm(pred_xy - gt_xy, axis=1)

    # Average across timesteps
    ade = float(np.mean(distances))

    if return_per_step:
        return ade, distances
    return ade


def compute_fde(
    pred_trajectory: np.ndarray,
    gt_trajectory: np.ndarray
) -> float:
    """
    Compute Final Displacement Error (FDE).

    FDE is the L2 distance between the predicted and ground truth
    final positions at the last timestep.

    Formula:
        FDE = ||pred_T - gt_T||_2

    Args:
        pred_trajectory: Predicted trajectory, shape (T, 2) or (T, 3)
        gt_trajectory: Ground truth trajectory, shape (T, 2) or (T, 3)

    Returns:
        fde: Final displacement error in meters

    Example:
        >>> pred = np.array([[0, 0], [1, 1], [2, 2]])
        >>> gt = np.array([[0, 0], [1, 1], [2.5, 1.5]])
        >>> fde = compute_fde(pred, gt)
        >>> print(f"FDE: {fde:.3f}m")
    """
    assert pred_trajectory.shape[0] == gt_trajectory.shape[0], \
        f"Length mismatch: {pred_trajectory.shape[0]} vs {gt_trajectory.shape[0]}"

    # Use only x, y coordinates
    pred_final = pred_trajectory[-1, :2]
    gt_final = gt_trajectory[-1, :2]

    # Compute L2 distance at final timestep
    fde = float(np.linalg.norm(pred_final - gt_final))

    return fde


def compute_heading_error(
    pred_trajectory: np.ndarray,
    gt_trajectory: np.ndarray
) -> Tuple[float, float]:
    """
    Compute Average and Final Heading Error (AHE, FHE).

    Args:
        pred_trajectory: Predicted trajectory with heading, shape (T, 3)
                        where columns are [x, y, heading]
        gt_trajectory: Ground truth trajectory with heading, shape (T, 3)

    Returns:
        ahe: Average heading error in radians
        fhe: Final heading error in radians
    """
    assert pred_trajectory.shape[1] >= 3 and gt_trajectory.shape[1] >= 3, \
        "Heading required for heading error computation"

    pred_heading = pred_trajectory[:, 2]
    gt_heading = gt_trajectory[:, 2]

    # Compute heading differences (handle angle wrapping)
    heading_diffs = np.arctan2(
        np.sin(pred_heading - gt_heading),
        np.cos(pred_heading - gt_heading)
    )

    ahe = float(np.mean(np.abs(heading_diffs)))
    fhe = float(np.abs(heading_diffs[-1]))

    return ahe, fhe


def compute_miss_rate(
    fde: float,
    threshold: float = 2.0
) -> float:
    """
    Compute binary miss rate based on FDE threshold.

    Args:
        fde: Final displacement error in meters
        threshold: Threshold for considering a prediction as "miss"
                  (default: 2.0m following nuPlan)

    Returns:
        miss_rate: 1.0 if FDE > threshold, else 0.0
    """
    return 1.0 if fde > threshold else 0.0


def compute_multi_horizon_metrics(
    pred_trajectory: np.ndarray,
    gt_trajectory: np.ndarray,
    timesteps_per_second: float = 2.0,  # NavSim uses 0.5s intervals
    horizons: Optional[List[float]] = None
) -> Dict[str, float]:
    """
    Compute ADE/FDE at multiple prediction horizons.

    Args:
        pred_trajectory: Full predicted trajectory
        gt_trajectory: Full ground truth trajectory
        timesteps_per_second: Number of timesteps per second
        horizons: List of horizon times in seconds

    Returns:
        Dict with ade_Xs and fde_Xs for each horizon
    """
    horizons = horizons or [3.0, 5.0, 8.0]
    results = {}

    for horizon in horizons:
        # Calculate number of timesteps for this horizon
        n_steps = int(horizon * timesteps_per_second)

        if n_steps > len(pred_trajectory):
            # Skip if trajectory is shorter than horizon
            continue

        # Slice trajectories
        pred_slice = pred_trajectory[:n_steps]
        gt_slice = gt_trajectory[:n_steps]

        # Compute metrics
        ade = compute_ade(pred_slice, gt_slice)
        fde = compute_fde(pred_slice, gt_slice)

        results[f'ade_{int(horizon)}s'] = ade
        results[f'fde_{int(horizon)}s'] = fde

    return results


def extract_trajectory_from_text(text: str) -> Optional[np.ndarray]:
    """
    Extract trajectory coordinates from prediction text.

    Supports formats:
    - Polar delta v2: D pd_R_T pd_R_T ... (inside <answer> tags)
    - Binned: x262 y47 x264 y94 ... (inside <answer> tags)
    - Legacy binned: <traj> x262 y47 ... </traj>
    - Float: [[x1, y1], [x2, y2], ...]
    - trajectory: [[...]] or waypoints: [[...]]

    Args:
        text: Model prediction text

    Returns:
        Array of shape (T, 2) with trajectory coordinates, or None if not found
    """
    # Polar delta v2 format: try decode first (9 tokens: gear + 8 delta)
    try:
        return _POLAR_CODEC.decode(text)
    except ValueError:
        pass

    # Bezier format: gear + 6 control point tokens (p1x, p1y, p2x, p2y, p3x, p3y)
    try:
        return _BEZIER_CODEC.decode(text)
    except ValueError:
        pass

    # Binned format: try decode (handles both <traj> and <answer> wrapped)
    try:
        return _TRAJ_CODEC.decode(text)
    except ValueError:
        pass

    # Pattern 1: Double bracket format [[x, y], [x, y]]
    pattern1 = r'\[\s*\[\s*(-?\d+\.?\d*)\s*,\s*(-?\d+\.?\d*)\s*\](?:\s*,\s*\[\s*(-?\d+\.?\d*)\s*,\s*(-?\d+\.?\d*)\s*\])*\s*\]'

    # Pattern 2: After "trajectory:" or "waypoints:"
    pattern2 = r'(?:trajectory|waypoints)\s*[:=]\s*(\[[\s\d,.\-\[\]]+\])'

    for pattern in [pattern1, pattern2]:
        match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
        if match:
            try:
                # Extract all numbers
                numbers = re.findall(r'-?\d+\.?\d*', match.group(0))
                coords = [float(n) for n in numbers]

                # Reshape to (T, 2)
                if len(coords) >= 4 and len(coords) % 2 == 0:
                    return np.array(coords).reshape(-1, 2)
            except Exception:
                continue

    return None


def compute_token_accuracy(predictions: List[str], references: List[str]) -> float:
    """Compute token-level accuracy for trajectory tokens."""
    if not predictions or not references:
        return 0.0

    total = 0
    correct = 0

    for pred, ref in zip(predictions, references):
        pred_tokens = pred.strip().split()
        ref_tokens = ref.strip().split()

        # Count matching tokens
        min_len = min(len(pred_tokens), len(ref_tokens))
        for i in range(min_len):
            if pred_tokens[i] == ref_tokens[i]:
                correct += 1
            total += 1

        # Count extra tokens as incorrect
        total += abs(len(pred_tokens) - len(ref_tokens))

    return correct / total if total > 0 else 0.0


def compute_openloop_metrics(
    predictions: List[Union[str, np.ndarray]],
    ground_truths: List[np.ndarray],
    miss_threshold: float = 2.0,
    compute_heading: bool = False
) -> Dict[str, float]:
    """
    Compute open-loop metrics for a batch of predictions.

    Args:
        predictions: List of predicted trajectories (arrays or text to parse)
        ground_truths: List of ground truth trajectories (arrays)
        miss_threshold: FDE threshold for miss rate computation
        compute_heading: Whether to compute heading errors (requires 3D trajectories)

    Returns:
        Dict with aggregated metrics:
            - ade: mean ADE
            - fde: mean FDE
            - miss_rate: percentage of predictions with FDE > threshold
            - ahe: mean AHE (if compute_heading=True)
            - fhe: mean FHE (if compute_heading=True)
    """
    assert len(predictions) == len(ground_truths), \
        "Number of predictions must match number of ground truths"

    all_metrics = []

    for pred, gt in zip(predictions, ground_truths):
        # Parse prediction if it's text
        if isinstance(pred, str):
            pred_traj = extract_trajectory_from_text(pred)
            if pred_traj is None:
                # Failed to extract trajectory
                continue
        else:
            pred_traj = pred

        # Ensure same length (truncate to shorter)
        pred_traj, gt_traj = _align_trajectories(pred_traj, gt)

        # Compute metrics
        try:
            ade = compute_ade(pred_traj, gt_traj)
            fde = compute_fde(pred_traj, gt_traj)
            miss = compute_miss_rate(fde, miss_threshold)

            metrics = {
                'ade': ade,
                'fde': fde,
                'miss_rate': miss
            }

            if compute_heading and pred_traj.shape[1] >= 3:
                ahe, fhe = compute_heading_error(pred_traj, gt_traj)
                metrics['ahe'] = ahe
                metrics['fhe'] = fhe

            all_metrics.append(metrics)
        except Exception as e:
            print(f"Warning: Error computing metrics: {e}")
            continue

    if not all_metrics:
        return {
            'ade': float('inf'),
            'fde': float('inf'),
            'miss_rate': 1.0
        }

    # Aggregate metrics
    aggregated = {
        'ade': np.mean([m['ade'] for m in all_metrics]),
        'fde': np.mean([m['fde'] for m in all_metrics]),
        'miss_rate': np.mean([m['miss_rate'] for m in all_metrics])
    }

    if compute_heading:
        aggregated['ahe'] = np.mean([m.get('ahe', 0) for m in all_metrics])
        aggregated['fhe'] = np.mean([m.get('fhe', 0) for m in all_metrics])

    return aggregated


class OpenLoopEvaluator:
    """
    Evaluator for open-loop trajectory prediction metrics.

    Usage:
        evaluator = OpenLoopEvaluator(miss_threshold=2.0)
        metrics = evaluator.evaluate_batch(predictions, ground_truths)
    """

    def __init__(
        self,
        miss_threshold: float = 2.0,
        compute_heading: bool = False,
        multi_horizon: bool = False
    ):
        self.miss_threshold = miss_threshold
        self.compute_heading = compute_heading
        self.multi_horizon = multi_horizon

    def evaluate_batch(
        self,
        predictions: List[Union[str, np.ndarray]],
        ground_truths: List[np.ndarray]
    ) -> Dict[str, float]:
        """Evaluate a batch of predictions."""
        return compute_openloop_metrics(
            predictions,
            ground_truths,
            miss_threshold=self.miss_threshold,
            compute_heading=self.compute_heading
        )

    def evaluate_single(
        self,
        prediction: Union[str, np.ndarray],
        ground_truth: np.ndarray
    ) -> TrajectoryMetrics:
        """Evaluate a single prediction."""
        # Parse if text
        if isinstance(prediction, str):
            pred_traj = extract_trajectory_from_text(prediction)
            if pred_traj is None:
                return TrajectoryMetrics(ade=float('inf'), fde=float('inf'), miss_rate=1.0)
        else:
            pred_traj = prediction

        # Ensure same length
        pred_traj, gt_traj = _align_trajectories(pred_traj, ground_truth)

        # Compute metrics
        ade = compute_ade(pred_traj, gt_traj)
        fde = compute_fde(pred_traj, gt_traj)
        miss = compute_miss_rate(fde, self.miss_threshold)

        metrics = TrajectoryMetrics(ade=ade, fde=fde, miss_rate=miss)

        if self.compute_heading and pred_traj.shape[1] >= 3:
            ahe, fhe = compute_heading_error(pred_traj, gt_traj)
            metrics.ahe = ahe
            metrics.fhe = fhe

        if self.multi_horizon:
            mh_metrics = compute_multi_horizon_metrics(pred_traj, gt_traj)
            metrics.ade_3s = mh_metrics.get('ade_3s')
            metrics.ade_5s = mh_metrics.get('ade_5s')
            metrics.fde_3s = mh_metrics.get('fde_3s')
            metrics.fde_5s = mh_metrics.get('fde_5s')

        return metrics
