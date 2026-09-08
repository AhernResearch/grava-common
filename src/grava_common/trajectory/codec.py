"""
Trajectory binning codec: encode/decode trajectory coordinates to/from discrete bin tokens.

Encoding schemes:
1. TrajectoryCodec: Absolute separate - `x262 y47 x264 y94 ...` (16 tokens)
2. CartesianPairCodec: Absolute pair - `cp_38_47 cp_38_94 ...` (8 tokens)
3. DeltaCartesianCodec: Delta pair + gear - `D dc_32_42 dc_33_42 ...` (9 tokens)
4. PolarDeltaCodec: Polar delta + gear - `D pd_7_33 pd_7_33 ...` (9 tokens)
5. DCTCodec: DCT coefficients + gear - `D cx0_255 cx1_150 ... cy0_500 ...` (9 tokens)
6. BezierCodec: Bezier control points + gear - `D p1x_40 p1y_130 p2x_80 p2y_215 p3x_220 p3y_350` (7 tokens)
"""

import re
from typing import List, Tuple, Optional

import numpy as np
from scipy.fft import dct, idct


class TrajectoryCodec:
    """Encode/decode trajectories between float coordinates and binned text tokens."""

    def __init__(
        self,
        x_range: Tuple[float, float] = (-12.0, 12.0),
        y_range: Tuple[float, float] = (-2.0, 55.0),
        n_bins: int = 512,
    ):
        self.x_min, self.x_max = x_range
        self.y_min, self.y_max = y_range
        self.n_bins = n_bins
        self.x_step = (self.x_max - self.x_min) / n_bins
        self.y_step = (self.y_max - self.y_min) / n_bins

    @property
    def vocab_size(self) -> int:
        return self.n_bins * 2  # x and y share same bin count

    def _val_to_bin(self, val: float, vmin: float, vmax: float) -> int:
        clamped = max(vmin, min(val, vmax))
        b = int((clamped - vmin) / (vmax - vmin) * self.n_bins)
        return min(b, self.n_bins - 1)

    def _bin_to_val(self, b: int, vmin: float, vmax: float) -> float:
        return vmin + (b + 0.5) * (vmax - vmin) / self.n_bins

    def encode(self, trajectory: List[List[float]]) -> str:
        """Encode float trajectory to binned token string.

        Args:
            trajectory: List of [x, y] pairs, e.g. [[0.07, 4.63], ...]

        Returns:
            Binned string like 'x262 y47 x264 y94 ...' (no wrapper tags)
        """
        tokens = []
        for x, y in trajectory:
            xb = self._val_to_bin(x, self.x_min, self.x_max)
            yb = self._val_to_bin(y, self.y_min, self.y_max)
            tokens.append(f"x{xb} y{yb}")
        return " ".join(tokens)

    def decode(self, text: str) -> np.ndarray:
        """Decode binned token string back to float trajectory.

        Args:
            text: String containing 'x262 y47 x264 y94 ...' (optionally wrapped in <traj> tags)

        Returns:
            np.ndarray of shape (T, 2)

        Raises:
            ValueError: If no valid x/y bin tokens found.
        """
        # Extract content - either from <traj> tags or directly from <answer> section
        traj_match = re.search(r"<traj>\s*(.*?)\s*</traj>", text, re.DOTALL)
        if traj_match:
            content = traj_match.group(1)
        else:
            # Look for content inside <answer> tags
            answer_match = re.search(r"<answer>\s*(.*?)\s*</answer>", text, re.DOTALL)
            if answer_match:
                content = answer_match.group(1)
            else:
                content = text

        pairs = re.findall(r"x(\d+)\s+y(\d+)", content)
        if not pairs:
            raise ValueError("No x/y bin tokens found")

        coords = []
        for xb_str, yb_str in pairs:
            x = self._bin_to_val(int(xb_str), self.x_min, self.x_max)
            y = self._bin_to_val(int(yb_str), self.y_min, self.y_max)
            coords.append([x, y])
        return np.array(coords)


class _EdgeBinMixin:
    """Shared edge-based binning: val→bin and bin→val via searchsorted."""

    @staticmethod
    def _val_to_bin(val: float, edges: np.ndarray, n_bins: int) -> int:
        b = int(np.searchsorted(edges, val, side='right')) - 1
        return max(0, min(b, n_bins - 1))

    @staticmethod
    def _bin_to_val(b: int, edges: np.ndarray, n_bins: int) -> float:
        b = max(0, min(b, n_bins - 1))
        return float((edges[b] + edges[b + 1]) / 2)


class CartesianPairCodec(_EdgeBinMixin):
    """Encode/decode trajectories using absolute (x,y) pair tokens.

    Output format: `cp_X_Y cp_X_Y ...` (one token per waypoint, 8 tokens total)

    Each waypoint's absolute (x, y) coordinate is jointly quantized into a single
    token from a 2D grid. Halves sequence length vs separate x/y tokens, but
    requires larger vocab for equivalent resolution.
    """

    def __init__(
        self,
        x_range: Tuple[float, float] = (-12.0, 12.0),
        y_range: Tuple[float, float] = (-2.0, 55.0),
        x_bins: int = 48,
        y_bins: int = 77,
    ):
        self.x_bins, self.y_bins = x_bins, y_bins
        self._x_edges = np.linspace(x_range[0], x_range[1], x_bins + 1)
        self._y_edges = np.linspace(y_range[0], y_range[1], y_bins + 1)

    @property
    def vocab_size(self) -> int:
        return self.x_bins * self.y_bins

    def encode(self, trajectory: List[List[float]]) -> str:
        tokens = []
        for x, y in trajectory:
            xb = self._val_to_bin(x, self._x_edges, self.x_bins)
            yb = self._val_to_bin(y, self._y_edges, self.y_bins)
            tokens.append(f"cp_{xb}_{yb}")
        return " ".join(tokens)

    def decode(self, text: str) -> np.ndarray:
        for tag in ['traj', 'answer']:
            match = re.search(rf"<{tag}>\s*(.*?)\s*</{tag}>", text, re.DOTALL)
            if match:
                text = match.group(1)
                break
        pairs = re.findall(r"cp_(\d+)_(\d+)", text)
        if not pairs:
            raise ValueError("No cp_X_Y tokens found")
        return np.array([
            [self._bin_to_val(int(xb), self._x_edges, self.x_bins),
             self._bin_to_val(int(yb), self._y_edges, self.y_bins)]
            for xb, yb in pairs
        ])


class DeltaCartesianCodec(_EdgeBinMixin):
    """Encode/decode trajectories using gear + delta (dx,dy) pair tokens.

    Output format: `D dc_X_Y dc_X_Y ...` or `R dc_X_Y dc_X_Y ...`
        - D/R: gear token (same reverse detection as PolarDeltaCodec)
        - dc_X_Y: Cartesian delta from previous point (first from origin)
        - 9 tokens total (1 gear + 8 deltas)
    """

    GEAR_FORWARD = "D"
    GEAR_REVERSE = "R"

    def __init__(
        self,
        dx_range: Tuple[float, float] = (-6.0, 6.0),
        dy_range: Tuple[float, float] = (-1.0, 7.0),
        dx_bins: int = 74,
        dy_bins: int = 50,
    ):
        self.dx_bins, self.dy_bins = dx_bins, dy_bins
        self._dx_edges = np.linspace(dx_range[0], dx_range[1], dx_bins + 1)
        self._dy_edges = np.linspace(dy_range[0], dy_range[1], dy_bins + 1)

    @property
    def vocab_size(self) -> int:
        return 2 + self.dx_bins * self.dy_bins

    def encode(self, trajectory: List[List[float]]) -> str:
        reverse = trajectory[-1][1] < trajectory[0][1]
        tokens = [self.GEAR_REVERSE if reverse else self.GEAR_FORWARD]
        prev_x, prev_y = 0.0, 0.0
        for x, y in trajectory:
            dx, dy = x - prev_x, y - prev_y
            if reverse:
                dy = -dy
            dxb = self._val_to_bin(dx, self._dx_edges, self.dx_bins)
            dyb = self._val_to_bin(dy, self._dy_edges, self.dy_bins)
            tokens.append(f"dc_{dxb}_{dyb}")
            prev_x, prev_y = x, y
        return " ".join(tokens)

    def decode(self, text: str) -> np.ndarray:
        for tag in ['traj', 'answer']:
            match = re.search(rf"<{tag}>\s*(.*?)\s*</{tag}>", text, re.DOTALL)
            if match:
                text = match.group(1)
                break
        text = text.strip()
        gear_match = re.search(r'([DR])\s+', text)
        if not gear_match:
            raise ValueError("No gear token 'D' or 'R' found")
        reverse = gear_match.group(1) == self.GEAR_REVERSE
        pairs = re.findall(r"dc_(\d+)_(\d+)", text)
        if not pairs:
            raise ValueError("No dc_X_Y tokens found")
        coords = []
        x, y = 0.0, 0.0
        for dxb, dyb in pairs:
            dx = self._bin_to_val(int(dxb), self._dx_edges, self.dx_bins)
            dy = self._bin_to_val(int(dyb), self._dy_edges, self.dy_bins)
            if reverse:
                dy = -dy
            x += dx
            y += dy
            coords.append([x, y])
        return np.array(coords)


class _AdditiveVocabCodec:
    """Base class for codecs with additive vocabulary (gear + multiple param tokens).

    Provides shared utilities:
        - Gear detection and token constants
        - Uniform binning helpers
        - Text extraction and parsing for decode
    """

    GEAR_FORWARD = "D"
    GEAR_REVERSE = "R"

    @staticmethod
    def _is_reverse(trajectory: List[List[float]]) -> bool:
        """Detect reverse: endpoint y < start y (net backward displacement)."""
        return trajectory[-1][1] < trajectory[0][1]

    @staticmethod
    def _val_to_bin(val: float, vmin: float, vmax: float, n_bins: int) -> int:
        """Map value to bin index (uniform binning)."""
        clamped = max(vmin, min(val, vmax))
        b = int((clamped - vmin) / (vmax - vmin) * n_bins)
        return min(b, n_bins - 1)

    @staticmethod
    def _bin_to_val(b: int, vmin: float, vmax: float, n_bins: int) -> float:
        """Map bin index to value (uniform binning, returns bin center)."""
        return vmin + (b + 0.5) * (vmax - vmin) / n_bins

    @staticmethod
    def _extract_content(text: str) -> str:
        """Extract content from <traj> or <answer> tags, or return text as-is."""
        for tag in ['traj', 'answer']:
            match = re.search(rf"<{tag}>\s*(.*?)\s*</{tag}>", text, re.DOTALL)
            if match:
                return match.group(1)
        return text

    @classmethod
    def _parse_gear(cls, content: str) -> Tuple[bool, str]:
        """Parse gear token from content start.

        Returns:
            (is_reverse, remaining_content)
        """
        content = content.strip()
        gear_match = re.search(r'([DR])\s+', content)
        if not gear_match:
            raise ValueError("No gear token 'D' or 'R' found")
        reverse = gear_match.group(1) == cls.GEAR_REVERSE
        return reverse, content[gear_match.end():]


class PolarDeltaCodec:
    """Encode/decode trajectories using gear token + polar delta tokens.

    Output format: `D pd_R_T pd_R_T ...` or `R pd_R_T pd_R_T ...`
        - D/R: gear token (D=forward, R=reverse with dy flipped)
        - pd_R_T: polar delta (R=distance bin, T=angle bin)
        - First delta: from origin (0,0) to first waypoint
        - Subsequent deltas: from previous waypoint

    Binning modes (all share the same encode/decode path via edges):
        1. Default: handcrafted 5-segment piecewise theta + uniform r
        2. Adaptive: data-driven quantile-based edges from config
        3. Mu-law: mu-law companding — smooth non-linear compression

    Reverse (R) handling:
        - Detect: trajectory endpoint y < start point y
        - Encode: flip dy of all deltas, then encode normally
        - Decode: decode normally, then flip dy back

    Coordinate system (NavSim):
        - x: lateral (positive = left)
        - y: longitudinal (positive = forward)
    """

    GEAR_FORWARD = "D"
    GEAR_REVERSE = "R"

    def __init__(
        self,
        r_max: float = 7.0,
        r_bins: int = 56,
        theta_max: float = 60.0,
        theta_bins: int = 66,
        theta_edges: Optional[np.ndarray] = None,
        r_edges: Optional[np.ndarray] = None,
    ):
        # Theta binning: custom edges or default handcrafted
        if theta_edges is not None:
            self._theta_edges = np.asarray(theta_edges, dtype=np.float64)
        else:
            self._theta_edges = self._build_default_theta_edges(
                theta_max=theta_max,
                theta_bins=theta_bins,
            )
        self.theta_bins = len(self._theta_edges) - 1

        # R binning: custom edges or default uniform
        if r_edges is not None:
            self._r_edges = np.asarray(r_edges, dtype=np.float64)
            self.r_bins = len(self._r_edges) - 1
            self.r_max = float(self._r_edges[-1])
        else:
            self._r_edges = np.linspace(0.0, r_max, r_bins + 1)
            self.r_bins = r_bins
            self.r_max = r_max

    # -- Factory methods --

    @classmethod
    def mu_law(cls, mu_r: float = 10.0, mu_theta: float = 2.0,
               r_max: float = 7.0, r_bins: int = 56,
               theta_max_deg: float = 60.0, theta_bins: int = 66) -> 'PolarDeltaCodec':
        """Create codec with mu-law companding for both r and theta.

        Mu-law: F(x) = sign(x) * ln(1 + mu*|x|) / ln(1 + mu)
        Inverse: F^{-1}(y) = sign(y) * ((1+mu)^|y| - 1) / mu

        Higher mu → more compression toward center, coarser at edges.
        mu=0 degenerates to uniform binning.

        Args:
            mu_r: Mu parameter for r (distance). Higher = finer near r=0.
            mu_theta: Mu parameter for theta (angle). Higher = finer near 0°.
            r_max: Maximum r value (meters).
            r_bins: Number of r bins.
            theta_max_deg: Maximum theta value (degrees, symmetric ±).
            theta_bins: Number of theta bins (must be even for symmetry).
        """
        r_edges = cls._build_mulaw_edges_onesided(mu_r, 0.0, r_max, r_bins)
        theta_max = np.radians(theta_max_deg)
        theta_edges = cls._build_mulaw_edges_symmetric(mu_theta, theta_max, theta_bins)
        return cls(theta_edges=theta_edges, r_edges=r_edges)

    @classmethod
    def from_config(cls, config_path: str) -> 'PolarDeltaCodec':
        """Load codec from edges config JSON."""
        import json
        with open(config_path) as f:
            config = json.load(f)
        return cls(
            theta_edges=np.array(config["theta_edges"]),
            r_edges=np.array(config["r_edges"]),
        )

    # -- Edge builders --

    @staticmethod
    def _build_default_theta_edges(
        theta_max: float = 60.0,
        theta_bins: int = 66,
    ) -> np.ndarray:
        """Build theta edges, preserving the legacy 66-bin default when unchanged."""
        if theta_bins == 66 and np.isclose(theta_max, 60.0):
            return np.concatenate([
                np.linspace(-60, -30, 10 + 1),     # 3°/bin, 10 bins
                np.linspace(-30, -15, 8 + 1)[1:],   # 2°/bin, 8 bins
                np.linspace(-15,  15, 30 + 1)[1:],  # 1°/bin, 30 bins
                np.linspace(15, 30, 8 + 1)[1:],     # 2°/bin, 8 bins
                np.linspace(30, 60, 10 + 1)[1:],    # 3°/bin, 10 bins
            ]) * np.pi / 180

        theta_max_rad = np.radians(theta_max)
        return np.linspace(-theta_max_rad, theta_max_rad, theta_bins + 1)

    @staticmethod
    def _mulaw_inverse(y: np.ndarray, mu: float) -> np.ndarray:
        """Inverse mu-law: map uniform [0,1] → compressed [0,1].

        F^{-1}(y) = ((1+mu)^y - 1) / mu
        """
        return ((1 + mu) ** y - 1) / mu

    @classmethod
    def _build_mulaw_edges_onesided(cls, mu: float, vmin: float, vmax: float,
                                     n_bins: int) -> np.ndarray:
        """Build mu-law compressed edges for [vmin, vmax]."""
        uniform = np.linspace(0, 1, n_bins + 1)
        compressed = cls._mulaw_inverse(uniform, mu)
        return vmin + compressed * (vmax - vmin)

    @classmethod
    def _build_mulaw_edges_symmetric(cls, mu: float, half_max: float,
                                      n_bins: int) -> np.ndarray:
        """Build mu-law compressed edges symmetric around 0 for [-half_max, half_max].

        n_bins must be even. Builds positive half, then mirrors.
        """
        assert n_bins % 2 == 0, f"n_bins must be even for symmetry, got {n_bins}"
        half_bins = n_bins // 2
        # Positive half: [0, half_max] with mu-law compression
        pos_edges = cls._build_mulaw_edges_onesided(mu, 0.0, half_max, half_bins)
        # Mirror: [-pos[::-1], pos[1:]]
        return np.concatenate([-pos_edges[::-1], pos_edges[1:]])

    def save_config(self, path: str) -> None:
        """Save current edges to JSON config."""
        import json
        from pathlib import Path
        config = {
            "theta_edges": self._theta_edges.tolist(),
            "r_edges": self._r_edges.tolist(),
            "theta_bins": self.theta_bins,
            "r_bins": self.r_bins,
            "r_max": self.r_max,
        }
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(config, f, indent=2)

    @property
    def vocab_size(self) -> int:
        """Total vocabulary: 2 gear tokens + r_bins × theta_bins delta tokens."""
        return 2 + self.r_bins * self.theta_bins

    # -- Binning (unified edge-based for both r and theta) --

    def _r_to_bin(self, r: float) -> int:
        b = int(np.searchsorted(self._r_edges, r, side='right')) - 1
        return max(0, min(b, self.r_bins - 1))

    def _bin_to_r(self, b: int) -> float:
        b = max(0, min(b, self.r_bins - 1))
        return float((self._r_edges[b] + self._r_edges[b + 1]) / 2)

    def _theta_to_bin(self, theta: float) -> int:
        b = int(np.searchsorted(self._theta_edges, theta, side='right')) - 1
        return max(0, min(b, self.theta_bins - 1))

    def _bin_to_theta(self, b: int) -> float:
        b = max(0, min(b, self.theta_bins - 1))
        return float((self._theta_edges[b] + self._theta_edges[b + 1]) / 2)

    # -- Coordinate conversion --

    def _cartesian_to_polar(self, dx: float, dy: float) -> Tuple[float, float]:
        r = np.sqrt(dx * dx + dy * dy)
        theta = np.arctan2(dx, dy)  # 0=forward, positive=left
        return r, theta

    def _polar_to_cartesian(self, r: float, theta: float) -> Tuple[float, float]:
        dx = r * np.sin(theta)
        dy = r * np.cos(theta)
        return dx, dy

    def _is_reverse(self, trajectory: List[List[float]]) -> bool:
        """Detect reverse: endpoint y < start y (net backward displacement)."""
        return trajectory[-1][1] < trajectory[0][1]

    # -- Encode / Decode --

    def encode(self, trajectory: List[List[float]]) -> str:
        """Encode trajectory to gear + polar delta tokens.

        Args:
            trajectory: List of [x, y] pairs (NavSim coords)

        Returns:
            Token string like 'D pd_7_33 pd_7_33 ...' (no wrapper tags)
        """
        if len(trajectory) < 1:
            raise ValueError("Trajectory must have at least one point")

        reverse = self._is_reverse(trajectory)
        gear = self.GEAR_REVERSE if reverse else self.GEAR_FORWARD
        tokens = [gear]

        prev_x, prev_y = 0.0, 0.0  # origin
        for x, y in trajectory:
            dx = x - prev_x
            dy = y - prev_y
            if reverse:
                dy = -dy

            r, theta = self._cartesian_to_polar(dx, dy)
            rb = self._r_to_bin(r)

            # When stopped (r_bin=0), theta is meaningless → force to bin 0
            tb = 0 if rb == 0 else self._theta_to_bin(theta)
            tokens.append(f"pd_{rb}_{tb}")

            prev_x, prev_y = x, y

        return " ".join(tokens)

    def encode_compensated(self, trajectory: List[List[float]]) -> str:
        """Encode with error compensation: track decoded cumulative position.

        Instead of computing deltas from GT previous point (greedy),
        compute deltas from the *decoded* cumulative position. This eliminates
        the encode/decode asymmetry that causes error accumulation.

        Complexity: O(n), same as greedy encode.
        """
        if len(trajectory) < 1:
            raise ValueError("Trajectory must have at least one point")

        reverse = self._is_reverse(trajectory)
        gear = self.GEAR_REVERSE if reverse else self.GEAR_FORWARD
        tokens = [gear]

        dec_x, dec_y = 0.0, 0.0  # decoded cumulative position
        for x, y in trajectory:
            dx = x - dec_x
            dy = y - dec_y
            if reverse:
                dy = -dy

            r, theta = self._cartesian_to_polar(dx, dy)
            rb = self._r_to_bin(r)
            tb = 0 if rb == 0 else self._theta_to_bin(theta)
            tokens.append(f"pd_{rb}_{tb}")

            # Simulate decoder to update tracked position
            r_dec = self._bin_to_r(rb)
            t_dec = self._bin_to_theta(0 if rb == 0 else tb)
            ddx, ddy = self._polar_to_cartesian(r_dec, t_dec)
            if reverse:
                ddy = -ddy
            dec_x += ddx
            dec_y += ddy

        return " ".join(tokens)

    def encode_beam(self, trajectory: List[List[float]], beam_width: int = 50) -> str:
        """Encode with beam search: maintain top-K candidates minimizing ADE.

        At each waypoint, expand all K beams by all V possible tokens,
        score by cumulative squared position error, and keep top K.

        Complexity: O(n * K * V) where V = 1 + (r_bins-1) * theta_bins.
        """
        if len(trajectory) < 1:
            raise ValueError("Trajectory must have at least one point")

        reverse = self._is_reverse(trajectory)
        gear = self.GEAR_REVERSE if reverse else self.GEAR_FORWARD

        # Precompute all valid (rb, tb) → world-space (dx, dy)
        opt_tokens = []  # (rb, tb) per option
        opt_deltas = []  # (dx, dy) per option
        for rb in range(self.r_bins):
            r = self._bin_to_r(rb)
            tb_range = [0] if rb == 0 else range(self.theta_bins)
            for tb in tb_range:
                t = self._bin_to_theta(tb)
                dx, dy = self._polar_to_cartesian(r, t)
                if reverse:
                    dy = -dy
                opt_tokens.append((rb, tb))
                opt_deltas.append([dx, dy])
        opt_deltas = np.array(opt_deltas)  # (V, 2)
        V = len(opt_tokens)

        # Beam state
        beam_scores = np.zeros(1)
        beam_pos = np.zeros((1, 2))
        beam_tok_idx: List[List[int]] = [[]]

        for gt_x, gt_y in trajectory:
            gt = np.array([gt_x, gt_y])
            K = len(beam_scores)

            # Expand: (K, V, 2) = (K,1,2) + (1,V,2)
            new_pos = beam_pos[:, None, :] + opt_deltas[None, :, :]
            errors = np.sum((new_pos - gt) ** 2, axis=2)  # (K, V)
            new_scores = beam_scores[:, None] + errors     # (K, V)

            # Select top beam_width
            flat = new_scores.ravel()
            n_keep = min(beam_width, len(flat))
            top_flat = np.argpartition(flat, n_keep)[:n_keep]

            bi = top_flat // V
            oi = top_flat % V

            beam_scores = flat[top_flat]
            beam_pos = new_pos[bi, oi]
            beam_tok_idx = [beam_tok_idx[b] + [o] for b, o in zip(bi, oi)]

        # Best beam → token string
        best = int(np.argmin(beam_scores))
        tokens = [gear]
        for o in beam_tok_idx[best]:
            rb, tb = opt_tokens[o]
            tokens.append(f"pd_{rb}_{tb}")
        return " ".join(tokens)

    def decode(self, text: str) -> np.ndarray:
        """Decode gear + polar delta tokens to trajectory.

        Args:
            text: String with 'D pd_R_T ...' or 'R pd_R_T ...' (optionally in tags)

        Returns:
            np.ndarray of shape (T, 2)

        Raises:
            ValueError: If no valid gear/delta tokens found
        """
        # Extract content from tags
        for tag in ['traj', 'answer']:
            match = re.search(rf"<{tag}>\s*(.*?)\s*</{tag}>", text, re.DOTALL)
            if match:
                content = match.group(1)
                break
        else:
            content = text

        content = content.strip()

        # Parse gear token
        gear_match = re.search(r'([DR])\s+', content)
        if not gear_match:
            raise ValueError("No gear token 'D' or 'R' found")
        reverse = gear_match.group(1) == self.GEAR_REVERSE

        # Parse polar delta tokens
        delta_matches = re.findall(r"pd_(\d+)_(\d+)", content)
        if not delta_matches:
            raise ValueError("No polar delta tokens 'pd_R_T' found")

        coords = []
        x, y = 0.0, 0.0  # start from origin
        for rb_str, tb_str in delta_matches:
            r = self._bin_to_r(int(rb_str))
            theta = self._bin_to_theta(int(tb_str))
            dx, dy = self._polar_to_cartesian(r, theta)
            if reverse:
                dy = -dy
            x += dx
            y += dy
            coords.append([x, y])

        return np.array(coords)


class DCTCodec(_AdditiveVocabCodec):
    """Encode/decode trajectories using DCT coefficients.

    Output format: [Gear] [X0] [X1] [X2] [X3] [Y0] [Y1] [Y2] [Y3] = 9 tokens
        - Gear: D/R (forward/reverse)
        - X0-X3: DCT coefficients of x coordinates (cx0_*, cx1_*, cx2_*, cx3_*)
        - Y0-Y3: DCT coefficients of y coordinates (cy0_*, cy1_*, cy2_*, cy3_*)

    DCT compresses 8 points into 4 coefficients per axis, capturing
    low-frequency shape information. Reconstruction via IDCT.
    """

    # X coefficient specs: (min, max, bins)
    _X_PARAMS = [
        (-75.0, 75.0, 500),   # X0 (DC): cx0_
        (-45.0, 45.0, 300),   # X1: cx1_
        (-12.0, 8.0, 200),    # X2: cx2_
        (-4.0, 8.0, 120),     # X3: cx3_
    ]

    # Y coefficient specs: (min, max, bins)
    _Y_PARAMS = [
        (-15.0, 295.0, 1000),  # Y0 (DC): cy0_
        (-105.0, 10.0, 400),   # Y1: cy1_
        (-12.0, 10.0, 220),    # Y2: cy2_
        (-16.0, 2.0, 180),     # Y3: cy3_
    ]

    _X_PREFIXES = ["cx0_", "cx1_", "cx2_", "cx3_"]
    _Y_PREFIXES = ["cy0_", "cy1_", "cy2_", "cy3_"]

    @property
    def vocab_size(self) -> int:
        """Total vocab = 2 (gear) + sum of all coefficient bins."""
        x_bins = sum(p[2] for p in self._X_PARAMS)
        y_bins = sum(p[2] for p in self._Y_PARAMS)
        return 2 + x_bins + y_bins

    def encode(self, trajectory: List[List[float]]) -> str:
        """Encode 8-point trajectory to 9 DCT tokens."""
        if len(trajectory) != 8:
            raise ValueError(f"DCTCodec requires exactly 8 points, got {len(trajectory)}")

        reverse = self._is_reverse(trajectory)
        gear = self.GEAR_REVERSE if reverse else self.GEAR_FORWARD

        # Extract x and y sequences
        x_seq = np.array([p[0] for p in trajectory], dtype=np.float64)
        y_seq = np.array([p[1] for p in trajectory], dtype=np.float64)

        # Flip y for reverse
        if reverse:
            y_seq = -y_seq

        # DCT-II (ortho-normalized)
        x_coeffs = dct(x_seq, type=2, norm='ortho')[:4]
        y_coeffs = dct(y_seq, type=2, norm='ortho')[:4]

        # Quantize coefficients
        tokens = [gear]
        for i, coef in enumerate(x_coeffs):
            vmin, vmax, nbins = self._X_PARAMS[i]
            b = self._val_to_bin(coef, vmin, vmax, nbins)
            tokens.append(f"{self._X_PREFIXES[i]}{b}")

        for i, coef in enumerate(y_coeffs):
            vmin, vmax, nbins = self._Y_PARAMS[i]
            b = self._val_to_bin(coef, vmin, vmax, nbins)
            tokens.append(f"{self._Y_PREFIXES[i]}{b}")

        return " ".join(tokens)

    def decode(self, text: str) -> np.ndarray:
        """Decode 9 DCT tokens to 8-point trajectory."""
        content = self._extract_content(text)
        reverse, remaining = self._parse_gear(content)

        # Parse X coefficients
        x_coeffs = np.zeros(8)
        for i, prefix in enumerate(self._X_PREFIXES):
            match = re.search(rf"{prefix}(\d+)", remaining)
            if not match:
                raise ValueError(f"Missing token {prefix}*")
            vmin, vmax, nbins = self._X_PARAMS[i]
            x_coeffs[i] = self._bin_to_val(int(match.group(1)), vmin, vmax, nbins)

        # Parse Y coefficients
        y_coeffs = np.zeros(8)
        for i, prefix in enumerate(self._Y_PREFIXES):
            match = re.search(rf"{prefix}(\d+)", remaining)
            if not match:
                raise ValueError(f"Missing token {prefix}*")
            vmin, vmax, nbins = self._Y_PARAMS[i]
            y_coeffs[i] = self._bin_to_val(int(match.group(1)), vmin, vmax, nbins)

        # IDCT reconstruction
        x_seq = idct(x_coeffs, type=2, norm='ortho')
        y_seq = idct(y_coeffs, type=2, norm='ortho')

        # Flip y back for reverse
        if reverse:
            y_seq = -y_seq

        return np.column_stack([x_seq, y_seq])


class BezierCodec(_AdditiveVocabCodec):
    """Encode/decode trajectories using gear + Bezier control points.

    Token format: [Gear] [p1x] [p1y] [p2x] [p2y] [p3x] [p3y] = 7 tokens

    The trajectory is approximated by a cubic Bezier curve with:
        - P0 = (0, 0) fixed at origin
        - P1, P2, P3 = control points (encoded as tokens)

    Control point ranges (all 0.10m resolution):
        - p1_x: [-4, 4], 80 bins
        - p1_y: [-1, 25], 260 bins
        - p2_x: [-8, 8], 160 bins
        - p2_y: [-1, 42], 430 bins
        - p3_x: [-22, 22], 440 bins
        - p3_y: [-5, 65], 700 bins

    Vocab = 2 + 80 + 260 + 160 + 430 + 440 + 700 = 2072
    """

    # Control point bin configs: (vmin, vmax, n_bins)
    _P1_X = (-4.0, 4.0, 80)
    _P1_Y = (-1.0, 25.0, 260)
    _P2_X = (-8.0, 8.0, 160)
    _P2_Y = (-1.0, 42.0, 430)
    _P3_X = (-22.0, 22.0, 440)
    _P3_Y = (-5.0, 65.0, 700)

    @property
    def vocab_size(self) -> int:
        return 2 + 80 + 260 + 160 + 430 + 440 + 700  # 2072

    @staticmethod
    def _bernstein(n: int, k: int, t: float) -> float:
        """Bernstein basis polynomial B_{k,n}(t)."""
        from scipy.special import comb
        return float(comb(n, k, exact=True) * (t ** k) * ((1 - t) ** (n - k)))

    def _resample_to_8(self, trajectory: List[List[float]]) -> np.ndarray:
        """Resample trajectory to exactly 8 points.

        If len(trajectory) == 8, returns as-is.
        If len(trajectory) > 8, uniformly resamples to 8 points.
        """
        traj = np.array(trajectory, dtype=np.float64)
        n = len(traj)
        if n == 8:
            return traj
        if n < 8:
            raise ValueError(f"Trajectory must have at least 8 points, got {n}")

        # Uniform resampling to 8 points
        # Map [0, 1, ..., n-1] to indices [0, ..., 7] in target
        indices = np.linspace(0, n - 1, 8)
        resampled = np.zeros((8, 2), dtype=np.float64)
        for i, idx in enumerate(indices):
            idx_floor = int(np.floor(idx))
            idx_ceil = min(idx_floor + 1, n - 1)
            frac = idx - idx_floor
            resampled[i] = (1 - frac) * traj[idx_floor] + frac * traj[idx_ceil]
        return resampled

    def encode(self, trajectory: List[List[float]]) -> str:
        """Encode trajectory to gear + Bezier control point tokens.

        Args:
            trajectory: List of [x, y] pairs (8 or more waypoints, resampled to 8)

        Returns:
            Token string like 'D p1x_40 p1y_130 p2x_80 p2y_215 p3x_220 p3y_350'
        """
        if len(trajectory) < 2:
            raise ValueError("Trajectory must have at least 2 points for Bezier fit")

        # Detect reverse
        reverse = self._is_reverse(trajectory)
        gear = self.GEAR_REVERSE if reverse else self.GEAR_FORWARD

        # Build Bernstein matrix B (8 x 4), B[i,j] = bernstein(3, j, t_i)
        # Sample at t = 1/8, 2/8, ..., 8/8
        t_vals = [(i + 1) / 8.0 for i in range(8)]
        B = np.array([[self._bernstein(3, j, t) for j in range(4)] for t in t_vals])

        # Waypoints as array (8, 2) - resample if necessary
        waypoints = self._resample_to_8(trajectory)
        if reverse:
            waypoints = waypoints.copy()
            waypoints[:, 1] = -waypoints[:, 1]  # flip y for reverse

        # P0 = (0, 0) fixed, solve for P1, P2, P3 via least squares
        # waypoints = B @ [P0, P1, P2, P3]^T
        # waypoints - B[:,0:1] @ P0 = B[:,1:] @ [P1, P2, P3]^T
        from numpy.linalg import lstsq
        P0 = np.array([0.0, 0.0])
        rhs = waypoints - B[:, 0:1] @ P0.reshape(1, 2)  # (8, 2)
        P_control, _, _, _ = lstsq(B[:, 1:], rhs, rcond=None)  # (3, 2)

        P1, P2, P3 = P_control[0], P_control[1], P_control[2]

        # Quantize control points
        p1x_b = self._val_to_bin(P1[0], *self._P1_X)
        p1y_b = self._val_to_bin(P1[1], *self._P1_Y)
        p2x_b = self._val_to_bin(P2[0], *self._P2_X)
        p2y_b = self._val_to_bin(P2[1], *self._P2_Y)
        p3x_b = self._val_to_bin(P3[0], *self._P3_X)
        p3y_b = self._val_to_bin(P3[1], *self._P3_Y)

        tokens = [gear,
                  f"p1x_{p1x_b}", f"p1y_{p1y_b}",
                  f"p2x_{p2x_b}", f"p2y_{p2y_b}",
                  f"p3x_{p3x_b}", f"p3y_{p3y_b}"]
        return " ".join(tokens)

    def decode(self, text: str) -> np.ndarray:
        """Decode gear + Bezier tokens to trajectory (8 waypoints).

        Args:
            text: String with gear and control point tokens

        Returns:
            np.ndarray of shape (8, 2)
        """
        content = self._extract_content(text)
        reverse, content = self._parse_gear(content)

        # Parse control point tokens
        def parse(prefix: str, cfg: Tuple[float, float, int]) -> float:
            match = re.search(rf"{prefix}_(\d+)", content)
            if not match:
                raise ValueError(f"No {prefix}_N token found")
            return self._bin_to_val(int(match.group(1)), *cfg)

        P1 = np.array([parse("p1x", self._P1_X), parse("p1y", self._P1_Y)])
        P2 = np.array([parse("p2x", self._P2_X), parse("p2y", self._P2_Y)])
        P3 = np.array([parse("p3x", self._P3_X), parse("p3y", self._P3_Y)])
        P0 = np.array([0.0, 0.0])

        # Sample Bezier curve at t = 1/8, 2/8, ..., 8/8
        control = np.array([P0, P1, P2, P3])  # (4, 2)
        waypoints = []
        for i in range(8):
            t = (i + 1) / 8.0
            point = sum(self._bernstein(3, j, t) * control[j] for j in range(4))
            waypoints.append(point)
        waypoints = np.array(waypoints)

        if reverse:
            waypoints[:, 1] = -waypoints[:, 1]

        return waypoints


class BiDeltaCodec(_AdditiveVocabCodec):
    """Bidirectional delta codec: endpoint + forward chain + backward chain.

    Token format: [Gear] [ex] [ey] [fd1] [fd2] [fd3] [fd4] [bd1] [bd2] = 9 tokens

    Encoding strategy:
        - Endpoint: absolute (x, y) quantized
        - Forward chain (4 deltas): origin → p1 → p2 → p3 → p4 (polar deltas)
        - Backward chain (2 deltas): endpoint → p7 → p6 (polar deltas)
        - p5: interpolated as (p4 + p6) / 2

    Vocab: 2 (gear) + 440 (bx) + 700 (by) + 28*33 (fd) + 28*33 (bd) = 2990
    """

    # Endpoint ranges
    END_X_RANGE = (-22.0, 22.0)
    END_Y_RANGE = (-5.0, 65.0)
    END_X_BINS = 440
    END_Y_BINS = 700

    # Polar delta ranges (shared by fd and bd)
    R_MAX = 7.0
    R_BINS = 28
    THETA_MAX_DEG = 60.0
    THETA_BINS = 33

    def __init__(self):
        self._theta_max = np.radians(self.THETA_MAX_DEG)

    @property
    def vocab_size(self) -> int:
        return (2 + self.END_X_BINS + self.END_Y_BINS +
                self.R_BINS * self.THETA_BINS * 2)

    # -- Binning --

    def _end_x_to_bin(self, x: float) -> int:
        return self._val_to_bin(x, *self.END_X_RANGE, self.END_X_BINS)

    def _bin_to_end_x(self, b: int) -> float:
        return self._bin_to_val(b, *self.END_X_RANGE, self.END_X_BINS)

    def _end_y_to_bin(self, y: float) -> int:
        return self._val_to_bin(y, *self.END_Y_RANGE, self.END_Y_BINS)

    def _bin_to_end_y(self, b: int) -> float:
        return self._bin_to_val(b, *self.END_Y_RANGE, self.END_Y_BINS)

    def _r_to_bin(self, r: float) -> int:
        return self._val_to_bin(r, 0.0, self.R_MAX, self.R_BINS)

    def _bin_to_r(self, b: int) -> float:
        return self._bin_to_val(b, 0.0, self.R_MAX, self.R_BINS)

    def _theta_to_bin(self, theta: float) -> int:
        return self._val_to_bin(theta, -self._theta_max, self._theta_max, self.THETA_BINS)

    def _bin_to_theta(self, b: int) -> float:
        return self._bin_to_val(b, -self._theta_max, self._theta_max, self.THETA_BINS)

    # -- Polar conversion (same as PolarDeltaCodec) --

    @staticmethod
    def _cartesian_to_polar(dx: float, dy: float) -> Tuple[float, float]:
        r = np.sqrt(dx * dx + dy * dy)
        theta = np.arctan2(dx, dy)  # 0=forward, positive=left
        return r, theta

    @staticmethod
    def _polar_to_cartesian(r: float, theta: float) -> Tuple[float, float]:
        dx = r * np.sin(theta)
        dy = r * np.cos(theta)
        return dx, dy

    # -- Encode --

    def encode(self, trajectory: List[List[float]]) -> str:
        """Encode 8-point trajectory to 9 tokens."""
        if len(trajectory) != 8:
            raise ValueError(f"BiDeltaCodec requires exactly 8 points, got {len(trajectory)}")

        reverse = self._is_reverse(trajectory)
        gear = self.GEAR_REVERSE if reverse else self.GEAR_FORWARD
        traj = np.array(trajectory)

        # Endpoint
        end_x, end_y = traj[-1]
        if reverse:
            end_y = -end_y
        ex_tok = f"bx_{self._end_x_to_bin(end_x)}"
        ey_tok = f"by_{self._end_y_to_bin(end_y)}"

        # Forward chain: origin → p1 → p2 → p3 → p4
        fd_tokens = []
        prev = np.array([0.0, 0.0])
        for i in range(4):
            curr = traj[i].copy()
            if reverse:
                curr[1] = -curr[1]
            dx, dy = curr - prev
            r, theta = self._cartesian_to_polar(dx, dy)
            rb = self._r_to_bin(r)
            tb = 0 if rb == 0 else self._theta_to_bin(theta)
            fd_tokens.append(f"fd_{rb}_{tb}")
            prev = curr

        # Backward chain: endpoint → p7 → p6
        bd_tokens = []
        prev = np.array([end_x, end_y])
        for i in [6, 5]:  # p7, p6
            curr = traj[i].copy()
            if reverse:
                curr[1] = -curr[1]
            dx, dy = prev - curr  # note: from endpoint backward
            r, theta = self._cartesian_to_polar(dx, dy)
            rb = self._r_to_bin(r)
            tb = 0 if rb == 0 else self._theta_to_bin(theta)
            bd_tokens.append(f"bd_{rb}_{tb}")
            prev = curr

        return " ".join([gear, ex_tok, ey_tok] + fd_tokens + bd_tokens)

    # -- Decode --

    def decode(self, text: str) -> np.ndarray:
        """Decode 9 tokens to 8-point trajectory."""
        content = self._extract_content(text).strip()

        # Parse gear
        reverse, rest = self._parse_gear(content)

        # Parse endpoint
        ex_match = re.match(r'bx_(\d+)\s+by_(\d+)', rest)
        if not ex_match:
            raise ValueError("No bx_/by_ endpoint tokens found")
        end_x = self._bin_to_end_x(int(ex_match.group(1)))
        end_y = self._bin_to_end_y(int(ex_match.group(2)))
        if reverse:
            end_y = -end_y

        rest = rest[ex_match.end():].strip()

        # Parse forward deltas (4)
        fd_matches = re.findall(r"fd_(\d+)_(\d+)", rest[:rest.find("bd_")] if "bd_" in rest else rest)
        if len(fd_matches) != 4:
            raise ValueError(f"Expected 4 fd tokens, got {len(fd_matches)}")

        # Parse backward deltas (2)
        bd_matches = re.findall(r"bd_(\d+)_(\d+)", rest)
        if len(bd_matches) != 2:
            raise ValueError(f"Expected 2 bd tokens, got {len(bd_matches)}")

        # Reconstruct trajectory
        coords = np.zeros((8, 2))

        # Forward chain: origin → p1 → p2 → p3 → p4
        x, y = 0.0, 0.0
        for i, (rb_str, tb_str) in enumerate(fd_matches):
            r = self._bin_to_r(int(rb_str))
            theta = self._bin_to_theta(int(tb_str))
            dx, dy = self._polar_to_cartesian(r, theta)
            if reverse:
                dy = -dy
            x += dx
            y += dy
            coords[i] = [x, y]

        # Backward chain: endpoint → p7 → p6
        # bd[0]: endpoint → traj[6], store in coords[6]
        # bd[1]: traj[6] → traj[5], store in coords[5]
        bx, by = end_x, end_y
        for i, (rb_str, tb_str) in enumerate(bd_matches):
            r = self._bin_to_r(int(rb_str))
            theta = self._bin_to_theta(int(tb_str))
            dx, dy = self._polar_to_cartesian(r, theta)
            if reverse:
                dy = -dy
            bx -= dx
            by -= dy
            coords[6 - i] = [bx, by]

        # Set endpoint (coords[7] = decoded endpoint)
        coords[7] = [end_x, end_y]

        # Interpolate p4 = (p3 + p5) / 2
        coords[4] = (coords[3] + coords[5]) / 2

        return coords


class AnchorLateralCodec(_AdditiveVocabCodec):
    """Encode/decode trajectories using gear + endpoint anchor + lateral offsets.

    Output format: `[Gear] [ex] [ey] [d1] [d2] [d3] [d4] [d5] [d6]` (9 tokens)
        - Gear: D/R (forward/reverse)
        - ex_XXX: endpoint x, [-22,22], 440 bins, 0.10m precision
        - ey_YYY: endpoint y, [-5,65], 700 bins, 0.10m precision
        - d1_..d6_: lateral offsets from baseline, [-6,6], 120 bins each

    Total vocab: 2 + 440 + 700 + 120*6 = 1862

    Principle:
        - Endpoint anchors trajectory direction
        - Baseline: line from origin to endpoint
        - d_i = signed perpendicular distance from waypoint to baseline
        - Stopped (||endpoint|| < 0.5m): d_i encodes waypoint x directly
    """

    EX_RANGE = (-22.0, 22.0)
    EX_BINS = 440
    EY_RANGE = (-5.0, 65.0)
    EY_BINS = 700
    D_RANGE = (-6.0, 6.0)
    D_BINS = 120
    STOPPED_THRESH = 0.5

    @property
    def vocab_size(self) -> int:
        return 2 + self.EX_BINS + self.EY_BINS + self.D_BINS * 6

    def encode(self, trajectory: List[List[float]]) -> str:
        if len(trajectory) != 8:
            raise ValueError(f"Expected 8 waypoints, got {len(trajectory)}")

        reverse = self._is_reverse(trajectory)
        gear = self.GEAR_REVERSE if reverse else self.GEAR_FORWARD

        end_x, end_y = trajectory[-1]
        exb = self._val_to_bin(end_x, *self.EX_RANGE, self.EX_BINS)
        eyb = self._val_to_bin(end_y, *self.EY_RANGE, self.EY_BINS)
        tokens = [gear, f"ex_{exb}", f"ey_{eyb}"]

        end_norm = np.sqrt(end_x ** 2 + end_y ** 2)
        stopped = end_norm < self.STOPPED_THRESH

        if stopped:
            for i in range(6):
                x = trajectory[i][0]
                db = self._val_to_bin(x, *self.D_RANGE, self.D_BINS)
                tokens.append(f"d{i+1}_{db}")
        else:
            ux, uy = end_x / end_norm, end_y / end_norm
            perp_x, perp_y = -uy, ux
            for i in range(6):
                wx, wy = trajectory[i]
                t = (wx * ux + wy * uy) / end_norm
                bl_x, bl_y = t * ux * end_norm, t * uy * end_norm
                lat = (wx - bl_x) * perp_x + (wy - bl_y) * perp_y
                db = self._val_to_bin(lat, *self.D_RANGE, self.D_BINS)
                tokens.append(f"d{i+1}_{db}")

        return " ".join(tokens)

    def decode(self, text: str) -> np.ndarray:
        content = self._extract_content(text)
        reverse, rest = self._parse_gear(content)

        ex_match = re.search(r"ex_(\d+)", rest)
        ey_match = re.search(r"ey_(\d+)", rest)
        if not ex_match or not ey_match:
            raise ValueError("Missing ex_ or ey_ tokens")
        end_x = self._bin_to_val(int(ex_match.group(1)), *self.EX_RANGE, self.EX_BINS)
        end_y = self._bin_to_val(int(ey_match.group(1)), *self.EY_RANGE, self.EY_BINS)

        d_vals = []
        for i in range(1, 7):
            m = re.search(rf"d{i}_(\d+)", rest)
            if not m:
                raise ValueError(f"Missing d{i}_ token")
            d_vals.append(self._bin_to_val(int(m.group(1)), *self.D_RANGE, self.D_BINS))

        coords = []
        end_norm = np.sqrt(end_x ** 2 + end_y ** 2)
        stopped = end_norm < self.STOPPED_THRESH

        if stopped:
            for i, x in enumerate(d_vals):
                y = (i + 1) / 7 * end_y
                coords.append([x, y])
            # Interpolate p7: average of p6 and endpoint
            p6 = coords[-1]
            p7 = [(p6[0] + end_x) / 2, (p6[1] + end_y) / 2]
            coords.append(p7)
        else:
            ux, uy = end_x / end_norm, end_y / end_norm
            perp_x, perp_y = -uy, ux
            for i, lat in enumerate(d_vals):
                t = (i + 1) / 8  # t = 1/8, 2/8, ..., 6/8 for trajectory[0..5]
                bl_x, bl_y = t * end_x, t * end_y
                wx = bl_x + lat * perp_x
                wy = bl_y + lat * perp_y
                coords.append([wx, wy])
            # Interpolate p7 (trajectory[6]): t = 7/8
            t7 = 7 / 8
            p7 = [t7 * end_x, t7 * end_y]
            coords.append(p7)

        coords.append([end_x, end_y])
        return np.array(coords)
