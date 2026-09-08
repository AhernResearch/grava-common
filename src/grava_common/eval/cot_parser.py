"""
COT (Chain-of-Thought) response parser for CoHC model outputs.

Parses <think>/<chain>/<answer> formatted responses and extracts
evaluation targets (trajectory, decision, chain) from the appropriate section.

Usage:
    from src.eval.cot_parser import COTParser

    result = COTParser.parse(model_output)
    trajectory = COTParser.extract_trajectory(model_output)
    decision = COTParser.extract_decision(model_output)
"""

from __future__ import annotations

import re
from typing import Optional
from dataclasses import dataclass

import numpy as np

from grava_common.trajectory.metrics import extract_trajectory_from_text
from grava_common.eval.chain_parser import ChainReward
from grava_common.constants import DECISION_LABELS, DECISION_ALIASES as _DECISION_ALIASES, EGO_LATERAL_LABELS, EGO_LONGITUDINAL_LABELS


@dataclass
class COTResult:
    """Parsed COT response."""
    think: Optional[str] = None
    chain: Optional[str] = None
    answer: Optional[str] = None
    raw: str = ""



class COTParser:
    """Parse <think>/<chain>/<answer> formatted model outputs."""

    _TAG_RE = {
        "think": re.compile(r"<think>(.*?)</think>", re.DOTALL | re.IGNORECASE),
        "chain": re.compile(r"<chain>(.*?)</chain>", re.DOTALL | re.IGNORECASE),
        "answer": re.compile(r"<answer>(.*?)</answer>", re.DOTALL | re.IGNORECASE),
    }
    _DECISION_RE = re.compile(r"decided:\s*(.+?)\.?\s*$", re.IGNORECASE | re.MULTILINE)
    _EGO_LATERAL_RE = re.compile(r"lateral\s*:\s*(\w+)", re.IGNORECASE)
    _EGO_LONGITUDINAL_RE = re.compile(r"longitudinal\s*:\s*(\w+(?:_\w+)*)", re.IGNORECASE)

    @staticmethod
    def parse(text: str) -> COTResult:
        """Parse text into COTResult with think/chain/answer sections."""
        result = COTResult(raw=text)
        for field, pattern in COTParser._TAG_RE.items():
            m = pattern.search(text)
            if m:
                setattr(result, field, m.group(1).strip())
        return result

    @staticmethod
    def extract_trajectory(text: str) -> Optional[np.ndarray]:
        """
        COT-aware trajectory extraction.
        Prioritizes <answer> section to avoid matching coordinates in <think>.
        """
        cot = COTParser.parse(text)
        if cot.answer:
            traj = extract_trajectory_from_text(cot.answer)
            if traj is not None:
                return traj
        # Fallback: full text (for non-COT outputs)
        return extract_trajectory_from_text(text)

    @staticmethod
    def extract_chain(text: str) -> Optional[str]:
        """Extract <chain> content (delegates to ChainReward)."""
        return ChainReward.extract_chain_from_output(text)

    @staticmethod
    def extract_decision(text: str) -> Optional[str]:
        """
        Extract and normalize decision label.
        1. COT: from <answer> section
        2. Plain: regex 'decided: {label}.'
        3. Normalize to canonical label set
        """
        cot = COTParser.parse(text)
        source = cot.answer if cot.answer else text

        # Try 'decided: ...' pattern
        m = COTParser._DECISION_RE.search(source)
        if m:
            return _normalize_decision(m.group(1).strip())

        # Try direct label match in source text
        return _match_decision_in_text(source)

    @staticmethod
    def has_cot_tags(text: str) -> bool:
        """Check if text contains any COT tags."""
        return any(p.search(text) for p in COTParser._TAG_RE.values())

    @staticmethod
    def extract_ego_decision(text: str) -> Optional[EgoDecisionResult]:
        """
        Extract ego-level decision (lateral + longitudinal) from <answer>.
        Expected format: 'lateral: keep_lane | longitudinal: decelerate'
        """
        cot = COTParser.parse(text)
        source = cot.answer if cot.answer else text

        lat_m = COTParser._EGO_LATERAL_RE.search(source)
        lon_m = COTParser._EGO_LONGITUDINAL_RE.search(source)

        lateral = lat_m.group(1).lower() if lat_m else None
        longitudinal = lon_m.group(1).lower() if lon_m else None

        # Validate against known labels
        if lateral and lateral not in EGO_LATERAL_LABELS:
            lateral = None
        if longitudinal and longitudinal not in EGO_LONGITUDINAL_LABELS:
            longitudinal = None

        if lateral is None and longitudinal is None:
            return None
        return EgoDecisionResult(lateral=lateral, longitudinal=longitudinal)


def _normalize_decision(raw: str) -> Optional[str]:
    """Normalize raw decision text to canonical label."""
    key = raw.lower().strip().rstrip(".")
    if key in _DECISION_ALIASES:
        return _DECISION_ALIASES[key]
    # Fuzzy: check if any canonical label is a substring
    for alias, label in _DECISION_ALIASES.items():
        if alias in key:
            return label
    return None


def _match_decision_in_text(text: str) -> Optional[str]:
    """Try to find a decision label directly in text."""
    text_lower = text.lower()
    # Check from most specific (longest) to least specific
    for label in ["nudge left", "nudge right", "overtake", "caution", "follow", "stop", "yield"]:
        if label in text_lower:
            return _DECISION_ALIASES[label]
    return None

from grava_common.types import EgoDecisionResult
