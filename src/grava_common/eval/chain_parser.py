"""
Chain parser and reward functions for CoHC GRPO training.

This module implements:
1. ChainParser: Parse chain syntax ("->" causal / "|" parallel / "entity.attribute" nodes)
2. ChainReward: Compute chain validity rewards for RL training

Usage:
    from src.eval.chain_reward import ChainParser, ChainReward

    # Parse chain
    parsed = ChainParser.parse("scene_context.navigation -> H.map_interaction -> decision")

    # Compute RL reward (validity check)
    reward = ChainReward.compute_rl_reward(
        pred_chain="scene_context.navigation -> H.map_interaction -> decision",
        scene_obstacles=["H", "A"],
        terminal="decision"
    )
"""

import re
from typing import List, Dict, Optional, Set, Tuple
from dataclasses import dataclass, field
from enum import Enum


class NodeType(Enum):
    """Types of nodes in CoHC chain."""
    SCENE_CONTEXT = "scene_context"
    OBJECT = "object"  # e.g., H.map_interaction
    EGO = "ego"  # ego-related nodes
    EGO_DECISION = "ego_decision"  # ego-level decision synthesis
    TRAJECTORY_CONTEXT = "trajectory_context"
    DECISION = "decision"
    TRAJECTORY = "trajectory"
    PLANNING = "planning_trajectory"
    UNKNOWN = "unknown"


@dataclass
class ChainNode:
    """A single node in the chain."""
    entity: str  # e.g., "scene_context", "H", "ego"
    attribute: str  # e.g., "navigation", "map_interaction"
    full_name: str  # e.g., "scene_context.navigation"
    node_type: NodeType = NodeType.UNKNOWN

    @classmethod
    def from_string(cls, node_str: str) -> "ChainNode":
        """Parse node from string like 'scene_context.navigation' or 'H'."""
        node_str = node_str.strip()
        if "." in node_str:
            parts = node_str.split(".", 1)
            entity = parts[0]
            attribute = parts[1]
        else:
            entity = node_str
            attribute = ""

        # Determine node type
        node_type = NodeType.UNKNOWN
        if entity == "scene_context":
            node_type = NodeType.SCENE_CONTEXT
        elif entity == "trajectory_context":
            node_type = NodeType.TRAJECTORY_CONTEXT
        elif entity == "ego":
            node_type = NodeType.EGO
        elif entity == "ego_decision":
            node_type = NodeType.EGO_DECISION
        elif entity == "decision":
            node_type = NodeType.DECISION
        elif entity == "trajectory":
            node_type = NodeType.TRAJECTORY
        elif entity == "planning_trajectory":
            node_type = NodeType.PLANNING
        elif len(entity) == 1 and entity.isalpha() and entity.isupper():
            # Single uppercase letter = obstacle
            node_type = NodeType.OBJECT

        return cls(
            entity=entity,
            attribute=attribute,
            full_name=node_str,
            node_type=node_type
        )


@dataclass
class ParsedChain:
    """Parsed chain structure."""
    raw_chain: str
    nodes: List[ChainNode] = field(default_factory=list)
    branches: List[List[ChainNode]] = field(default_factory=list)  # For parallel chains
    is_valid: bool = True
    errors: List[str] = field(default_factory=list)

    @property
    def terminal(self) -> Optional[str]:
        """Get terminal node name."""
        if not self.nodes:
            return None
        return self.nodes[-1].full_name

    @property
    def obstacle_refs(self) -> Set[str]:
        """Get all obstacle references (uppercase single letters)."""
        obstacles = set()
        for node in self.nodes:
            if node.node_type == NodeType.OBJECT:
                obstacles.add(node.entity)
        return obstacles

    @property
    def all_node_names(self) -> Set[str]:
        """Get all node full names."""
        return {node.full_name for node in self.nodes}


class ChainParser:
    """
    Parse CoHC chain syntax.

    Syntax:
    - "->" : Causal/sequential dependency
    - "|"  : Parallel/branches
    - "entity.attribute" : Node reference

    Example:
        "scene_context.navigation -> H.map_interaction | H.intention -> H.ego_interaction -> decision"
    """

    @classmethod
    def parse(cls, chain_str: str) -> ParsedChain:
        """
        Parse chain string into structured format.

        Args:
            chain_str: Chain string like "A -> B | C -> D"

        Returns:
            ParsedChain object
        """
        if not chain_str or not isinstance(chain_str, str):
            return ParsedChain(
                raw_chain=chain_str or "",
                is_valid=False,
                errors=["Empty or invalid chain string"]
            )

        chain_str = chain_str.strip()
        result = ParsedChain(raw_chain=chain_str)

        try:
            # Split by "|" for parallel branches (simplified: treat as sequential for now)
            # Full implementation would handle true parallel branches
            if "|" in chain_str:
                branches_str = [b.strip() for b in chain_str.split("|")]
                for branch_str in branches_str:
                    branch_nodes = cls._parse_sequence(branch_str)
                    result.branches.append(branch_nodes)
                    result.nodes.extend(branch_nodes)
            else:
                result.nodes = cls._parse_sequence(chain_str)
                result.branches = [result.nodes]

        except Exception as e:
            result.is_valid = False
            result.errors.append(f"Parse error: {str(e)}")

        return result

    @classmethod
    def _parse_sequence(cls, seq_str: str) -> List[ChainNode]:
        """Parse a sequential chain (no | operators)."""
        nodes = []
        # Split by "->"
        parts = [p.strip() for p in seq_str.split("->")]

        for part in parts:
            if part:
                node = ChainNode.from_string(part)
                nodes.append(node)

        return nodes


class ChainReward:
    """
    Compute chain-based rewards for RL training.

    For RL stage: Only checks validity (not matching GT chain)
    For eval stage: Can compare with GT chain
    """

    # Valid terminal nodes for L3 and L4
    VALID_L3_TERMINALS = {"decision", "action", "ego_decision"}
    VALID_L4_TERMINALS = {"trajectory", "planning_trajectory", "waypoints"}

    # Valid node patterns (for basic validation)
    VALID_ENTITIES = {
        "scene_context", "object_perception", "map_interaction",
        "motion_state", "ego_interaction", "intention",
        "decision", "action", "trajectory_context",
        "trajectory", "planning_trajectory", "navigation",
        "traffic_sign", "congestion"
    }

    @classmethod
    def compute_rl_reward(
        cls,
        pred_chain: str,
        scene_obstacles: List[str],
        terminal_type: str = "l3"  # "l3" or "l4"
    ) -> Dict[str, float]:
        """
        Compute RL reward for chain validity (not matching GT).

        Args:
            pred_chain: Predicted chain string
            scene_obstacles: List of valid obstacle IDs in scene ["H", "A", ...]
            terminal_type: "l3" for decision, "l4" for trajectory

        Returns:
            Dict with reward components:
            - format_valid: 1.0 if syntax valid
            - node_valid: fraction of valid nodes
            - terminal_ok: 1.0 if terminal correct
            - chain_reward: weighted sum
        """
        parsed = ChainParser.parse(pred_chain)

        # 1. Format validity
        format_valid = 1.0 if parsed.is_valid and len(parsed.nodes) > 0 else 0.0

        # 2. Node validity (check obstacles exist in scene)
        node_valid = cls._check_node_validity(parsed, set(scene_obstacles))

        # 3. Terminal check
        terminal_ok = cls._check_terminal(parsed, terminal_type)

        # Weighted chain reward
        chain_reward = (
            0.4 * format_valid +
            0.4 * node_valid +
            0.2 * terminal_ok
        )

        return {
            "format_valid": format_valid,
            "node_valid": node_valid,
            "terminal_ok": terminal_ok,
            "chain_reward": chain_reward
        }

    # Node types that are always valid (no external reference needed)
    _ALWAYS_VALID_TYPES = {
        NodeType.SCENE_CONTEXT, NodeType.TRAJECTORY_CONTEXT, NodeType.EGO,
        NodeType.EGO_DECISION, NodeType.DECISION, NodeType.TRAJECTORY,
        NodeType.PLANNING,
    }

    @classmethod
    def _check_node_validity(cls, parsed: ParsedChain, valid_obstacles: Set[str]) -> float:
        """Check if referenced nodes are valid."""
        if not parsed.nodes:
            return 0.0

        valid_count = 0
        for node in parsed.nodes:
            if node.node_type in cls._ALWAYS_VALID_TYPES:
                is_valid = True
            elif node.node_type == NodeType.OBJECT:
                # Obstacle must exist in scene
                is_valid = node.entity in valid_obstacles
            else:
                is_valid = False

            if is_valid:
                valid_count += 1

        return valid_count / len(parsed.nodes)

    @classmethod
    def _check_terminal(cls, parsed: ParsedChain, terminal_type: str) -> float:
        """Check if terminal node is valid."""
        terminal = parsed.terminal
        if not terminal:
            return 0.0

        if terminal_type == "l3":
            # Terminal should be decision or action
            return 1.0 if any(t in terminal.lower() for t in cls.VALID_L3_TERMINALS) else 0.0
        elif terminal_type == "l4":
            # Terminal should be trajectory or planning_trajectory
            return 1.0 if any(t in terminal.lower() for t in cls.VALID_L4_TERMINALS) else 0.0

        return 0.0

    @classmethod
    def extract_chain_from_output(cls, text: str) -> Optional[str]:
        """
        Extract chain from model output text.

        Looks for:
        - <chain>...</chain> tags
        - Chain: ... pattern

        Returns:
            Chain string or None
        """
        # Try <chain> tags
        match = re.search(r"<chain>(.*?)</chain>", text, re.DOTALL | re.IGNORECASE)
        if match:
            return match.group(1).strip()

        # Try "Chain: ..." pattern
        match = re.search(r"Chain:\s*(.+?)(?:\n|$)", text, re.IGNORECASE)
        if match:
            return match.group(1).strip()

        return None
