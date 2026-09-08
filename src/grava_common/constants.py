"""
Shared constants for NOUS training.

The default stage mapping follows the current bbox-grounding training chain.
The older pre-bbox CoHC QA mapping is still exposed under LEGACY_* names for
archival scripts only.
"""

# Current bbox-grounding Stage -> QA types mapping.
#
# Stage12 data stores Stage 1 and Stage 2 records in one file:
#   $AFS/train_data/bbox_grounding/front_only_v1/stage12.jsonl
# Stage 3 uses hybrid_v6 trajectory planner records:
#   stage3_official_kimi_selfrollout_full_hybrid_v6_e21_empty5310_filled_sft.jsonl
STAGE_MAPPING = {
    1: ['Scene', 'ObjectPerception'],
    2: ['ObjectInteraction'],
    3: ['Trajectory'],
}

# Reverse mapping: QA type -> stage number for the active bbox path.
QA_TYPE_TO_STAGE = {qa: stage for stage, qas in STAGE_MAPPING.items() for qa in qas}

# Stage training configs (name, data_file, description, learning_rate) for the
# active bbox path.
STAGE_CONFIGS = {
    1: {
        'name': 'bbox_stage1_grounding',
        'data_file': 'stage12',
        'description': 'Scene + ObjectPerception bbox grounding',
        'learning_rate': 2e-5,
        'num_train_epochs': 3,
    },
    2: {
        'name': 'bbox_stage2_interaction',
        'data_file': 'stage12',
        'description': 'ObjectInteraction bbox grounding',
        'learning_rate': 1e-5,
        'num_train_epochs': 3,
    },
    3: {
        'name': 'bbox_stage3_hybrid_v6_planning',
        'data_file': (
            'stage3_official_kimi_selfrollout_full_hybrid_v6_e21_empty5310_filled_sft'
        ),
        'description': 'Trajectory planning with hybrid_v6 planner JSON',
        'learning_rate': 5e-6,
        'num_train_epochs': 3,
    },
}

# Historical pre-bbox CoHC QA stage mapping. Do not use this for current
# bbox-grounding data generation or training.
LEGACY_COHC_STAGE_MAPPING = {
    1: ['Scene', 'ObjectPerception'],
    2: ['ObjectInteraction'],
    3: ['Decision', 'DecisionChain', 'ObjectDecision', 'EgoDecision', 'EgoChain',
        'ObjectDecisionChain', 'EgoDecisionChain'],
    4: ['Trajectory', 'TrajectoryChain'],
}

LEGACY_COHC_QA_TYPE_TO_STAGE = {
    qa: stage for stage, qas in LEGACY_COHC_STAGE_MAPPING.items() for qa in qas
}

LEGACY_COHC_STAGE_CONFIGS = {
    1: {
        'name': 'L1_Perception',
        'data_file': 'nous_stage1',
        'description': 'Scene + ObjectPerception (legacy L1 - Perception Layer)',
        'learning_rate': 2e-5,
        'num_train_epochs': 3,
    },
    2: {
        'name': 'L2_Interaction',
        'data_file': 'nous_stage2',
        'description': 'ObjectInteraction (legacy L2 - Interaction Understanding)',
        'learning_rate': 1e-5,
        'num_train_epochs': 3,
    },
    3: {
        'name': 'L3_Decision',
        'data_file': 'nous_stage3',
        'description': 'Decision (legacy L3 - Decision Making)',
        'learning_rate': 1e-5,
        'num_train_epochs': 1,
    },
    4: {
        'name': 'L4_Planning',
        'data_file': 'nous_stage4',
        'description': 'Trajectory + TrajectoryChain (legacy L4 - Planning)',
        'learning_rate': 5e-6,
        'num_train_epochs': 3,
    },
}

# Camera view name -> camera key mapping (used in dataset index)
CAMERA_KEY_MAP = {
    'front_center': 'cam_front_center',
    'front_left': 'cam_front_left',
    'front_right': 'cam_front_right',
    'back_left': 'cam_back_left',
    'back_right': 'cam_back_right',
    'side_left': 'cam_side_left',
    'side_right': 'cam_side_right',
    'back': 'cam_back',
}

# Camera view name -> overlay filename mapping
CAMERA_OVERLAY_MAP = {
    view: f'{cam_key}_overlay.png'
    for view, cam_key in CAMERA_KEY_MAP.items()
}

# Decision labels (7 classes)
DECISION_LABELS = [
    "Follow", "Stop", "Yield", "Nudge Left", "Nudge Right",
    "Overtake", "Caution",
]

# Fuzzy matching map: lowercased variant -> canonical label
DECISION_ALIASES = {
    "follow": "Follow",
    "stop": "Stop",
    "yield": "Yield",
    "nudge left": "Nudge Left",
    "nudgeleft": "Nudge Left",
    "nudge_left": "Nudge Left",
    "nudge right": "Nudge Right",
    "nudgeright": "Nudge Right",
    "nudge_right": "Nudge Right",
    "overtake": "Overtake",
    "pass": "Overtake",
    "caution": "Caution",
    "monitor": "Caution",
}

# Ego decision labels
EGO_LATERAL_LABELS = [
    "keep_lane", "lane_change_left", "lane_change_right",
    "turn_left", "turn_right", "u_turn",
]
EGO_LONGITUDINAL_LABELS = [
    "accelerate", "cruise", "decelerate", "decelerate_to_crawl", "stop",
]

# View configs
VIEW_CONFIGS = {
    "front_only": ["front_center"],
    "front_with_bev": ["front_center", "bev"],
}

# Trajectory parameters in EgoFrame.NOUS:
# - x: lateral (positive = right)
# - y: longitudinal (positive = forward)
TRAJECTORY_POINTS_NAVSIM = 8
TRAJECTORY_POINTS_INTERNAL = 16
TRAJECTORY_X_RANGE = (-12.0, 12.0)
TRAJECTORY_Y_RANGE = (-2.0, 55.0)
TRAJECTORY_N_BINS = 512
