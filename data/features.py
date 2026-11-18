"""Turn an episode into one row of numbers that a model can learn from."""

from data.episode import Episode, FloatArray

JOINT_NAMES = [
    "shoulder_pitch",
    "shoulder_roll",
    "shoulder_yaw",
    "elbow",
    "wrist_roll",
    "wrist_pitch",
    "wrist_yaw",
    "thumb_metacarpal",
    "thumb_proximal",
    "index",
    "middle",
    "ring",
    "pinky",
]
STATISTICS = ["mean", "std", "min", "max", "last"]
FEATURE_NAMES = ["duration"] + [f"{stat}_{joint}" for stat in STATISTICS for joint in JOINT_NAMES]


def episode_features(episode: Episode) -> FloatArray:
    """Summarise the arm that does the task, one value per entry in FEATURE_NAMES.

    Only the active arm is used, so a left-arm and a right-arm episode fill the same columns.
    """
    raise NotImplementedError
