"""One recorded episode and how to read it from disk.

An episode is two files with the same name:

    episode_000.npz
        observation.state   (frames, 31)   where each joint is
        action              (frames, 31)   where each joint was told to go
        timestamp           (frames,)      seconds since the start of the episode
    episode_000.json
        episode.arm         "left" or "right", the arm that does the task
        outcome.success     true or false, this is the label

The 31 columns of state and action are the same in every episode:

    0-6    left arm        14-19  left hand      26-27  head
    7-13   right arm       20-25  right hand     28-30  base velocity

The .npz also has privileged.object_pose and info.phase. They only exist in the simulator,
so they are not loaded.
"""

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float64]


@dataclass(frozen=True)
class Episode:
    """What a real robot would also know about an episode, plus the label."""

    name: str
    state: FloatArray
    action: FloatArray
    timestamp: FloatArray
    arm: str
    success: bool


def load_episode(npz_path: Path) -> Episode:
    """Read one episode from its .npz file and the .json file next to it."""
    arrays = np.load(npz_path)
    meta = json.loads(npz_path.with_suffix(".json").read_text())
    return Episode(
        name=npz_path.stem,
        state=arrays["observation.state"].astype(np.float64),
        action=arrays["action"].astype(np.float64),
        timestamp=arrays["timestamp"].astype(np.float64),
        arm=meta["episode"]["arm"],
        success=meta["outcome"]["success"],
    )
