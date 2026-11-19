"""One recorded episode and how to read it from disk.

An episode is two files with the same name:

    episode_000.npz
        observation.state   (frames, 31)   where each joint is
        action              (frames, 31)   where each joint was told to go (not loaded)
        timestamp           (frames,)      seconds since the start of the episode
    episode_000.json
        episode.arm         "left" or "right", the arm that does the task
        outcome.success     true or false, this is the label

The 31 columns of state and action are the same in every episode:

    0-6    left arm        14-19  left hand      26-27  head
    7-13   right arm       20-25  right hand     28-30  base velocity

The .npz also has privileged.object_pose and info.phase. They only exist in the simulator,
so they are not loaded. The action is not loaded either, because no feature uses it so far.
"""

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int64]

STATE_DIM = 31
# Columns of state and action that belong to each arm and hand.
ARM_COLUMNS = {"left": slice(0, 7), "right": slice(7, 14)}
HAND_COLUMNS = {"left": slice(14, 20), "right": slice(20, 26)}


@dataclass(frozen=True)
class Episode:
    """What a real robot would also know about an episode, plus the label.

    Raises a ValueError when it is created with data that does not fit the format above.
    """

    name: str
    state: FloatArray
    timestamp: FloatArray
    arm: str
    success: bool

    def __post_init__(self) -> None:
        """Check the format, so that code using an Episode does not have to."""
        frames = len(self.timestamp)
        if frames == 0:
            raise ValueError(f"{self.name}: has no frames")
        if self.state.shape != (frames, STATE_DIM):
            raise ValueError(
                f"{self.name}: expected a state of shape ({frames}, {STATE_DIM}), "
                f"got {self.state.shape}"
            )
        if self.arm not in ARM_COLUMNS:
            raise ValueError(f"{self.name}: arm must be left or right, got {self.arm!r}")


def load_episode(npz_path: Path) -> Episode:
    """Read one episode from its .npz file and the .json file next to it."""
    arrays = np.load(npz_path)
    meta = json.loads(npz_path.with_suffix(".json").read_text())
    return Episode(
        name=npz_path.stem,
        state=arrays["observation.state"].astype(np.float64),
        timestamp=arrays["timestamp"].astype(np.float64),
        arm=meta["episode"]["arm"],
        success=meta["outcome"]["success"],
    )
