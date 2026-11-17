import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import pytest


@pytest.fixture
def write_episode(tmp_path: Path) -> Callable[..., Path]:
    """Write a small episode in the dataset's file format and return the .npz path."""

    def write(
        name: str = "episode_000",
        frames: int = 10,
        arm: str = "left",
        success: bool = True,
        action_frames: int | None = None,
    ) -> Path:
        npz_path = tmp_path / f"{name}.npz"
        arrays: dict[str, npt.NDArray[Any]] = {
            "observation.state": np.zeros((frames, 31), dtype=np.float32),
            "action": np.ones((action_frames or frames, 31), dtype=np.float32),
            "timestamp": np.arange(frames) / 30.0,
            "privileged.object_pose": np.zeros((frames, 7), dtype=np.float32),
        }
        np.savez(npz_path, allow_pickle=False, **arrays)
        meta = {"episode": {"arm": arm}, "outcome": {"success": success}}
        npz_path.with_suffix(".json").write_text(json.dumps(meta))
        return npz_path

    return write
