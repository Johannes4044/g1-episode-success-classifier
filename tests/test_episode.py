from collections.abc import Callable
from pathlib import Path

import pytest

from data.episode import load_episode


@pytest.mark.integration
def test_load_episode_reads_arrays_and_label(write_episode: Callable[..., Path]) -> None:
    npz_path = write_episode(name="episode_007", frames=10, arm="right", success=False)

    episode = load_episode(npz_path)

    assert episode.name == "episode_007"
    assert episode.state.shape == (10, 31)
    assert episode.action.shape == (10, 31)
    assert episode.timestamp.shape == (10,)
    assert episode.arm == "right"
    assert episode.success is False
