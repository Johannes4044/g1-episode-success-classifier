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


@pytest.mark.integration
def test_load_episode_rejects_different_frame_counts(write_episode: Callable[..., Path]) -> None:
    npz_path = write_episode(frames=10, action_frames=9)

    with pytest.raises(ValueError, match="episode_000"):
        load_episode(npz_path)


@pytest.mark.integration
def test_load_episode_rejects_unknown_arm(write_episode: Callable[..., Path]) -> None:
    npz_path = write_episode(arm="middle")

    with pytest.raises(ValueError, match="arm"):
        load_episode(npz_path)


@pytest.mark.integration
def test_load_episode_rejects_an_episode_without_frames(write_episode: Callable[..., Path]) -> None:
    npz_path = write_episode(frames=0)

    with pytest.raises(ValueError, match="no frames"):
        load_episode(npz_path)
