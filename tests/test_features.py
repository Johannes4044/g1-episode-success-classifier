from collections.abc import Callable

import numpy as np
import pytest

from data.episode import Episode
from data.features import FEATURE_NAMES, episode_features


@pytest.mark.unit
def test_one_value_per_feature_name(make_episode: Callable[..., Episode]) -> None:
    features = episode_features(make_episode())

    assert len(FEATURE_NAMES) == 66
    assert features.shape == (66,)


@pytest.mark.unit
def test_duration_is_the_last_timestamp(make_episode: Callable[..., Episode]) -> None:
    features = episode_features(make_episode(frames=31))

    assert features[FEATURE_NAMES.index("duration")] == pytest.approx(1.0)


@pytest.mark.unit
def test_statistics_of_one_joint(make_episode: Callable[..., Episode]) -> None:
    state = np.zeros((5, 31))
    state[:, 16] = [0.0, 0.5, 1.0, 1.5, 0.5]  # column 16 is the left index finger

    features = episode_features(make_episode(arm="left", state=state))

    assert features[FEATURE_NAMES.index("mean_index")] == pytest.approx(0.7)
    assert features[FEATURE_NAMES.index("min_index")] == pytest.approx(0.0)
    assert features[FEATURE_NAMES.index("max_index")] == pytest.approx(1.5)
    assert features[FEATURE_NAMES.index("last_index")] == pytest.approx(0.5)
    assert features[FEATURE_NAMES.index("std_index")] == pytest.approx(np.std(state[:, 16]))


@pytest.mark.unit
def test_left_and_right_arm_fill_the_same_columns(make_episode: Callable[..., Episode]) -> None:
    rng = np.random.default_rng(0)
    arm_motion = rng.random((8, 7))
    hand_motion = rng.random((8, 6))
    left_state = np.zeros((8, 31))
    left_state[:, 0:7] = arm_motion
    left_state[:, 14:20] = hand_motion
    right_state = np.zeros((8, 31))
    right_state[:, 7:14] = arm_motion
    right_state[:, 20:26] = hand_motion

    left = episode_features(make_episode(arm="left", state=left_state))
    right = episode_features(make_episode(arm="right", state=right_state))

    assert left == pytest.approx(right)
