from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest

from data.dataset import build_xy, load_episodes, split_off_test, split_train_validation
from data.episode import Episode

# 100 episodes, every fourth one failed. Each row is its own index, so rows can be told apart.
X = np.arange(100, dtype=np.float64).reshape(100, 1)
y = np.array([0, 1, 1, 1] * 25)


@pytest.mark.integration
def test_load_episodes_returns_them_sorted_by_name(write_episode: Callable[..., Path]) -> None:
    folder = write_episode(name="episode_002").parent
    write_episode(name="episode_000")
    write_episode(name="episode_001")

    episodes = load_episodes(folder)

    assert [episode.name for episode in episodes] == ["episode_000", "episode_001", "episode_002"]


@pytest.mark.integration
def test_the_recorded_dataset_loads() -> None:
    episodes = load_episodes()

    assert len(episodes) == 400
    assert sum(episode.success for episode in episodes) == 303


@pytest.mark.unit
def test_build_xy_has_one_row_and_one_label_per_episode(
    make_episode: Callable[..., Episode],
) -> None:
    episodes = [make_episode(success=True), make_episode(success=False), make_episode()]

    features, labels = build_xy(episodes)

    assert features.shape == (3, 66)
    assert labels.tolist() == [1, 0, 1]


@pytest.mark.unit
def test_split_off_test_takes_a_fifth_and_keeps_the_class_balance() -> None:
    X_rest, X_test, y_rest, y_test = split_off_test(X, y)

    assert len(X_rest) == 80
    assert len(X_test) == 20
    assert y_rest.mean() == pytest.approx(0.75)
    assert y_test.mean() == pytest.approx(0.75)


@pytest.mark.unit
def test_split_off_test_is_the_same_every_time() -> None:
    first_test_set = split_off_test(X, y)[1]
    second_test_set = split_off_test(X, y)[1]

    assert first_test_set.tolist() == second_test_set.tolist()


@pytest.mark.unit
def test_train_and_validation_do_not_overlap() -> None:
    X_train, X_val, _, _ = split_train_validation(X, y, seed=0)

    assert len(X_train) == 75
    assert len(X_val) == 25
    assert set(X_train[:, 0]).isdisjoint(set(X_val[:, 0]))


@pytest.mark.unit
def test_a_different_seed_gives_a_different_validation_set() -> None:
    first_val = split_train_validation(X, y, seed=0)[1]
    second_val = split_train_validation(X, y, seed=1)[1]

    assert first_val.tolist() != second_val.tolist()
