"""Load all episodes, turn them into X and y, and split them."""

from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split

from data.episode import Episode, FloatArray, IntArray, load_episode
from data.features import episode_features

EPISODES_DIR = Path(__file__).parent / "raw" / "place_perturbed_400" / "episodes"

TEST_SPLIT_SEED = 0
TEST_FRACTION = 0.2
# share of what is left after the test set, so validation is also 20 % of all episodes
VALIDATION_FRACTION = 0.25

Split = tuple[FloatArray, FloatArray, IntArray, IntArray]


def load_episodes(folder: Path = EPISODES_DIR) -> list[Episode]:
    """Load every episode in a folder, sorted by file name."""
    return [load_episode(path) for path in sorted(folder.glob("*.npz"))]


def build_xy(episodes: list[Episode]) -> tuple[FloatArray, IntArray]:
    """Build X (one row of features per episode) and y (1 = success, 0 = failure)."""
    X = np.array([episode_features(episode) for episode in episodes])
    y = np.array([int(episode.success) for episode in episodes])
    return X, y


def split_off_test(X: FloatArray, y: IntArray) -> Split:
    """Put 20 % aside as test set. Returns X_rest, X_test, y_rest, y_test.

    The split is always the same, so the test set stays untouched between notebooks.
    """
    X_rest, X_test, y_rest, y_test = train_test_split(
        X, y, test_size=TEST_FRACTION, stratify=y, random_state=TEST_SPLIT_SEED
    )
    return X_rest, X_test, y_rest, y_test


def split_train_validation(X: FloatArray, y: IntArray, seed: int) -> Split:
    """Split what is left after split_off_test. Returns X_train, X_val, y_train, y_val."""
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=VALIDATION_FRACTION, stratify=y, random_state=seed
    )
    return X_train, X_val, y_train, y_val
