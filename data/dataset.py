"""Load all episodes, turn them into X and y, and split them."""

from pathlib import Path

from data.episode import Episode, FloatArray, IntArray

EPISODES_DIR = Path(__file__).parent / "raw" / "place_perturbed_400" / "episodes"

Split = tuple[FloatArray, FloatArray, IntArray, IntArray]


def load_episodes(folder: Path = EPISODES_DIR) -> list[Episode]:
    """Load every episode in a folder, sorted by file name."""
    raise NotImplementedError


def build_xy(episodes: list[Episode]) -> tuple[FloatArray, IntArray]:
    """Build X (one row of features per episode) and y (1 = success, 0 = failure)."""
    raise NotImplementedError


def split_off_test(X: FloatArray, y: IntArray) -> Split:
    """Put 20 % aside as test set. Returns X_rest, X_test, y_rest, y_test.

    The split is always the same, so the test set stays untouched between notebooks.
    """
    raise NotImplementedError


def split_train_validation(X: FloatArray, y: IntArray, seed: int) -> Split:
    """Split what is left after split_off_test. Returns X_train, X_val, y_train, y_val."""
    raise NotImplementedError
