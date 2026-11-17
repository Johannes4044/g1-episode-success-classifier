"""A benchmark that needs no learning: always guess the most common class."""

import numpy as np

from data.episode import FloatArray, IntArray


class MajorityGuessingBenchmark:
    """A model that always predicts the class it saw most often in training."""

    def __init__(self) -> None:
        """Create the model. It knows no majority class until fit is called."""
        self.majority_class: int | None = None

    def fit(self, X: FloatArray, y: IntArray) -> None:  # noqa: ARG002
        """Remember the most common label in y. X is only here to match other models."""
        labels, counts = np.unique(y, return_counts=True)
        self.majority_class = int(labels[np.argmax(counts)])

    def predict(self, X: FloatArray) -> IntArray:
        """Predict the majority class for every row in X."""
        if self.majority_class is None:
            raise RuntimeError("call fit before predict")
        return np.full(len(X), self.majority_class)
