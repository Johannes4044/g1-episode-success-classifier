"""A benchmark that needs no learning: always guess the most common class."""

from data.episode import FloatArray, IntArray


class MajorityGuessingBenchmark:
    """A model that always predicts the class it saw most often in training."""

    def __init__(self) -> None:
        """Create the model. It knows no majority class until fit is called."""
        self.majority_class: int | None = None

    def fit(self, X: FloatArray, y: IntArray) -> None:
        """Remember the most common label in y. X is only here to match other models."""
        raise NotImplementedError

    def predict(self, X: FloatArray) -> IntArray:
        """Predict the majority class for every row in X."""
        raise NotImplementedError
