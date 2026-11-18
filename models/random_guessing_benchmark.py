"""A benchmark that needs no learning: guess a random class."""

from data.episode import FloatArray, IntArray


class RandomGuessingBenchmark:
    """A model that predicts one of the training labels at random, each equally likely."""

    def __init__(self, seed: int) -> None:
        """Create the model. The seed makes the guesses repeatable."""
        self.seed = seed
        self.labels: IntArray | None = None

    def fit(self, X: FloatArray, y: IntArray) -> None:
        """Remember which labels exist. X is only here to match other models."""
        raise NotImplementedError

    def predict(self, X: FloatArray) -> IntArray:
        """Predict a random label for every row in X."""
        raise NotImplementedError
