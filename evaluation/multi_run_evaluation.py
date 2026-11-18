"""Train a model several times and summarise the scores."""

from collections.abc import Callable
from typing import Protocol

from data.episode import FloatArray, IntArray
from evaluation.evaluation_metrics import EvaluationMetrics


class Model(Protocol):
    """Anything with fit and predict: the scikit-learn models and the benchmarks."""

    def fit(self, X: FloatArray, y: IntArray) -> object:
        """Learn from the training data."""
        ...

    def predict(self, X: FloatArray) -> IntArray:
        """Predict one label per row."""
        ...


class MultiRunEvaluation:
    """Trains a model on several train/validation splits and records the scores of each run."""

    def __init__(self, model_creation: Callable[[int], Model]) -> None:
        """Set up the evaluation. model_creation gets a seed and returns a new, untrained model."""
        self.model_creation = model_creation
        self.training_results: list[EvaluationMetrics] = []
        self.validation_results: list[EvaluationMetrics] = []

    def evaluate(self, nr_runs: int, X: FloatArray, y: IntArray) -> None:
        """Train nr_runs times. Run number i uses i as seed for the split and for the model."""
        raise NotImplementedError

    def get_training_macro_f1s(self) -> list[float]:
        """Return the macro F1 on the training data, one value per run."""
        raise NotImplementedError

    def get_validation_macro_f1s(self) -> list[float]:
        """Return the macro F1 on the validation data, one value per run."""
        raise NotImplementedError

    def get_validation_failure_f1s(self) -> list[float]:
        """Return the F1 of the failure class on the validation data, one value per run."""
        raise NotImplementedError

    def print_metrics(self) -> None:
        """Print minimum, maximum, mean and standard deviation of the scores."""
        raise NotImplementedError

    @staticmethod
    def get_metrics_summary(values: list[float]) -> dict[str, float]:
        """Return minimum, maximum, mean and standard deviation of a list of scores."""
        raise NotImplementedError
