"""Train a model several times and summarise the scores."""

import statistics
from collections.abc import Callable
from typing import Protocol

from data.dataset import split_train_validation
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
        for run in range(nr_runs):
            X_train, X_val, y_train, y_val = split_train_validation(X, y, seed=run)

            model = self.model_creation(run)
            model.fit(X_train, y_train)

            self.training_results.append(
                EvaluationMetrics(y_true=y_train, y_pred=model.predict(X_train))
            )
            self.validation_results.append(
                EvaluationMetrics(y_true=y_val, y_pred=model.predict(X_val))
            )

    def get_training_macro_f1s(self) -> list[float]:
        """Return the macro F1 on the training data, one value per run."""
        return [metrics.macro_f1 for metrics in self.training_results]

    def get_validation_macro_f1s(self) -> list[float]:
        """Return the macro F1 on the validation data, one value per run."""
        return [metrics.macro_f1 for metrics in self.validation_results]

    def get_validation_failure_f1s(self) -> list[float]:
        """Return the F1 of the failure class on the validation data, one value per run."""
        return [metrics.f1_failure for metrics in self.validation_results]

    def print_metrics(self) -> None:
        """Print minimum, maximum, mean and standard deviation of the scores."""
        print(f"{'score':16s} {'min':>6s} {'max':>6s} {'mean':>6s} {'std':>6s}")
        scores = {
            "macro_f1_train": self.get_training_macro_f1s(),
            "macro_f1_val": self.get_validation_macro_f1s(),
            "f1_failure_val": self.get_validation_failure_f1s(),
        }
        for name, values in scores.items():
            summary = self.get_metrics_summary(values)
            print(
                f"{name:16s} {summary['min']:6.2f} {summary['max']:6.2f} "
                f"{summary['mean']:6.2f} {summary['std_dev']:6.2f}"
            )

    @staticmethod
    def get_metrics_summary(values: list[float]) -> dict[str, float]:
        """Return minimum, maximum, mean and standard deviation of a list of scores."""
        return {
            "min": min(values),
            "max": max(values),
            "mean": statistics.mean(values),
            # the standard deviation needs at least two values
            "std_dev": statistics.stdev(values) if len(values) > 1 else 0.0,
        }
