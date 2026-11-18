"""Scores that compare predicted labels with the true labels."""

from data.episode import IntArray

FAILURE = 0
SUCCESS = 1


class EvaluationMetrics:
    """Accuracy, F1 score per class, macro F1 and the confusion matrix of one prediction."""

    def __init__(self, y_true: IntArray, y_pred: IntArray) -> None:
        """Calculate the metrics by comparing predicted labels with the true labels."""
        self.accuracy: float
        self.f1_failure: float
        self.f1_success: float
        self.macro_f1: float
        self.confusion_matrix: IntArray
        raise NotImplementedError

    def print_evaluation_report(self, description: str) -> None:
        """Print a summary of the metrics."""
        raise NotImplementedError
