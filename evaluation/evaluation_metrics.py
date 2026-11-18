"""Scores that compare predicted labels with the true labels."""

from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

from data.episode import IntArray

FAILURE = 0
SUCCESS = 1


class EvaluationMetrics:
    """Accuracy, F1 score per class, macro F1 and the confusion matrix of one prediction."""

    def __init__(self, y_true: IntArray, y_pred: IntArray) -> None:
        """Calculate the metrics by comparing predicted labels with the true labels."""
        f1_scores = f1_score(
            y_true, y_pred, average=None, labels=[FAILURE, SUCCESS], zero_division=0
        )

        self.accuracy = float(accuracy_score(y_true, y_pred))
        self.f1_failure = float(f1_scores[FAILURE])
        self.f1_success = float(f1_scores[SUCCESS])
        self.macro_f1 = (self.f1_failure + self.f1_success) / 2
        # rows: true failure, true success. columns: predicted failure, predicted success
        self.confusion_matrix: IntArray = confusion_matrix(
            y_true, y_pred, labels=[FAILURE, SUCCESS]
        )

    def print_evaluation_report(self, description: str) -> None:
        """Print a summary of the metrics."""
        print(description)
        print("- accuracy:", round(self.accuracy, 2))
        print("- f1 failure:", round(self.f1_failure, 2), " f1 success:", round(self.f1_success, 2))
        print("- macro f1:", round(self.macro_f1, 2))
        print("- confusion matrix (rows: true failure, true success):")
        print(self.confusion_matrix)
