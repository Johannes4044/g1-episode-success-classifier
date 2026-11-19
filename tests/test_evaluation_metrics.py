import numpy as np
import pytest

from evaluation.evaluation_metrics import EvaluationMetrics

y_true = np.array([0, 0, 1, 1, 1, 1])


@pytest.mark.unit
def test_perfect_predictions_score_one_everywhere() -> None:
    metrics = EvaluationMetrics(y_true=y_true, y_pred=y_true)

    assert metrics.accuracy == 1.0
    assert metrics.f1_failure == 1.0
    assert metrics.f1_success == 1.0
    assert metrics.macro_f1 == 1.0


@pytest.mark.unit
def test_always_predicting_success_finds_no_failure() -> None:
    metrics = EvaluationMetrics(y_true=y_true, y_pred=np.ones(6, dtype=np.int64))

    assert metrics.accuracy == pytest.approx(4 / 6)
    assert metrics.f1_failure == 0.0
    assert metrics.f1_success == pytest.approx(0.8)
    assert metrics.macro_f1 == pytest.approx(0.4)


@pytest.mark.unit
def test_confusion_matrix_has_true_labels_in_rows_and_failure_first() -> None:
    y_pred = np.array([0, 1, 1, 1, 1, 0])

    metrics = EvaluationMetrics(y_true=y_true, y_pred=y_pred)

    assert metrics.confusion_matrix.tolist() == [[1, 1], [1, 3]]


@pytest.mark.unit
def test_report_names_the_run_and_shows_the_macro_f1(capsys: pytest.CaptureFixture[str]) -> None:
    metrics = EvaluationMetrics(y_true=y_true, y_pred=y_true)

    metrics.print_evaluation_report("Some Model - Validation Set")

    printed = capsys.readouterr().out
    assert "Some Model - Validation Set" in printed
    assert "macro f1: 1.0" in printed


@pytest.mark.unit
def test_metrics_still_cover_both_classes_when_only_successes_occur() -> None:
    only_successes = np.ones(4, dtype=np.int64)

    metrics = EvaluationMetrics(y_true=only_successes, y_pred=only_successes)

    assert metrics.f1_success == 1.0
    assert metrics.f1_failure == 0.0
    assert metrics.confusion_matrix.tolist() == [[0, 0], [0, 4]]
