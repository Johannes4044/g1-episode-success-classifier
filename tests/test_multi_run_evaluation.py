import numpy as np
import pytest
from sklearn.tree import DecisionTreeClassifier

from data.dataset import build_xy, load_episodes
from evaluation.multi_run_evaluation import MultiRunEvaluation
from models.majority_guessing_benchmark import MajorityGuessingBenchmark

# 100 samples, every fourth one is a failure
X = np.zeros((100, 2))
y = np.array([0, 1, 1, 1] * 25)


@pytest.mark.integration
def test_evaluate_records_one_result_per_run() -> None:
    evaluation = MultiRunEvaluation(model_creation=lambda _seed: MajorityGuessingBenchmark())

    evaluation.evaluate(nr_runs=3, X=X, y=y)

    assert len(evaluation.get_training_macro_f1s()) == 3
    assert len(evaluation.get_validation_macro_f1s()) == 3
    assert len(evaluation.get_validation_failure_f1s()) == 3


@pytest.mark.integration
def test_every_run_gets_its_own_seed() -> None:
    seeds = []

    def create_model(seed: int) -> MajorityGuessingBenchmark:
        seeds.append(seed)
        return MajorityGuessingBenchmark()

    MultiRunEvaluation(model_creation=create_model).evaluate(nr_runs=3, X=X, y=y)

    assert seeds == [0, 1, 2]


@pytest.mark.integration
def test_majority_guessing_scores_the_same_in_every_run() -> None:
    evaluation = MultiRunEvaluation(model_creation=lambda _seed: MajorityGuessingBenchmark())

    evaluation.evaluate(nr_runs=3, X=X, y=y)

    # all predictions are "success": F1 success = 0.857, F1 failure = 0, the mean is 0.429
    assert evaluation.get_validation_macro_f1s() == pytest.approx([0.4286] * 3, abs=0.01)
    assert evaluation.get_validation_failure_f1s() == [0.0, 0.0, 0.0]


@pytest.mark.unit
def test_metrics_summary() -> None:
    summary = MultiRunEvaluation.get_metrics_summary([0.5, 0.7, 0.9])

    assert summary["min"] == 0.5
    assert summary["max"] == 0.9
    assert summary["mean"] == pytest.approx(0.7)
    assert summary["std_dev"] == pytest.approx(0.2)


@pytest.mark.integration
def test_print_metrics_shows_one_line_per_score(capsys: pytest.CaptureFixture[str]) -> None:
    evaluation = MultiRunEvaluation(model_creation=lambda _seed: MajorityGuessingBenchmark())
    evaluation.evaluate(nr_runs=2, X=X, y=y)

    evaluation.print_metrics()

    printed = capsys.readouterr().out
    assert "macro_f1_train" in printed
    assert "macro_f1_val" in printed
    assert "f1_failure_val" in printed


@pytest.mark.unit
def test_metrics_summary_of_a_single_run_has_no_spread() -> None:
    summary = MultiRunEvaluation.get_metrics_summary([0.8])

    assert summary["mean"] == 0.8
    assert summary["std_dev"] == 0.0


@pytest.mark.integration
def test_evaluating_twice_gives_the_same_scores() -> None:
    rng = np.random.default_rng(0)
    noise = rng.random((60, 3))
    labels = rng.integers(0, 2, size=60)

    def create_tree(seed: int) -> DecisionTreeClassifier:
        return DecisionTreeClassifier(random_state=seed)

    first = MultiRunEvaluation(model_creation=create_tree)
    second = MultiRunEvaluation(model_creation=create_tree)
    first.evaluate(nr_runs=5, X=noise, y=labels)
    second.evaluate(nr_runs=5, X=noise, y=labels)

    assert first.get_validation_macro_f1s() == second.get_validation_macro_f1s()


@pytest.mark.integration
def test_a_tree_beats_majority_guessing_on_the_recorded_episodes() -> None:
    features, labels = build_xy(load_episodes())
    tree = MultiRunEvaluation(lambda seed: DecisionTreeClassifier(max_depth=2, random_state=seed))
    majority = MultiRunEvaluation(lambda _seed: MajorityGuessingBenchmark())

    tree.evaluate(nr_runs=2, X=features, y=labels)
    majority.evaluate(nr_runs=2, X=features, y=labels)

    assert min(tree.get_validation_macro_f1s()) > max(majority.get_validation_macro_f1s()) + 0.3
