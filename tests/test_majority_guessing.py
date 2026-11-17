import numpy as np
import pytest

from models.majority_guessing_benchmark import MajorityGuessingBenchmark


@pytest.mark.unit
def test_predicts_the_most_common_training_label() -> None:
    model = MajorityGuessingBenchmark()
    model.fit(X=np.zeros((5, 2)), y=np.array([1, 0, 1, 1, 0]))

    predictions = model.predict(X=np.zeros((3, 2)))

    assert predictions.tolist() == [1, 1, 1]


@pytest.mark.unit
def test_predict_before_fit_is_an_error() -> None:
    model = MajorityGuessingBenchmark()

    with pytest.raises(RuntimeError, match="fit"):
        model.predict(X=np.zeros((3, 2)))
