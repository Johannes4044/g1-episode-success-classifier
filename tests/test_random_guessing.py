import numpy as np
import pytest

from models.random_guessing_benchmark import RandomGuessingBenchmark

X = np.zeros((200, 2))
y = np.array([0, 1] * 100)


@pytest.mark.unit
def test_predicts_one_known_label_per_row() -> None:
    model = RandomGuessingBenchmark(seed=0)
    model.fit(X=X, y=y)

    predictions = model.predict(X=X)

    assert predictions.shape == (200,)
    assert set(predictions.tolist()) == {0, 1}


@pytest.mark.unit
def test_same_seed_gives_same_guesses() -> None:
    first = RandomGuessingBenchmark(seed=3)
    second = RandomGuessingBenchmark(seed=3)
    first.fit(X=X, y=y)
    second.fit(X=X, y=y)

    assert first.predict(X=X).tolist() == second.predict(X=X).tolist()


@pytest.mark.unit
def test_predict_before_fit_is_an_error() -> None:
    model = RandomGuessingBenchmark(seed=0)

    with pytest.raises(RuntimeError, match="fit"):
        model.predict(X=X)
