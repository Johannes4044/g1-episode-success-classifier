# g1-episode-success-classifier

Can a model tell from the joint data of a G1 "place the object in the tray" episode whether
the episode succeeded?

## The problem

- Input: one recorded episode (joint positions of the robot over time)
- Output: success or failure

The label comes from the simulator, which knows where the object ended up. The model only gets
data a real robot would also have. `privileged.object_pose` is never used as input.

## Results

Macro F1, random forest against the benchmark:

| | majority guessing | random forest |
|---|---|---|
| validation, 10 splits (mean ± std) | 0.43 ± 0.00 | 0.94 ± 0.02 |
| test set (80 episodes, used once) | 0.43 | 0.93 |

The forest finds 18 of the 19 failures in the test set. The catch: a rule that only looks at
the thumb position in the last frame reaches 0.90. The model mostly learned whether the scripted
expert finished its routine or gave up, not where the object is. The evaluation notebook
explains this and what follows from it.

## Structure

The structure follows the SE_15 example project.

- `data/`: reading episodes (`episode.py`), features (`features.py`), loading everything and
  splitting (`dataset.py`). The recorded episodes are in `data/raw/`.
- `models/`: the two benchmark models (majority guessing, random guessing)
- `evaluation/`: metrics for one prediction, and repeated runs over several splits
- `notebooks/`: the entry points, see below
- `tests/`: automated tests for the code in the three folders above

## Notebooks

1. `01_data_exploration.ipynb`: what is in an episode, what do failures look like
2. `02_data_preparation.ipynb`: X, y and the train / validation / test split
3. `03_model_selection.ipynb`: benchmarks, logistic regression, random forest
4. `04_evaluation.ipynb`: the chosen model on the test set, error analysis, limitations

## Setup

    uv sync
    uv run pytest                 # tests with coverage
    uv run pytest -m unit         # only the fast ones
    uv run ruff check . && uv run ruff format --check . && uv run mypy
    uv run --with mutmut mutmut run   # optional: do the tests notice broken code?

For the notebooks, open them in VS Code and select the interpreter in `.venv`.

## How it was built

`git log --reverse --oneline` shows the steps in order:

1. **Explore** (notebook 01): look at one episode, then at all of them.
2. **Loader, test first** (`data/episode.py`): every `test:` commit fails on purpose, the
   commit after it makes it pass.
3. **Try things in notebooks** (02 and 03): features, split and metrics were first written
   directly in the notebooks, and copied from one notebook to the other.
4. **Refactor**: the notebook code moved into `data/` and `evaluation/` with tests, then the
   notebooks were changed to import it. The printed results did not change.
5. **Evaluate**: repeated runs, then the test set once (notebook 04).
6. **Review**: the code was checked against the Clean Code review checklist and with mutation
   testing. That found a test that could not tell the last frame from the second one, a crash
   with a single run, an unused field and unnamed numbers. Each got its own small commit.

## Open points

Things a code review could still argue about:

- `Episode` is frozen, but the numpy arrays inside it can still be changed.
- `arm` is a plain string. It is checked when an `Episode` is created, an enum would be stricter.
- `evaluation/` imports the split from `data/`. Passing the split function in would decouple them.
- Calling `MultiRunEvaluation.evaluate` twice on the same object adds to the earlier results.
- The two `print_...` methods are only tested for a few words of their output.
- The type aliases `FloatArray` and `IntArray` live in `data/episode.py` although `models/` and
  `evaluation/` use them too.

## Not done

- No hyperparameter search. Tree depth made no difference on validation, so it was left out.
- Arm joints are not mirrored between left and right arm.
- No camera images. That would be the next step, see the conclusion in notebook 04.
