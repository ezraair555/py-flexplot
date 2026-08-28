"""Correctness tests for pyflexplot.flex_nn that would have caught the
v0.2.1 metric-dispatch bug, plus ranking-correctness checks for
permutation_importance."""

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch", reason="torch is not installed")

from pyflexplot.flex_nn import (  # noqa: E402
    NeuralNetFit,
    permutation_importance,
    set_response_var,
)


def _build_regression_data(n: int = 100, p: int = 3, seed: int = 0):
    """Build data where the first predictor has the strongest effect."""
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, p))
    # x1 has the strongest coefficient, x3 the weakest.
    y = 2.0 * X[:, 0] + 0.5 * X[:, 1] + 0.1 * X[:, 2] + rng.normal(scale=0.1, size=n)
    df = pd.DataFrame(X, columns=[f"x{j+1}" for j in range(p)])
    return df, y


def _build_binary_data(n: int = 200, p: int = 3, seed: int = 0):
    """Build binary classification data where x1 is the dominant feature."""
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, p))
    # x1 strongly predicts the label via a wide-margin logistic.
    score = 2.5 * X[:, 0] + 0.3 * X[:, 1] + 0.05 * X[:, 2]
    prob = 1.0 / (1.0 + np.exp(-score))
    y = (rng.uniform(size=n) < prob).astype(int)
    df = pd.DataFrame(X, columns=[f"x{j+1}" for j in range(p)])
    return df, y


def _train_torch_model(X: np.ndarray, y: np.ndarray, *, out_dim: int = 1):
    torch.manual_seed(0)
    model = torch.nn.Sequential(
        torch.nn.Linear(X.shape[1], 8),
        torch.nn.ReLU(),
        torch.nn.Linear(8, out_dim),
    ).eval()
    optim = torch.optim.Adam(model.parameters(), lr=0.05)
    X_t = torch.as_tensor(X, dtype=torch.float32)
    y_t = torch.as_tensor(y, dtype=torch.float32)
    target_shape = y_t.shape if out_dim > 1 else y_t
    for _ in range(400):
        optim.zero_grad()
        out = model(X_t).squeeze(-1)
        loss = torch.nn.functional.mse_loss(out, target_shape)
        loss.backward()
        optim.step()
    return model


# ---------------------------------------------------------------------------
# BUG-2 regression: all named metrics must work without UnboundLocalError
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "metric",
    ["mse", "mae", "rmse", "r2", "loss"],
)
def test_regression_metric_runs(metric):
    df, y = _build_regression_data()
    model = _train_torch_model(df.to_numpy(), y)
    set_response_var(model, "y")
    fit = NeuralNetFit(
        model=model,
        response_var="y",
        predictor_names=list(df.columns),
    )
    # This used to UnboundLocalError on `auc`, `precision`, `recall`, `f1`,
    # and `loss` before v0.2.2.
    result = permutation_importance(fit, df, y, metric=metric, random_state=0)
    assert len(result) == df.shape[1]


@pytest.mark.parametrize(
    "metric",
    ["accuracy", "auc", "precision", "recall", "f1"],
)
def test_binary_metric_runs(metric):
    df, y = _build_binary_data()
    model = _train_torch_model(df.to_numpy(), y.astype(np.float32))
    set_response_var(model, "y")
    fit = NeuralNetFit(
        model=model,
        response_var="y",
        predictor_names=list(df.columns),
    )
    result = permutation_importance(fit, df, y, metric=metric, random_state=0)
    assert len(result) == df.shape[1]


def test_auc_raises_loudly_on_nonbinary_y():
    df, y = _build_regression_data()  # continuous y, not binary
    model = _train_torch_model(df.to_numpy(), y)
    set_response_var(model, "y")
    fit = NeuralNetFit(
        model=model,
        response_var="y",
        predictor_names=list(df.columns),
    )
    with pytest.raises(ValueError, match="binary"):
        permutation_importance(fit, df, y, metric="auc", random_state=0)


# ---------------------------------------------------------------------------
# Ranking-correctness tests
# ---------------------------------------------------------------------------

class TestPermutationImportanceRanking:
    """The variable with the strongest true coefficient should rank first
    on permutation importance (modulo noise).  This is a contract test --
    if it ever fails, the importance computation has regressed."""

    def test_regression_x1_ranks_first(self):
        df, y = _build_regression_data(seed=0)
        model = _train_torch_model(df.to_numpy(), y)
        set_response_var(model, "y")
        fit = NeuralNetFit(
            model=model,
            response_var="y",
            predictor_names=list(df.columns),
        )
        result = permutation_importance(fit, df, y, metric="mse", random_state=0)
        assert result.iloc[0]["variable"] == "x1", (
            f"Expected x1 (strongest coefficient) to rank first; "
            f"got {result['variable'].tolist()}"
        )

    def test_regression_importance_values_are_nonnegative(self):
        df, y = _build_regression_data(seed=0)
        model = _train_torch_model(df.to_numpy(), y)
        set_response_var(model, "y")
        fit = NeuralNetFit(
            model=model,
            response_var="y",
            predictor_names=list(df.columns),
        )
        result = permutation_importance(fit, df, y, metric="mse", random_state=0)
        # For MSE (lower-is-better), importance = permuted - baseline >= 0.
        assert (result["importance"] >= 0).all()

    def test_higher_is_better_metric_sign_flips(self):
        df, y = _build_regression_data(seed=0)
        model = _train_torch_model(df.to_numpy(), y)
        set_response_var(model, "y")
        fit = NeuralNetFit(
            model=model,
            response_var="y",
            predictor_names=list(df.columns),
        )
        # r2 is higher-is-better; importance should still be non-negative
        # (baseline - permuted) and the ranking should still match.
        result = permutation_importance(fit, df, y, metric="r2", random_state=0)
        assert (result["importance"] >= 0).all()
        assert result.iloc[0]["variable"] == "x1"