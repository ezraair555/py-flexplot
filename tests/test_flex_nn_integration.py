"""Integration tests: NeuralNetFit vs OLS in compare_fits."""

import numpy as np
import pandas as pd
import pytest
import statsmodels.formula.api as smf
from plotnine import ggplot

torch = pytest.importorskip("torch", reason="torch is not installed")

from pyflexplot import compare_fits  # noqa: E402
from pyflexplot.flex_nn import NeuralNetFit, set_response_var  # noqa: E402


def _build_neural_fit(X: np.ndarray, y: np.ndarray, *, seed: int = 0):
    """Train a tiny 1-hidden-layer network and wrap it as NeuralNetFit."""
    torch.manual_seed(seed)
    n, p = X.shape
    model = torch.nn.Sequential(
        torch.nn.Linear(p, 8),
        torch.nn.ReLU(),
        torch.nn.Linear(8, 1),
    ).eval()
    optim = torch.optim.Adam(model.parameters(), lr=0.05)
    X_t = torch.as_tensor(X, dtype=torch.float32)
    y_t = torch.as_tensor(y, dtype=torch.float32)
    for _ in range(300):
        optim.zero_grad()
        loss = torch.nn.functional.mse_loss(model(X_t).squeeze(-1), y_t)
        loss.backward()
        optim.step()
    set_response_var(model, "y")
    return NeuralNetFit(
        model=model,
        response_var="y",
        predictor_names=[f"x{i}" for i in range(p)],
    )


def test_compare_fits_accepts_neural_net_fit():
    rng = np.random.default_rng(0)
    n = 80
    X = rng.normal(size=(n, 2))
    y = X[:, 0] * 1.5 + X[:, 1] * -0.7 + rng.normal(scale=0.2, size=n)
    df = pd.DataFrame(X, columns=["x0", "x1"])
    df["y"] = y

    ols = smf.ols("y ~ x0 + x1", data=df).fit()
    nn_fit = _build_neural_fit(X, y)

    p = compare_fits("y ~ x0", data=df, model1=ols, model2=nn_fit)
    assert isinstance(p, ggplot)