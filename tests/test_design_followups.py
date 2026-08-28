"""Tests for the v0.3.0 design-followup improvements:

1. visualize() supports NeuralNetFit
2. flexplot() method validation (no silent typo acceptance)
3. flexplot() given-variable validation (no silent 3+ drop)
4. bluepill polynomials 'to' key is optional (no API wart)
"""

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch", reason="torch is not installed")

from pyflexplot import flexplot, visualize  # noqa: E402
from pyflexplot.bluepill import mixed_model  # noqa: E402
from pyflexplot.flex_nn import NeuralNetFit, set_response_var  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _train_tiny_model(n: int = 200, p: int = 3, seed: int = 0):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, p)).astype(np.float32)
    y = (
        1.5 * X[:, 0] - 2.0 * X[:, 1] + 0.7 * X[:, 2]
        + rng.normal(scale=0.1, size=n).astype(np.float32)
    )
    model = torch.nn.Sequential(
        torch.nn.Linear(p, 8),
        torch.nn.ReLU(),
        torch.nn.Linear(8, 1),
    ).eval()
    optim = torch.optim.Adam(model.parameters(), lr=0.05)
    X_t = torch.as_tensor(X)
    y_t = torch.as_tensor(y)
    for _ in range(300):
        optim.zero_grad()
        loss = torch.nn.functional.mse_loss(model(X_t).squeeze(-1), y_t)
        loss.backward()
        optim.step()
    set_response_var(model, "y")
    fit = NeuralNetFit(
        model=model,
        response_var="y",
        predictor_names=[f"x{j+1}" for j in range(p)],
    )
    return fit, pd.DataFrame(X, columns=[f"x{j+1}" for j in range(p)]), y


# ---------------------------------------------------------------------------
# 1. visualize() accepts NeuralNetFit
# ---------------------------------------------------------------------------

class TestVisualizeNeuralNetFit:
    def test_returns_ggplot(self):
        from plotnine import ggplot
        fit, X_df, y = _train_tiny_model()
        X_df = X_df.copy()
        X_df["y"] = y
        p = visualize(fit, data=X_df)
        assert isinstance(p, ggplot)

    def test_explicit_x_is_honoured(self):
        from plotnine import ggplot
        fit, X_df, y = _train_tiny_model()
        X_df["y"] = y
        p = visualize(fit, data=X_df, x="x2")
        assert isinstance(p, ggplot)

    def test_missing_data_raises(self):
        fit, _, _ = _train_tiny_model()
        with pytest.raises(ValueError, match="requires data"):
            visualize(fit)

    def test_missing_response_raises(self):
        fit, X_df, _ = _train_tiny_model()
        # response_var='y' but data has no 'y' column
        with pytest.raises(ValueError, match="not found in data"):
            visualize(fit, data=X_df)

    def test_unknown_explicit_x_raises(self):
        fit, X_df, y = _train_tiny_model()
        X_df["y"] = y
        with pytest.raises(ValueError, match="not found in data"):
            visualize(fit, data=X_df, x="not_a_column")

    def test_non_dataframe_raises(self):
        fit, _, _ = _train_tiny_model()
        with pytest.raises(TypeError, match="DataFrame"):
            visualize(fit, data={"x1": [1.0, 2.0]})

    def test_empty_dataframe_raises(self):
        fit, _, _ = _train_tiny_model()
        empty = pd.DataFrame(columns=["x1", "x2", "x3", "y"])
        with pytest.raises(ValueError, match="non-empty"):
            visualize(fit, data=empty)


# ---------------------------------------------------------------------------
# 2. flexplot() method validation
# ---------------------------------------------------------------------------

class TestFlexplotMethodValidation:
    def _df(self):
        return pd.DataFrame({
            "y": np.random.normal(size=100),
            "x": np.random.normal(size=100),
        })

    def test_unknown_method_raises(self):
        with pytest.raises(ValueError, match="method must be one of"):
            flexplot("y ~ x", data=self._df(), method="loes")  # typo for loess

    def test_known_methods_accepted(self):
        for method in ("auto", "lm", "loess"):
            p = flexplot("y ~ x", data=self._df(), method=method)
            assert p is not None


# ---------------------------------------------------------------------------
# 3. flexplot() given-variable validation
# ---------------------------------------------------------------------------

class TestFlexplotGivenValidation:
    def _df(self):
        return pd.DataFrame({
            "y": np.random.normal(size=100),
            "x": np.random.normal(size=100),
            "a": np.random.choice(["A", "B"], size=100),
            "b": np.random.choice(["X", "Y"], size=100),
            "c": np.random.choice(["P", "Q"], size=100),
        })

    def test_two_given_is_fine(self):
        p = flexplot("y ~ x | a + b", data=self._df())
        assert p is not None

    def test_three_given_raises(self):
        with pytest.raises(ValueError, match="at most 2 given variables"):
            flexplot("y ~ x | a + b + c", data=self._df())

    def test_no_given_is_fine(self):
        p = flexplot("y ~ x", data=self._df())
        assert p is not None


# ---------------------------------------------------------------------------
# 4. bluepill polynomials 'to' key is optional
# ---------------------------------------------------------------------------

class TestPolynomialsOptionalToKey:
    def _vars(self):
        return {
            "y": (10.0, 3.0, 0),
            "x1": (22.0, 7.0, 0),
            "x2": (5.0, 2.0, 0),
            "cluster": ["A", "B", "C", "D", "E"],
        }

    def test_polynomials_without_to_key_accepted(self):
        df = mixed_model(
            fixed=[0.0, 0.2, 0.5],
            random=[0.1, 0.1, 0.1],
            sigma=0.3,
            clusters=5,
            n_per=[10, 2],
            vars=self._vars(),
            polynomials={"from": [1, 2], "coef": [0.2, 0.3]},
            seed=0,
        )
        assert len(df) > 0

    def test_polynomials_with_to_key_still_works_backward_compatibly(self):
        # Existing users may still pass 'to' -- we should accept it (and ignore it).
        df = mixed_model(
            fixed=[0.0, 0.2, 0.5],
            random=[0.1, 0.1, 0.1],
            sigma=0.3,
            clusters=5,
            n_per=[10, 2],
            vars=self._vars(),
            polynomials={"from": [1, 2], "to": [1, 2], "coef": [0.2, 0.3]},
            seed=0,
        )
        assert len(df) > 0

    def test_polynomials_with_unknown_extra_key_raises(self):
        with pytest.raises(ValueError, match="polynomials dict must have only"):
            mixed_model(
                fixed=[0.0, 0.2, 0.5],
                random=[0.1, 0.1, 0.1],
                sigma=0.3,
                clusters=5,
                n_per=[10, 2],
                vars=self._vars(),
                polynomials={"from": [1], "coef": [0.1], "bogus": [42]},
            )

    def test_polynomials_missing_coef_raises(self):
        with pytest.raises(ValueError, match="must have both"):
            mixed_model(
                fixed=[0.0, 0.2, 0.5],
                random=[0.1, 0.1, 0.1],
                sigma=0.3,
                clusters=5,
                n_per=[10, 2],
                vars=self._vars(),
                polynomials={"from": [1, 2]},  # missing 'coef'
            )

    def test_polynomials_length_mismatch_raises(self):
        with pytest.raises(ValueError, match="inconsistent lengths"):
            mixed_model(
                fixed=[0.0, 0.2, 0.5],
                random=[0.1, 0.1, 0.1],
                sigma=0.3,
                clusters=5,
                n_per=[10, 2],
                vars=self._vars(),
                polynomials={"from": [1, 2], "coef": [0.1]},  # 2 vs 1
            )

    def test_interactions_still_require_to(self):
        # Regression guard: splitting the validators must not break the
        # interactions path.  Interactions still need 'to'.
        with pytest.raises(ValueError, match="interactions dict must have"):
            mixed_model(
                fixed=[0.0, 0.2, 0.5],
                random=[0.1, 0.1, 0.1],
                sigma=0.3,
                clusters=5,
                n_per=[10, 2],
                vars=self._vars(),
                interactions={"from": [1, 2], "coef": [0.1, 0.2]},  # no 'to'
            )