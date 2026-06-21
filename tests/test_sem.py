import pandas as pd
import numpy as np
import pytest
from plotnine import ggplot

from pyflexplot import hopper_plot, disturbance_plot, measurement_plot


def test_sem_scaffolding():
    # Create dummy model class to mimic semopy Model if not present
    class MockModel:
        def __init__(self):
            self.mx_cov = pd.DataFrame(np.eye(3), index=["x1", "x2", "x3"], columns=["x1", "x2", "x3"])

        def calc_sigma(self):
            return (np.eye(3) * 0.9, None)

        def predict(self, data):
            return data * 0.8

        def predict_factors(self, data):
            return pd.DataFrame({"F1": np.random.normal(size=len(data))}, index=data.index)

    model = MockModel()
    df = pd.DataFrame({
        "x1": np.random.normal(size=10),
        "x2": np.random.normal(size=10),
        "x3": np.random.normal(size=10)
    })

    p1 = hopper_plot(model)
    assert isinstance(p1, ggplot)

    p2 = disturbance_plot(model, "x1", "x2", df)
    assert isinstance(p2, ggplot)

    p3 = measurement_plot(model, "F1", "x1", df)
    assert isinstance(p3, ggplot)


def test_hopper_plot_real_semopy():
    semopy = pytest.importorskip("semopy")
    from semopy import Model

    desc = "y =~ x1 + x2 + x3"
    np.random.seed(0)
    df = pd.DataFrame(np.random.randn(50, 4), columns=["y", "x1", "x2", "x3"])
    model = Model(desc)
    model.fit(df)

    p = hopper_plot(model)
    assert isinstance(p, ggplot)
    # Should render without error.
    p.draw()


def test_disturbance_plot_real_semopy():
    semopy = pytest.importorskip("semopy")
    from semopy import Model

    desc = "y =~ x1 + x2 + x3"
    np.random.seed(0)
    df = pd.DataFrame(np.random.randn(50, 4), columns=["y", "x1", "x2", "x3"])
    model = Model(desc)
    model.fit(df)

    p = disturbance_plot(model, "x1", "x2", df)
    assert isinstance(p, ggplot)


def test_measurement_plot_real_semopy():
    semopy = pytest.importorskip("semopy")
    from semopy import Model

    desc = "y =~ x1 + x2 + x3"
    np.random.seed(0)
    df = pd.DataFrame(np.random.randn(50, 4), columns=["y", "x1", "x2", "x3"])
    model = Model(desc)
    model.fit(df)

    p = measurement_plot(model, "y", "x1", df)
    assert isinstance(p, ggplot)
