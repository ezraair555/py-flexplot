import numpy as np
import pandas as pd
import pytest
from plotnine import ggplot

from pyflexplot import flexplot


def _linear_mixed_df(seed=0, n_groups=12, per_group=12):
    rng = np.random.default_rng(seed)
    g = np.repeat(np.arange(n_groups), per_group)
    x = rng.normal(size=n_groups * per_group)
    re = rng.normal(scale=0.9, size=n_groups)[g]
    y = 1.2 + 1.8 * x + re + rng.normal(scale=0.45, size=len(x))
    return pd.DataFrame({"y": y, "x": x, "group": g})


def _binary_mixed_df(seed=1, n_groups=10, per_group=14):
    rng = np.random.default_rng(seed)
    g = np.repeat(np.arange(n_groups), per_group)
    x = rng.normal(size=n_groups * per_group)
    re = rng.normal(scale=0.7, size=n_groups)[g]
    eta = -0.4 + 1.1 * x + re
    p = 1.0 / (1.0 + np.exp(-eta))
    y = rng.binomial(1, p)
    return pd.DataFrame({"y": y, "x": x, "group": g})


def test_mixedlm_requires_random_effects():
    df = _linear_mixed_df()
    with pytest.raises(ValueError, match="random_effects"):
        flexplot("y ~ x", data=df, method="mixedlm")


def test_mixedlm_with_group_column_runs():
    df = _linear_mixed_df()
    p = flexplot("y ~ x", data=df, method="mixedlm", random_effects="group")
    assert isinstance(p, ggplot)
    layer_types = [layer.geom.__class__.__name__ for layer in p.layers]
    assert "geom_line" in layer_types


def test_lmer_alias_with_lme4_style_random_effects_runs():
    df = _linear_mixed_df()
    p = flexplot("y ~ x", data=df, method="lmer", random_effects="(1|group)")
    assert isinstance(p, ggplot)


def test_glmer_runs_for_binary_outcome():
    df = _binary_mixed_df()
    p = flexplot("y ~ x", data=df, method="glmer", random_effects="group")
    assert isinstance(p, ggplot)
    layer_types = [layer.geom.__class__.__name__ for layer in p.layers]
    assert "geom_line" in layer_types


def test_glmer_non_binary_outcome_raises():
    df = _linear_mixed_df()
    with pytest.raises(ValueError, match="binary 0/1"):
        flexplot("y ~ x", data=df, method="glmer", random_effects="group")
