# Neural-Network Integration (flex_nn Port)

`pyflexplot.flex_nn` is a Python port of the spirit (not the surface) of
Dustin Fife's [`flex_nn`](https://github.com/dustinfife/flex_nn) R package.
It provides thin wrappers so that fitted neural networks can be used as
first-class models in py-flexplot's visualization API -- in particular,
`compare_fits()`.

## `NeuralNetFit`

```python
from pyflexplot.flex_nn import NeuralNetFit, set_response_var
import torch

# Train or load a model.
model = ...  # torch.nn.Module

# Attach the response-variable name so the wrapper knows which column the
# network predicts.
set_response_var(model, "y")

fit = NeuralNetFit(
    model=model,
    response_var="y",
    predictor_names=["x1", "x2", "x3"],
    x_means=None,  # set if the model was trained on z-scored inputs
    x_sds=None,
    history=None,  # free-form training history (Keras History, list, ...)
)
```

`NeuralNetFit.predict(data)` returns a `pandas.Series` aligned to
`data.index`, exactly like the statsmodels predictions, so it slots into
`compare_fits()`:

```python
from pyflexplot import compare_fits
p = compare_fits("y ~ x1", data=df, model1=ols_fit, model2=nn_fit)
```

## `set_response_var`

```python
from pyflexplot.flex_nn import set_response_var
set_response_var(model, "y_target")
```

Attaches the response-variable name as a private attribute on *model*.
Required for `NeuralNetFit` to know which column of the input DataFrame
the network predicts.

## `prepare_torch_data`

```python
from pyflexplot.flex_nn import prepare_torch_data
X = prepare_torch_data(df, categorical_vars=["color", "group"])
```

Converts a DataFrame into a dense float matrix suitable for a torch
network.  Categorical columns are integer-encoded starting at zero.  The
function does **not** impute missing values -- imputation is the caller's
responsibility.

## `permutation_importance`

```python
from pyflexplot.flex_nn import permutation_importance
result = permutation_importance(fit, X=df[["x1", "x2", "x3"]], y=df["y"],
                                metric="mse", random_state=0)
print(result)
#    variable  importance
# 0        x1    0.842...
# 1        x2    0.103...
# 2        x3    0.017...
```

Column-shuffling variable importance.  *metric* can be `"mse"`, `"mae"`,
`"rmse"`, `"r2"`, `"accuracy"`, or a callable `(y_true, y_pred) -> float`
combined with `higher_is_better=True/False`.  The result's `.attrs`
attribute exposes the baseline metric value.

## Backend selection

The default backend is **torch**.  Keras 3 is supported opportunistically
when `keras` is importable; no keras dependency is declared in
`pyproject.toml`.  Use `is_torch_model(obj)` / `is_keras_model(obj)` to
inspect whether an object is recognised by the wrappers.

## Why this design

The R `flex_nn` package is fundamentally a thin S3-method layer that
hooks Keras models into flexplot's `compare.fits()`.  In Python we
replicate that idea with a small dataclass wrapper (`NeuralNetFit`) and
duck-typed `.predict()` semantics.  No attempt is made to fit networks
inside py-flexplot -- fitting is the caller's job, as it is in the R
package once Keras/TensorFlow is set up.