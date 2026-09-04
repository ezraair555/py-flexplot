"""Datasets used by the statistical examples in the py-flexplot docs.

The loaders return a fresh :class:`pandas.DataFrame` on every call and retain
column names from the upstream ``flexplot`` R package for reproducibility.
"""

from importlib.resources import files
from io import TextIOWrapper
from typing import Final

import pandas as pd

_DATA_FILES: Final = {
    "avengers": "avengers.csv",
    "diet": "diet.csv",
    "exercise_data": "exercise_data.csv",
}


def load_dataset(name: str) -> pd.DataFrame:
    """Load one of the statistical example datasets.

    Parameters
    ----------
    name:
        One of ``"avengers"``, ``"diet"``, or ``"exercise_data"``.

    Returns
    -------
    pandas.DataFrame
        A new DataFrame containing the upstream dataset.

    Raises
    ------
    ValueError
        If *name* is not a supported dataset.
    """
    try:
        filename = _DATA_FILES[name]
    except KeyError as exc:
        choices = ", ".join(sorted(_DATA_FILES))
        raise ValueError(f"Unknown dataset {name!r}; choose from {choices}") from exc

    resource = files("pyflexplot.datasets").joinpath("data", filename)
    with resource.open("rb") as raw:
        # TextIOWrapper keeps decoding explicit and works with importlib resources.
        with TextIOWrapper(raw, encoding="utf-8", newline="") as text:
            return pd.read_csv(text)


def load_avengers() -> pd.DataFrame:
    """Load the simulated final Avengers battle dataset."""
    return load_dataset("avengers")


def load_diet() -> pd.DataFrame:
    """Load the paired pre-treatment and six-week diet dataset."""
    return load_dataset("diet")


def load_exercise_data() -> pd.DataFrame:
    """Load the simulated exercise/therapy weight-loss dataset."""
    return load_dataset("exercise_data")


__all__ = ["load_dataset", "load_avengers", "load_diet", "load_exercise_data"]
