import pandas as pd
import pytest
import statsmodels.formula.api as smf
from pyflexplot.datasets import (
    load_avengers,
    load_dataset,
    load_diet,
    load_exercise_data,
)
from scipy import stats


def test_loaders_match_upstream_shapes_and_columns():
    avengers = load_avengers()
    diet = load_diet()
    exercise = load_exercise_data()

    assert avengers.shape == (812, 15)
    assert diet.shape == (78, 7)
    assert exercise.shape == (200, 11)
    assert "agility" in avengers.columns
    assert {"pre.weight", "weight6weeks"} <= set(diet.columns)
    assert {"therapy.type", "weight.loss"} <= set(exercise.columns)


def test_load_dataset_dispatch_and_unknown_name():
    pd.testing.assert_frame_equal(load_dataset("diet"), load_diet())
    with pytest.raises(ValueError, match="Unknown dataset"):
        load_dataset("not_a_dataset")


def test_loaders_return_independent_frames():
    first = load_avengers()
    first.loc[0, "iq"] = -1
    second = load_avengers()
    assert second.loc[0, "iq"] != -1


def test_textbook_one_sample_and_regression_results():
    avengers = load_avengers()
    one_sample = stats.ttest_1samp(avengers["iq"], 100)
    assert one_sample.statistic == pytest.approx(35.555866, abs=1e-5)

    model = smf.ols("agility ~ speed", data=avengers).fit()
    assert model.params["speed"] == pytest.approx(43.085687, abs=1e-5)
    assert model.rsquared == pytest.approx(0.238689, abs=1e-6)


def test_textbook_paired_t_test_result():
    diet = load_diet()
    paired = stats.ttest_rel(diet["pre.weight"], diet["weight6weeks"])
    assert paired.statistic == pytest.approx(13.308754, abs=1e-5)
    assert (diet["weight6weeks"] - diet["pre.weight"]).mean() == pytest.approx(-3.844872, abs=1e-6)


def test_textbook_anova_result():
    exercise = load_exercise_data()
    model = smf.ols('Q("weight.loss") ~ C(Q("therapy.type"))', data=exercise).fit()
    assert model.fvalue == pytest.approx(8.501012, abs=1e-5)
