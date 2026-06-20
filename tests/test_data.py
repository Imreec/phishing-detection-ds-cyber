"""Sanity tests for the schema-flexible data loaders.

These run on small synthetic frames (no downloaded data needed), so they are fast,
deterministic, and safe to run anywhere. They guard the column-resolution and
label-normalization logic that the rest of the pipeline depends on.
"""

import numpy as np
import pandas as pd

from src import data


def test_find_column_is_case_insensitive():
    df = pd.DataFrame(columns=["Subject", "BODY", "Label"])
    assert data._find_column(df, ["body"]) == "BODY"
    assert data._find_column(df, ["label"]) == "Label"
    assert data._find_column(df, ["nonexistent"]) is None


def test_normalize_label_passes_through_numeric():
    series = pd.Series([0, 1, 1, 0])
    assert data._normalize_label(series).tolist() == [0, 1, 1, 0]


def test_normalize_label_maps_text_classes():
    series = pd.Series(["spam", "ham", "Phishing Email", "Safe Email"])
    result = data._normalize_label(series)
    assert result.tolist() == [1, 0, 1, 0]


def test_normalize_label_unknown_becomes_nan():
    result = data._normalize_label(pd.Series(["mystery"]))
    assert np.isnan(result.iloc[0])


def test_build_text_combines_subject_and_body():
    df = pd.DataFrame({"subject": ["Hi"], "body": ["there"]})
    assert data._build_text(df).iloc[0] == "Hi there"


def test_build_text_falls_back_to_body_only():
    df = pd.DataFrame({"body": ["only body"]})
    assert data._build_text(df).iloc[0] == "only body"


def test_build_text_raises_without_text_columns():
    df = pd.DataFrame({"label": [1]})
    try:
        data._build_text(df)
        assert False, "expected ValueError"
    except ValueError:
        pass
