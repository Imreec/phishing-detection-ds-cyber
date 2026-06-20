"""Feature extraction: TF-IDF vectorization plus interpretable text statistics.

Two feature families live here:
  * TF-IDF (the source paper's representation) — for reproduction and modeling.
  * Lightweight hand-features (length, URL count, urgency lexicon, punctuation) —
    used both in EDA (to compare phishing vs. legitimate distributions) and as an
    auxiliary, interpretable feature set. Keeping them here keeps EDA and modeling
    on the same definitions (DRY).
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer

from . import config

# A small urgency/credential lexicon. Phishing leans on urgency and authority
# (see the course's Social Engineering lecture); these terms operationalize that.
URGENCY_TERMS: tuple[str, ...] = (
    "urgent", "immediately", "verify", "suspended", "confirm", "password",
    "account", "click", "alert", "security", "update", "login", "bank",
    "limited", "expire", "act now", "congratulations", "winner", "free",
    "warning", "validate", "unauthorized", "restricted",
)

_URL_RE = r"https?://|www\."
_URGENCY_RE = "|".join(re.escape(term) for term in URGENCY_TERMS)


def add_text_stats(df: pd.DataFrame, text_col: str = config.TEXT_COL) -> pd.DataFrame:
    """Append interpretable text-statistic columns (vectorized, no Python loops).

    Columns added:
      char_count, word_count  - message size
      url_count               - count of http(s):// or www. occurrences
      urgency_count           - hits from the urgency/credential lexicon
      exclaim_count           - '!' count (a common phishing emphasis marker)
      upper_ratio             - fraction of letters that are uppercase (SHOUTING)
    """
    text = df[text_col].fillna("")
    out = df.copy()
    out["char_count"] = text.str.len()
    out["word_count"] = text.str.split().str.len()
    out["url_count"] = text.str.count(_URL_RE)
    out["urgency_count"] = text.str.lower().str.count(_URGENCY_RE)
    out["exclaim_count"] = text.str.count("!")
    letters = text.str.count(r"[A-Za-z]")
    uppers = text.str.count(r"[A-Z]")
    out["upper_ratio"] = np.where(letters > 0, uppers / letters.where(letters > 0, 1), 0.0)
    return out


TEXT_STAT_COLS: tuple[str, ...] = (
    "char_count", "word_count", "url_count",
    "urgency_count", "exclaim_count", "upper_ratio",
)


def build_tfidf(params: dict | None = None) -> TfidfVectorizer:
    """Create a TF-IDF vectorizer from the project's configured parameters."""
    return TfidfVectorizer(**(params or config.TFIDF_PARAMS))


def fit_transform_tfidf(
    train_texts, test_texts, params: dict | None = None
) -> tuple[csr_matrix, csr_matrix, TfidfVectorizer]:
    """Fit TF-IDF on training texts only (no leakage) and transform both splits."""
    vectorizer = build_tfidf(params)
    x_train = vectorizer.fit_transform(train_texts)
    x_test = vectorizer.transform(test_texts)
    return x_train, x_test, vectorizer
