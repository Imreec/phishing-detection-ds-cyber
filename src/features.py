"""Feature extraction: TF-IDF vectorization (the source paper's representation).

Engineered hand-features (length, URL counts, urgency lexicon, header signals)
are added in Phase 2; this module starts with the TF-IDF pipeline needed to
reproduce the paper's baseline.
"""

from __future__ import annotations

from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer

from . import config


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
