"""Model definitions and training.

We train four models under one protocol. LinearSVC is the reproduction anchor for
the paper's claim; the other three let us show the ~99%-then-collapse pattern is
model-agnostic (it is the data, not the model).
"""

from __future__ import annotations

from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC

from . import config


def build_models(seed: int = config.RANDOM_SEED) -> dict:
    """Return the four classifiers, all with fixed seeds where applicable."""
    return {
        "LinearSVC": LinearSVC(random_state=seed),
        "MultinomialNB": MultinomialNB(),
        "LogisticRegression": LogisticRegression(
            max_iter=1000, random_state=seed
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=200, n_jobs=-1, random_state=seed
        ),
    }


def build_linsvc(seed: int = config.RANDOM_SEED) -> LinearSVC:
    """The reproduction anchor: TF-IDF + Linear SVM, as in the source paper."""
    return LinearSVC(random_state=seed)


def train(estimator, x_train, y_train):
    """Fit an estimator and return it (kept tiny for a uniform call site)."""
    estimator.fit(x_train, y_train)
    return estimator


def fit_all(estimators: dict, x_train, y_train) -> dict:
    """Clone and fit each estimator, returning {name: fitted_model}.

    Cloning keeps the input definitions pristine so the same dict can be reused
    across experiments (e.g. the in-distribution fit and later cross-corpus fits).
    """
    return {name: clone(est).fit(x_train, y_train) for name, est in estimators.items()}
