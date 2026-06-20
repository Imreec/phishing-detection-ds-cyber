"""Helpers for the critical-evaluation experiments (the project's centerpiece).

These operationalize the critique: quantify the corpus<->label association, read off
which tokens a linear model relies on, measure cross-corpus generalization
(leave-one-corpus-out), and translate balanced-test rates to realistic prevalence.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency

from . import config, evaluate, features, models


def cramers_v(x, y) -> float:
    """Cramer's V association (0..1) between two categorical series, from chi-squared."""
    table = pd.crosstab(x, y)
    chi2 = chi2_contingency(table)[0]
    n = table.to_numpy().sum()
    r, k = table.shape
    return float(np.sqrt((chi2 / n) / min(r - 1, k - 1)))


def top_tokens(linear_model, feature_names, n: int = 20) -> pd.DataFrame:
    """Most phishing- and most legitimate-leaning tokens of a fitted linear model.

    Positive coefficients push toward phishing (class 1), negative toward legitimate.
    Reveals whether the model keys on phishing semantics or source/corpus artifacts.
    """
    coef = np.ravel(linear_model.coef_)
    order = np.argsort(coef)
    legit = [(feature_names[i], round(float(coef[i]), 3)) for i in order[:n]]
    phish = [(feature_names[i], round(float(coef[i]), 3)) for i in order[-n:][::-1]]
    return pd.DataFrame({"phishing_token": [t for t, _ in phish],
                         "phishing_weight": [w for _, w in phish],
                         "legit_token": [t for t, _ in legit],
                         "legit_weight": [w for _, w in legit]})


def leave_one_corpus_out(merged: pd.DataFrame, estimators: dict | None = None,
                         params: dict | None = None, beta: float = 2.0) -> pd.DataFrame:
    """Train on all corpora but one, test on the held-out corpus; repeat for each.

    This is the realistic-deployment test: performance on an email source unseen in
    training. Returns one row per (held-out corpus, model). For phishing-only corpora
    the test set is single-class, so MCC/ROC-AUC are undefined (set NaN) and recall is
    the interpretable metric.
    """
    estimators = estimators or models.build_models()
    rows = []
    for corpus in merged[config.CORPUS_COL].unique():
        train = merged[merged[config.CORPUS_COL] != corpus]
        test = merged[merged[config.CORPUS_COL] == corpus]
        x_train, x_test, _ = features.fit_transform_tfidf(
            train[config.TEXT_COL], test[config.TEXT_COL], params
        )
        single_class = test[config.LABEL_COL].nunique() == 1
        fitted = models.fit_all(estimators, x_train, train[config.LABEL_COL].to_numpy())
        y_test = test[config.LABEL_COL].to_numpy()
        for name, model in fitted.items():
            metrics = evaluate.evaluate_model(model, x_test, y_test, beta=beta)
            if single_class:  # MCC/ROC-AUC are meaningless on a one-class test set
                metrics["mcc"] = np.nan
                metrics.pop("roc_auc", None)
            rows.append({
                "held_out": corpus, "model": name, "n_test": len(test),
                "pct_phish": round(float(test[config.LABEL_COL].mean()), 2), **metrics,
            })
    return pd.DataFrame(rows)


def recall_fpr(cm: np.ndarray) -> tuple[float, float]:
    """Return (recall=TPR, FPR) from a confusion matrix with label order [0, 1]."""
    tn, fp, fn, tp = cm[0, 0], cm[0, 1], cm[1, 0], cm[1, 1]
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    fpr = fp / (fp + tn) if (fp + tn) else 0.0
    return float(recall), float(fpr)


def precision_at_prevalence(recall: float, fpr: float, prevalence: float) -> float:
    """Precision implied by a model's (recall, FPR) at a given phishing prevalence.

    recall and FPR are prevalence-independent, so we can re-express precision at any
    base rate: P = (recall*pi) / (recall*pi + fpr*(1-pi)). At a realistic low prevalence
    this exposes the precision a balanced test set hides.
    """
    tp = recall * prevalence
    fp = fpr * (1 - prevalence)
    return tp / (tp + fp) if (tp + fp) else 0.0


def precision_prevalence_curve(recall: float, fpr: float, prevalences: np.ndarray) -> np.ndarray:
    """Vectorized precision_at_prevalence over an array of prevalences."""
    return np.array([precision_at_prevalence(recall, fpr, p) for p in prevalences])
