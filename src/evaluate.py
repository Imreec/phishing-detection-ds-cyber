"""Unified evaluation: one metric suite used everywhere for fair comparison.

The metric choice is itself a critique point: the source paper reports only
accuracy/precision/recall/F1. We add MCC, F-beta, and ROC-AUC because they are
the metrics that expose imbalance and threshold sensitivity.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    fbeta_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)


def get_scores(model, x) -> np.ndarray | None:
    """Continuous scores for ROC-AUC: predict_proba if available, else decision_function."""
    if hasattr(model, "predict_proba"):
        return model.predict_proba(x)[:, 1]
    if hasattr(model, "decision_function"):
        return model.decision_function(x)
    return None


def evaluate_predictions(
    y_true, y_pred, y_score=None, beta: float = 2.0
) -> dict[str, float]:
    """Compute the full metric suite from labels (and optional scores)."""
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        f"f{beta:g}": fbeta_score(y_true, y_pred, beta=beta, zero_division=0),
        "mcc": matthews_corrcoef(y_true, y_pred),
    }
    if y_score is not None and len(np.unique(y_true)) > 1:
        metrics["roc_auc"] = roc_auc_score(y_true, y_score)
    return metrics


def evaluate_model(model, x_test, y_test, beta: float = 2.0) -> dict[str, float]:
    """Predict with a fitted model and return its metric suite."""
    y_pred = model.predict(x_test)
    y_score = get_scores(model, x_test)
    return evaluate_predictions(y_test, y_pred, y_score, beta=beta)


def confusion(y_true, y_pred) -> np.ndarray:
    """2x2 confusion matrix with fixed label order [0=legit, 1=phishing]."""
    return confusion_matrix(y_true, y_pred, labels=[0, 1])


def compare_models(fitted: dict, x_test, y_test, beta: float = 2.0) -> pd.DataFrame:
    """Evaluate several fitted models on one test set; rows = models, cols = metrics."""
    rows = {name: evaluate_model(model, x_test, y_test, beta=beta) for name, model in fitted.items()}
    return pd.DataFrame(rows).T
