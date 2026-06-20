"""Reusable plotting helpers, so the notebook stays a thin narrative.

Every figure is saved to ``figures/`` (via :func:`save_fig`) at a consistent size
and resolution so the report can embed the exact same images.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import RocCurveDisplay

from . import config

# One consistent look across every figure.
sns.set_theme(style="whitegrid", context="notebook")
_PALETTE = {0: "#4C72B0", 1: "#C44E52"}  # 0 = legitimate (blue), 1 = phishing (red)
_LABEL_NAMES = {0: "legitimate", 1: "phishing"}


def save_fig(fig: plt.Figure, name: str) -> Path:
    """Save a figure to figures/<name>.png and return its path."""
    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    path = config.FIGURES_DIR / f"{name}.png"
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    return path


def dist_by_class(
    df, value_col: str, label_col: str = config.LABEL_COL,
    *, log_x: bool = False, clip_quantile: float | None = 0.99, title: str | None = None,
):
    """Overlaid distributions of a numeric feature, split by class.

    clip_quantile trims the long right tail for readability (common on text-length
    and URL-count features); log_x is offered for heavy-tailed counts.
    """
    data = df[[value_col, label_col]].copy()
    if clip_quantile is not None:
        upper = data[value_col].quantile(clip_quantile)
        data = data[data[value_col] <= upper]

    fig, ax = plt.subplots(figsize=(7, 4))
    for label, group in data.groupby(label_col):
        sns.kdeplot(
            group[value_col], ax=ax, fill=True, alpha=0.4,
            label=_LABEL_NAMES.get(label, str(label)),
            color=_PALETTE.get(label),
            log_scale=log_x,
        )
    ax.set_title(title or f"{value_col} by class")
    ax.set_xlabel(value_col)
    ax.legend(title="class")
    return fig, ax


def grouped_bar(series, *, title: str, xlabel: str = "", ylabel: str = "count", rotate: int = 0):
    """Bar chart from a pre-aggregated Series (e.g. value_counts)."""
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.barplot(x=series.index, y=series.values, ax=ax, color="#4C72B0")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if rotate:
        ax.tick_params(axis="x", rotation=rotate)
    return fig, ax


def heatmap(
    matrix, *, title: str, fmt: str = ".2f", cmap: str = "coolwarm",
    annot: bool = True, center: float | None = 0,
):
    """Annotated heatmap for correlation matrices (center=0) or crosstabs (center=None)."""
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.heatmap(matrix, annot=annot, fmt=fmt, cmap=cmap, ax=ax, center=center)
    ax.set_title(title)
    return fig, ax


def confusion_heatmap(cm, *, title: str):
    """Render a 2x2 confusion matrix (rows=true, cols=pred; 0=legit, 1=phishing)."""
    names = ["legitimate", "phishing"]
    fig, ax = plt.subplots(figsize=(4.5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", center=None, cbar=False,
                xticklabels=names, yticklabels=names, ax=ax)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title(title)
    return fig, ax


def roc_curves(scored: dict, y_test, *, title: str = "ROC curves"):
    """Overlay ROC curves from {model_name: continuous_scores}."""
    fig, ax = plt.subplots(figsize=(6, 5))
    for name, scores in scored.items():
        if scores is not None:
            RocCurveDisplay.from_predictions(y_test, scores, name=name, ax=ax)
    ax.plot([0, 1], [0, 1], linestyle="--", color="grey", linewidth=1)
    ax.set_title(title)
    return fig, ax


def scatter_2d(
    coords, hue, *, title: str, hue_name: str = "group", sample: int = 5000, seed: int = 0,
):
    """2-D scatter of reduced features, colored by a categorical (corpus/label).

    Subsamples for legibility; used to show how separable the corpora are.
    """
    rng = np.random.default_rng(seed)
    n = coords.shape[0]
    idx = rng.choice(n, size=min(sample, n), replace=False)
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.scatterplot(
        x=coords[idx, 0], y=coords[idx, 1], hue=np.asarray(hue)[idx],
        s=10, alpha=0.5, linewidth=0, ax=ax,
    )
    ax.set_xlabel("component 1")
    ax.set_ylabel("component 2")
    ax.set_title(title)
    ax.legend(title=hue_name, markerscale=2, fontsize=8)
    return fig, ax
