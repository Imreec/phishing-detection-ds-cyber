"""Central configuration: paths, random seed, dataset layout, and model hyperparameters.

Importing this module is the single source of truth for everything that must stay
fixed across the notebook and the `src` package (seeds, corpus->label mapping,
TF-IDF settings). Keeping it here avoids magic numbers scattered through the code.
"""

from __future__ import annotations

from pathlib import Path

# --------------------------------------------------------------------------- #
# Reproducibility
# --------------------------------------------------------------------------- #
RANDOM_SEED: int = 42
TEST_SIZE: float = 0.2

# --------------------------------------------------------------------------- #
# Paths (all relative to the repository root, resolved from this file)
# --------------------------------------------------------------------------- #
ROOT_DIR: Path = Path(__file__).resolve().parent.parent
DATA_RAW_DIR: Path = ROOT_DIR / "data" / "raw"
DATA_PROCESSED_DIR: Path = ROOT_DIR / "data" / "processed"
FIGURES_DIR: Path = ROOT_DIR / "figures"

# --------------------------------------------------------------------------- #
# Dataset: "Phish No More" (the source paper's own dataset)
# https://www.kaggle.com/datasets/naserabdullahalam/phishing-email-dataset
# Six per-corpus CSVs are merged into one frame. The `corpus` tag we attach is
# the linchpin of the whole critique (it lets us test corpus-vs-label leakage).
# --------------------------------------------------------------------------- #
KAGGLE_DATASET: str = "naserabdullahalam/phishing-email-dataset"

# Canonical corpus -> expected filename. `data.load_corpora` resolves these
# case-insensitively by substring, so minor filename variants still match.
CORPUS_FILES: dict[str, str] = {
    "Enron": "Enron.csv",
    "Ling": "Ling.csv",
    "CEAS": "CEAS_08.csv",
    "Nazario": "Nazario.csv",
    "Nigerian_Fraud": "Nigerian_Fraud.csv",
    "SpamAssassin": "SpamAssasin.csv",  # note: dataset ships this spelling
}

# Ground-truth provenance of each corpus. This mapping is exactly what makes the
# dataset confounded: the *source* of an email is almost perfectly predictive of
# its *label*, so a classifier can win by detecting the corpus, not the phishing.
LEGIT_CORPORA: frozenset[str] = frozenset({"Enron", "Ling"})
PHISH_CORPORA: frozenset[str] = frozenset(
    {"CEAS", "Nazario", "Nigerian_Fraud", "SpamAssassin"}
)

# Column names in the merged frame (kept after harmonization).
TEXT_COL: str = "text"
LABEL_COL: str = "label"          # 1 = phishing, 0 = legitimate
CORPUS_COL: str = "corpus"

# --------------------------------------------------------------------------- #
# Feature extraction (matches the source paper's TF-IDF setup)
# --------------------------------------------------------------------------- #
TFIDF_PARAMS: dict = {
    "lowercase": True,
    "stop_words": "english",
    "max_features": 20_000,
    "ngram_range": (1, 1),
    "sublinear_tf": True,
}

# Real-world inbox prevalence used for the honest re-evaluation (Exp 6).
# Phishing is rare in practice; ~5% is a deliberately generous illustrative value.
REALISTIC_PHISH_PREVALENCE: float = 0.05
