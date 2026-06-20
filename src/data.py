"""Data acquisition and loading for the Phish No More dataset.

Responsibilities:
  * download_data()  - fetch the Kaggle dataset into data/raw/ (or use a manual copy)
  * load_corpora()   - load each per-corpus CSV, tagged with its `corpus` of origin
  * build_merged()   - concatenate the corpora into one analysis frame

The `corpus` tag attached here is what the entire critique depends on: it lets us
test whether the label is predictable from the *source corpus* alone.

The loaders are schema-flexible (columns are resolved by candidate names), because
the per-corpus CSVs in this dataset do not all share an identical header.
"""

from __future__ import annotations

import shutil
import warnings
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from . import config

# --------------------------------------------------------------------------- #
# Column-resolution helpers
# --------------------------------------------------------------------------- #
_TEXT_CANDIDATES = ["body", "text", "text_combined", "email text", "message", "content"]
_SUBJECT_CANDIDATES = ["subject"]
_DATE_CANDIDATES = ["date", "datetime", "timestamp", "time"]
_LABEL_CANDIDATES = ["label", "class", "target", "type", "email type", "spam", "category"]

# Strings that indicate the positive (phishing/spam) class, lower-cased.
_PHISH_TOKENS = {"1", "phishing", "phish", "spam", "phishing email", "fraud", "malicious"}
_LEGIT_TOKENS = {"0", "legitimate", "legit", "ham", "safe", "safe email", "benign", "normal"}


def _find_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
    """Return the first column whose lower-cased name matches a candidate."""
    lookup = {col.lower().strip(): col for col in df.columns}
    for name in candidates:
        if name in lookup:
            return lookup[name]
    return None


def _normalize_label(series: pd.Series) -> pd.Series:
    """Map heterogeneous label values to 1 (phishing) / 0 (legitimate)."""
    if pd.api.types.is_numeric_dtype(series):
        return series.astype(int)

    def to_binary(value: object) -> int | float:
        token = str(value).strip().lower()
        if token in _PHISH_TOKENS:
            return 1
        if token in _LEGIT_TOKENS:
            return 0
        return float("nan")

    return series.map(to_binary)


def _build_text(df: pd.DataFrame) -> pd.Series:
    """Construct the model input text: 'subject + body' when both exist."""
    subject_col = _find_column(df, _SUBJECT_CANDIDATES)
    text_col = _find_column(df, _TEXT_CANDIDATES)
    if text_col is None and subject_col is None:
        raise ValueError(f"No text-like column found among {list(df.columns)}")

    body = df[text_col].fillna("") if text_col else ""
    subject = df[subject_col].fillna("") if subject_col else ""
    combined = (subject.astype(str) + " " + body.astype(str)) if subject_col else body.astype(str)
    return combined.str.strip()


# --------------------------------------------------------------------------- #
# Download
# --------------------------------------------------------------------------- #
def download_data(force: bool = False) -> Path:
    """Download the Kaggle dataset into data/raw/ and return that directory.

    Needs Kaggle API credentials (a kaggle.json token, or KAGGLE_USERNAME /
    KAGGLE_KEY env vars). Manual fallback: download the dataset from
    https://www.kaggle.com/datasets/naserabdullahalam/phishing-email-dataset
    and unzip its CSVs into data/raw/.
    """
    config.DATA_RAW_DIR.mkdir(parents=True, exist_ok=True)
    existing = list(config.DATA_RAW_DIR.glob("*.csv"))
    if existing and not force:
        print(f"Found {len(existing)} CSV(s) in {config.DATA_RAW_DIR}; skipping download.")
        return config.DATA_RAW_DIR

    import kagglehub  # imported lazily so the package is optional for graders

    cache_path = Path(kagglehub.dataset_download(config.KAGGLE_DATASET))
    for csv in cache_path.rglob("*.csv"):
        shutil.copy2(csv, config.DATA_RAW_DIR / csv.name)
    print(f"Downloaded dataset to {config.DATA_RAW_DIR}")
    return config.DATA_RAW_DIR


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
def _resolve_corpus_file(corpus: str, filename: str) -> Path | None:
    """Locate a corpus CSV in data/raw/, tolerating filename variants."""
    exact = config.DATA_RAW_DIR / filename
    if exact.exists():
        return exact
    stem = filename.split(".")[0].lower()
    for path in config.DATA_RAW_DIR.glob("*.csv"):
        if stem in path.stem.lower() or corpus.lower() in path.stem.lower():
            return path
    return None


def load_corpus(corpus: str) -> pd.DataFrame:
    """Load one corpus into a frame: [text, label, corpus, date]."""
    filename = config.CORPUS_FILES[corpus]
    path = _resolve_corpus_file(corpus, filename)
    if path is None:
        raise FileNotFoundError(
            f"Corpus '{corpus}' ({filename}) not found in {config.DATA_RAW_DIR}. "
            "Run download_data() or place the CSV there manually."
        )

    raw = pd.read_csv(path, low_memory=False)
    frame = pd.DataFrame({config.TEXT_COL: _build_text(raw)})

    label_col = _find_column(raw, _LABEL_CANDIDATES)
    if label_col is not None:
        frame[config.LABEL_COL] = _normalize_label(raw[label_col])
    else:
        # Fall back to the corpus's known provenance.
        frame[config.LABEL_COL] = 1 if corpus in config.PHISH_CORPORA else 0

    date_col = _find_column(raw, _DATE_CANDIDATES)
    frame["date"] = (
        pd.to_datetime(raw[date_col], errors="coerce", utc=True) if date_col else pd.NaT
    )
    frame[config.CORPUS_COL] = corpus
    return frame


def load_corpora() -> dict[str, pd.DataFrame]:
    """Load every configured corpus, returning {corpus_name: frame}."""
    return {corpus: load_corpus(corpus) for corpus in config.CORPUS_FILES}


def build_merged(cache: bool = True) -> pd.DataFrame:
    """Concatenate all corpora into one analysis frame (optionally cached)."""
    cache_path = config.DATA_PROCESSED_DIR / "merged.pkl"
    if cache and cache_path.exists():
        return pd.read_pickle(cache_path)

    frames = list(load_corpora().values())
    merged = pd.concat(frames, ignore_index=True)

    # Basic hygiene: drop empty texts and unresolved labels.
    before = len(merged)
    merged = merged[merged[config.TEXT_COL].str.len() > 0]
    merged = merged.dropna(subset=[config.LABEL_COL])
    merged[config.LABEL_COL] = merged[config.LABEL_COL].astype(int)
    if len(merged) < before:
        warnings.warn(f"Dropped {before - len(merged)} rows (empty text or NaN label).")

    if cache:
        config.DATA_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
        merged.to_pickle(cache_path)
    return merged.reset_index(drop=True)


def split_data(
    df: pd.DataFrame,
    test_size: float = config.TEST_SIZE,
    seed: int = config.RANDOM_SEED,
    stratify_on: str = config.LABEL_COL,
):
    """Stratified train/test split returning (train_df, test_df).

    Stratifying on the label keeps class proportions stable across splits; the
    fixed seed makes the partition reproducible.
    """
    return train_test_split(
        df,
        test_size=test_size,
        random_state=seed,
        stratify=df[stratify_on],
    )


def inspect_raw() -> None:
    """Print the schema and head of every CSV in data/raw/ (used during setup)."""
    for path in sorted(config.DATA_RAW_DIR.glob("*.csv")):
        df = pd.read_csv(path, nrows=3, low_memory=False)
        print(f"\n=== {path.name} ({path.stat().st_size / 1e6:.1f} MB) ===")
        print(f"columns: {list(df.columns)}")
        print(df.head(2).to_string())
