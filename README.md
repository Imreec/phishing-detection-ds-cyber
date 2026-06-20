# Phishing Detection — A Critical Evaluation of the "99% Accuracy" Claim

> Final project for *Data Science Methods in Cyber Security* (University of Haifa, Dr. Uri Itai).

## Project description
This project critically evaluates a published phishing-email-detection result that
reports **99.1% accuracy** using TF-IDF + a Linear SVM on a merged multi-corpus
dataset. We reproduce the result and then test a single question:

> **Does the reported 99% reflect generalizable phishing detection, or only
> in-distribution performance on this particular pool of corpora?**

We reproduce the headline result exactly, then show that it does **not generalize**:
under a leave-one-corpus-out protocol (testing on an email source unseen in
training — the realistic deployment scenario), performance degrades sharply, with
recall on an unseen phishing corpus collapsing to ~0.47. We trace this to the model
relying on **source-, era-, and campaign-specific artifacts** rather than
generalizable phishing semantics, and we re-evaluate honestly with imbalance-aware
metrics (F1, MCC, F-beta, ROC-AUC) and realistic class prevalence. Our verdict: the
claim is reproducible, but the conclusion of real-world efficacy is overstated.

## Links
- **Source under evaluation:** Al-Subaiey et al., *"Novel Interpretable and Robust
  Web-based AI Platform for Phishing Email Detection"* — https://arxiv.org/abs/2405.11619
- **Original implementation:** the paper publishes no training-code repository; it
  provides the dataset and a described TF-IDF + SVM pipeline (reproducibility is
  analyzed in the report).
- **Dataset source:** "Phishing Email Dataset" (Phish No More), Kaggle —
  https://www.kaggle.com/datasets/naserabdullahalam/phishing-email-dataset

## Repository contents
- `report/report.pdf` — full written report (8 sections).
- `notebooks/` — the complete, executable, documented notebook.
- `src/` — supporting modules (data, features, models, evaluation, plots).
- `requirements.txt` — pinned dependencies.

## Execution instructions
Requires Python 3.11.

```bash
# 1. Environment
python -m venv .venv
source .venv/Scripts/activate        # Windows (Git Bash);  use .venv/bin/activate on macOS/Linux
pip install -r requirements.txt

# 2. Get the data (needs a free Kaggle account / API token)
python -c "from src.data import download_data; download_data()"
#    Manual fallback: download the dataset from the Kaggle link above and unzip
#    its CSVs into data/raw/.

# 3. Run the analysis
jupyter notebook notebooks/
```

## Key findings
- **The 99% claim reproduces** almost exactly (TF-IDF + LinearSVC: accuracy 0.9905, F1 0.9909),
  and even the metrics the source omitted are strong *in-distribution* (MCC 0.981, ROC-AUC 0.999),
  consistently across four model families.
- **But it does not generalize.** Under leave-one-corpus-out, MCC collapses from ~0.98 to
  **0.56–0.86** on unseen mixed corpora, and **recall on the unseen Nazario corpus falls to
  0.26–0.48** — a "99%" detector missing half or more of phishing from a new source. The
  collapse is **model-agnostic**.
- **The model learned source identity, not phishing.** A classifier predicts the corpus from
  text at **95.8% accuracy**, and the model's "legitimate" tokens are Enron/business artifacts.
- **Temporal drift** (within CEAS): MCC 0.66 (past→future) vs 0.995 (random split).
- **Realistic prevalence:** at 5% phishing, cross-corpus precision drops to **0.14**.
- **Verdict:** the claim is reproducible, but the conclusion of real-world efficacy is overstated.

See [`report/report.pdf`](report/) for the full analysis and [`docs/KNOWN_LIMITATIONS.md`](docs/KNOWN_LIMITATIONS.md) for scoping caveats.

## Rebuilding the report (optional)
The report PDF is committed at [`report/report.pdf`](report/report.pdf). To regenerate it from
`report/report.md` you need [Pandoc](https://pandoc.org) and any LaTeX engine (e.g. Tectonic,
MiKTeX, or TeX Live):

```bash
pandoc report/report.md -o report/report.pdf --resource-path=report --pdf-engine=<engine>
```

## License
See [LICENSE](LICENSE).
