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
_(Filled in after the analysis — see Phase 4.)_

## License
See [LICENSE](LICENSE).
