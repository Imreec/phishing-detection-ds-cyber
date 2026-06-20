# CLAUDE.md — Project conventions & context

> **Course:** Using Data Science Methods in Cybersecurity (University of Haifa) · **Instructor:** Dr. Uri Itai.
> **Source of truth:** the assignment brief (the project PDF). When in doubt, the PDF wins.
> Requirement coverage is tracked in [`docs/REQUIREMENTS_CHECKLIST.md`](docs/REQUIREMENTS_CHECKLIST.md);
> honest scoping caveats in [`docs/KNOWN_LIMITATIONS.md`](docs/KNOWN_LIMITATIONS.md).
> **Deadline:** 2026-07-10 (internal target ~07-07). Individual project.

## What this project is
A **critical-evaluation** project. The goal is **not** to build the best phishing
detector — it is to reproduce a published phishing-detection result and rigorously
test whether its headline claim is supported by the evidence. The grade lives in the
critique (20 pts), summary (15), EDA (15), and model comparison (15); code quality is 5.

## The source under evaluation
- **Paper:** Al-Subaiey et al., *"Novel Interpretable and Robust Web-based AI
  Platform for Phishing Email Detection"* — arXiv:2405.11619.
- **Claim:** TF-IDF + Linear SVM reaches **99.1% accuracy / 0.99 F1** on a merged
  multi-corpus email dataset. The paper reports **no MCC, no ROC-AUC, and no
  cross-corpus generalization test**.

## The dataset
"Phish No More" (Kaggle: `naserabdullahalam/phishing-email-dataset`), authored by
a co-author of the paper. It merges six corpora (82,486 emails; 42,891 phishing /
39,595 legitimate). These are the standard research spam/phishing corpora, and
**most contain BOTH classes** (verified):

| corpus | legit | phishing | has dates |
|---|---|---|---|
| Enron (Enron-Spam) | 15,791 | 13,976 | no |
| Ling (Ling-Spam) | 2,401 | 458 | no |
| CEAS-08 | 17,312 | 21,842 | yes |
| SpamAssassin | 4,091 | 1,718 | yes |
| Nazario | 0 | 1,565 | yes |
| Nigerian_Fraud | 0 | 3,332 | yes |

So `corpus` is only *moderately* predictive of `label` (Cramer's V = 0.306) — the
naive "corpus = label" story is false.

## The central thesis (what we are testing)
The reported 99% is **real but in-distribution only**. We reproduce it exactly
(TF-IDF + LinearSVC: accuracy 0.9905, F1 0.9909, MCC 0.981, ROC-AUC 0.999 on a
pooled stratified split). But the paper never tests **generalization**. Under
leave-one-corpus-out (train on the rest, test on an unseen source — the realistic
deployment case), performance degrades sharply: MCC falls to 0.62-0.75 on unseen
mixed corpora and **recall on the Nazario phishing corpus collapses to 0.47**
(a "99%" detector would miss ~53% of unseen-source phishing). The model has
learned **source-, era-, and campaign-specific artifacts**, not generalizable
phishing semantics. The authors' accuracy-only, single-pooled-split methodology
(no cross-corpus or temporal test, no MCC/ROC-AUC, no realistic prevalence)
overstates real-world efficacy.

**Verdict:** the claim is *reproducible*, but the *conclusion of real-world
efficacy is overstated*.

## Repository layout
```
src/            Focused modules imported by the notebook
  config.py     Seeds, paths, corpus->label map, TF-IDF params (single source of truth)
  data.py       Download, load per-corpus (tags `corpus`), build merged frame
  features.py   TF-IDF + engineered features
  models.py     Train the four models under one protocol
  evaluate.py   Unified metric suite (acc, P, R, F1, Fbeta, MCC, ROC-AUC, confusion)
  plots.py      Reusable plotting helpers
notebooks/      The single narrative notebook (runs top-to-bottom from repo root)
report/         report.md -> report.pdf (Pandoc)
data/raw/       Downloaded corpora (git-ignored)
data/processed/ Cached merged/processed frames (git-ignored)
figures/        Exported figures used by the report
```

## Models
TF-IDF + **LinearSVC** (reproduction anchor) · **Multinomial NB** ·
**Logistic Regression** · **Random Forest** (cross-family). One stratified split,
fixed seed, one shared evaluation protocol.

## Critique experiments (centerpiece = Exp 4)
1. Cramer's V between `corpus` and `label` (= 0.306) — honestly retire the naive
   "corpus = label" confound and motivate the generalization question
2. Provenance classifier — predict the corpus from text; explains *why* cross-corpus
   generalization fails (sources are stylistically separable)
3. Top-token autopsy — top weights are source/era/campaign artifacts (enron/hou/ect),
   not phishing semantics
4. **Leave-one-corpus-out across all four models (model-agnostic collapse)** — the
   centerpiece evidence
5. Temporal split — clean within-corpus split on CEAS (dated, both classes); discuss
   the ham-has-no-dates era confound honestly
6. Honest re-evaluation: in-distribution honest metrics are good; realistic-prevalence
   precision + cross-corpus metrics tell the real story

## Conventions
- Python 3.11, `venv` + pinned `requirements.txt`.
- All randomness goes through `config.RANDOM_SEED`.
- `src` functions: short, single-purpose, English docstrings/comments, no duplication.
- The notebook reads as a narrative: markdown "what & why" -> a few calls -> outputs.
- Raw data is never committed; it is fetched by `src/data.download_data()`.
- **No absolute/user paths in committed files** (`C:\Users\...`, `D:\...`) — paths come
  from `config.py`, resolved relative to the repo root.

## Requirements & gates
- Every claimed deliverable must trace to a row in `docs/REQUIREMENTS_CHECKLIST.md`.
- Never report work as done unless it exists in the tree and the verifying command was
  run; show the output. Checklist boxes are ticked as work lands, never as aspiration.
- Public docs (README, report, CLAUDE.md) must never contradict the repo state — the
  doc↔repo gap is the one fatal failure mode.

## Process
- **Conventional Commits** (`feat/fix/docs/refactor/chore`), one atomic concern each,
  imperative subject. **No `Co-Authored-By` / AI-author trailer** — authorship is the
  student's, consistent with the course's own-work requirement.
- Strategic planning lives in `docs/planning/` (git-ignored). Committed docs are neutral
  and submission-appropriate.

## Anti-patterns banned
- Aspirational README/report (claims ahead of the tree).
- Leaked absolute filesystem paths in committed files.
- Treating a lecturer's "for example" value as a locked spec (e.g. the metric list,
  the ~5% prevalence figure — justify choices, don't copy blindly).
- Accuracy-only reporting (the very flaw we critique) in our own evaluation.
- Data leakage: fitting any transform on the test split; mixing preprocessing across splits.
- Overclaiming the critique beyond what the data supports (the thesis was already
  refined once when the data falsified the naive version — keep that discipline).

## How to run
```bash
python -m venv .venv && source .venv/Scripts/activate   # Windows Git Bash
pip install -r requirements.txt
python -c "from src.data import download_data; download_data()"
jupyter notebook notebooks/
```

## Key decisions log
- Source locked to arXiv:2405.11619 (canonical paper behind the dataset; verified
  claim + verified metric omissions). No public training code -> we reimplement,
  and "no code released" is itself a reproducibility finding.
- `src/` package + one narrative notebook (Code Quality rubric).
- Report authored in Markdown, rendered to PDF via Pandoc.
