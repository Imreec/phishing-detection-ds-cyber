# Known Limitations

Honest disclosure of this study's limitations and scoping decisions. A documented
limitation is a sign of rigor; an undisclosed one a reader discovers is a flaw.
Updated as the analysis proceeds.

| # | Limitation | Severity | Note / mitigation |
|---|---|---|---|
| 1 | The source paper publishes **no training code**; our reproduction is a faithful re-implementation of its described TF-IDF + Linear SVM pipeline, not a rerun of the authors' exact code. | Low | We match the reported headline (acc 0.9905 vs 0.991). Discussed in Report §4 (Reproducibility) — itself a finding. |
| 2 | **Temporal analysis is partial.** The ham-bearing corpora (Enron, Ling) ship with no `date` column, so a clean cross-class train-past/test-future split over the whole dataset is impossible. | Medium | We run a within-corpus temporal split on CEAS (dated, both classes) and discuss the era↔corpus entanglement explicitly. |
| 3 | **Leave-one-corpus-out on pure-phish corpora** (Nazario, Nigerian_Fraud) yields a single-class test set, so MCC/ROC-AUC are undefined and only recall is interpretable there. | Low | Reported as recall; mixed corpora (Enron/Ling/CEAS/SpamAssassin) carry the full-metric LOCO evidence. |
| 4 | **Realistic-prevalence figures are illustrative.** Real inbox phishing prevalence varies by population; we use ~5% as a deliberately generous reference and show a precision-vs-prevalence curve rather than a single number. | Low | Curve makes the dependence explicit. |
| 5 | **Corpus provenance is taken from the dataset's file partition.** We cannot independently verify each email's true origin beyond the dataset's own labeling. | Low | Provenance is used to study generalization, not as a ground-truth identity claim. |
| 6 | **Near-duplicate / leaked emails** across the train/test split could inflate in-distribution scores. | Low | EDA found 45 duplicated text bodies / 23 fully duplicated rows out of 82,486 (<0.06%) — too few to materially affect results, but flagged. |
| 7 | **Corrupt timestamps in CEAS-08.** Its `date` field ranges 1980–2100 (impossible for a 2008 corpus), indicating sentinel/parse errors. | Low | Flagged in EDA (§1.4) as a data-quality red flag; the temporal experiment restricts CEAS to a plausible 2007–2009 window before splitting. |
