# Requirements Checklist

Traceability of every assignment requirement (from the course project brief) to where
it is satisfied in this repository. Maintained as work lands; a ticked box means the
item exists in the tree, not that it is planned.

Legend: ✅ done · 🔄 in progress · ⬜ pending · location = report section / notebook section / file.

---

## Source selection criteria
| Requirement | Status | Where |
|---|---|---|
| Topic within the allowed list (Phishing Detection) | ✅ | Source: arXiv:2405.11619 |
| Source clearly defines a problem | ✅ | Report §1 |
| Source proposes a solution | ✅ | Report §1 |
| Includes an implementation / repository | 🔄 | Described pipeline + dataset; reproduced by us; see Report §4 |
| Provides data / enough info to reproduce | ✅ | Kaggle dataset; Report §4 |

## PDF report — required sections
| # | Section | Status | Where |
|---|---|---|---|
| 1 | Summary of the Source (problem, importance, solution, dataset, methodology) | ✅ | `report/report.md` §1 |
| 2 | Critical Evaluation (claims, evidence, methodology, weaknesses, conclusions) | ✅ | `report/report.md` §2 |
| 3 | Feature Engineering Analysis (transforms, redundancy, meaningfulness, extra features) | ✅ | `report/report.md` §3 |
| 4 | Reproducibility Analysis (runs? deps? hidden preprocessing? overall) | ✅ | `report/report.md` §4 |
| 5 | Experimental Results (experiments, modifications, models, metrics, results) | ✅ | `report/report.md` §5 |
| 6 | Conclusions (findings, lessons, strengths/weaknesses, future work) | ✅ | `report/report.md` §6 |
| 7 | Executive Summary (~1 page) | ✅ | `report/report.md` §7 |
| 8 | Summing It Up (problem, source, dataset, methodology, findings, claims held?, insights, recommendation, conclusion) | ✅ | `report/report.md` §8 |
| — | Written in English, PDF | ✅ | `report/report.pdf` (14 pp, Pandoc+Tectonic) |

## Python notebook — required content
| # | Section | Status | Where |
|---|---|---|---|
| 1 | Data Loading (load, inspect, size, types, temporal, missing, index/column sanity, single-value/duplicate features) | ✅ | notebook §1 |
| 2 | EDA (distributions, missing, outliers, temporal, crosstab/group-by, **justified correlation**, class imbalance/prevalence, visualizations) | ✅ | notebook §2 |
| 3 | Feature Engineering (encoding+why, scaling, creation, selection, dim. reduction) | ✅ | notebook §3 |
| 4 | Model Training (≥2 models — we train 4) | ✅ | notebook §4 |
| 5 | Evaluation (every metric: math definition + cyber interpretation; justify chosen/excluded) | 🔄 | notebook §5 (suite done; full math in report §5) |
| 6 | Error Analysis (failures, error patterns, cyber implications, FP/FN tradeoff) | ✅ | notebook §7 |
| — | Complete, executable, clearly documented; runs top-to-bottom | ✅ | notebook (executes, 0 errors) |

## Code quality
| Requirement | Status | Where |
|---|---|---|
| Short, focused functions | ✅ | `src/` |
| Meaningful variable names | ✅ | `src/` |
| No unnecessary loops; proper pandas/numpy/sklearn | ✅ | `src/` (vectorized) |
| Clear separation: preprocessing / EDA / training / evaluation | ✅ | `src/{data,features,models,evaluate,critique}.py` |
| English comments; no duplicated code | ✅ | `src/` |
| Fixed random seeds | ✅ | `src/config.py` (RANDOM_SEED=42) |
| Train/test split or cross-validation | ✅ | `src/data.split_data` (stratified) |

## Submission requirements
| Requirement | Status | Where |
|---|---|---|
| Public GitHub repo | ✅ | github.com/Imreec/phishing-detection-ds-cyber |
| PDF report | ✅ | `report/report.pdf` |
| Python notebook | ✅ | `notebooks/phishing_critique.ipynb` |
| Supporting code files | ✅ | `src/` |
| README: description | ✅ | `README.md` |
| README: link to source | ✅ | `README.md` |
| README: link to original repo | ✅ | `README.md` (no code repo — noted) |
| README: execution instructions | ✅ | `README.md` |
| README: dataset source | ✅ | `README.md` |
| Repo link emailed to examiner | ⬜ | at submission |

## Grading rubric coverage (100 pts)
| Component | Pts | Primary evidence |
|---|---|---|
| Problem Understanding & Source Selection | 10 | Report §1; source vetting |
| Summary Quality | 15 | Report §1, §7, §8 |
| Critical Evaluation of the Author's Claims | 20 | Report §2; notebook critique battery (Exp 1–6) |
| Feature Engineering Analysis | 10 | Report §3; notebook §3 |
| Exploratory Data Analysis (EDA) | 15 | notebook §2 |
| Model Training and Comparison | 15 | notebook §4 (4 models) |
| Evaluation and Error Analysis | 10 | notebook §5–6; Report §5 |
| Code Quality and Software Engineering | 5 | `src/`, this checklist, tests |
