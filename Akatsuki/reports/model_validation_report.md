# Akatsuki Business Entity Resolution — Model & Metric Validation Report

**Member 3:** ML & Evaluation Lead  
**Timestamp:** 2026-09-25 16:18:54  
**Evaluation Scope:** Out-of-Fold (OOF) 5-Fold GroupKFold Cross-Validation  

---

## Executive Summary

This report documents the independent audit and empirical validation of the LightGBM candidate matching model, GroupKFold cross-validation partitioning, threshold selection, and exact competition **Macro $F_{0.5}$** evaluation logic for the Amazon ML Challenge 2026.

### Key Performance Highlights:
- **Optimal Decision Threshold:** `0.5`
- **Overall Macro $F_{0.5}$ Score:** **`0.9980`**
- **Macro Precision:** `0.9982` (weighted $2\times$ heavily per competition rules)
- **Macro Recall:** `0.9979`
- **Singleton Accuracy:** **`99.63%`** (272/273 singletons perfectly identified)
- **Multi-Match $F_{0.5}$:** `0.9981`
- **GroupKFold Entity Leakage:** **`0.00%`** (100% verified zero $S_1$ entity leakage across folds)

---

## 1. GroupKFold Partitioning & Data Leakage Audit

### Audit Requirement:
Verify `group = source1_entity_id` during 5-fold cross-validation and guarantee zero $S_1$ entity leakage between training and validation sets.

### Audit Findings:
- **Root Cause of Baseline Flaw:** In `trainer.py`, the baseline implementation utilized standard `StratifiedKFold`. Because multiple candidate pairs share the same `source1_id`, standard stratified splits randomly distributed candidate pairs of a single $S_1$ entity across both train and test splits, causing severe **entity leakage** and overestimating performance.
- **Audited Implementation:** Upgraded cross-validation in `trainer.py` and `validate_model_eval.py` to `GroupKFold(n_splits=5)` using `groups = df['source1_id']`.
- **Empirical Verification:**
  - Fold 1 Overlap: `0` $S_1$ entities (`0.00%`)
  - Fold 2 Overlap: `0` $S_1$ entities (`0.00%`)
  - Fold 3 Overlap: `0` $S_1$ entities (`0.00%`)
  - Fold 4 Overlap: `0` $S_1$ entities (`0.00%`)
  - Fold 5 Overlap: `0` $S_1$ entities (`0.00%`)
- **Conclusion:** GroupKFold partitioning is **100% verified**. Zero $S_1$ entity leakage exists across cross-validation folds.

---

## 2. Macro $F_{0.5}$ Evaluator Audit & Singleton Rule Integration

### Metric Definition:
The competition evaluates predictions using **Macro $F_{0.5}$** computed per $S_1$ entity and averaged across all $N$ entities in the dataset:

$$F_{0.5} = \frac{(1 + 0.5^2) \cdot P \cdot R}{(0.5^2 \cdot P) + R} = \frac{1.25 \cdot P \cdot R}{0.25 \cdot P + R}$$

### Ground Truth & Singleton Handling Audit:
1. **Singletons (Ground Truth Set $T_i = \emptyset$):**
   - If predicted set $P_i = \emptyset$ (no links predicted) $\rightarrow P_i = 1.0, R_i = 1.0, F_{0.5, i} = 1.0$ (Perfect match credit).
   - If predicted set $P_i \neq \emptyset$ (false links predicted) $\rightarrow P_i = 0.0, R_i = 0.0, F_{0.5, i} = 0.0$ (Zero credit penalty).
2. **Non-Singletons (Ground Truth Set $|T_i| \ge 1$):**
   - If predicted set $P_i = \emptyset \rightarrow P_i = 0.0, R_i = 0.0, F_{0.5, i} = 0.0$.
   - If predicted set $P_i \neq \emptyset \rightarrow P_i = \frac{|T_i \cap P_i|}{|P_i|}, R_i = \frac{|T_i \cap P_i|}{|T_i|}, F_{0.5, i} = \text{F\_beta}(P_i, R_i, \beta=0.5)$.

---

## 3. Threshold Grid Sweep Results

The model probability output was evaluated across decision thresholds from `0.50` to `0.95`. Precision weighting ($2\times$) strongly favors higher thresholds (`0.65`–`0.80`) to minimize False Positives.

### Global Threshold Sweep Grid:
| Global_Threshold | Macro F0.5 | Macro Precision | Macro Recall | Singleton Score | Multi-Match F0.5 | Total Predicted Links |
| --- | --- | --- | --- | --- | --- | --- |
| 0.5 | 0.998 | 0.9982 | 0.9979 | 0.9963 | 0.9981 | 17242.0 |
| 0.55 | 0.9977 | 0.998 | 0.9976 | 0.9963 | 0.9978 | 17237.0 |
| 0.6 | 0.9977 | 0.998 | 0.9973 | 0.9963 | 0.9977 | 17232.0 |
| 0.65 | 0.9977 | 0.9981 | 0.9971 | 0.9963 | 0.9978 | 17226.0 |
| 0.7 | 0.9976 | 0.9982 | 0.9966 | 0.9963 | 0.9977 | 17214.0 |
| 0.75 | 0.9976 | 0.9983 | 0.9963 | 0.9963 | 0.9977 | 17208.0 |
| 0.8 | 0.9975 | 0.9983 | 0.996 | 0.9963 | 0.9976 | 17203.0 |
| 0.85 | 0.9974 | 0.9984 | 0.9952 | 0.9963 | 0.9974 | 17185.0 |
| 0.9 | 0.9973 | 0.9988 | 0.9941 | 1.0 | 0.9972 | 17160.0 |
| 0.95 | 0.9965 | 0.9986 | 0.9914 | 1.0 | 0.9963 | 17108.0 |

### Source-Specific Threshold Analysis ($S_2$ vs $S_3$):
- **Optimal $S_2$ Threshold:** `0.50`
- **Optimal $S_3$ Threshold:** `0.50`
- **Source-Split Macro $F_{0.5}$:** `0.9980`

---

## 4. Multi-Match Cardinality & Singleton Performance Breakdown

Performance across true match cardinalities (from 0 singletons up to 11 matches per entity):

| Match_Cardinality | Entity_Count | Percentage_of_Total | Mean_Precision | Mean_Recall | Mean_F0.5 | Exact_Match_Pct |
| --- | --- | --- | --- | --- | --- | --- |
| 0.0 | 273.0 | 5.46 | 0.9963 | 0.9963 | 0.9963 | 99.63 |
| 1.0 | 255.0 | 5.1 | 0.9941 | 0.9961 | 0.9943 | 99.22 |
| 2.0 | 885.0 | 17.7 | 0.9966 | 0.9972 | 0.9962 | 98.42 |
| 3.0 | 1242.0 | 24.84 | 0.9978 | 0.9979 | 0.9975 | 98.47 |
| 4.0 | 1066.0 | 21.32 | 0.9996 | 0.9972 | 0.999 | 98.69 |
| 5.0 | 705.0 | 14.1 | 0.9993 | 0.9966 | 0.9986 | 98.01 |
| 6.0 | 372.0 | 7.44 | 0.9996 | 0.9964 | 0.9989 | 97.58 |
| 7.0 | 152.0 | 3.04 | 1.0 | 0.9953 | 0.9989 | 96.71 |
| 8.0 | 37.0 | 0.74 | 1.0 | 0.9966 | 0.9992 | 97.3 |
| 9.0 | 12.0 | 0.24 | 1.0 | 1.0 | 1.0 | 100.0 |
| 10.0 | 1.0 | 0.02 | 1.0 | 1.0 | 1.0 | 100.0 |

### Singleton Deep Dive:
- Total Singletons in Validation Sample: `273`
- Correctly Predicted Singletons ($P_i = \emptyset$): `272`
- Misclassified Singletons (False Links): `1`
- Singleton Precision / Accuracy: **`99.63%`**

---

## 5. Error Analysis & Key Failure Modes

Out of 25,104 out-of-fold candidate predictions, the model generated `80` total errors:
- **False Positives (FP):** `28` pairs (model predicted match, ground truth was non-match)
- **False Negatives (FN):** `52` pairs (model predicted non-match, ground truth was true match)

### Primary Root Causes of Errors:
1. **False Positives:** Driven by shared generic business suffixes (e.g. "LLC", "INC", "CORP") combined with high address Jaccard similarity in multi-tenant commercial buildings.
2. **False Negatives:** Driven by severe name transliteration differences, missing street numbers, or abbreviated company acronyms where standard string metrics yield low similarity.

---

## 6. Verification & Deliverables Manifest

All 5 required validation deliverables have been generated, verified, and exported:
1. `Akatsuki/reports/oof_predictions.parquet` — Full out-of-fold predictions with fold & probability
2. `Akatsuki/reports/threshold_results.csv` — Full threshold sweep grid results
3. `Akatsuki/reports/model_validation_report.md` — This executive validation report
4. `Akatsuki/reports/singleton_analysis.csv` — Performance breakdown by match cardinality & singletons
5. `Akatsuki/reports/error_analysis.csv` — Extracted false positive and false negative error pairs

---
*Report compiled by Member 3 (ML & Evaluation Lead), Akatsuki Team.*
