# Member 1 — Data Ingestion & Data Quality Layer

This module provides the complete, production-grade data ingestion and validation framework for the **Business Entity Resolution Challenge (Amazon ML Challenge 2026)**.

---

## Deliverables in `member1/`

| File | Purpose |
|---|---|
| [`data_loader.py`](file:///D:/Akatsuki/member1/data_loader.py) | Reusable data loading module with explicit TSV parsing (`sep="\t"`), chunked streaming, and ground truth lookup structures. |
| [`data_validation.py`](file:///D:/Akatsuki/member1/data_validation.py) | High-performance, streaming validation engine checking columns, types, row counts, duplicates, missingness, prefixes, and ground truth referential integrity. |
| [`data_quality_report.csv`](file:///D:/Akatsuki/member1/data_quality_report.csv) | Machine-readable tabular summary of row counts, unique IDs, missing rates, and country breakdown across all 6 source datasets. |
| [`ground_truth_quality_report.json`](file:///D:/Akatsuki/member1/ground_truth_quality_report.json) | Detailed referential integrity audit of `train_ground_truth.tsv`, match counts, and source distributions. |
| [`validation_summary.json`](file:///D:/Akatsuki/member1/validation_summary.json) | Full structural audit summary with execution timings and readiness verdicts. |

---

## Executive Summary: Member 1 Primary Question

> **"Is the raw data clean, complete, and structurally valid enough to enter preprocessing?"**

### **Verdict: YES — with 2 critical structural observations.**

1. **Structural Validity (100% PASS):**
   - Every file adheres strictly to the required TSV schema and header naming.
   - **Zero invalid or corrupted rows** (no bad delimiter counts or line breaks).
   - **Zero duplicate entity IDs** across all files (every entity ID is globally unique within its table).
   - **Zero malformed IDs**: All IDs strictly adhere to `S1-<int>`, `S2-<int>`, and `S3-<int>` formats.
   - **Zero missing business names** across all 24.2 million records (100% complete).
   - **Zero missing country labels** across all 24.2 million records (100% complete).
   - **100% referential integrity**: Every S1 entity in `train_source1.tsv` has exactly one corresponding row in `train_ground_truth.tsv` (a perfect 1-to-1 bijection).
   - **Zero orphan match IDs**: All 7,638,365 matched entity IDs in ground truth exist in `train_source2.tsv` or `train_source3.tsv`.
   - **Zero cross-country matches**: In ground truth, matched pairs share 100% identical countries (Country is a hard blocking partition).

2. **Crucial Observations for Preprocessing & Modeling:**
   - **Missing Addresses in S2 & S3:** While Reference Source 1 has **0% missing addresses**, Sources 2 and 3 exhibit missing addresses (~3.35% in Train, ~2.65% in Test). The pipeline must implement a fallback matching mechanism (e.g. name-only / town-fallback similarity) when addresses are empty.
   - **France Appears ONLY in Test:** `France` accounts for ~15.0% of Test S1 (259,452 records) and does not exist in Train. Country-level normalization and blocking must remain zero-shot and language/format-agnostic for French addresses and entities.

---

## Data Quality Report

Tabular summary from [`data_quality_report.csv`](file:///D:/Akatsuki/member1/data_quality_report.csv):

| Dataset | Rows | Unique IDs | Missing Name | Missing Address | Missing Country | Duplicate IDs | Invalid IDs | Countries |
|---|---|---|---|---|---|---|---|---|
| **S1 Train** | 2,206,821 | 2,206,821 | 0 (0.0%) | 0 (0.0%) | 0 (0.0%) | 0 | 0 | India: 883,188, US: 1,323,633 |
| **S2 Train** | 5,034,616 | 5,034,616 | 0 (0.0%) | 168,967 (3.36%) | 0 (0.0%) | 0 | 0 | India: 2,017,799, US: 3,016,817 |
| **S3 Train** | 5,285,603 | 5,285,603 | 0 (0.0%) | 175,916 (3.33%) | 0 (0.0%) | 0 | 0 | India: 2,115,547, US: 3,170,056 |
| **S1 Test** | 1,732,544 | 1,732,544 | 0 (0.0%) | 0 (0.0%) | 0 (0.0%) | 0 | 0 | France: 259,452, India: 809,986, US: 663,106 |
| **S2 Test** | 4,887,273 | 4,887,273 | 0 (0.0%) | 129,408 (2.65%) | 0 (0.0%) | 0 | 0 | France: 703,378, India: 2,312,565, US: 1,871,330 |
| **S3 Test** | 5,082,316 | 5,082,316 | 0 (0.0%) | 136,098 (2.68%) | 0 (0.0%) | 0 | 0 | France: 731,615, India: 2,405,000, US: 1,945,701 |

---

## Ground Truth & Match Topology (Insights for Member 2 & Member 3)

From [`ground_truth_quality_report.json`](file:///D:/Akatsuki/member1/ground_truth_quality_report.json):

- **Total S1 Ground Truth Records:** 2,206,821
- **Total Matched Entity Pairs:** 7,638,365 (S2: 3,693,619; S3: 3,944,746)
- **Mean Matches per S1:** 3.46 matches

### 1. Match Cardinality Distribution (0 / 1 / Many)
- **0 Matches (Singletons):** 123,247 (5.58%)
- **1 Match:** 119,157 (5.40%)
- **2 Matches:** 375,212 (17.00%)
- **3 Matches:** 530,841 (24.05%)
- **4 Matches:** 484,115 (21.94%)
- **5 Matches:** 321,957 (14.59%)
- **6 Matches:** 164,868 (7.47%)
- **7 Matches:** 63,968 (2.90%)
- **8–11 Matches:** 23,456 (1.06%)
- **Max Matches for single S1:** 11

### 2. Source Breakdown Distribution
- **Matches in Both S2 & S3:** 1,776,047 (80.48%)
- **Matches in S3 Only:** 164,498 (7.45%)
- **Matches in S2 Only:** 143,029 (6.48%)
- **Zero Matches (Singletons):** 123,247 (5.58%)

### 3. Cluster Uniqueness (Vital Finding)
- Every matched S2 ID appears **at most 1 time** in the ground truth.
- Every matched S3 ID appears **at most 1 time** in the ground truth.
- **Conclusion:** The relationship is strictly **1-to-N**; an S2 or S3 entity can belong to **at most ONE** S1 entity. There is no many-to-many overlap across S1 entities.

---

## Quick Usage Guide

### Using `data_loader.py` in your code:

```python
from member1.data_loader import DataLoader

loader = DataLoader()

# Load reference source
s1_train = loader.load_source("s1", split="train", nrows=10000)

# Memory-efficient chunked loading for large files
for chunk in loader.load_source("s2", split="train", chunksize=100_000):
    # Process chunk
    pass

# Load ground truth dictionary for rapid evaluation
gt_dict = loader.load_ground_truth_dict()
# {'S1-965667': {'S2-681193310', 'S2-743505751', 'S3-775321672', ...}}
```

### Re-running validation:

```bash
python member1/data_validation.py --dataset-dir D:/Akatsuki/student_resource/dataset
```
