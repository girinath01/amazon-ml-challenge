# Member 1 Handoff Document: Preprocessing & Normalization Layer
**Team Akatsuki — Amazon ML Challenge 2026 (Business Entity Resolution)**
**Author / Responsibility:** Member 1 (Faizur Rahman — Data Ingestion & Preprocessing)
**Pipeline Execution Timestamp:** 2026-09-25 13:02:00
**Status:** ALL 16 INTEGRITY CHECKS PASSED (100% Zero Data Loss)

---

## 1. Executive Summary & Architecture Overview

Member 1 is responsible for the foundational transformation:
```
RAW DATA
  -> NAME NORMALIZATION
  -> ADDRESS NORMALIZATION
  -> TRANSLITERATION
  -> COUNTRY NORMALIZATION
  -> INTEGRATED PREPROCESSING
  -> DATA INTEGRITY VALIDATION
  -> NORMALIZATION REPORTS
  -> MEMBER 2 & 3 HANDOFF
```

### High-Throughput Performance Metrics
* **Total Rows Processed in Benchmark Run:** 400,000
* **Total Execution Time:** 32.81 seconds
* **Throughput:** **12,192 rows/second**
* **Memory Management:** Vectorized operations with streaming chunk iterator (`chunksize=50,000`).
* **Idempotency & Determinism:** 100% verified. Re-running the pipeline on identical inputs produces bitwise identical data.

---

## 2. Complete Data Dictionary (Normalized Schema)

Every output file preserves **all original raw columns** and appends **17 standardized normalized fields**:

| Column Name | Category | Type | Description | Example Raw -> Processed |
|:---|:---|:---|:---|:---|
| `entity_id` | **Raw Identity** | `str` | Unique entity identifier (strictly preserved) | `S1-925783039` |
| `business_name` | **Raw Name** | `str` | Original business name from input TSV | `Orelee's Barbershop & Salon Inc.` |
| `business_address` | **Raw Address** | `str` | Original address from input TSV | `1795 Westchester Rd, Ste 4B` |
| `country` | **Raw Country** | `str` | Original country string from input TSV | `US` |
| `name_raw` | **Name** | `str` | Clean copy of original business name string | `Orelee's Barbershop & Salon Inc.` |
| `name_norm` | **Name** | `str` | NFKC-normalized, lowercase, cleaned punctuation, canonicalized legal forms | `orelees barbershop and salon inc` |
| `name_core` | **Name** | `str` | Core business name with legal entity forms stripped | `orelees barbershop and salon` |
| `name_tokens` | **Name** | `list[str]` | Whitespace-tokenized list of normalized words | `['orelees', 'barbershop', 'and', 'salon', 'inc']` |
| `name_translit` | **Name** | `str` | Pure ASCII Latin transliteration (handles Devanagari, Tamil, etc.) | `ram marketing pvt ltd` |
| `name_has_digits` | **Name** | `bool` | Flag indicating digits present in business name | `False` |
| `name_length` | **Name** | `int` | Character length of `name_norm` | `32` |
| `address_raw` | **Address** | `str` | Clean copy of original address string (`""` if null) | `1795 Westchester Rd, Ste 4B` |
| `address_norm` | **Address** | `str` | Standardized address with abbreviations expanded (`rd`->`road`, `ste`->`suite`) | `1795 westchester road suite 4b` |
| `address_tokens` | **Address** | `list[str]` | Whitespace-tokenized list of normalized address words | `['1795', 'westchester', 'road', 'suite', '4b']` |
| `address_numbers` | **Address** | `list[str]` | All numeric sequences extracted from address | `['1795', '4']` |
| `house_number` | **Address** | `str` | Conservatively extracted street/house/building number | `1795` |
| `postal_code` | **Address** | `str` | Extracted 5-digit ZIP, 6-digit Indian PIN, or French postal code | `27262` |
| `address_has_digits` | **Address** | `bool` | Flag indicating digits present in raw address | `True` |
| `address_missing` | **Address** | `int` | Binary indicator (`1` if raw address is missing/null/empty, `0` otherwise) | `0` |
| `address_length` | **Address** | `int` | Character length of `address_norm` | `30` |
| `country_norm` | **Country** | `str` | Canonical ISO country representation (`US`, `India`, `France`, open-set title case) | `US` |

---

## 3. Dataset-by-Dataset Integrity Audit

| Metric | Source 1 (Reference) | Source 2 (Secondary) | Source 3 (Tertiary) |
|:---|:---|:---|:---|
| **Rows Audited** | 100,000 | 100,000 | 100,000 |
| **Throughput (rows/sec)** | 15,596 | 14,908 | 16,177 |
| **Row Count Preserved** | 100% MATCH | 100% MATCH | 100% MATCH |
| **ID Referential Integrity** | 100% EXACT ORDER | 100% EXACT ORDER | 100% EXACT ORDER |
| **Duplicate IDs Introduced** | **0** | **0** | **0** |
| **Missing Business Names** | **0 (0.00%)** | **0 (0.00%)** | **0 (0.00%)** |
| **Missing Addresses** | **0 (0.00%)** | **3,330 (3.33%)** | **3,352 (3.35%)** |
| **House Numbers Extracted** | 93,961 | 87,304 | 87,203 |
| **Postal Codes Extracted** | 695 | 642 | 809 |
| **Devanagari / Translit Names** | 0 | 15,061 | 11,664 |
| **Validation Verdict** | **PASS** | **PASS** | **PASS** |

---

## 4. Key Findings & Guidelines for Member 2 & Member 3

### Guidance for Member 2 (EDA & Feature Engineering):
1. **Use `name_core` for Token Jaccard / Levenshtein:** Legal suffixes (`Inc`, `Corp`, `LLC`, `Pvt Ltd`) dominate names. Comparing `name_core` rather than `name_norm` increases distinguishing power by eliminating false common-token overlaps.
2. **Use `name_translit` for Cross-Script Distance:** When comparing Indic script records against English script records, compute phonetic/edit distance between `name_translit` fields.
3. **Use `address_missing` as an Interaction Feature:** Do not impute missing addresses with synthetic text. Member 2 should compute an interaction term: `has_address_match = (1 - S1_addr_missing) * (1 - S2_addr_missing) * sim(addr1, addr2)`.
4. **House Number & Postal Code Exact Matching:** When `house_number` and `postal_code` are non-empty, an exact match provides extremely high precision signals for pairwise classification.

### Guidance for Member 3 (Candidate Blocking & Ground Truth Pairing):
1. **Country is a Hard Blocking Partition (100% Strict Boundary):** Zero cross-country matches exist in Ground Truth. Member 3 should block strictly on `country_norm`.
2. **First-Token Core Name Blocking:** Block on the first 4 characters of `name_core` or `name_translit`.
3. **Postal Code Blocking:** Within the same country, block candidates sharing identical non-empty `postal_code`.

---

## 5. Deliverables & Output File Locations

The following generated files are available for downstream usage:

1. **Normalized Source Tables:**
   * Source 1: `output/preprocessed/s1_normalized.tsv`
   * Source 2: `output/preprocessed/s2_normalized.tsv`
   * Source 3: `output/preprocessed/s3_normalized.tsv`
   * Test Sample: `output/preprocessed/s1_test_sample_normalized.tsv`

2. **Audit Reports:**
   * Representative Examples: `reports/normalization_examples.csv`
   * Full Dataset Statistics: `reports/normalization_statistics.json`
   * Difficult & Edge Cases: `reports/normalization_difficult_cases.csv`
   * Country Diagnostics: `reports/country_diagnostics.csv`
   * Unit Test Suite: `tests/test_preprocessing.py` (26 tests passing)

---

## 6. How to Reproduce Preprocessing

To re-run the complete pipeline and regenerate all reports:
```powershell
# Run the pipeline runner
py src/preprocessing/run_pipeline.py

# Run the test suite
py -m unittest tests/test_preprocessing.py
```
