# Step 1 Final Report — Team Akatsuki
## Amazon ML Challenge 2026 · Business Entity Resolution

> **Purpose:** This document consolidates all findings from Member 1 (Data Validation), Member 2 (EDA), and Member 3 (Ground Truth Analysis) into a single Step 1 completion report. Every number is sourced directly from executed code against the actual dataset.

---

## ✅ Step 1 Completion Checklist

| Criterion | Status | Source |
|---|---|---|
| S1/S2/S3 row counts | ✅ Complete | Member 1 |
| Missing-value percentages | ✅ Complete | Member 1 & 2 |
| Duplicate statistics | ✅ Complete | Member 1 |
| Country distribution | ✅ Complete | Member 1 & 2 |
| Name statistics | ✅ Complete | Member 2 |
| Address statistics | ✅ Complete | Member 2 |
| 0/1/many match distribution | ✅ Complete | Member 1 & 3 |
| S2-only / S3-only / both distribution | ✅ Complete | Member 1 & 3 |
| Examples of true matches | ✅ Complete | Member 3 |
| Examples of difficult matches | ✅ Complete | Member 3 |
| Initial conclusions for normalization + blocking | ✅ Complete | All Members |

---

## Section 1 — Dataset Row Counts

| Source | Train Records | Test Records |
|---|---|---|
| Source 1 (Reference) | **2,206,821** | **1,732,544** |
| Source 2 | **5,034,616** | **4,887,273** |
| Source 3 | **5,285,603** | **5,082,316** |
| **Total** | **12,527,040** | **11,702,133** |

- All entity IDs are **100% unique** within every source file — zero duplicates across all 6 files.
- All IDs conform strictly to the `S1-<int>`, `S2-<int>`, `S3-<int>` format — zero malformed IDs.

---

## Section 2 — Data Quality & Missing Values

| Source | Missing Name | Missing Address | Missing Country |
|---|---|---|---|
| S1 Train | 0 (0.00%) | **0 (0.00%)** | 0 (0.00%) |
| S2 Train | 0 (0.00%) | **168,967 (3.36%)** | 0 (0.00%) |
| S3 Train | 0 (0.00%) | **175,916 (3.33%)** | 0 (0.00%) |
| S1 Test  | 0 (0.00%) | **0 (0.00%)** | 0 (0.00%) |
| S2 Test  | 0 (0.00%) | **129,408 (2.65%)** | 0 (0.00%) |
| S3 Test  | 0 (0.00%) | **136,098 (2.68%)** | 0 (0.00%) |

**Key findings:**
- Business names and country labels are **100% complete** across all 24.2 million records.
- Source 1 (the reference source) has **0% missing addresses** in both train and test.
- Sources 2 and 3 have ~2.65–3.36% missing addresses — a **fallback matching mechanism** (name-only similarity) is required for these records.
- There are **zero duplicate entity IDs** and **zero invalid records** in any file.

---

## Section 3 — Name Characteristics

| Metric | S1 Train | S2 Train | S3 Train | S1 Test |
|---|---|---|---|---|
| Record count | 2,206,821 | 5,034,614 | 5,285,590 | 1,732,544 |
| Unique names | 1,539,229 | 4,402,008 | 4,651,608 | 1,238,867 |
| Duplicate name % | 30.25% | 12.57% | 11.99% | 28.49% |
| Average length | 24.03 | 25.10 | 25.20 | 23.84 |
| Median length | 24 | 25 | 25 | 24 |
| Min length | 3 | 2 | 2 | 3 |
| Max length | 105 | 104 | 123 | 92 |

**Legal suffix noise patterns (across all train sources combined):**

| Pattern | Occurrence Count |
|---|---|
| Pvt / Private | **2,131,313** |
| Ltd / Limited | **2,699,409** |
| LLC | **1,455,401** |
| Inc / Incorporated | **1,106,422** |
| Corp / Corporation | **451,183** |
| Ampersand (&) | **537,910** |

**What transformations are necessary:**
- Legal suffix canonicalization is **mandatory** — "Private Limited", "Pvt. Ltd.", "Pvt Ltd", "Pvt. Limited" must all map to a single canonical form before comparison.
- Case normalization (lowercasing).
- Punctuation stripping (periods, hyphens, extra spaces).
- Script/transliteration handling — Member 3 found **Devanagari and Tamil script names** as true matches against Latin-script S1 names (see Section 8). Pure string similarity gives 0.0 on these pairs.

---

## Section 4 — Address Characteristics

| Metric | S1 Train | S2 Train | S3 Train | S1 Test |
|---|---|---|---|---|
| Average length | 52.07 | 47.83 | 48.32 | 57.21 |
| Median length | 41 | 37 | 42 | 50 |
| Min / Max length | 11 / 256 | 8 / 249 | 2 / 240 | 11 / 268 |
| Has numeric token % | **96.51%** | 93.79% | 93.93% | 95.85% |
| Has postal code (5-6 digit) % | 6.67% | 7.59% | 7.55% | 4.37% |

**Address abbreviation noise (all train sources combined):**

| Pattern | Count |
|---|---|
| "Rd" (abbreviation) | 676,918 |
| "Road" (full form) | 1,750,197 |
| "St" (abbreviation) | 681,083 |
| "Street" (full form) | 1,003,784 |

**Key findings:**
- Addresses are **highly noisy**: abbreviations, word reordering, landmark references, partial addresses, component omissions all confirmed at scale.
- Postal codes (5–6 digit) are present in only **4.37–7.59%** of records — they **cannot** be used as mandatory blocking keys.
- Numeric tokens appear in >93% of addresses — numeric overlap (house/plot numbers) is a strong confirming feature, especially when names diverge.
- Member 3 found cases where **names are completely different but the address is identical** (e.g., DBA names, domain-name style aliases like `precisionfrontierestate.com`). Address-based blocking is essential for these.

---

## Section 5 — Country Distribution

### Train Set
| Country | S1 Train | S2 Train | S3 Train |
|---|---|---|---|
| US | 1,323,633 | 3,016,817 | 3,170,056 |
| India | 883,188 | 2,017,799 | 2,115,547 |

### Test Set
| Country | S1 Test | S2 Test | S3 Test |
|---|---|---|---|
| US | 663,106 | 1,871,330 | 1,945,701 |
| India | 809,986 | 2,312,565 | 2,405,000 |
| **France** | **259,452** | **703,378** | **731,615** |

**France verified in test:** ✅ **YES** — France constitutes **~15% of test S1** (259,452 records).

**Critical implication:** Country **MUST** be treated as an open-set string label. Do NOT one-hot encode, do NOT filter on `{US, India}`. The pipeline must process France records without retraining.

**Confirmed by Member 1:** There are **zero cross-country matches** in training ground truth. Country is therefore a valid and reliable **hard pre-blocking partition** — records with different countries will never be true matches.

---

## Section 6 — Ground Truth Analysis

### Match Cardinality Distribution

| Match Count | S1 Entities | Percentage |
|---|---|---|
| 0 (singletons) | 123,247 | 5.58% |
| 1 match | 119,157 | 5.40% |
| 2 matches | 375,212 | 17.00% |
| 3 matches | 530,841 | 24.05% |
| 4 matches | 484,115 | 21.94% |
| 5 matches | 321,957 | 14.59% |
| 6 matches | 164,868 | 7.47% |
| 7 matches | 63,968 | 2.90% |
| 8–11 matches | 23,456 | 1.06% |
| **Max matches** | **11** | — |

### Source Breakdown (among entities with ≥1 match)

| Category | Count | % of total S1 |
|---|---|---|
| Both S2 & S3 | 1,776,047 | 80.48% |
| S3 only | 164,498 | 7.45% |
| S2 only | 143,029 | 6.48% |
| Zero matches | 123,247 | 5.58% |

**Key structural finding (Member 1):**  
The ground truth follows a strict **1-to-N** structure — every S2 or S3 entity appears **at most once** across all match lists. There is no many-to-many overlap. This means matched records are partitioned into disjoint clusters around each S1 entity.

---

## Section 7 — Feature Strength from Positive Pair Analysis (Member 3)

Member 3 sampled true positive pairs and measured similarity scores across features:

| Feature | Mean Sim | Median | ≥0.8 (%) | ≥0.5 (%) | <0.2 (%) |
|---|---|---|---|---|---|
| Name Token Jaccard | 0.615 | 0.667 | 31.75% | 76.80% | 14.83% |
| Name Char 3-gram Sim | 0.610 | 0.667 | 27.23% | 71.45% | 10.45% |
| Address Token Jaccard | 0.597 | 0.625 | 24.39% | 67.42% | 6.53% |
| Address Char 3-gram | 0.649 | 0.687 | 30.25% | 77.50% | 6.04% |
| Address Digit Overlap | 0.725 | 1.000 | 63.00% | 77.46% | 17.55% |
| **Country Agreement** | **1.000** | **1.000** | **100%** | **100%** | **0%** |

**Country agreement is 100% on all true positive pairs** — confirms country as a hard partition.

**By country (India vs US):**
- Indian business names are noisier (mean Jaccard 0.53 vs 0.67 for US) due to transliteration variance and Devanagari/Tamil scripts.
- Indian addresses are actually more similar on average (Jaccard 0.66) than US addresses (0.55) because Indian addresses tend to include more shared landmark tokens.

---

## Section 8 — Difficult Match Categories (Member 3)

Member 3 identified four major categories of hard cases with real examples:

### 8.1 — Missing Address in Target (~3.3% of S2/S3)
The target record has an empty address field. The model must rely exclusively on name similarity.
- Example: `S1: "Maure Williams Colombier Inc" | S2: "Maure Wilblims Colombier Inc" [address: empty]`
- Name Token Jaccard = 0.60, Address Jaccard = 0.0
- **Implication:** Name-only fallback path is mandatory in the matching model.

### 8.2 — Transliteration / Script Mismatch (India only)
S1 is in Latin script; matched S2/S3 record is in Devanagari or Tamil. Latin string similarity gives 0.0.
- Example: `S1: "Royal Surya Management" | S2: "रॉयल सूर्य मैनेजमेंट"` — Name Jaccard = 0.0, but Address Digit Overlap = 0.5
- Example: `S1: "Prime Properties Private Limited" | S2: "பிரைம் புராப்பர்ட்டீஸ் பிரைவேட் லிமிடெட்"` — Tamil script, name sim = 0.0
- **Implication:** Address-based and digit-overlap blocking must be the fallback anchor for India pairs. Script detection or transliteration normalization would help but may not be feasible within contest rules.

### 8.3 — Different Name, Same Address (DBA / Domain Names)
The business uses a different trading name or domain name but shares the same physical address.
- Example: `S1: "Precision Frontier Estate LLC" | S2: "precisionfrontierestate.com"` — Name Jaccard = 0.0, Address Jaccard = 0.625
- Example: `S1: "Cascade Ministries" | S2: "LYRABELOMIRA"` — Name Jaccard = 0.0, Address Jaccard = 1.0
- **Implication:** Pure name blocking will **completely miss these pairs**. Address-based blocking is essential for capturing this class.

### 8.4 — Similar Name, Low Address Similarity (Reordering / Typos / Abbreviations)
Names are near-identical but address tokens differ substantially due to reordering, abbreviations, or typos.
- Example: `S1: "Cornerstone Advanced Tavia Inc." at "1415 3rd Street, Lanham, MD" | S3: "CORNERSTONE ADVANCED TAVIA INC." at "01415 Third St, Lanham, Maryland"` — Name Jaccard = 1.0, Address Jaccard = 0.111
- **Implication:** Address similarity should never be a disqualifying gate — low address similarity can coexist with a true match.

---

## Section 9 — Initial Conclusions for Normalization & Blocking Design

### 9.1 Normalization (Before Any Comparison)

| Field | Required Transformation |
|---|---|
| business_name | Lowercase, strip punctuation, canonicalize legal suffixes (Corp/Corporation, Pvt/Private, Ltd/Limited, Inc/Incorporated, LLC) |
| business_name | Handle & → "and" substitution |
| business_address | Lowercase, strip punctuation, expand abbreviations (Rd→Road, St→Street, Ave→Avenue, Blvd→Boulevard) |
| business_address | Tokenize, remove stop-words, sort tokens for normalized comparison |
| country | Use as-is string for equality comparison — never encode as fixed-vocabulary categorical |

### 9.2 Blocking Strategy

Based on the confirmed noise patterns and difficult cases from all three members:

| Blocking Pass | Keys | Purpose |
|---|---|---|
| Pass 1: Hard partition | `country` | Eliminates 100% of cross-country false candidates (validated by Member 1: 0 cross-country true matches) |
| Pass 2: Name token index | Top-3 name tokens (sorted, lowercased) | Captures ~77% of true pairs with name Jaccard ≥ 0.5 |
| Pass 3: Phonetic keys | Soundex/Metaphone on name tokens | Captures typos and transliteration variants |
| Pass 4: Address digit index | Shared house/plot numbers | Captures DBA cases and transliteration cases where name fails completely |
| Pass 5: TF-IDF cosine | Name n-gram similarity ≥ 0.25 | Fuzzy name backup for partial matches |

**Blocking design principle:** Passes 2–5 run independently and are **OR-combined** into the candidate set. The candidate set is what feeds the matching model.

### 9.3 Matching Model Recommendations

- **Strongest features:** Country agreement (binary), address digit overlap (median 1.0 on true pairs), name char 3-gram similarity, address char 3-gram similarity.
- **Required fallback path:** When target address is empty (~3.3% of S2/S3), use name-only features with a lower confidence threshold.
- **Threshold tuning:** F_0.5 is precision-heavy (precision weighted 2×). Tune the classification threshold to favour precision over recall.
- **Singleton handling:** 5.58% of S1 entities are singletons (no true matches). The model must produce high-confidence negative predictions for these — false merges on singletons incur full F_0.5 = 0.0 penalty.

---

## Section 10 — Step 1 Answer Summary

| Question | Answer |
|---|---|
| How many S1/S2/S3 records? | Train: S1=2.2M, S2=5.0M, S3=5.3M · Test: S1=1.7M, S2=4.9M, S3=5.1M |
| How many unique IDs? | 100% unique within every file — no duplicates |
| How much missing data? | Names: 0% everywhere · Addresses: 0% in S1, ~3.3% in S2/S3 train, ~2.7% in S2/S3 test · Country: 0% everywhere |
| Are there duplicates? | No duplicate entity IDs anywhere |
| Any invalid records? | Zero malformed records or columns |
| How noisy are names? | High noise — legal suffix variants, case, punctuation, word-order confirmed at millions of records; Devanagari/Tamil script in India subset |
| What transformations are necessary? | Suffix canonicalization, lowercasing, punctuation stripping, phonetic encoding for India records |
| How noisy are addresses? | High noise — abbreviations, reordering, partial addresses, landmark references; ~7% have postal codes |
| How often are numbers/postcodes available? | Numeric tokens: ~94–96% · Postal codes (5-6 digit): only ~4–8% |
| Countries in train? | US, India |
| Countries in test? | US, India, **France** |
| France in test? | **YES** — 259,452 S1 records (~15%) |
| Country representation? | Open-set string equality feature only — never one-hot encode |
| How many S1 entities have zero matches? | **123,247 (5.58%)** |
| How many have one match? | **119,157 (5.40%)** |
| How many have multiple matches? | **1,964,417 (89.01%)** |
| S2-only vs S3-only vs both? | Both: 1,776,047 (80.48%) · S3-only: 164,498 (7.45%) · S2-only: 143,029 (6.48%) |
| Which fields appear strongest? | Country (100% agreement on true pairs) > Address digit overlap (median 1.0) > Name/address char n-gram similarity |
| Likely false positives? | Businesses with same/similar names at different addresses in the same city; generic names like "Morgan Partners", "Internal Medicine Clinic" |
| Likely hard negatives? | DBA / domain-name records (completely different name, same address); Devanagari/Tamil transliterations of Indian names |

---

*Step 1 complete. Team Akatsuki is ready to proceed to Step 2: Normalization + Blocking.*
