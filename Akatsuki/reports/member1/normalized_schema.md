# Normalized Schema Specification & Information-Loss Audit
**Team Akatsuki — Amazon ML Challenge 2026 (Business Entity Resolution)**  
**Author / Role:** Member 1 (Faizur Rahman — Preprocessing & Normalization Lead)  
**Deliverable:** `reports/member1/normalized_schema.md`  

---

## 1. Schema Overview

The Preprocessing and Normalization Layer processes raw records from Source 1 (Reference), Source 2 (Secondary), and Source 3 (Tertiary) without altering or mutating raw source columns. All transformations append standardized, deterministic normalized fields.

* **Raw Input Columns:** 4 columns (`entity_id`, `business_name`, `business_address`, `country`)
* **Normalized Added Columns:** 18 fields (9 name fields, 8 address fields, 1 country field)
* **Total Output Columns:** 22 columns
* **Row Preservation:** Bitwise exact row counts preserved ($100\%$ zero rows dropped, $100\%$ zero duplicate IDs introduced, exact row order retained).

---

## 2. Complete Field Dictionary

| Field Name | Data Type | Meaning / Purpose | Example Value | Source Scope | Safe for Downstream Blocking? | Safe for Downstream Features? |
|:---|:---|:---|:---|:---|:---|:---|
| `entity_id` | `str` | Unique primary record identifier | `S1-925783039` | S1, S2, S3 | **YES** (Primary ID / Key) | N/A (Key identifier) |
| `business_name` | `str` | Raw original business name from input TSV | `Orelee's Barbershop & Salon Inc.` | S1, S2, S3 | NO (Unnormalized noise) | NO (Use `name_norm` / `name_core`) |
| `business_address` | `str` | Raw original address from input TSV | `1795 Westchester Rd, Ste 4B` | S1, S2, S3 | NO (Unnormalized noise) | NO (Use `address_norm`) |
| `country` | `str` | Raw original country string from input TSV | `US` | S1, S2, S3 | NO (Casing/punct variants) | NO (Use `country_norm`) |
| `name_raw` | `str` | Clean string copy of raw name (`""` if null) | `Orelee's Barbershop & Salon Inc.` | S1, S2, S3 | NO (Preserves raw noise) | YES (Audit / debug baseline) |
| `name_norm` | `str` | NFKC-normalized, lowercase, cleaned punctuation, canonicalized legal forms | `orelees barbershop and salon inc` | S1, S2, S3 | **YES** (Exact name index B0) | **YES** (Jaro-Winkler, Levenshtein) |
| `name_core` | `str` | Core business name with legal entity forms stripped | `orelees barbershop and salon` | S1, S2, S3 | **YES** (Core name index B0/B1) | **YES** (Token Jaccard, Cosine) |
| `name_tokens` | `list[str]` | Whitespace-tokenized list of normalized tokens | `['orelees', 'barbershop', 'and', 'salon', 'inc']` | S1, S2, S3 | **YES** (Rare token inverted index B1) | **YES** (Set intersection / overlap) |
| `name_sorted_tokens` | `str` | Alphabetically sorted normalized tokens joined by space (word-order invariant) | `and barbershop inc orelees salon` | S1, S2, S3 | **YES** (Word-order invariant index B0/B1) | **YES** (Token Sort similarity) |
| `name_translit` | `str` | Clean Latin ASCII transliteration of non-Latin scripts (Devanagari, Tamil, etc.) | `ram marketing pvt ltd` | S1, S2, S3 | **YES** (Cross-script index B3) | **YES** (Transliteration similarity) |
| `name_phonetic` | `str` | 4-character American Soundex phonetic representation of `name_core` | `O642` | S1, S2, S3 | **YES** (Phonetic typo blocking B1) | **YES** (Phonetic agreement flag) |
| `name_has_digits` | `bool` | Boolean flag indicating presence of numeric digits in raw business name | `False` | S1, S2, S3 | NO (Too low entropy) | **YES** (Numeric interaction feature) |
| `name_length` | `int` | Character length of `name_norm` | `32` | S1, S2, S3 | NO (Too low entropy) | **YES** (Length ratio feature) |
| `address_raw` | `str` | Clean string copy of raw address (`""` if null) | `1795 Westchester Rd, Ste 4B` | S1, S2, S3 | NO (Preserves raw noise) | YES (Audit / debug baseline) |
| `address_norm` | `str` | Standardized address with expanded street abbreviations, lowercase, cleaned punctuation | `1795 westchester road suite 4b` | S1, S2, S3 | **YES** (Address token blocking B2) | **YES** (Address Jaccard, 3-gram Cosine) |
| `address_tokens` | `list[str]` | Whitespace-tokenized list of normalized address words | `['1795', 'westchester', 'road', 'suite', '4b']` | S1, S2, S3 | **YES** (Rare address token index B2) | **YES** (Address token overlap) |
| `address_numbers` | `list[str]` | All numeric string sequences extracted from address | `['1795', '4']` | S1, S2, S3 | **YES** (Numeric locality index B2) | **YES** (Digit overlap ratio) |
| `house_number` | `str` | Conservatively extracted street, building, door, or plot number (supports sub-unit letters) | `1795` | S1, S2, S3 | **YES** (House number exact blocking B2) | **YES** (House number exact match binary) |
| `postal_code` | `str` | Extracted 5-digit ZIP, 6-digit Indian PIN, or French 5-digit postal code | `27262` | S1, S2, S3 | **YES** (Postal locality blocking B2) | **YES** (Postal exact match binary) |
| `address_has_digits` | `bool` | Boolean flag indicating presence of digits in address | `True` | S1, S2, S3 | NO (Too low entropy) | **YES** (Numeric interaction feature) |
| `address_missing` | `int` | Binary indicator (1 if address is empty/null/whitespace, else 0) | `0` | S1, S2, S3 | **YES** (Address gate / filter) | **YES** (Missingness interaction term) |
| `address_length` | `int` | Character length of `address_norm` | `30` | S1, S2, S3 | NO (Too low entropy) | **YES** (Address length ratio feature) |
| `country_norm` | `str` | Standardized ISO 3166-1 country representation (`US`, `India`, `France`, open-set title-case) | `US` | S1, S2, S3 | **YES** (Hard blocking partition Pass 1) | **YES** (Country agreement binary: 100% strict) |

---

## 3. Information-Loss Audit

Every transformation in the Member 1 pipeline was evaluated against two criteria:
1. **Matching Representation Improvement:** Does the transformation increase candidate recall or feature discriminatory power?
2. **Information Preservation:** Does the transformation accidentally discard or corrupt identity information?

### 3.1 Legal Suffix Normalization & Stripping
* **Observed Data Pattern:** Over 7.8 million records in the challenge training set contain legal entity designations (`Limited/Ltd`: 2.70M, `Private/Pvt`: 2.13M, `LLC`: 1.46M, `Inc/Incorporated`: 1.11M, `Corp/Corporation`: 0.45M).
* **Improvement:** In `name_norm`, abbreviations and full words are standardized to canonical tokens (`corporation` $\to$ `corp`, `l.l.c.` $\to$ `llc`, `private limited` $\to$ `pvt ltd`). This eliminates superficial string mismatches.
* **Information Preservation:** 
  - To prevent false merges between distinct entities (e.g., "General Medical LLC" vs. "General Medical Inc"), `name_norm` retains the canonical legal token.
  - Simultaneously, `name_core` strips legal tokens exclusively from the outer boundaries, allowing token similarity models (Member 3) to focus on the distinctive business identity.
  - **Verdict: ZERO INFORMATION LOSS.** Both `name_norm` and `name_core` are emitted side-by-side.

### 3.2 Punctuation Handling & Indic Matra Safety
* **Observed Data Pattern:** Direct regex removal like `[^\w\s]` erroneously deletes Unicode combining characters (such as Devanagari matras `\u093e`, `\u0947` in Indian records).
* **Improvement:** We employ category-based Unicode parsing:
  ```python
  [ " " if unicodedata.category(c)[0] in ("P", "S") else c for c in text ]
  ```
  Punctuation (`P*`) and symbols (`S*`) are replaced with spaces, while letters (`L*`), numbers (`N*`), and combining marks (`M*`) are preserved intact.
* **Apostrophe / Quote Handling:** Quotation marks and apostrophes are deleted *without* adding spaces (e.g. `Orelee's` $\to$ `orelees`), preventing possessive contractions from splitting into orphaned `'s'` tokens.
* **Verdict: ZERO INFORMATION LOSS.** Native script glyphs and word structures remain uncorrupted.

### 3.3 Word-Order Inversion & Token Ordering
* **Observed Data Pattern:** Business names frequently swap tokens between database sources (e.g., `Cornerstone Advanced Tavia` in S1 vs `Tavia Cornerstone Advanced` in S2).
* **Improvement:** We generate `name_sorted_tokens` (`" ".join(sorted(norm_tokens))`). Inverted names produce bitwise identical sorted strings (e.g. `"advanced cornerstone tavia"`), enabling $O(1)$ inverted index lookups in Pass B0/B1.
* **Information Preservation:** `name_norm` preserves the original natural-language token sequence for n-gram and Jaro-Winkler calculations.
* **Verdict: ZERO INFORMATION LOSS.**

### 3.4 Transliteration (AnyAscii Engine)
* **Observed Data Pattern:** 11–15% of Indian business records in S2 and S3 are written in Devanagari or regional scripts, while reference S1 records are primarily in Latin English. Standard string metrics on `("रॉयल सूर्य", "Royal Surya")` yield 0.0 similarity.
* **Improvement:** `name_translit` converts non-Latin scripts to clean ASCII Latin (`"royal surya"`), enabling cross-script candidate blocking in Pass B3 and feature computation.
* **Information Preservation:** The native script is preserved in `name_raw` and `name_norm`. Transliteration is isolated to `name_translit`.
* **Verdict: ZERO INFORMATION LOSS.**

### 3.5 Numeric & Address Component Extraction
* **Observed Data Pattern:** >93% of addresses contain numbers, while postal codes are present in only ~4–8% of records.
* **Improvement:**
  - `address_numbers`: Collects all numeric sequences for digit-overlap calculations.
  - `house_number`: Conservatively isolates building, plot, and door numbers, including complex Indian sub-plots (`16-11-23/37/A`, `KH NO. -570/13`) and French street numbers (`20 Rue René Panhard` $\to$ `20`).
  - `postal_code`: Extracts US ZIP, Indian PIN, and French postal codes without confusing leading house numbers.
  - `address_missing`: Flags empty addresses with binary `1` rather than imputing synthetic text (e.g. `"unknown"`), which would cause massive false-positive collisions.
* **Information Preservation:** `address_norm` retains the entire normalized address text with expanded abbreviations (`rd` $\to$ `road`, `st` $\to$ `street`).
* **Verdict: ZERO INFORMATION LOSS.**

---

## 4. Downstream Integration Guidance

### For Member 2 (Candidate Blocking Lead):
1. **Hard Country Partition (Pass 1):** Block strictly on `country_norm`. Ground truth audit proves 0% cross-country matches.
2. **Exact & Core Name Blocking (Pass B0/B1):** Index on `(country_norm, name_norm)`, `(country_norm, name_core)`, and `(country_norm, name_sorted_tokens)`.
3. **Phonetic & Typo Blocking (Pass B1):** Index on `(country_norm, name_phonetic)`.
4. **Cross-Script Transliteration Blocking (Pass B3):** Index on `(country_norm, name_translit)`.
5. **Address & Numeric Locality Blocking (Pass B2):** Index on `(country_norm, house_number)` and `(country_norm, postal_code)` where non-empty.

### For Member 3 (ML Classification & Inference Lead):
1. **Core Similarity:** Use `name_core` rather than `name_norm` for Token Jaccard and Levenshtein to avoid artificial inflation from shared legal suffixes.
2. **Address Missing Interaction:** In LightGBM, interact address features with `address_missing`:
   $$\text{effective\_addr\_sim} = (1 - S1\text{.address\_missing}) \times (1 - S2\text{.address\_missing}) \times \text{sim}(\text{addr}_1, \text{addr}_2)$$
3. **Discrete Confirming Features:** When `house_number` and `postal_code` are non-empty, binary equality provides high-precision features.
