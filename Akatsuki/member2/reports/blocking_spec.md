# Technical Blocking Specification (`blocking_spec.md`)

## 1. Objective & Target Performance
The Member 2 Blocking Engine converts normalized Source 1, 2, and 3 records into candidate matching pairs for pairwise classification.

Target Optimization Objective:
$$\boxed{\text{very high true-match recall} + \text{manageable candidate volume}}$$

---

## 2. Empirical Data Evidence Grounding
- **Postal Code Coverage**: Raw address regex audit reveals 5/6-digit postal codes are present in only **6.67% of S1**, **7.33% of S2**, and **7.30% of S3** records. Postcode is treated as a secondary rule.
- **Name Collisions**: 30.25% of S1 business names collide across distinct entities. Exact name is a candidate rule, not a match decision.
- **Digit Overlap**: 62.5% of true positive pairs share exact digits. Numeric locality tokens receive dedicated indexing.
- **Script Variations**: Indic non-Latin scripts (Devanagari/Tamil) use transliteration indexes (`name_translit`).

---

## 3. 8-Layer Multi-Pass Architecture

| Pass Code | Pass Name | Index Keys | Frequency Cap / Constraint |
|---|---|---|---|
| **B0** | Country Partition | `country_norm` + `UNKNOWN` fallback | Dynamic open-set partition |
| **B1a** | Exact Name Match | `country \| name_norm` | Max 1000 candidates |
| **B1b** | Core Name Match | `country \| name_core` | Max 200 candidates |
| **B1c** | Sorted Tokens Name | `country \| ' '.join(sorted(tokens))` | Max 600 candidates (Word-order invariant) |
| **B2a** | Rare Token Match | `country \| rare_token` (`doc_freq <= 50`) | Max 50 candidates / token |
| **B2b** | Short Name 1-Del Hash | `country \| 1-deletion-variant` (`len <= 12`) | Max 30 candidates / variant |
| **B3a** | Postcode Match | `country \| postal_code` | Uncapped (7% coverage) |
| **B3b** | House Num + Rare Addr | `country \| house_number \| rare_addr_token` | Max 200 candidates |
| **B3c** | Numeric Locality | `country \| num_token \| locality_token` | Max 200 candidates |
| **B4** | Script Transliteration | `country \| name_translit` & rare tokens | Max 50 candidates |
| **B5** | Hybrid Word+Char TF-IDF | Dual Word (1-2) + Char (3-5) n-gram TF-IDF | Top-30, Cosine threshold >= 0.50 |
| **B6** | Soundex Phonetic Token | `country \| soundex(token)` | Max 100 doc freq cap |
| **B7** | Union & Adaptive Prune | Priority ordering & address-adaptive budget | Cap 75 (address) / 200 (name-only) |

---

## 4. Internal Provenance Matrix Schema (`candidate_pairs_long.parquet`)
- `source1_id`: S1 Entity ID
- `candidate_id`: Matched S2/S3 Candidate ID
- `block_exact_name`: Boolean
- `block_core`: Boolean
- `block_rare_token`: Boolean
- `block_address`: Boolean
- `block_numeric`: Boolean
- `block_translit`: Boolean
- `block_ann`: Boolean
- `block_phonetic`: Boolean
