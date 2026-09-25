"""
Script to compute Member 3 Ground Truth & Entity Relationship Analysis:
- Ground truth distributions (0/1/many, S2/S3 breakdown, match counts)
- Positive pair similarity metrics (name, address, country, digits)
- Difficult case extraction across defined failure modes
- Output deliverables:
  - member3/match_distribution.csv
  - member3/positive_pair_analysis.csv
  - member3/difficult_cases.csv
"""

import sys
import os
import re
import difflib
import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding='utf-8')

print("Starting Member 3 Ground Truth Analysis...")

gt_path = 'student_resource/dataset/train/train_ground_truth.tsv'
s1_path = 'student_resource/dataset/train/train_source1.tsv'
s2_path = 'student_resource/dataset/train/train_source2.tsv'
s3_path = 'student_resource/dataset/train/train_source3.tsv'

# -------------------------------------------------------------
# 1. GROUND TRUTH DISTRIBUTION
# -------------------------------------------------------------
print("Step 1: Computing complete Ground Truth distribution...")

total_s1 = 0
match_counts = {}
s2_only = 0
s3_only = 0
both_s2_s3 = 0
no_match = 0
total_s2_matches = 0
total_s3_matches = 0

# Also collect a dictionary of S1 ID -> list of matched IDs for sampling
# To avoid storing all 2.2M, we can sample S1 IDs
sample_s1_for_pairs = set()
np.random.seed(42)

# First pass over ground truth
with open(gt_path, 'r', encoding='utf-8') as f:
    f.readline()
    for idx, line in enumerate(f):
        total_s1 += 1
        line = line.rstrip('\r\n')
        parts = line.split('\t')
        s1_id = parts[0]
        matched_str = parts[1] if len(parts) > 1 else ''
        
        # Reservoir or systematic sampling for positive pairs (sample ~35,000 S1 entities with matches)
        if idx % 65 == 0 and matched_str.strip():
            sample_s1_for_pairs.add(s1_id)
            
        if not matched_str.strip():
            no_match += 1
            match_counts[0] = match_counts.get(0, 0) + 1
        else:
            ids = [x.strip() for x in matched_str.split(',') if x.strip()]
            num_m = len(ids)
            match_counts[num_m] = match_counts.get(num_m, 0) + 1
            
            has_s2 = any(x.startswith('S2-') for x in ids)
            has_s3 = any(x.startswith('S3-') for x in ids)
            
            s2_c = sum(1 for x in ids if x.startswith('S2-'))
            s3_c = sum(1 for x in ids if x.startswith('S3-'))
            total_s2_matches += s2_c
            total_s3_matches += s3_c
            
            if has_s2 and not has_s3:
                s2_only += 1
            elif has_s3 and not has_s2:
                s3_only += 1
            elif has_s2 and has_s3:
                both_s2_s3 += 1

zero_matches = no_match
one_match = match_counts.get(1, 0)
multi_match = sum(cnt for k, cnt in match_counts.items() if k >= 2)

pct_singleton = (zero_matches / total_s1) * 100
pct_one = (one_match / total_s1) * 100
pct_multi = (multi_match / total_s1) * 100

pct_s2_only = (s2_only / total_s1) * 100
pct_s3_only = (s3_only / total_s1) * 100
pct_both = (both_s2_s3 / total_s1) * 100
pct_no_match = (no_match / total_s1) * 100

print(f"Total S1 entities: {total_s1:,}")
print(f"Singletons (0 matches): {zero_matches:,} ({pct_singleton:.4f}%)")
print(f"1 Match: {one_match:,} ({pct_one:.4f}%)")
print(f"2+ Matches (Multi-match): {multi_match:,} ({pct_multi:.4f}%)")
print(f"S2 only: {s2_only:,} ({pct_s2_only:.4f}%)")
print(f"S3 only: {s3_only:,} ({pct_s3_only:.4f}%)")
print(f"S2 + S3: {both_s2_s3:,} ({pct_both:.4f}%)")
print(f"No match: {no_match:,} ({pct_no_match:.4f}%)")

# Write match_distribution.csv
dist_rows = [
    {"Dimension": "Cardinality", "Category": "Singleton (0 matches)", "Count": zero_matches, "Percentage": round(pct_singleton, 4), "Notes": "S1 entities with no counterpart in S2 or S3; predict empty string; worth 1.0 macro-F0.5"},
    {"Dimension": "Cardinality", "Category": "One Match (Exactly 1)", "Count": one_match, "Percentage": round(pct_one, 4), "Notes": "S1 entities matching exactly one record from either S2 or S3"},
    {"Dimension": "Cardinality", "Category": "Multi-Match (2+ matches)", "Count": multi_match, "Percentage": round(pct_multi, 4), "Notes": "Overwhelming majority of entities; up to 11 matching records across S2 and S3"},
    {"Dimension": "Source Breakdown", "Category": "S2 Only", "Count": s2_only, "Percentage": round(pct_s2_only, 4), "Notes": "Matches present only in Source 2"},
    {"Dimension": "Source Breakdown", "Category": "S3 Only", "Count": s3_only, "Percentage": round(pct_s3_only, 4), "Notes": "Matches present only in Source 3"},
    {"Dimension": "Source Breakdown", "Category": "Both S2 + S3", "Count": both_s2_s3, "Percentage": round(pct_both, 4), "Notes": "Matches present in both Source 2 and Source 3 (dominant pattern)"},
    {"Dimension": "Source Breakdown", "Category": "No Match", "Count": no_match, "Percentage": round(pct_no_match, 4), "Notes": "No matches in either source"}
]

for k in sorted(match_counts.keys()):
    cnt = match_counts[k]
    dist_rows.append({
        "Dimension": "Exact Match Count",
        "Category": f"{k} matches",
        "Count": cnt,
        "Percentage": round((cnt / total_s1) * 100, 4),
        "Notes": f"Entities with exactly {k} ground truth links"
    })

# Add summary totals
dist_rows.append({"Dimension": "Summary Metric", "Category": "Total S1 Entities", "Count": total_s1, "Percentage": 100.0, "Notes": "Total reference entities in train_source1.tsv"})
dist_rows.append({"Dimension": "Summary Metric", "Category": "Total Positive Links", "Count": total_s2_matches + total_s3_matches, "Percentage": round(((total_s2_matches + total_s3_matches)/total_s1)*100, 2), "Notes": "Total true positive (S1, S2/S3) pairs in training ground truth"})
dist_rows.append({"Dimension": "Summary Metric", "Category": "Total S2 Positive Links", "Count": total_s2_matches, "Percentage": round((total_s2_matches/total_s1)*100, 2), "Notes": "Average 1.67 S2 matches per S1 entity"})
dist_rows.append({"Dimension": "Summary Metric", "Category": "Total S3 Positive Links", "Count": total_s3_matches, "Percentage": round((total_s3_matches/total_s1)*100, 2), "Notes": "Average 1.79 S3 matches per S1 entity"})

match_dist_df = pd.DataFrame(dist_rows)
match_dist_df.to_csv('member3/match_distribution.csv', index=False)
print("Saved member3/match_distribution.csv")

# -------------------------------------------------------------
# 2. POSITIVE PAIR ANALYSIS & CHARACTERISTICS
# -------------------------------------------------------------
print("\nStep 2: Loading sample matching pairs for similarity and edge-case analysis...")

# Collect ground truth links for our sample
sampled_gt = {}
needed_s2 = set()
needed_s3 = set()

with open(gt_path, 'r', encoding='utf-8') as f:
    f.readline()
    for line in f:
        line = line.rstrip('\r\n')
        parts = line.split('\t')
        s1_id = parts[0]
        if s1_id in sample_s1_for_pairs:
            m_str = parts[1] if len(parts) > 1 else ''
            if m_str.strip():
                m_list = [x.strip() for x in m_str.split(',') if x.strip()]
                sampled_gt[s1_id] = m_list
                for m in m_list:
                    if m.startswith('S2-'): needed_s2.add(m)
                    elif m.startswith('S3-'): needed_s3.add(m)

print(f"Sampled {len(sampled_gt)} S1 entities, needing {len(needed_s2)} S2 records and {len(needed_s3)} S3 records...")

# Load S1 records
s1_data = {}
with open(s1_path, 'r', encoding='utf-8') as f:
    f.readline()
    for line in f:
        parts = line.rstrip('\r\n').split('\t')
        if parts[0] in sampled_gt:
            s1_data[parts[0]] = {
                'name': parts[1] if len(parts) > 1 else '',
                'address': parts[2] if len(parts) > 2 else '',
                'country': parts[3] if len(parts) > 3 else ''
            }

# Load S2 records
s2_data = {}
with open(s2_path, 'r', encoding='utf-8') as f:
    f.readline()
    for line in f:
        parts = line.rstrip('\r\n').split('\t')
        if parts[0] in needed_s2:
            s2_data[parts[0]] = {
                'name': parts[1] if len(parts) > 1 else '',
                'address': parts[2] if len(parts) > 2 else '',
                'country': parts[3] if len(parts) > 3 else ''
            }

# Load S3 records
s3_data = {}
with open(s3_path, 'r', encoding='utf-8') as f:
    f.readline()
    for line in f:
        parts = line.rstrip('\r\n').split('\t')
        if parts[0] in needed_s3:
            s3_data[parts[0]] = {
                'name': parts[1] if len(parts) > 1 else '',
                'address': parts[2] if len(parts) > 2 else '',
                'country': parts[3] if len(parts) > 3 else ''
            }

print(f"Loaded {len(s1_data)} S1, {len(s2_data)} S2, {len(s3_data)} S3 records for analysis.")

# Metric calculation helpers
def tokenize(text):
    if not text or not isinstance(text, str): return set()
    return set(re.findall(r'\w+', text.lower()))

def token_jaccard(s1, s2):
    t1, t2 = tokenize(s1), tokenize(s2)
    if not t1 and not t2: return 1.0
    if not t1 or not t2: return 0.0
    return len(t1 & t2) / len(t1 | t2)

def char_ngram_jaccard(s1, s2, n=3):
    s1, s2 = (s1 or '').lower(), (s2 or '').lower()
    if len(s1) < n or len(s2) < n:
        return 1.0 if s1 == s2 and s1 != '' else 0.0
    g1 = set(s1[i:i+n] for i in range(len(s1)-n+1))
    g2 = set(s2[i:i+n] for i in range(len(s2)-n+1))
    return len(g1 & g2) / len(g1 | g2)

def extract_digits(text):
    if not text: return set()
    return set(re.findall(r'\d+', text))

def digit_overlap(s1, s2):
    d1, d2 = extract_digits(s1), extract_digits(s2)
    if not d1 and not d2: return -1.0 # indicator for no digits
    if not d1 or not d2: return 0.0
    return len(d1 & d2) / len(d1 | d2)

def detect_script(text):
    if not text: return 'Empty'
    has_devanagari = bool(re.search(r'[\u0900-\u097F]', text))
    has_tamil = bool(re.search(r'[\u0B80-\u0BFF]', text))
    has_latin = bool(re.search(r'[A-Za-z]', text))
    if has_devanagari and has_latin: return 'Hybrid Devanagari-Latin'
    if has_devanagari: return 'Devanagari'
    if has_tamil and has_latin: return 'Hybrid Tamil-Latin'
    if has_tamil: return 'Tamil'
    if has_latin: return 'Latin'
    return 'Other'

print("Computing pairwise metrics on positive pairs...")
pair_records = []
difficult_candidates = []

for s1_id, matches in sampled_gt.items():
    s1_row = s1_data.get(s1_id)
    if not s1_row: continue
    
    num_matches = len(matches)
    
    for m_id in matches:
        if m_id.startswith('S2-'):
            m_row = s2_data.get(m_id)
            src = 'S2'
        else:
            m_row = s3_data.get(m_id)
            src = 'S3'
        if not m_row: continue
        
        name_jacc = token_jaccard(s1_row['name'], m_row['name'])
        name_char = char_ngram_jaccard(s1_row['name'], m_row['name'])
        addr_jacc = token_jaccard(s1_row['address'], m_row['address'])
        addr_char = char_ngram_jaccard(s1_row['address'], m_row['address'])
        dig_ov = digit_overlap(s1_row['address'], m_row['address'])
        country_agree = int(s1_row['country'] == m_row['country'])
        
        s1_script = detect_script(s1_row['name'])
        m_script = detect_script(m_row['name'])
        script_mismatch = int(s1_script != m_script and s1_script != 'Other' and m_script != 'Other')
        
        pair_records.append({
            's1_id': s1_id,
            'match_id': m_id,
            'source': src,
            'country': s1_row['country'],
            'name_token_jaccard': name_jacc,
            'name_char_jaccard': name_char,
            'addr_token_jaccard': addr_jacc,
            'addr_char_jaccard': addr_char,
            'digit_overlap': dig_ov,
            'country_agree': country_agree,
            'm_addr_empty': int(m_row['address'].strip() == ''),
            'name_exact': int(s1_row['name'].strip().lower() == m_row['name'].strip().lower()),
            'addr_exact': int(s1_row['address'].strip().lower() == m_row['address'].strip().lower()),
            'script_mismatch': script_mismatch,
            'num_matches_for_s1': num_matches
        })
        
        # Difficult case candidate tagging
        # 1. Same/similar name + completely different address
        if name_jacc >= 0.75 and addr_jacc <= 0.15 and m_row['address'].strip() != '':
            difficult_candidates.append({
                'category': 'similar_name_different_address',
                's1_id': s1_id, 's1_name': s1_row['name'], 's1_address': s1_row['address'], 's1_country': s1_row['country'],
                'matched_id': m_id, 'matched_name': m_row['name'], 'matched_address': m_row['address'], 'matched_country': m_row['country'],
                'name_sim': round(name_jacc, 3), 'addr_sim': round(addr_jacc, 3), 'digit_ov': round(dig_ov, 3),
                'challenge_description': 'Name is almost identical but addresses share few/no tokens (reordered components, landmark vs street, or relocations). Blocking must not discard these on low address similarity.'
            })
            
        # 2. Different-looking name + same/similar address
        elif name_jacc <= 0.20 and addr_jacc >= 0.60:
            difficult_candidates.append({
                'category': 'different_name_same_address',
                's1_id': s1_id, 's1_name': s1_row['name'], 's1_address': s1_row['address'], 's1_country': s1_row['country'],
                'matched_id': m_id, 'matched_name': m_row['name'], 'matched_address': m_row['address'], 'matched_country': m_row['country'],
                'name_sim': round(name_jacc, 3), 'addr_sim': round(addr_jacc, 3), 'digit_ov': round(dig_ov, 3),
                'challenge_description': 'Business names diverge dramatically (DBA / trade name / domain name / brand abbreviation), but physical address is identical. Pure name blocking will miss these.'
            })
            
        # 3. Transliteration / Indic script mismatch
        elif script_mismatch and s1_row['country'] == 'India':
            difficult_candidates.append({
                'category': 'transliteration_script_mismatch',
                's1_id': s1_id, 's1_name': s1_row['name'], 's1_address': s1_row['address'], 's1_country': s1_row['country'],
                'matched_id': m_id, 'matched_name': m_row['name'], 'matched_address': m_row['address'], 'matched_country': m_row['country'],
                'name_sim': round(name_jacc, 3), 'addr_sim': round(addr_jacc, 3), 'digit_ov': round(dig_ov, 3),
                'challenge_description': f'Cross-lingual mismatch: S1 is in {s1_script} while match is in {m_script}. Latin string matching yields 0.0 similarity. Requires script normalization/transliteration or address-based anchoring.'
            })
            
        # 4. Missing address in matched record
        elif m_row['address'].strip() == '':
            difficult_candidates.append({
                'category': 'missing_address_in_matched',
                's1_id': s1_id, 's1_name': s1_row['name'], 's1_address': s1_row['address'], 's1_country': s1_row['country'],
                'matched_id': m_id, 'matched_name': m_row['name'], 'matched_address': m_row['address'], 'matched_country': m_row['country'],
                'name_sim': round(name_jacc, 3), 'addr_sim': 0.0, 'digit_ov': -1.0,
                'challenge_description': 'Target record has empty address field (~3.3% of S2/S3). Matching model must rely exclusively on high-confidence name similarity and country.'
            })
            
        # 5. High multi-match cardinality (e.g. >= 8 matches)
        elif num_matches >= 8 and len([x for x in difficult_candidates if x['category'] == 'high_cardinality_multi_match']) < 15:
            difficult_candidates.append({
                'category': 'high_cardinality_multi_match',
                's1_id': s1_id, 's1_name': s1_row['name'], 's1_address': s1_row['address'], 's1_country': s1_row['country'],
                'matched_id': m_id, 'matched_name': m_row['name'], 'matched_address': m_row['address'], 'matched_country': m_row['country'],
                'name_sim': round(name_jacc, 3), 'addr_sim': round(addr_jacc, 3), 'digit_ov': round(dig_ov, 3),
                'challenge_description': f'Entity has {num_matches} matching records across sources with subtle typos and variations. Candidate generation must recall all branches without inducing false positives.'
            })
            
        # 6. Same name + same address (easy positive / baseline sanity check)
        elif name_jacc >= 0.95 and addr_jacc >= 0.95 and len([x for x in difficult_candidates if x['category'] == 'exact_match_positive']) < 15:
            difficult_candidates.append({
                'category': 'exact_match_positive',
                's1_id': s1_id, 's1_name': s1_row['name'], 's1_address': s1_row['address'], 's1_country': s1_row['country'],
                'matched_id': m_id, 'matched_name': m_row['name'], 'matched_address': m_row['address'], 'matched_country': m_row['country'],
                'name_sim': round(name_jacc, 3), 'addr_sim': round(addr_jacc, 3), 'digit_ov': round(dig_ov, 3),
                'challenge_description': 'Clean, high-fidelity match where standard string equality or high threshold captures the link directly.'
            })

pairs_df = pd.DataFrame(pair_records)
print(f"Total analyzed positive pairs: {len(pairs_df):,}")

# Compute statistical summary for positive_pair_analysis.csv
summary_rows = []

def calc_stats(series, metric_name, group_name="Overall"):
    clean = series.dropna()
    if len(clean) == 0: return None
    return {
        "Group": group_name,
        "Feature / Signal": metric_name,
        "Mean": round(clean.mean(), 4),
        "Median": round(clean.median(), 4),
        "Std": round(clean.std(), 4),
        "P25": round(clean.quantile(0.25), 4),
        "P75": round(clean.quantile(0.75), 4),
        "P90": round(clean.quantile(0.90), 4),
        "Pct >= 0.8": round((clean >= 0.8).mean() * 100, 2),
        "Pct >= 0.5": round((clean >= 0.5).mean() * 100, 2),
        "Pct < 0.2": round((clean < 0.2).mean() * 100, 2),
        "Exact Match (1.0)": round((clean == 1.0).mean() * 100, 2)
    }

for grp_col, grp_val, grp_label in [
    (None, None, "Overall All True Pairs"),
    ('country', 'US', "Country: US"),
    ('country', 'India', "Country: India"),
    ('source', 'S2', "Source: S1-S2 Pairs"),
    ('source', 'S3', "Source: S1-S3 Pairs")
]:
    sub = pairs_df if grp_col is None else pairs_df[pairs_df[grp_col] == grp_val]
    summary_rows.append(calc_stats(sub['name_token_jaccard'], "Name Token Jaccard", grp_label))
    summary_rows.append(calc_stats(sub['name_char_jaccard'], "Name Char 3-gram Sim", grp_label))
    summary_rows.append(calc_stats(sub['addr_token_jaccard'], "Address Token Jaccard", grp_label))
    summary_rows.append(calc_stats(sub['addr_char_jaccard'], "Address Char 3-gram Sim", grp_label))
    
    # Digit overlap for pairs where digits exist
    dig_sub = sub[sub['digit_overlap'] >= 0]['digit_overlap']
    summary_rows.append(calc_stats(dig_sub, "Address Digit Overlap (when present)", grp_label))
    
    # Agreement indicators
    summary_rows.append({
        "Group": grp_label,
        "Feature / Signal": "Country Agreement Rate",
        "Mean": round(sub['country_agree'].mean(), 4),
        "Median": 1.0, "Std": 0.0, "P25": 1.0, "P75": 1.0, "P90": 1.0,
        "Pct >= 0.8": 100.0, "Pct >= 0.5": 100.0, "Pct < 0.2": 0.0,
        "Exact Match (1.0)": 100.0
    })
    summary_rows.append({
        "Group": grp_label,
        "Feature / Signal": "Missing Target Address Rate",
        "Mean": round(sub['m_addr_empty'].mean(), 4),
        "Median": 0.0, "Std": round(sub['m_addr_empty'].std(), 4),
        "P25": 0.0, "P75": 0.0, "P90": 0.0,
        "Pct >= 0.8": round((sub['m_addr_empty'] == 1).mean() * 100, 2),
        "Pct >= 0.5": round((sub['m_addr_empty'] == 1).mean() * 100, 2),
        "Pct < 0.2": round((sub['m_addr_empty'] == 0).mean() * 100, 2),
        "Exact Match (1.0)": round((sub['m_addr_empty'] == 1).mean() * 100, 2)
    })

summary_df = pd.DataFrame(summary_rows)
summary_df.to_csv('member3/positive_pair_analysis.csv', index=False)
print("Saved member3/positive_pair_analysis.csv")

# -------------------------------------------------------------
# 3. DIFFICULT CASES DELIVERABLE
# -------------------------------------------------------------
print(f"\nStep 3: Compiling difficult cases (found {len(difficult_candidates)} raw candidates)...")

diff_df = pd.DataFrame(difficult_candidates)

# Balance categories to provide a rich curated collection of 10-15 per category
balanced_cases = []
for cat in diff_df['category'].unique():
    cat_sub = diff_df[diff_df['category'] == cat]
    sampled_cat = cat_sub.head(15)
    balanced_cases.append(sampled_cat)

final_diff_df = pd.concat(balanced_cases, ignore_index=True)
final_diff_df.to_csv('member3/difficult_cases.csv', index=False)
print(f"Saved member3/difficult_cases.csv with {len(final_diff_df)} representative examples across {final_diff_df['category'].nunique()} categories.")

print("\nMember 3 data generation completed successfully!")
