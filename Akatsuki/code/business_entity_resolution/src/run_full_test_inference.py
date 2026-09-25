"""
Akatsuki/code/business_entity_resolution/src/run_full_test_inference.py
-------------------------------------------------------------------------
High-Throughput Full 1.73M S1 Test Inference Engine & Submission Validator.

Optimization Architecture:
  1. Pre-indexes S2 (4.88M) & S3 (5.08M) candidate structures EXACTLY ONCE.
  2. Processes all 1,732,544 test S1 records in fast batch queries.
  3. Guarantees 1,732,544 output rows in matching_results.tsv with zero ID loss.
"""

import sys
import time
import pickle
import logging
from pathlib import Path
import numpy as np
import pandas as pd

# Setup paths
THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parent.parent.parent  # Akatsuki/
PROJ_ROOT = REPO_ROOT.parent.parent

if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from data_loader import DataLoader
from preprocessing.preprocessor import preprocess_source
from blocking.country_partition import build_country_indexes
from blocking.exact_name_block import build_name_indexes, block_exact_and_core_name
from blocking.token_block import build_rare_token_index, block_rare_tokens
from blocking.address_block import block_address_rare
from blocking.numeric_block import build_numeric_locality_index, block_numeric_locality
from blocking.transliteration_block import build_translit_index, block_transliteration
from blocking.ann_name_retrieval import build_and_query_ann_per_country
from blocking.phonetic_block import build_phonetic_index, block_phonetic
from blocking.candidate_union import union_candidate_passes
from blocking.candidate_pruning import prune_candidate_matrix
from features.pair_features import compute_pair_features

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("Full_Test_Inference")

MODEL_PKL = REPO_ROOT / "model" / "lgbm_entity_match.pkl"
OUTPUT_TSV = REPO_ROOT / "output" / "matching_results.tsv"
REPORT_CSV = REPO_ROOT / "reports" / "inference_report.csv"

FEATURE_COLS = [
    "name_exact", "name_jaccard", "name_levenshtein",
    "name_jaro_winkler", "name_token_sort", "name_token_set",
    "name_char3_cosine", "name_char4_cosine",
    "name_core_similarity", "name_translit_similarity",
    "address_jaccard", "address_levenshtein", "address_char3_cosine",
    "address_digit_overlap", "house_number_match",
    "postal_code_match", "address_exact", "address_missing",
    "country_match", "name_length_ratio", "address_length_ratio",
    "name_token_count_difference", "address_token_count_difference",
    "numeric_conflict",
    "name_address_similarity_product", "name_high_address_low",
    "name_high_address_high", "name_address_conflict",
]


def run_full_test_inference(batch_size: int = 100000, threshold: float = 0.50):
    t_start = time.time()
    logger.info("=" * 70)
    logger.info("AKATSUKI OPTIMIZED 1.73M TEST INFERENCE & SUBMISSION GENERATOR")
    logger.info("=" * 70)

    # 1. Load Model
    if not MODEL_PKL.exists():
        raise FileNotFoundError(f"Model file not found: {MODEL_PKL}")
    with open(MODEL_PKL, "rb") as f:
        model = pickle.load(f)
    logger.info(f"Loaded LightGBM model from {MODEL_PKL}")
    logger.info(f"Using Decision Threshold tau = {threshold:.2f}")

    # 2. Load Full Test Datasets
    loader = DataLoader()
    logger.info("Loading FULL test data split...")
    raw_s1_test = loader.load_source("s1", split="test")
    raw_s2_test = loader.load_source("s2", split="test")
    raw_s3_test = loader.load_source("s3", split="test")

    total_test_s1 = len(raw_s1_test)
    logger.info(f"Loaded Raw Records: S1={total_test_s1:,}, S2={len(raw_s2_test):,}, S3={len(raw_s3_test):,}")

    t_prep = time.time()
    logger.info("Preprocessing all test records...")
    df_s1_test = preprocess_source(raw_s1_test, verbose=False)
    df_s2_test = preprocess_source(raw_s2_test, verbose=False)
    df_s3_test = preprocess_source(raw_s3_test, verbose=False)
    logger.info(f"Preprocessing complete in {time.time() - t_prep:.1f}s.")

    # 3. Pre-build S2/S3 Candidate Indexes ONCE
    t_idx = time.time()
    logger.info("Pre-building candidate blocking indexes over S2 (4.88M) and S3 (5.08M)...")
    exact_idx, core_idx, sorted_idx = build_name_indexes(df_s2_test, df_s3_test)
    rare_idx, _, del_idx = build_rare_token_index(df_s2_test, df_s3_test, max_doc_freq=50)
    num_loc_idx, _ = build_numeric_locality_index(df_s2_test, df_s3_test, max_locality_freq=100)
    translit_idx = build_translit_index(df_s2_test, df_s3_test)
    phonetic_idx, _ = build_phonetic_index(df_s2_test, df_s3_test, max_freq_cap=100)
    logger.info(f"S2/S3 Indexes pre-built in {time.time() - t_idx:.1f}s.")

    # Prepare lookup dicts
    s2_map = df_s2_test.set_index("entity_id").to_dict("index")
    s3_map = df_s3_test.set_index("entity_id").to_dict("index")
    empty_rec = {"business_name_norm": "", "business_address_norm": "", "country_norm": ""}

    # 4. Chunked Batch Querying Across S1 Entities
    all_s1_ids = df_s1_test["entity_id"].tolist()
    num_chunks = (total_test_s1 + batch_size - 1) // batch_size
    logger.info(f"Executing Batch Candidate Generation & Scoring ({num_chunks} batches of {batch_size:,})...")

    results_map = {}
    total_pairs_scored = 0

    for b_idx in range(num_chunks):
        t_b0 = time.time()
        start_idx = b_idx * batch_size
        end_idx = min(start_idx + batch_size, total_test_s1)
        sub_s1 = df_s1_test.iloc[start_idx:end_idx]
        sub_s1_ids = sub_s1["entity_id"].tolist()

        pass_results = {}

        # Fast Pass Queries using pre-built indexes
        cand_exact, cand_core = block_exact_and_core_name(sub_s1, exact_idx, core_idx, sorted_idx)
        pass_results["block_exact_name"] = cand_exact
        pass_results["block_core"] = cand_core

        cand_rare = block_rare_tokens(sub_s1, rare_idx, del_idx, max_candidates_per_s1=50)
        pass_results["block_rare_token"] = cand_rare

        cand_addr = block_address_rare(sub_s1, df_s2_test, df_s3_test, max_candidates_per_s1=50)
        pass_results["block_address"] = cand_addr

        cand_num = block_numeric_locality(sub_s1, num_loc_idx, max_candidates_per_s1=30)
        pass_results["block_numeric"] = cand_num

        cand_translit = block_transliteration(sub_s1, translit_idx, max_candidates_per_s1=50)
        pass_results["block_translit"] = cand_translit

        cand_ann = build_and_query_ann_per_country(sub_s1, df_s2_test, df_s3_test, top_k=30, min_similarity=0.50)
        pass_results["block_ann"] = cand_ann

        cand_phonetic = block_phonetic(sub_s1, phonetic_idx, max_candidates_per_s1=30)
        pass_results["block_phonetic"] = cand_phonetic

        # Union & Prune
        df_long_b = union_candidate_passes(pass_results, sub_s1_ids)
        df_long_pruned = prune_candidate_matrix(df_long_b, max_total_per_s1=100, adaptive_pruning=True)

        if len(df_long_pruned) > 0:
            s1_sub_map = sub_s1.set_index("entity_id").to_dict("index")
            rows = []
            pair_keys = []
            for _, r in df_long_pruned.iterrows():
                s1_id = r["source1_id"]
                cand_id = r["candidate_id"]

                rec1 = s1_sub_map.get(s1_id, empty_rec)
                rec2 = s2_map.get(cand_id, s3_map.get(cand_id, empty_rec))

                feats = compute_pair_features(
                    name1=rec1.get("business_name_norm", ""),
                    name2=rec2.get("business_name_norm", ""),
                    addr1=rec1.get("business_address_norm", ""),
                    addr2=rec2.get("business_address_norm", ""),
                    country1=rec1.get("country_norm", ""),
                    country2=rec2.get("country_norm", ""),
                )
                rows.append([feats[c] for c in FEATURE_COLS])
                pair_keys.append((s1_id, cand_id))

            X_b = np.array(rows, dtype=np.float32)
            preds = model.predict(X_b)
            total_pairs_scored += len(preds)

            for (s1_id, cand_id), score in zip(pair_keys, preds):
                if score >= threshold:
                    if s1_id not in results_map:
                        results_map[s1_id] = []
                    results_map[s1_id].append(cand_id)

        t_elapsed = time.time() - t_b0
        logger.info(f"  Batch {b_idx + 1}/{num_chunks} ({end_idx:,}/{total_test_s1:,} S1 records) scored in {t_elapsed:.1f}s")

    # 5. Export Official Submission TSV File
    logger.info("\nWriting Official Competition Submission TSV File...")
    sub_rows = []
    for s1_id in all_s1_ids:
        matched = results_map.get(s1_id, [])
        sub_rows.append({
            "source1_entity_id": s1_id,
            "matched_entity_ids": ",".join(sorted(set(matched))),
        })

    df_sub = pd.DataFrame(sub_rows)
    OUTPUT_TSV.parent.mkdir(parents=True, exist_ok=True)
    df_sub.to_csv(OUTPUT_TSV, sep="\t", index=False)
    logger.info(f"Saved Submission TSV -> {OUTPUT_TSV}")

    # 6. Submission Integrity Audit Assertions
    logger.info("\nExecuting Submission Integrity Sanity Audit Assertions...")
    assert OUTPUT_TSV.exists(), "Submission file does not exist!"
    assert len(df_sub) == total_test_s1, f"Row count mismatch! Expected {total_test_s1:,}, got {len(df_sub):,}"
    assert df_sub["source1_entity_id"].nunique() == total_test_s1, "Duplicate source1_entity_ids found!"
    assert list(df_sub.columns) == ["source1_entity_id", "matched_entity_ids"], f"Invalid columns: {list(df_sub.columns)}"
    assert df_sub["source1_entity_id"].isnull().sum() == 0, "Null source1_entity_ids found!"

    n_matched = int((df_sub["matched_entity_ids"].fillna("") != "").sum())
    n_singletons = total_test_s1 - n_matched
    match_pct = (n_matched / total_test_s1) * 100

    logger.info(f"  ALL AUDIT CHECKS PASSED PERFECTLY:")
    logger.info(f"    Total Test S1 Rows  : {total_test_s1:,}")
    logger.info(f"    Unique S1 IDs       : {df_sub['source1_entity_id'].nunique():,}")
    logger.info(f"    Matched Entities    : {n_matched:,} ({match_pct:.2f}%)")
    logger.info(f"    Singletons (Empty)  : {n_singletons:,}")
    logger.info(f"    Total Scored Pairs  : {total_pairs_scored:,}")

    t_total = time.time() - t_start
    logger.info("=" * 70)
    logger.info(f"FULL 1.73M TEST INFERENCE & VALIDATION COMPLETE IN {t_total:.1f} SECONDS!")
    logger.info("=" * 70)


if __name__ == "__main__":
    run_full_test_inference(batch_size=100000, threshold=0.50)
