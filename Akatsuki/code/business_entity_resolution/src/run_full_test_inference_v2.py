"""
Akatsuki/code/business_entity_resolution/src/run_full_test_inference_v2.py
--------------------------------------------------------------------------
ROBUST Chunked 1.73M S1 Test Inference Engine - V2

Key fixes over V1:
  1. Preprocesses S2 and S3 in CHUNKS of 200K rows (not all-at-once)
     -> Prevents the silent memory hang on 5M+ record preprocessing
  2. Logs progress every chunk with timing
  3. Uses threshold=0.50 (empirically optimal per OOF sweep: Macro F0.5=0.9980)
  4. Clears intermediate DataFrames to free RAM between chunks
"""

import gc
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

if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from data_loader import DataLoader
from preprocessing.preprocessor import EntityPreprocessor
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

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("FullTestInference_V2")

MODEL_PKL  = REPO_ROOT / "model"  / "lgbm_entity_match.pkl"
OUTPUT_TSV = REPO_ROOT / "output" / "matching_results.tsv"
REPORT_CSV = REPO_ROOT / "reports"/ "inference_report_v3.csv"

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

PREP_CHUNK = 200_000   # rows per preprocessing chunk (avoids memory hang)
S1_BATCH   = 100_000   # S1 entities per scoring batch


def preprocess_in_chunks(df_raw: pd.DataFrame, source_name: str) -> pd.DataFrame:
    """Preprocess a large DataFrame in chunks to avoid memory hang."""
    preprocessor = EntityPreprocessor()
    n = len(df_raw)
    chunks = []
    for start in range(0, n, PREP_CHUNK):
        end = min(start + PREP_CHUNK, n)
        chunk = df_raw.iloc[start:end].copy()
        proc  = preprocessor.preprocess_dataframe(chunk)
        # Add non-latin flag (needed by some blockers)
        proc["name_is_non_latin"] = chunk["business_name"].apply(
            lambda s: any(ord(c) > 127 for c in str(s)) if pd.notna(s) else False
        )
        chunks.append(proc)
        logger.info(f"  [{source_name}] Preprocessed {end:,}/{n:,} rows "
                    f"({100*end/n:.1f}%)")
        gc.collect()
    return pd.concat(chunks, ignore_index=True)


def run_full_test_inference(s1_batch: int = S1_BATCH, threshold: float = 0.50):
    t_start = time.time()
    logger.info("=" * 70)
    logger.info("AKATSUKI V2 — ROBUST CHUNKED 1.73M TEST INFERENCE ENGINE")
    logger.info(f"Decision Threshold tau = {threshold:.2f}  |  Prep chunk = {PREP_CHUNK:,}")
    logger.info("=" * 70)

    # ------------------------------------------------------------------ #
    # 1. Load Model
    # ------------------------------------------------------------------ #
    if not MODEL_PKL.exists():
        raise FileNotFoundError(f"Model not found: {MODEL_PKL}")
    with open(MODEL_PKL, "rb") as f:
        model = pickle.load(f)
    logger.info(f"Loaded LightGBM model: {MODEL_PKL}")

    # ------------------------------------------------------------------ #
    # 2. Load Raw Test Data
    # ------------------------------------------------------------------ #
    loader = DataLoader()
    logger.info("Loading FULL test split raw records...")
    t0 = time.time()
    raw_s1 = loader.load_source("s1", split="test")
    raw_s2 = loader.load_source("s2", split="test")
    raw_s3 = loader.load_source("s3", split="test")
    logger.info(
        f"Raw loaded in {time.time()-t0:.1f}s  |  "
        f"S1={len(raw_s1):,}  S2={len(raw_s2):,}  S3={len(raw_s3):,}"
    )
    total_s1 = len(raw_s1)

    # ------------------------------------------------------------------ #
    # 3. Chunked Preprocessing  (the V1 hang-point — now fixed)
    # ------------------------------------------------------------------ #
    logger.info("--- STEP 3: Chunked Preprocessing ---")

    logger.info("Preprocessing S1 test...")
    t0 = time.time()
    df_s1 = preprocess_in_chunks(raw_s1, "S1")
    logger.info(f"S1 preprocessing done in {time.time()-t0:.1f}s")
    del raw_s1; gc.collect()

    logger.info("Preprocessing S2 test...")
    t0 = time.time()
    df_s2 = preprocess_in_chunks(raw_s2, "S2")
    logger.info(f"S2 preprocessing done in {time.time()-t0:.1f}s  |  rows={len(df_s2):,}")
    del raw_s2; gc.collect()

    logger.info("Preprocessing S3 test...")
    t0 = time.time()
    df_s3 = preprocess_in_chunks(raw_s3, "S3")
    logger.info(f"S3 preprocessing done in {time.time()-t0:.1f}s  |  rows={len(df_s3):,}")
    del raw_s3; gc.collect()

    # ------------------------------------------------------------------ #
    # 4. Pre-build S2/S3 Candidate Indexes ONCE
    # ------------------------------------------------------------------ #
    logger.info("--- STEP 4: Building blocking indexes over S2+S3 ---")
    t0 = time.time()
    exact_idx, core_idx, sorted_idx = build_name_indexes(df_s2, df_s3)
    rare_idx, _, del_idx = build_rare_token_index(df_s2, df_s3, max_doc_freq=50)
    num_loc_idx, _ = build_numeric_locality_index(df_s2, df_s3, max_locality_freq=100)
    translit_idx = build_translit_index(df_s2, df_s3)
    phonetic_idx, _ = build_phonetic_index(df_s2, df_s3, max_freq_cap=100)
    logger.info(f"All indexes built in {time.time()-t0:.1f}s")

    # Fast lookup dicts
    s2_map = df_s2.set_index("entity_id").to_dict("index")
    s3_map = df_s3.set_index("entity_id").to_dict("index")
    empty_rec = {"business_name_norm": "", "business_address_norm": "", "country_norm": ""}

    # ------------------------------------------------------------------ #
    # 5. Batch Scoring: S1 entities -> candidates -> features -> LightGBM
    # ------------------------------------------------------------------ #
    all_s1_ids = df_s1["entity_id"].tolist()
    num_batches = (total_s1 + s1_batch - 1) // s1_batch
    logger.info(f"--- STEP 5: Scoring {total_s1:,} S1 entities in {num_batches} batches ---")

    results_map = {}
    total_pairs_scored = 0
    report_rows = []

    for b_idx in range(num_batches):
        t_b = time.time()
        start = b_idx * s1_batch
        end   = min(start + s1_batch, total_s1)
        sub_s1     = df_s1.iloc[start:end]
        sub_s1_ids = sub_s1["entity_id"].tolist()

        # Multi-pass blocking
        pass_results = {}
        cand_exact, cand_core = block_exact_and_core_name(sub_s1, exact_idx, core_idx, sorted_idx)
        pass_results["block_exact_name"] = cand_exact
        pass_results["block_core"]       = cand_core

        cand_rare = block_rare_tokens(sub_s1, rare_idx, del_idx, max_candidates_per_s1=50)
        pass_results["block_rare_token"] = cand_rare

        cand_addr = block_address_rare(sub_s1, df_s2, df_s3, max_candidates_per_s1=50)
        pass_results["block_address"] = cand_addr

        cand_num = block_numeric_locality(sub_s1, num_loc_idx, max_candidates_per_s1=30)
        pass_results["block_numeric"] = cand_num

        cand_translit = block_transliteration(sub_s1, translit_idx, max_candidates_per_s1=50)
        pass_results["block_translit"] = cand_translit

        cand_ann = build_and_query_ann_per_country(sub_s1, df_s2, df_s3, top_k=30, min_similarity=0.50)
        pass_results["block_ann"] = cand_ann

        cand_phonetic = block_phonetic(sub_s1, phonetic_idx, max_candidates_per_s1=30)
        pass_results["block_phonetic"] = cand_phonetic

        # Union + prune
        df_long = union_candidate_passes(pass_results, sub_s1_ids)
        df_long  = prune_candidate_matrix(df_long, max_total_per_s1=100, adaptive_pruning=True)

        n_cands = len(df_long)
        n_scored = 0

        if n_cands > 0:
            s1_sub_map = sub_s1.set_index("entity_id").to_dict("index")
            rows = []
            pair_keys = []
            for _, r in df_long.iterrows():
                s1_id   = r["source1_id"]
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

            X_b   = np.array(rows, dtype=np.float32)
            preds = model.predict(X_b)
            n_scored = len(preds)
            total_pairs_scored += n_scored

            for (s1_id, cand_id), score in zip(pair_keys, preds):
                if score >= threshold:
                    results_map.setdefault(s1_id, []).append(cand_id)

        t_elapsed = time.time() - t_b
        pct_done  = 100 * end / total_s1
        logger.info(
            f"  Batch {b_idx+1:>3}/{num_batches} | "
            f"S1 {end:>9,}/{total_s1:,} ({pct_done:5.1f}%) | "
            f"cands={n_cands:>7,} scored={n_scored:>7,} | "
            f"{t_elapsed:.1f}s"
        )
        report_rows.append({
            "batch": b_idx + 1,
            "s1_start": start,
            "s1_end": end,
            "candidates": n_cands,
            "scored_pairs": n_scored,
            "elapsed_s": round(t_elapsed, 2),
        })

    # ------------------------------------------------------------------ #
    # 6. Write Submission File
    # ------------------------------------------------------------------ #
    logger.info("\n--- STEP 6: Writing official submission TSV ---")
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
    logger.info(f"Submission written -> {OUTPUT_TSV}")

    # ------------------------------------------------------------------ #
    # 7. Sanity Assertions
    # ------------------------------------------------------------------ #
    logger.info("\n--- STEP 7: Integrity Audit ---")
    assert len(df_sub) == total_s1, \
        f"ROW COUNT MISMATCH! Expected {total_s1:,}, got {len(df_sub):,}"
    assert df_sub["source1_entity_id"].nunique() == total_s1, \
        "DUPLICATE S1 IDs detected!"
    assert list(df_sub.columns) == ["source1_entity_id", "matched_entity_ids"], \
        f"Column format wrong: {list(df_sub.columns)}"
    assert df_sub["source1_entity_id"].isnull().sum() == 0, \
        "NULL source1_entity_ids found!"

    n_matched   = int((df_sub["matched_entity_ids"].fillna("") != "").sum())
    n_singletons = total_s1 - n_matched
    match_pct   = n_matched / total_s1 * 100

    logger.info("  ALL SANITY CHECKS PASSED ✅")
    logger.info(f"    Total S1 rows      : {total_s1:,}")
    logger.info(f"    Unique S1 IDs      : {df_sub['source1_entity_id'].nunique():,}")
    logger.info(f"    Matched entities   : {n_matched:,} ({match_pct:.2f}%)")
    logger.info(f"    Singletons (empty) : {n_singletons:,}")
    logger.info(f"    Total pairs scored : {total_pairs_scored:,}")

    # Save batch report
    pd.DataFrame(report_rows).to_csv(REPORT_CSV, index=False)
    logger.info(f"Batch report -> {REPORT_CSV}")

    t_total = time.time() - t_start
    logger.info("=" * 70)
    logger.info(f"COMPLETE: Full {total_s1:,} S1 Inference finished in {t_total/60:.1f} minutes")
    logger.info("=" * 70)


if __name__ == "__main__":
    run_full_test_inference(s1_batch=S1_BATCH, threshold=0.50)
