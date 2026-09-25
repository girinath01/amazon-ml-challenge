"""
run_full_test_inference_v3.py
------------------------------
DISK-BACKED 1.73M Inference Engine — V3 (CORRECTED)

Memory architecture:
  1. Preprocess S1 -> parquet cache -> del raw_s1, del df_s1
  2. Preprocess S2 -> parquet cache -> del raw_s2, del df_s2
  3. Preprocess S3 -> parquet cache -> del raw_s3, del df_s3
  4. Reload S2+S3 from parquet -> build all blocking indexes
     -> keep SLIM df_s2/df_s3 (only columns needed by address/ANN blockers)
     -> build lookup dicts (entity_id -> flat dict) for feature computation
  5. Stream S1 parquet in batches of 50K -> block -> score -> collect
  6. Write matching_results.tsv (1,732,544 rows) -> sanity audit

Key corrections vs V2:
  - No psutil dependency (removed)
  - block_address_rare and build_and_query_ann_per_country receive slim
    S2/S3 DataFrames kept in RAM (much smaller than full preprocessed DFs)
  - PREP_CHUNK enlarged to 300K (fewer chunks, same safety margin)
  - force_reprocess=False uses cached parquet on rerun (major time saver)
"""

import gc
import sys
import time
import pickle
import logging
from pathlib import Path
import numpy as np
import pandas as pd

# ------------------------------------------------------------------ #
# Paths
# ------------------------------------------------------------------ #
THIS_DIR  = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parent.parent.parent          # Akatsuki/
CACHE_DIR = REPO_ROOT / "cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

S1_PARQUET = CACHE_DIR / "test_s1_prep.parquet"
S2_PARQUET = CACHE_DIR / "test_s2_prep.parquet"
S3_PARQUET = CACHE_DIR / "test_s3_prep.parquet"
MODEL_PKL  = REPO_ROOT / "model"  / "lgbm_entity_match.pkl"
OUTPUT_TSV = REPO_ROOT / "output" / "matching_results.tsv"
REPORT_CSV = REPO_ROOT / "reports"/ "inference_report_v3.csv"

if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from data_loader import DataLoader
from preprocessing.preprocessor import EntityPreprocessor
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
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("InferenceV3")

# Columns needed by address + ANN blockers (keep slim copies in RAM)
SLIM_COLS = [
    "entity_id", "name_norm", "name_core", "name_sorted_tokens",
    "name_translit", "name_phonetic", "name_is_non_latin",
    "address_norm", "address_numbers", "house_number", "postal_code",
    "country_norm", "address_missing",
]

# Columns needed for feature computation lookup dict
LOOKUP_COLS = [
    "entity_id", "business_name_norm", "business_address_norm", "country_norm",
]

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

PREP_CHUNK = 300_000   # rows per preprocessing chunk
S1_BATCH   = 50_000    # S1 entities per scoring batch


# ------------------------------------------------------------------ #
# Helpers
# ------------------------------------------------------------------ #

def preprocess_and_save(raw_df: pd.DataFrame, source_name: str,
                        out_path: Path, force: bool = False) -> None:
    """Preprocess raw_df in chunks and save to parquet. No-op if cached."""
    if out_path.exists() and not force:
        size_mb = out_path.stat().st_size / 1e6
        logger.info(f"  [CACHE HIT] {source_name} -> {out_path} ({size_mb:.0f} MB)")
        return

    prep = EntityPreprocessor()
    n    = len(raw_df)
    chunks = []
    for start in range(0, n, PREP_CHUNK):
        end   = min(start + PREP_CHUNK, n)
        chunk = raw_df.iloc[start:end].copy()
        proc  = prep.preprocess_dataframe(chunk)
        proc["name_is_non_latin"] = chunk["business_name"].apply(
            lambda s: any(ord(c) > 127 for c in str(s)) if pd.notna(s) else False
        )
        chunks.append(proc)
        logger.info(f"  [{source_name}] {end:>10,}/{n:,} ({100*end/n:5.1f}%)")
        gc.collect()

    df_out = pd.concat(chunks, ignore_index=True)
    logger.info(f"  [{source_name}] Saving {len(df_out):,} rows to parquet ...")

    # Serialise list-type columns for parquet compatibility
    for col in ["name_tokens", "address_tokens", "address_numbers"]:
        if col in df_out.columns:
            df_out[col] = df_out[col].apply(
                lambda v: "|".join(v) if isinstance(v, list) else str(v)
            )
    df_out.to_parquet(out_path, index=False, compression="snappy")
    logger.info(f"  [{source_name}] Saved: {out_path.stat().st_size/1e6:.0f} MB")
    del df_out, chunks
    gc.collect()


def load_slim(parquet_path: Path, source_name: str) -> pd.DataFrame:
    """Load only the columns needed by blockers (much smaller than full DF)."""
    available = pd.read_parquet(parquet_path, columns=None).columns.tolist()
    cols = [c for c in SLIM_COLS if c in available]
    df = pd.read_parquet(parquet_path, columns=cols)
    # Restore address_numbers list column if it was serialised as pipe-joined string
    if "address_numbers" in df.columns:
        df["address_numbers"] = df["address_numbers"].apply(
            lambda v: v.split("|") if isinstance(v, str) and v else []
        )
    logger.info(f"  Loaded slim {source_name}: {len(df):,} rows, {len(cols)} cols")
    return df


def build_lookup(parquet_path: Path) -> dict:
    """entity_id -> {business_name_norm, business_address_norm, country_norm}."""
    avail = pd.read_parquet(parquet_path, columns=None).columns.tolist()
    cols  = [c for c in LOOKUP_COLS if c in avail]
    df    = pd.read_parquet(parquet_path, columns=cols)
    lkp   = df.set_index("entity_id").to_dict("index")
    del df; gc.collect()
    return lkp


# ------------------------------------------------------------------ #
# Main
# ------------------------------------------------------------------ #

def run_inference(threshold: float = 0.50, force_reprocess: bool = False):
    t_start = time.time()
    logger.info("=" * 70)
    logger.info("AKATSUKI V3 — DISK-BACKED 1.73M TEST INFERENCE (CORRECTED)")
    logger.info(f"tau={threshold:.2f}  prep_chunk={PREP_CHUNK:,}  s1_batch={S1_BATCH:,}")
    logger.info(f"Cache: {CACHE_DIR}")
    logger.info("=" * 70)

    # ---- Load model ------------------------------------------------ #
    with open(MODEL_PKL, "rb") as f:
        model = pickle.load(f)
    logger.info(f"Model loaded: {MODEL_PKL}")

    # ---- Load raw data --------------------------------------------- #
    loader = DataLoader()
    logger.info("Loading raw test records...")
    t0 = time.time()
    raw_s1 = loader.load_source("s1", split="test")
    raw_s2 = loader.load_source("s2", split="test")
    raw_s3 = loader.load_source("s3", split="test")
    total_s1 = len(raw_s1)
    logger.info(
        f"Loaded in {time.time()-t0:.1f}s | "
        f"S1={total_s1:,}  S2={len(raw_s2):,}  S3={len(raw_s3):,}"
    )

    # ==================================================================
    # STEPS 1-3: Preprocess to parquet — one source at a time
    # ==================================================================
    logger.info("\n--- STEP 1: Preprocess + cache S1 ---")
    preprocess_and_save(raw_s1, "S1", S1_PARQUET, force=force_reprocess)
    del raw_s1; gc.collect()

    logger.info("\n--- STEP 2: Preprocess + cache S2 ---")
    preprocess_and_save(raw_s2, "S2", S2_PARQUET, force=force_reprocess)
    del raw_s2; gc.collect()

    logger.info("\n--- STEP 3: Preprocess + cache S3 ---")
    preprocess_and_save(raw_s3, "S3", S3_PARQUET, force=force_reprocess)
    del raw_s3; gc.collect()

    # ==================================================================
    # STEP 4: Build blocking indexes from slim S2+S3 DataFrames
    # ==================================================================
    logger.info("\n--- STEP 4: Load slim S2+S3 and build indexes ---")
    t0 = time.time()
    df_s2_slim = load_slim(S2_PARQUET, "S2")
    df_s3_slim = load_slim(S3_PARQUET, "S3")
    logger.info(f"Slim DFs loaded in {time.time()-t0:.1f}s")

    t0 = time.time()
    exact_idx, core_idx, sorted_idx = build_name_indexes(df_s2_slim, df_s3_slim)
    rare_idx, _, del_idx            = build_rare_token_index(df_s2_slim, df_s3_slim, max_doc_freq=50)
    num_loc_idx, _                  = build_numeric_locality_index(df_s2_slim, df_s3_slim, max_locality_freq=100)
    translit_idx                    = build_translit_index(df_s2_slim, df_s3_slim)
    phonetic_idx, _                 = build_phonetic_index(df_s2_slim, df_s3_slim, max_freq_cap=100)
    logger.info(f"All indexes built in {time.time()-t0:.1f}s")

    # Feature lookup dicts (flat, only 3 fields per entity)
    logger.info("Building feature lookup dicts from parquet ...")
    s2_map = build_lookup(S2_PARQUET)
    s3_map = build_lookup(S3_PARQUET)
    logger.info(f"Lookups ready: S2={len(s2_map):,}  S3={len(s3_map):,}")

    empty_rec = {"business_name_norm": "", "business_address_norm": "", "country_norm": ""}

    # ==================================================================
    # STEP 5: Stream S1 in batches -> block -> score -> accumulate
    # ==================================================================
    df_s1 = pd.read_parquet(S1_PARQUET)
    all_s1_ids  = df_s1["entity_id"].tolist()
    num_batches = (total_s1 + S1_BATCH - 1) // S1_BATCH
    logger.info(f"\n--- STEP 5: Scoring {total_s1:,} S1 in {num_batches} batches ---")

    results_map        = {}
    total_pairs_scored = 0
    report_rows        = []

    for b_idx in range(num_batches):
        t_b   = time.time()
        start = b_idx * S1_BATCH
        end   = min(start + S1_BATCH, total_s1)
        sub_s1     = df_s1.iloc[start:end]
        sub_s1_ids = sub_s1["entity_id"].tolist()

        # Multi-pass blocking (all blockers receive correct DF types)
        pass_res = {}
        cand_exact, cand_core = block_exact_and_core_name(sub_s1, exact_idx, core_idx, sorted_idx)
        pass_res["block_exact_name"] = cand_exact
        pass_res["block_core"]       = cand_core
        pass_res["block_rare_token"] = block_rare_tokens(
            sub_s1, rare_idx, del_idx, max_candidates_per_s1=50)
        pass_res["block_address"]    = block_address_rare(
            sub_s1, df_s2_slim, df_s3_slim, max_candidates_per_s1=50)
        pass_res["block_numeric"]    = block_numeric_locality(
            sub_s1, num_loc_idx, max_candidates_per_s1=30)
        pass_res["block_translit"]   = block_transliteration(
            sub_s1, translit_idx, max_candidates_per_s1=50)
        pass_res["block_ann"]        = build_and_query_ann_per_country(
            sub_s1, df_s2_slim, df_s3_slim, top_k=30, min_similarity=0.50)
        pass_res["block_phonetic"]   = block_phonetic(
            sub_s1, phonetic_idx, max_candidates_per_s1=30)

        df_long  = union_candidate_passes(pass_res, sub_s1_ids)
        df_long  = prune_candidate_matrix(df_long, max_total_per_s1=100, adaptive_pruning=True)
        n_cands  = len(df_long)
        n_scored = 0

        if n_cands > 0:
            s1_sub_map = sub_s1.set_index("entity_id").to_dict("index")
            rows, pair_keys = [], []
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
        logger.info(
            f"  Batch {b_idx+1:>3}/{num_batches} | "
            f"S1 {end:>9,}/{total_s1:,} ({100*end/total_s1:5.1f}%) | "
            f"cands={n_cands:>6,} scored={n_scored:>6,} | {t_elapsed:.1f}s"
        )
        report_rows.append(dict(
            batch=b_idx+1, s1_start=start, s1_end=end,
            candidates=n_cands, scored=n_scored, elapsed_s=round(t_elapsed, 2)
        ))

    # ==================================================================
    # STEP 6: Write submission TSV + sanity audit
    # ==================================================================
    logger.info("\n--- STEP 6: Writing submission TSV ---")
    sub_rows = [
        {"source1_entity_id": sid,
         "matched_entity_ids": ",".join(sorted(set(results_map.get(sid, []))))}
        for sid in all_s1_ids
    ]
    df_sub = pd.DataFrame(sub_rows)
    OUTPUT_TSV.parent.mkdir(parents=True, exist_ok=True)
    df_sub.to_csv(OUTPUT_TSV, sep="\t", index=False)

    # Assertions
    assert len(df_sub) == total_s1,                            f"ROW MISMATCH: {len(df_sub)} vs {total_s1}"
    assert df_sub["source1_entity_id"].nunique() == total_s1,  "DUPLICATE S1 IDs!"
    assert df_sub["source1_entity_id"].isnull().sum() == 0,    "NULL S1 IDs!"
    assert list(df_sub.columns) == ["source1_entity_id", "matched_entity_ids"]

    n_matched    = int((df_sub["matched_entity_ids"].fillna("") != "").sum())
    n_singletons = total_s1 - n_matched

    logger.info("  ALL SANITY CHECKS PASSED ✅")
    logger.info(f"  Total S1 rows      : {total_s1:,}")
    logger.info(f"  Unique S1 IDs      : {df_sub['source1_entity_id'].nunique():,}")
    logger.info(f"  Matched entities   : {n_matched:,} ({n_matched/total_s1*100:.2f}%)")
    logger.info(f"  Singletons (empty) : {n_singletons:,}")
    logger.info(f"  Total scored pairs : {total_pairs_scored:,}")

    pd.DataFrame(report_rows).to_csv(REPORT_CSV, index=False)
    logger.info(f"  Batch report       : {REPORT_CSV}")

    t_total = time.time() - t_start
    logger.info("=" * 70)
    logger.info(f"COMPLETE in {t_total/60:.1f} min — {OUTPUT_TSV}")
    logger.info("=" * 70)


if __name__ == "__main__":
    run_inference(threshold=0.50, force_reprocess=False)
