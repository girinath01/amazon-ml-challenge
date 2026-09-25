"""
Akatsuki/run_full_end_to_end.py
--------------------------------
Master End-to-End Execution Pipeline for Amazon ML Challenge 2026:
Business Entity Resolution.

Executes all 4 components sequentially:
  Step 1: Data Ingestion & Preprocessing (Member 1)
  Step 2: 8-Pass Candidate Blocking B0–B7 & Provenance Matrix (Member 2)
  Step 3: 28-Feature Extraction & LightGBM Model Training + CV Threshold Search (Member 3)
  Step 4: Inference Engine & Official Competition Submission Export (matching_results.tsv)
  Step 5: Submission Format & Integrity Validation Audit
"""

import sys
import time
import json
import logging
from pathlib import Path
import pandas as pd

# Setup paths
THIS_DIR = Path(__file__).resolve().parent
SRC_DIR = THIS_DIR / "code" / "business_entity_resolution" / "src"
MEMBER2_DIR = THIS_DIR / "member2"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
if str(MEMBER2_DIR) not in sys.path:
    sys.path.insert(0, str(MEMBER2_DIR))

# Setup logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger("Akatsuki_Master_Pipeline")


def run_master_pipeline(train_sample_rows: int = 5000, run_optuna: bool = False):
    t_start = time.time()
    logger.info("=" * 70)
    logger.info("AKATSUKI BUSINESS ENTITY RESOLUTION — MASTER END-TO-END PIPELINE")
    logger.info("=" * 70)

    # -------------------------------------------------------------
    # Step 1: Data Ingestion & Preprocessing
    # -------------------------------------------------------------
    logger.info("\n[STEP 1/5] Loading & Preprocessing Raw Data Sources...")
    from data_loader import DataLoader
    from preprocessing.preprocessor import preprocess_source

    loader = DataLoader()
    raw_s1_train = loader.load_source("s1", split="train", nrows=train_sample_rows)
    raw_s2_train = loader.load_source("s2", split="train", nrows=train_sample_rows * 5)
    raw_s3_train = loader.load_source("s3", split="train", nrows=train_sample_rows * 5)

    df_s1_tr = preprocess_source(raw_s1_train, verbose=False)
    df_s2_tr = preprocess_source(raw_s2_train, verbose=False)
    df_s3_tr = preprocess_source(raw_s3_train, verbose=False)

    logger.info(f"  Preprocessed Train S1={len(df_s1_tr):,}, S2={len(df_s2_tr):,}, S3={len(df_s3_tr):,}")

    # -------------------------------------------------------------
    # Step 2: Member 2 Candidate Blocking B0–B7
    # -------------------------------------------------------------
    logger.info("\n[STEP 2/5] Running 8-Layer Multi-Pass Candidate Blocking (B0–B7)...")
    from blocking.candidate_generator import run_candidate_generation_pipeline

    df_long_tr, df_tsv_tr, timing_dict = run_candidate_generation_pipeline(
        df_s1_tr, df_s2_tr, df_s3_tr, max_total_per_s1=200
    )

    out_parquet = THIS_DIR / "member2" / "output" / "candidate_pairs_long.parquet"
    out_tsv = THIS_DIR / "output" / "candidate_pairs.tsv"
    out_parquet.parent.mkdir(parents=True, exist_ok=True)
    out_tsv.parent.mkdir(parents=True, exist_ok=True)

    df_long_tr.to_parquet(out_parquet, index=False)
    df_tsv_tr.to_csv(out_tsv, sep="\t", index=False)
    logger.info(f"  Candidate Generation Complete! Generated {len(df_long_tr):,} pairs.")
    logger.info(f"  Saved candidates to {out_parquet.name} and {out_tsv.name}")

    # -------------------------------------------------------------
    # Step 3: Feature Extraction & LightGBM Training
    # -------------------------------------------------------------
    logger.info("\n[STEP 3/5] Extracting Features & Training LightGBM Model...")
    from features.pair_features import compute_pair_features, FEATURE_NAMES

    # Load GT dictionary to label candidates
    gt_dict = loader.load_ground_truth_dict()

    s1_map = df_s1_tr.set_index("entity_id").to_dict("index")
    s2_map = df_s2_tr.set_index("entity_id").to_dict("index")
    s3_map = df_s3_tr.set_index("entity_id").to_dict("index")

    empty_rec = {"business_name_norm": "", "business_address_norm": "", "country_norm": ""}
    feat_rows = []

    for _, row in df_long_tr.iterrows():
        s1_id = row["source1_id"]
        cand_id = row["candidate_id"]
        label = 1 if cand_id in gt_dict.get(s1_id, set()) else 0

        rec1 = s1_map.get(s1_id, empty_rec)
        rec2 = s2_map.get(cand_id, s3_map.get(cand_id, empty_rec))

        feats = compute_pair_features(
            name1=rec1.get("business_name_norm", ""),
            name2=rec2.get("business_name_norm", ""),
            addr1=rec1.get("business_address_norm", ""),
            addr2=rec2.get("business_address_norm", ""),
            country1=rec1.get("country_norm", ""),
            country2=rec2.get("country_norm", ""),
        )
        feats["source1_id"] = s1_id
        feats["candidate_id"] = cand_id
        feats["label"] = label
        feat_rows.append(feats)

    df_feats = pd.DataFrame(feat_rows)
    data_dir = THIS_DIR / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    parquet_feat_path = data_dir / "training_pair_features.parquet"
    df_feats.to_parquet(parquet_feat_path, index=False)
    logger.info(f"  Saved {len(df_feats):,} training feature rows to {parquet_feat_path.name}")

    # Train model
    from trainer import EntityMatchTrainer
    trainer = EntityMatchTrainer(parquet_path=parquet_feat_path)
    trainer.run_full_training(run_optuna=run_optuna, optuna_trials=20)

    # -------------------------------------------------------------
    # Step 4: Inference Engine & Official Submission Generation
    # -------------------------------------------------------------
    logger.info("\n[STEP 4/5] Executing Test Inference & Submission TSV Export...")
    from inference import EntityMatchInference
    inf = EntityMatchInference()
    inf.run()

    # -------------------------------------------------------------
    # Step 5: Submission Integrity Audit
    # -------------------------------------------------------------
    logger.info("\n[STEP 5/5] Running Submission Format Sanity Audit...")
    sub_tsv_path = THIS_DIR / "output" / "matching_results.tsv"
    assert sub_tsv_path.exists(), "matching_results.tsv does not exist!"

    df_sub = pd.read_csv(sub_tsv_path, sep="\t", dtype=str)
    assert list(df_sub.columns) == ["source1_entity_id", "matched_entity_ids"], \
        f"Incorrect header columns: {list(df_sub.columns)}"
    assert len(df_sub) > 0, "Submission file is empty!"
    assert df_sub["source1_entity_id"].isnull().sum() == 0, "Null source1_entity_ids found!"

    n_matched = int((df_sub["matched_entity_ids"].fillna("") != "").sum())
    n_single = len(df_sub) - n_matched
    match_pct = (n_matched / len(df_sub)) * 100

    logger.info(f"  Submission Sanity Check PASSED:")
    logger.info(f"    Total S1 Records    : {len(df_sub):,}")
    logger.info(f"    Matched Records     : {n_matched:,} ({match_pct:.2f}%)")
    logger.info(f"    Singleton Records   : {n_single:,}")
    logger.info(f"    Submission File     : {sub_tsv_path}")

    t_total = time.time() - t_start
    logger.info("\n" + "=" * 70)
    logger.info(f"ALL COMPONENTS EXECUTED SUCCESSFULLY IN {t_total:.1f} SECONDS!")
    logger.info("=" * 70)


if __name__ == "__main__":
    run_master_pipeline(train_sample_rows=5000, run_optuna=False)
