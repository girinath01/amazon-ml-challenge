"""
run_member3_complete.py
=======================
Complete Member 3 pipeline — runs ALL steps end to end:

  Phase A  (if parquet does not exist)
    Step 1 : Load data + build benchmark pair dataset
    Step 2 : Extract 28 pairwise features
    Step 3 : Feature quality checks

  Phase B  (always runs)
    Step 4 : Optuna HP search (40 trials)
    Step 5 : 5-fold Stratified Cross-Validation
    Step 6 : Final LightGBM model training
    Step 7 : Save model + training metadata

  Phase C  (always runs)
    Step 8 : Inference on available candidate pairs
    Step 9 : Write matching_results.tsv (submission format)
    Step 10: Write inference_report.csv

Usage:
    python run_member3_complete.py            # full pipeline
    python run_member3_complete.py --no-optuna  # skip Optuna (faster)
    python run_member3_complete.py --infer-only # skip training, only run inference
"""

import sys
import time
import argparse
from pathlib import Path

t_start = time.time()

# ── Argument parsing ───────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="Member 3 Complete Pipeline")
parser.add_argument("--no-optuna",   action="store_true",
                    help="Skip Optuna HP search (use baseline params)")
parser.add_argument("--infer-only",  action="store_true",
                    help="Skip training; run inference only (model must exist)")
parser.add_argument("--no-infer",    action="store_true",
                    help="Skip inference after training")
parser.add_argument("--optuna-trials", type=int, default=40,
                    help="Number of Optuna trials (default: 40)")
args = parser.parse_args()

# ── Path setup ─────────────────────────────────────────────────────────────
THIS       = Path(__file__).resolve().parent
REPO_ROOT  = THIS / "amazon-ml-challenge" / "Akatsuki"
SRC_DIR    = REPO_ROOT / "code" / "business_entity_resolution" / "src"
DATA_DIR   = REPO_ROOT / "data"
REPORT_DIR = REPO_ROOT / "reports"
MODEL_DIR  = REPO_ROOT / "model"

for d in [DATA_DIR, REPORT_DIR, MODEL_DIR]:
    d.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(SRC_DIR))

FEATURE_PARQUET = DATA_DIR / "training_pair_features.parquet"
MODEL_PKL       = MODEL_DIR / "lgbm_entity_match.pkl"

# ══════════════════════════════════════════════════════════════════════════
print("=" * 70)
print("MEMBER 3 — COMPLETE PIPELINE (Feature Eng + Training + Inference)")
print("=" * 70)

# ══ Phase A: Feature engineering (if parquet missing) ════════════════════
if not FEATURE_PARQUET.exists() or args.infer_only is False:
    if not FEATURE_PARQUET.exists():
        print("\n>>> PHASE A: Building feature matrix (parquet not found)...")

        from pair_builder import PairBuilder
        from features.pair_features import compute_pair_features, FEATURE_NAMES
        import pandas as pd
        import numpy as np

        builder = PairBuilder()
        print("\n>>> STEP 1-2: Constructing Training Pair Dataset...")
        pairs_df, stats, s1_recs, s2_recs, s3_recs = builder.build_benchmark_pairs(
            num_s1_entities=10000, neg_to_pos_ratio=2.0
        )
        print("\n--- Training Pair Statistics ---")
        for k, v in stats.items():
            print(f"  • {k}: {v:,}" if isinstance(v, int) else f"  • {k}: {v}")

        print(f"\n>>> STEP 3: Extracting Features for {len(pairs_df):,} pairs...")
        feat_df = builder.generate_feature_matrix(pairs_df, s1_recs, s2_recs, s3_recs)
        print(f"Feature matrix shape: {feat_df.shape}")

        # Quality checks
        feature_cols = FEATURE_NAMES
        nan_count = int(feat_df[feature_cols].isna().sum().sum())
        inf_count = int((feat_df[feature_cols].abs() == float("inf")).sum().sum())
        const_feats = [c for c in feature_cols if feat_df[c].nunique() <= 1]
        print(f"\n--- Feature Quality ---")
        print(f"  ✓ NaN values : {nan_count}")
        print(f"  ✓ Inf values : {inf_count}")
        print(f"  ✓ Constant   : {const_feats or 'None'}")

        feat_df.to_parquet(FEATURE_PARQUET, index=False)
        print(f"\n  Saved → {FEATURE_PARQUET} ({FEATURE_PARQUET.stat().st_size:,} bytes)")
    else:
        print(f"\n>>> PHASE A: Skipping — parquet already exists: {FEATURE_PARQUET}")

# ══ Phase B: LightGBM Training ════════════════════════════════════════════
if not args.infer_only:
    print("\n>>> PHASE B: LightGBM Training")

    from trainer import EntityMatchTrainer

    trainer = EntityMatchTrainer(parquet_path=FEATURE_PARQUET)
    trainer.run_full_training(
        run_optuna=not args.no_optuna,
        optuna_trials=args.optuna_trials,
    )

    print("\n>>> PHASE B COMPLETE")
    import numpy as np
    cv_mean = float(np.mean(trainer.cv_scores))
    cv_std  = float(np.std(trainer.cv_scores))
    print(f"  CV Mean F0.5  : {cv_mean:.4f} ± {cv_std:.4f}")
    print(f"  Optimal thr.  : {trainer.best_threshold:.2f}")
    print(f"  Model saved   : {MODEL_PKL}")

    print("\n  Top 10 Features by Gain:")
    print(trainer.feature_importance.head(10)[["feature","gain"]].to_string(index=False))

# ══ Phase C: Inference ════════════════════════════════════════════════════
if not args.no_infer:
    print("\n>>> PHASE C: Inference — Generating matching_results.tsv")

    from inference import EntityMatchInference

    inf = EntityMatchInference()
    inf.run()

    print("\n>>> PHASE C COMPLETE")

# ══ Final Summary ══════════════════════════════════════════════════════════
elapsed = round(time.time() - t_start, 1)
print("\n" + "=" * 70)
print("ALL PHASES COMPLETE")
print("=" * 70)
print(f"  Total time            : {elapsed}s")
print(f"  Feature matrix        : {FEATURE_PARQUET}")
print(f"  LightGBM model        : {MODEL_PKL}")
print(f"  Training metadata     : {MODEL_DIR / 'training_metadata.json'}")
print(f"  Feature importance    : {REPORT_DIR / 'feature_importance.csv'}")
print(f"  Submission output     : {REPO_ROOT / 'output' / 'matching_results.tsv'}")
print(f"  Inference report      : {REPORT_DIR / 'inference_report.csv'}")
print("=" * 70)
