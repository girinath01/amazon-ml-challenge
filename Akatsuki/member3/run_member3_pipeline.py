"""
run_member3_pipeline.py - Complete Feature Engineering & Validation Pipeline for Member 3
Executes:
1. Pair dataset construction with strict ID preservation and ground truth labels (0/1)
2. Pairwise feature extraction (10 name + 8 address + 6 context + 4 interaction = 28 features)
3. Automated feature quality check (duplicates, NaNs, Infs, constants, target leakage checks)
4. Positive vs Negative distribution analysis (separability audit)
5. Hard-negative identification and mining pool generation
6. Macro F_0.5 evaluation engine & threshold simulation
7. Output deliverables generation:
   - data/training_pair_features.parquet
   - reports/feature_statistics.csv
   - reports/positive_negative_comparison.csv
   - reports/hard_negative_candidates.csv
"""

import sys
import os
from pathlib import Path
import numpy as np
import pandas as pd

# Set UTF-8 output
sys.stdout.reconfigure(encoding='utf-8')

# Ensure src is on path
current_dir = Path(__file__).resolve().parent
src_dir = current_dir / "amazon-ml-challenge" / "Akatsuki" / "code" / "business_entity_resolution" / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from pair_builder import PairBuilder
from features.pair_features import FEATURE_NAMES, compute_pair_features
from evaluation.evaluator import evaluate_predictions, test_thresholds


def main():
    print("=" * 70)
    print("MEMBER 3: PAIRWISE FEATURE ENGINEERING & VALIDATION PIPELINE")
    print("=" * 70)

    # Output paths
    repo_data_dir = current_dir / "amazon-ml-challenge" / "Akatsuki" / "data"
    repo_reports_dir = current_dir / "amazon-ml-challenge" / "Akatsuki" / "reports"
    root_data_dir = current_dir / "data"
    root_reports_dir = current_dir / "reports"

    for d in [repo_data_dir, repo_reports_dir, root_data_dir, root_reports_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------------
    # STEP 2: BUILD TRAINING PAIR TABLE
    # ---------------------------------------------------------
    print("\n>>> STEP 2: Constructing Training Pair Dataset...")
    builder = PairBuilder(seed=42)
    
    # We sample 10,000 S1 entities to generate a clean, rigorous pair table
    # covering thousands of true matches and hard negative distractors
    pairs_df, pair_stats, s1_rec, s2_rec, s3_rec = builder.build_benchmark_pairs(
        num_s1_entities=10000,
        neg_to_pos_ratio=1.5
    )

    print("\n--- Training Pair Dataset Statistics ---")
    for k, v in pair_stats.items():
        print(f"  • {k}: {v:,}" if isinstance(v, int) else f"  • {k}: {v}")

    # Validate pair table schema
    assert "pair_id" in pairs_df.columns
    assert "source1_id" in pairs_df.columns
    assert "candidate_id" in pairs_df.columns
    assert "label" in pairs_df.columns
    assert pair_stats["duplicate_pairs"] == 0, "Duplicate pair IDs detected!"
    assert pair_stats["missing_ids"] == 0, "Missing entity IDs detected!"

    # ---------------------------------------------------------
    # STEP 3 - 6: COMPUTE ALL 28 PAIRWISE FEATURES
    # ---------------------------------------------------------
    print(f"\n>>> STEPS 3–6: Extracting 28 Pairwise Features for {len(pairs_df):,} pairs...")
    feat_df = builder.generate_feature_matrix(pairs_df, s1_rec, s2_rec, s3_rec)
    print(f"Feature matrix generated with shape: {feat_df.shape}")

    # ---------------------------------------------------------
    # STEP 7: FEATURE QUALITY CHECK
    # ---------------------------------------------------------
    print("\n>>> STEP 7: Automated Feature Quality & Integrity Checks...")
    quality_issues = []

    # 1. Duplicate check
    dup_pairs = feat_df.duplicated(subset=["source1_id", "candidate_id"]).sum()
    if dup_pairs > 0:
        quality_issues.append(f"Found {dup_pairs} duplicate entity pairs")

    # 2. Check NaNs
    nan_counts = feat_df[FEATURE_NAMES].isna().sum()
    total_nans = nan_counts.sum()
    if total_nans > 0:
        quality_issues.append(f"Found {total_nans} NaN values across features: {nan_counts[nan_counts > 0].to_dict()}")

    # 3. Check Infs
    num_cols = feat_df[FEATURE_NAMES].select_dtypes(include=[np.number]).columns
    inf_counts = np.isinf(feat_df[num_cols]).sum().sum()
    if inf_counts > 0:
        quality_issues.append(f"Found {inf_counts} infinite values")

    # 4. Check constant features
    constant_cols = [c for c in FEATURE_NAMES if feat_df[c].nunique() <= 1]
    if constant_cols:
        quality_issues.append(f"Constant features detected: {constant_cols}")

    # 5. Target leakage verification
    assert "label" not in FEATURE_NAMES, "CRITICAL: 'label' leaked into feature names!"
    assert "source1_id" not in FEATURE_NAMES, "CRITICAL: 'source1_id' leaked into features!"
    assert "candidate_id" not in FEATURE_NAMES, "CRITICAL: 'candidate_id' leaked into features!"

    print(f"  ✓ Total Rows: {len(feat_df):,}")
    print(f"  ✓ Total Feature Columns: {len(FEATURE_NAMES)}")
    print(f"  ✓ NaN Values: {total_nans}")
    print(f"  ✓ Infinite Values: {inf_counts}")
    print(f"  ✓ Constant Features: {constant_cols if constant_cols else 'None (All 28 features vary)'}")
    print(f"  ✓ Target Leakage Check: PASSED (label and IDs isolated)")
    print(f"  ✓ Quality Verdict: {'PASSED (Clean)' if not quality_issues else 'ISSUES DETECTED: ' + str(quality_issues)}")

    # ---------------------------------------------------------
    # EXPORT FEATURE TABLE TO PARQUET
    # ---------------------------------------------------------
    parquet_repo_path = repo_data_dir / "training_pair_features.parquet"
    parquet_root_path = root_data_dir / "training_pair_features.parquet"
    feat_df.to_parquet(parquet_repo_path, index=False)
    feat_df.to_parquet(parquet_root_path, index=False)
    print(f"\nSaved feature table to Parquet: {parquet_repo_path} ({os.path.getsize(parquet_repo_path):,} bytes)")

    # ---------------------------------------------------------
    # STEP 8: POSITIVE VS NEGATIVE ANALYSIS
    # ---------------------------------------------------------
    print("\n>>> STEP 8: Computing Positive vs Negative Feature Comparison...")
    pos_df = feat_df[feat_df["label"] == 1]
    neg_df = feat_df[feat_df["label"] == 0]

    comparison_rows = []
    feature_stats_rows = []

    for feat in FEATURE_NAMES:
        p_vals = pos_df[feat]
        n_vals = neg_df[feat]

        p_mean = p_vals.mean()
        n_mean = n_vals.mean()
        p_med = p_vals.median()
        n_med = n_vals.median()
        p_std = p_vals.std()
        n_std = n_vals.std()
        
        # Separation metric (Difference in means normalized by pooled std)
        pooled_std = np.sqrt((p_std**2 + n_std**2) / 2.0) if (p_std**2 + n_std**2) > 0 else 1.0
        cohen_d = (p_mean - n_mean) / pooled_std if pooled_std > 0 else 0.0

        comparison_rows.append({
            "Feature": feat,
            "Positive_Mean": round(p_mean, 4),
            "Negative_Mean": round(n_mean, 4),
            "Difference": round(p_mean - n_mean, 4),
            "Positive_Median": round(p_med, 4),
            "Negative_Median": round(n_med, 4),
            "Cohen_d_Separation": round(cohen_d, 4),
            "Positive_Missing_Rate": round((p_vals == -1.0).mean() * 100, 2) if (p_vals == -1.0).any() else 0.0,
            "Negative_Missing_Rate": round((n_vals == -1.0).mean() * 100, 2) if (n_vals == -1.0).any() else 0.0
        })

        # Overall distribution stats for feature_statistics.csv
        all_vals = feat_df[feat]
        feature_stats_rows.append({
            "Feature": feat,
            "Mean": round(all_vals.mean(), 4),
            "Median": round(all_vals.median(), 4),
            "Std": round(all_vals.std(), 4),
            "Min": round(all_vals.min(), 4),
            "P25": round(all_vals.quantile(0.25), 4),
            "P75": round(all_vals.quantile(0.75), 4),
            "Max": round(all_vals.max(), 4),
            "Zeros_Pct": round((all_vals == 0.0).mean() * 100, 2),
            "Ones_Pct": round((all_vals == 1.0).mean() * 100, 2)
        })

    comp_df = pd.DataFrame(comparison_rows).sort_values(by="Cohen_d_Separation", ascending=False)
    comp_repo_path = repo_reports_dir / "positive_negative_comparison.csv"
    comp_root_path = root_reports_dir / "positive_negative_comparison.csv"
    comp_df.to_csv(comp_repo_path, index=False)
    comp_df.to_csv(comp_root_path, index=False)

    stats_repo_path = repo_reports_dir / "feature_statistics.csv"
    stats_root_path = root_reports_dir / "feature_statistics.csv"
    pd.DataFrame(feature_stats_rows).to_csv(stats_repo_path, index=False)
    pd.DataFrame(feature_stats_rows).to_csv(stats_root_path, index=False)

    print("\n--- Top 10 Most Discriminative Features (by Cohen's d Separation) ---")
    print(comp_df[["Feature", "Positive_Mean", "Negative_Mean", "Difference", "Cohen_d_Separation"]].head(10).to_string(index=False))

    # ---------------------------------------------------------
    # STEP 9: HARD-NEGATIVE ANALYSIS
    # ---------------------------------------------------------
    print("\n>>> STEP 9: Identifying and Mining Hard Negatives...")
    # Negative pairs with high name similarity (>= 0.70) or high overall similarity but label = 0
    hard_negatives = []

    for idx, row in feat_df[feat_df["label"] == 0].iterrows():
        name_sim = row["name_token_sort"]
        addr_sim = row["address_jaccard"]
        num_conf = row["numeric_conflict"]
        dig_ov = row["address_digit_overlap"]
        
        reason = None
        if name_sim >= 0.75 and num_conf == 1.0:
            reason = "High name similarity but conflicting address numbers (Branch / Different address collision)"
        elif name_sim >= 0.75 and addr_sim <= 0.15:
            reason = "High name similarity but completely different address text (Same business name, different entity)"
        elif addr_sim >= 0.75 and name_sim <= 0.20:
            reason = "High address similarity but completely different business name"
        elif name_sim >= 0.85:
            reason = "Very high name similarity (>0.85) but ground-truth non-match"

        if reason:
            s1_id = row["source1_id"]
            cand_id = row["candidate_id"]
            s1_info = s1_rec.get(s1_id, {})
            cand_info = s2_rec.get(cand_id, {}) if cand_id.startswith("S2-") else s3_rec.get(cand_id, {})
            
            hard_negatives.append({
                "pair_id": row["pair_id"],
                "source1_id": s1_id,
                "candidate_id": cand_id,
                "s1_name": s1_info.get("name", ""),
                "cand_name": cand_info.get("name", ""),
                "s1_address": s1_info.get("address", ""),
                "cand_address": cand_info.get("address", ""),
                "name_token_sort": round(name_sim, 3),
                "address_jaccard": round(addr_sim, 3),
                "digit_overlap": round(dig_ov, 3),
                "numeric_conflict": int(num_conf),
                "difficulty_reason": reason
            })

    hard_neg_df = pd.DataFrame(hard_negatives)
    hard_repo_path = repo_reports_dir / "hard_negative_candidates.csv"
    hard_root_path = root_reports_dir / "hard_negative_candidates.csv"
    hard_neg_df.to_csv(hard_repo_path, index=False)
    hard_neg_df.to_csv(hard_root_path, index=False)
    print(f"Catalogued {len(hard_neg_df):,} hard negative pairs in {hard_repo_path}")

    # ---------------------------------------------------------
    # STEP 10 - 12: EVALUATION & THRESHOLD SIMULATION
    # ---------------------------------------------------------
    print("\n>>> STEPS 10–12: Running Evaluation Engine & Threshold Framework...")
    # Load ground truth for the evaluated S1 entities
    sample_gt = {s1: set() for s1 in feat_df["source1_id"].unique()}
    gt_dict = builder.load_ground_truth_dict()
    for s1 in sample_gt:
        sample_gt[s1] = gt_dict.get(s1, set())

    # Build composite similarity score to test threshold framework
    # Composite baseline score = 0.55 * name_token_set + 0.35 * address_jaccard + 0.10 * digit_overlap
    clean_dig = feat_df["address_digit_overlap"].apply(lambda x: max(0.0, x))
    feat_df["composite_score"] = (
        0.55 * feat_df["name_token_set"] +
        0.35 * feat_df["address_jaccard"] +
        0.10 * clean_dig
    )

    threshold_results = test_thresholds(
        pair_scores_df=feat_df,
        ground_truth=sample_gt,
        score_col="composite_score",
        thresholds=[0.40, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90]
    )

    print("\n--- Macro F0.5 Threshold Simulation Table ---")
    print(threshold_results.to_string(index=False))

    # Evaluate at optimal baseline threshold
    best_row = threshold_results.loc[threshold_results["Macro F0.5"].idxmax()]
    print(f"\nOptimal Baseline Threshold: {best_row['Threshold']} -> Macro F0.5 = {best_row['Macro F0.5']:.4f}")
    print(f"  • Precision: {best_row['Macro Precision']:.4f}")
    print(f"  • Recall: {best_row['Macro Recall']:.4f}")
    print(f"  • Singleton Score: {best_row['Singleton Score']:.4f}")
    print(f"  • Multi-Match F0.5: {best_row['Multi-Match F0.5']:.4f}")

    print("\n" + "=" * 70)
    print("MEMBER 3 PIPELINE COMPLETED SUCCESSFULLY!")
    print("=" * 70)


if __name__ == "__main__":
    main()
