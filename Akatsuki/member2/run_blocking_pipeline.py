"""
run_blocking_pipeline.py
-------------------------
Main Execution Entry Point for Member 2 — Candidate Generation / Blocking Engineer.

Runs the complete multi-pass blocking pipeline (B0-B5) on train or test split.
Measures recall against Ground Truth (train split), exports candidates to Parquet & TSV,
and writes comprehensive evaluation reports.

Usage:
  python run_blocking_pipeline.py [--split train|test] [--nrows 50000]
"""

import argparse
import logging
import sys
from pathlib import Path
import pandas as pd

# Add src to Python Path
SRC_PATH = Path(__file__).resolve().parent.parent / "code" / "business_entity_resolution" / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from data_loader import DataLoader
from preprocessing.preprocessor import preprocess_source
from blocking.candidate_generator import run_candidate_generation_pipeline
from evaluation.blocking_evaluation import (
    evaluate_blocking_recall,
    compute_candidate_volume_stats,
    compute_blocking_audit_summary,
    generate_ablation_report,
    analyze_blocking_failures,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)
logger = logging.getLogger("Member2_BlockingPipeline")


def main():
    parser = argparse.ArgumentParser(description="Member 2 Candidate Generation Pipeline")
    parser.add_argument("--split", type=str, default="train", choices=["train", "test"], help="Dataset split to run on")
    parser.add_argument("--nrows", type=int, default=None, help="Number of rows to sample per source file (for testing)")
    parser.add_argument("--max_candidates", type=int, default=200, help="Max candidates per S1 entity")
    args = parser.parse_args()

    member2_dir = Path(__file__).resolve().parent
    reports_dir = member2_dir / "reports"
    output_dir = member2_dir / "output"
    reports_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"--- MEMBER 2: CANDIDATE GENERATION PIPELINE ({args.split.upper()}) ---")
    if args.nrows:
        logger.info(f"Sampling limit: {args.nrows} rows per file")

    # Step 1: Load Data
    loader = DataLoader()
    logger.info("Loading raw TSV sources...")
    raw_s1 = loader.load_source("s1", split=args.split, nrows=args.nrows)
    raw_s2 = loader.load_source("s2", split=args.split, nrows=args.nrows)
    raw_s3 = loader.load_source("s3", split=args.split, nrows=args.nrows)

    # Step 2: Preprocess Sources
    logger.info("Preprocessing names, addresses, countries, transliterations...")
    df_s1 = preprocess_source(raw_s1)
    df_s2 = preprocess_source(raw_s2)
    df_s3 = preprocess_source(raw_s3)

    # Step 3: Run Multi-Pass Blocking Pipeline (B0 -> B5)
    df_long, df_tsv, timing = run_candidate_generation_pipeline(
        df_s1, df_s2, df_s3, max_total_per_s1=args.max_candidates
    )

    # Step 4: Export Candidates
    long_parquet_path = output_dir / "candidate_pairs_long.parquet"
    tsv_path = output_dir / "candidate_pairs.tsv"

    logger.info(f"Saving long candidate matrix to: {long_parquet_path}")
    df_long.to_parquet(long_parquet_path, index=False)

    logger.info(f"Saving submission TSV format to: {tsv_path}")
    df_tsv.to_csv(tsv_path, sep="\t", index=False)

    # Step 5: Evaluate (if train split)
    if args.split == "train":
        logger.info("Loading train ground truth dictionary...")
        gt_dict = loader.load_ground_truth_dict(nrows=args.nrows)

        recall, total_gt, found_gt, pass_recalls = evaluate_blocking_recall(df_long, gt_dict)
        vol_stats = compute_candidate_volume_stats(df_long, df_s1["entity_id"].tolist())
        audit_stats = compute_blocking_audit_summary(
            df_candidates_long=df_long,
            df_s1=df_s1,
            df_s2=df_s2,
            df_s3=df_s3,
            found_gt_pairs=found_gt,
            total_gt_pairs=total_gt,
        )

        logger.info("=" * 60)
        logger.info(f"BLOCKING EVALUATION RESULTS:")
        logger.info(f"  Total Ground Truth Pairs: {total_gt:,}")
        logger.info(f"  Found Matching Pairs:     {found_gt:,}")
        logger.info(f"  OVERALL BLOCKING RECALL:  {recall:.3f}%")
        logger.info(f"  Candidate Recall:         {audit_stats['candidate_recall']:.6f}")
        logger.info(f"  Naive Pair Space:         {int(audit_stats['naive_pair_count']):,}")
        logger.info(f"  Candidate Pairs:          {int(audit_stats['candidate_pair_count']):,}")
        logger.info(f"  Pair Space Retained:      {audit_stats['candidate_retained_pct']:.6f}%")
        logger.info(f"  Pair Space Reduced:       {audit_stats['candidate_reduction_pct']:.6f}%")
        logger.info(f"  Candidate Stats (per S1): Mean={vol_stats['mean_candidates_per_s1']:.2f}, "
                    f"Median={vol_stats['median_candidates_per_s1']:.1f}, "
                    f"P95={vol_stats['p95_candidates_per_s1']:.1f}, "
                    f"P99={vol_stats['p99_candidates_per_s1']:.1f}, "
                    f"Max={vol_stats['max_candidates_per_s1']:.0f}")
        logger.info("=" * 60)

        # Write reports
        metrics_df = pd.DataFrame([{
            "split": args.split,
            "overall_recall_pct": round(recall, 3),
            "total_gt_pairs": total_gt,
            "found_gt_pairs": found_gt,
            **vol_stats,
            **audit_stats
        }])
        metrics_df.to_csv(reports_dir / "blocking_metrics.csv", index=False)

        pd.DataFrame([audit_stats]).to_csv(reports_dir / "blocking_audit_summary.csv", index=False)

        # Size distribution report
        dist_df = pd.DataFrame(list(vol_stats.items()), columns=["metric", "value"])
        dist_df.to_csv(reports_dir / "candidate_size_distribution.csv", index=False)

        # Failure analysis
        failure_df = analyze_blocking_failures(df_long, gt_dict, df_s1, df_s2, df_s3)
        failure_df.to_csv(reports_dir / "blocking_failure_analysis.csv", index=False)
        logger.info(f"Failure analysis saved ({len(failure_df)} sampled missed pairs).")

    print("\nMember 2 Blocking Pipeline execution completed successfully!")


if __name__ == "__main__":
    main()
