"""
evaluation/blocking_evaluation.py
----------------------------------
Evaluation metrics for Member 2 — Blocking & Candidate Generation.

Calculates:
  1. Overall Blocking Recall against ground truth pairs
  2. Candidate distribution statistics (Mean, Median, P95, P99, Max)
  3. Per-Pass Ablation Table (incremental recall gain per pass B0-B4)
  4. Failure Analysis (missed ground-truth pairs broken down by noise category)
"""

from collections import Counter, defaultdict
import logging
from typing import Dict, List, Set, Tuple
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def evaluate_blocking_recall(
    df_candidates_long: pd.DataFrame,
    gt_dict: Dict[str, Set[str]],
) -> Tuple[float, int, int, Dict[str, float]]:
    """
    Evaluates total blocking recall against ground truth dictionary.
    
    Args:
        df_candidates_long: Long matrix with 'source1_id', 'candidate_id'
        gt_dict: Dict[s1_id -> Set[matched_candidate_ids]]
        
    Returns:
        recall: Recall percentage (0.0 to 100.0)
        total_gt_pairs: Total ground truth matching pairs
        found_gt_pairs: Number of ground truth matching pairs retained in candidates
        pass_recalls: Dict[pass_column -> Recall % for that pass alone]
    """
    total_gt_pairs = sum(len(matched) for matched in gt_dict.values())
    if total_gt_pairs == 0:
        return 0.0, 0, 0, {}

    # Build candidate lookup dict
    retrieved_dict: Dict[str, Set[str]] = defaultdict(set)
    if not df_candidates_long.empty:
        for s1_id, group in df_candidates_long.groupby("source1_id", sort=False):
            retrieved_dict[s1_id] = set(group["candidate_id"].tolist())

    found_gt_pairs = 0
    for s1_id, gt_matches in gt_dict.items():
        if not gt_matches:
            continue
        cands = retrieved_dict.get(s1_id, set())
        found_gt_pairs += len(gt_matches.intersection(cands))

    recall = (found_gt_pairs / total_gt_pairs) * 100.0

    # Per-pass recall breakdown
    pass_recalls = {}
    flag_cols = [c for c in df_candidates_long.columns if c.startswith("block_")]

    for col in flag_cols:
        pass_df = df_candidates_long[df_candidates_long[col] == True]
        pass_retrieved: Dict[str, Set[str]] = defaultdict(set)
        for s1_id, group in pass_df.groupby("source1_id", sort=False):
            pass_retrieved[s1_id] = set(group["candidate_id"].tolist())

        p_found = 0
        for s1_id, gt_matches in gt_dict.items():
            if not gt_matches:
                continue
            cands = pass_retrieved.get(s1_id, set())
            p_found += len(gt_matches.intersection(cands))

        pass_recalls[col] = (p_found / total_gt_pairs) * 100.0

    return recall, total_gt_pairs, found_gt_pairs, pass_recalls


def compute_candidate_volume_stats(
    df_candidates_long: pd.DataFrame,
    all_s1_ids: List[str],
) -> Dict[str, float]:
    """
    Calculates statistical metrics for candidates generated per S1 entity.
    """
    counts_map = {s1_id: 0 for s1_id in all_s1_ids}
    if not df_candidates_long.empty:
        counts = df_candidates_long.groupby("source1_id").size()
        for s1_id, cnt in counts.items():
            counts_map[s1_id] = cnt

    counts_arr = np.array(list(counts_map.values()))

    return {
        "total_s1_records": float(len(all_s1_ids)),
        "total_candidate_pairs": float(len(df_candidates_long)),
        "mean_candidates_per_s1": float(np.mean(counts_arr)),
        "median_candidates_per_s1": float(np.median(counts_arr)),
        "p95_candidates_per_s1": float(np.percentile(counts_arr, 95)),
        "p99_candidates_per_s1": float(np.percentile(counts_arr, 99)),
        "max_candidates_per_s1": float(np.max(counts_arr)),
        "zero_candidate_s1_count": float(np.sum(counts_arr == 0)),
    }


def compute_blocking_audit_summary(
    df_candidates_long: pd.DataFrame,
    df_s1: pd.DataFrame,
    df_s2: pd.DataFrame,
    df_s3: pd.DataFrame,
    found_gt_pairs: int,
    total_gt_pairs: int,
) -> Dict[str, float]:
    """
    Computes end-to-end blocking audit metrics:
      - candidate recall
      - naive pair count (S1 x (S2+S3))
      - candidate reduction vs naive all-pairs
    """
    s1_count = int(len(df_s1))
    s2_count = int(len(df_s2))
    s3_count = int(len(df_s3))
    naive_pairs = int(s1_count * (s2_count + s3_count))
    candidate_pairs = int(len(df_candidates_long))

    candidate_recall = float(found_gt_pairs / total_gt_pairs) if total_gt_pairs > 0 else 0.0
    candidate_retained_pct = float((candidate_pairs / naive_pairs) * 100.0) if naive_pairs > 0 else 0.0
    candidate_reduction_pct = float(100.0 - candidate_retained_pct) if naive_pairs > 0 else 0.0

    return {
        "s1_count": float(s1_count),
        "s2_count": float(s2_count),
        "s3_count": float(s3_count),
        "naive_pair_count": float(naive_pairs),
        "candidate_pair_count": float(candidate_pairs),
        "candidate_recall": round(candidate_recall, 6),
        "candidate_retained_pct": round(candidate_retained_pct, 6),
        "candidate_reduction_pct": round(candidate_reduction_pct, 6),
    }


def generate_ablation_report(
    pass_results: Dict[str, Dict[str, Set[str]]],
    gt_dict: Dict[str, Set[str]],
    timing_dict: Dict[str, float],
) -> pd.DataFrame:
    """
    Generates an ablation table showing cumulative recall & candidate volume gains per pass.
    """
    total_gt = sum(len(m) for m in gt_dict.values())
    rows = []

    cumulative_candidates: Dict[str, Set[str]] = defaultdict(set)
    pass_order = [
        ("B0", "block_exact_name", "Exact Name"),
        ("B1_core", "block_core", "Core Name"),
        ("B1_rare", "block_rare_token", "Rare Token"),
        ("B2_addr", "block_address", "Address Rare Token"),
        ("B2_num", "block_numeric", "Numeric Locality"),
        ("B3_trans", "block_translit", "Transliteration"),
        ("B4_ann", "block_ann", "ANN Vector Cosine"),
    ]

    for code, col_name, label in pass_order:
        if col_name not in pass_results:
            continue

        cands_dict = pass_results[col_name]
        for s1_id, c_set in cands_dict.items():
            cumulative_candidates[s1_id].update(c_set)

        # Count found GT
        found = 0
        total_cands = 0
        for s1_id, gt_matches in gt_dict.items():
            if not gt_matches:
                continue
            retrieved = cumulative_candidates.get(s1_id, set())
            found += len(gt_matches.intersection(retrieved))

        for c_set in cumulative_candidates.values():
            total_cands += len(c_set)

        cum_recall = (found / total_gt * 100.0) if total_gt > 0 else 0.0
        duration = timing_dict.get(code, timing_dict.get(col_name, 0.0))

        rows.append({
            "pass_code": code,
            "pass_label": label,
            "cumulative_found_gt": found,
            "total_gt_pairs": total_gt,
            "cumulative_recall_pct": round(cum_recall, 3),
            "cumulative_total_candidates": total_cands,
            "runtime_sec": round(duration, 2),
        })

    return pd.DataFrame(rows)


def analyze_blocking_failures(
    df_candidates_long: pd.DataFrame,
    gt_dict: Dict[str, Set[str]],
    df_s1: pd.DataFrame,
    df_s2: pd.DataFrame,
    df_s3: pd.DataFrame,
    sample_limit: int = 50,
) -> pd.DataFrame:
    """
    Identifies ground truth pairs missed by blocking and categorizes the root cause of failure.
    """
    retrieved_dict: Dict[str, Set[str]] = defaultdict(set)
    if not df_candidates_long.empty:
        for s1_id, group in df_candidates_long.groupby("source1_id", sort=False):
            retrieved_dict[s1_id] = set(group["candidate_id"].tolist())

    s1_map = df_s1.set_index("entity_id").to_dict("index")
    s23_df = pd.concat([df_s2, df_s3], ignore_index=True)
    s23_map = s23_df.set_index("entity_id").to_dict("index")

    missed_rows = []
    count = 0

    for s1_id, gt_matches in gt_dict.items():
        if not gt_matches:
            continue
        cands = retrieved_dict.get(s1_id, set())
        missed = gt_matches - cands

        for target_id in missed:
            s1_info = s1_map.get(s1_id, {})
            target_info = s23_map.get(target_id, {})

            s1_country = s1_info.get("country_norm", "")
            t_country = target_info.get("country_norm", "")

            # Failure categorization
            if s1_country and t_country and s1_country != t_country:
                reason = "Country Mismatch"
            elif not s1_info.get("address_norm") or not target_info.get("address_norm"):
                reason = "Missing Address"
            elif s1_info.get("is_non_latin") or target_info.get("is_non_latin"):
                reason = "Cross-Script Transliteration Miss"
            else:
                reason = "Fuzzy Name / Major Typo Discrepancy"

            missed_rows.append({
                "source1_id": s1_id,
                "s1_name": s1_info.get("business_name", ""),
                "s1_country": s1_country,
                "missed_target_id": target_id,
                "target_name": target_info.get("business_name", ""),
                "target_country": t_country,
                "failure_reason": reason,
            })
            count += 1
            if count >= sample_limit:
                break
        if count >= sample_limit:
            break

    return pd.DataFrame(missed_rows)
