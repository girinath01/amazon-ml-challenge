"""
blocking/candidate_pruning.py
------------------------------
Blocking Pass B5 — Adaptive Candidate Pruning & Budget Capping.

Enforces:
1. Priority ordering across blocking passes:
   Exact Name > Core Name > Rare Token > Transliteration > Address > Numeric > ANN
2. Adaptive candidate budgeting based on address availability & match quality.
"""

from typing import Dict, List, Optional, Set
import pandas as pd

PASS_PRIORITY = {
    "block_exact_name": 1,
    "block_core": 2,
    "block_rare_token": 3,
    "block_translit": 4,
    "block_address": 5,
    "block_numeric": 6,
    "block_ann": 7,
    "block_phonetic": 8,
}


def prune_candidate_matrix(
    df_candidates: pd.DataFrame,
    max_total_per_s1: int = 200,
    adaptive_pruning: bool = True,
) -> pd.DataFrame:
    """
    Prunes long-form candidate DataFrame using priority scoring and adaptive budget allocation.
    
    Args:
        df_candidates: DataFrame output of candidate_union
        max_total_per_s1: Default ceiling on maximum candidates per S1 entity
        adaptive_pruning: If True, dynamically adjusts budget based on match strength
        
    Returns:
        Pruned DataFrame with identical schema
    """
    if df_candidates.empty:
        return df_candidates

    flag_cols = [c for c in df_candidates.columns if c.startswith("block_")]

    def get_candidate_priority(row):
        priorities = [PASS_PRIORITY.get(col, 99) for col in flag_cols if row[col]]
        # Give extra boost if candidate matches multiple passes simultaneously
        num_passes = sum(1 for col in flag_cols if row[col])
        base_priority = min(priorities) if priorities else 99
        return base_priority - (0.1 * num_passes)

    df_candidates["_priority"] = df_candidates.apply(get_candidate_priority, axis=1)

    pruned_dfs = []

    for s1_id, group in df_candidates.groupby("source1_id", sort=False):
        has_address_match = False
        if "block_address" in group.columns and (group["block_address"] == True).any():
            has_address_match = True

        # Adaptive budget limit:
        # High confidence address matches -> tight cap (75) to save downstream RAM
        # Low confidence / name-only -> larger cap (max_total_per_s1) for maximum recall
        effective_limit = max_total_per_s1
        if adaptive_pruning and has_address_match and len(group) > 75:
            effective_limit = min(max_total_per_s1, 100)

        if len(group) <= effective_limit:
            pruned_dfs.append(group)
            continue

        # Sort group by priority (highest priority first)
        group_sorted = group.sort_values(by="_priority", ascending=True)
        group_pruned = group_sorted.iloc[:effective_limit]
        pruned_dfs.append(group_pruned)

    result_df = pd.concat(pruned_dfs, ignore_index=True)
    result_df.drop(columns=["_priority"], errors="ignore", inplace=True)
    return result_df


def format_to_tsv(
    df_candidates: pd.DataFrame,
    all_s1_ids: List[str],
) -> pd.DataFrame:
    """
    Formats long-form candidate DataFrame into official TSV format:
    source1_entity_id \t matched_entity_ids (comma-separated)
    
    Ensures EVERY S1 entity_id has exactly one row, even if candidate list is empty.
    """
    s1_to_cands: Dict[str, List[str]] = {s1_id: [] for s1_id in all_s1_ids}

    if not df_candidates.empty:
        for s1_id, group in df_candidates.groupby("source1_id", sort=False):
            s1_to_cands[s1_id] = group["candidate_id"].tolist()

    rows = []
    for s1_id in all_s1_ids:
        cands = s1_to_cands.get(s1_id, [])
        cand_str = ",".join(cands) if cands else ""
        rows.append({"source1_entity_id": s1_id, "candidate_entity_ids": cand_str})

    return pd.DataFrame(rows)
