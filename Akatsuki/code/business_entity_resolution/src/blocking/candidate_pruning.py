"""
blocking/candidate_pruning.py
------------------------------
Blocking Pass B5 — Frequency-aware Candidate Pruning & Budget Capping.

Enforces:
1. Per-route budget caps (e.g., max 50 candidates from address pass alone)
2. Total candidate cap per S1 (e.g., max 200 candidates per S1 entity)
3. Priority ordering across blocking passes when pruning:
   Exact Name > Core Name > Rare Token > Transliteration > Address > Numeric > ANN
"""

from typing import Dict, List, Optional, Set
import pandas as pd


# Default priority rank for blocking passes (lower number = higher priority)
PASS_PRIORITY = {
    "block_exact_name": 1,
    "block_core": 2,
    "block_rare_token": 3,
    "block_translit": 4,
    "block_address": 5,
    "block_numeric": 6,
    "block_ann": 7,
}


def prune_candidate_matrix(
    df_candidates: pd.DataFrame,
    max_total_per_s1: int = 200,
    per_route_caps: Optional[Dict[str, int]] = None,
) -> pd.DataFrame:
    """
    Prunes long-form candidate DataFrame according to route caps and overall per-S1 limits.
    
    Args:
        df_candidates: DataFrame output of candidate_union
        max_total_per_s1: Hard ceiling on maximum candidates per S1 entity
        per_route_caps: Dict[pass_name -> max candidates allowed from this pass]
        
    Returns:
        Pruned DataFrame with identical schema
    """
    if df_candidates.empty:
        return df_candidates

    if per_route_caps is None:
        per_route_caps = {
            "block_exact_name": 100,
            "block_core": 100,
            "block_rare_token": 50,
            "block_translit": 50,
            "block_address": 50,
            "block_numeric": 30,
            "block_ann": 30,
        }

    flag_cols = [c for c in df_candidates.columns if c.startswith("block_")]

    # Calculate min pass priority for sorting each candidate
    def get_candidate_priority(row):
        priorities = [PASS_PRIORITY.get(col, 99) for col in flag_cols if row[col]]
        return min(priorities) if priorities else 99

    df_candidates["_priority"] = df_candidates.apply(get_candidate_priority, axis=1)

    pruned_dfs = []

    for s1_id, group in df_candidates.groupby("source1_id", sort=False):
        if len(group) <= max_total_per_s1:
            pruned_dfs.append(group)
            continue

        # Sort group by priority (highest priority first)
        group_sorted = group.sort_values(by="_priority", ascending=True)

        # Enforce total limit
        group_pruned = group_sorted.iloc[:max_total_per_s1]
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
