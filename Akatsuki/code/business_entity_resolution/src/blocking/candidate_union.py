"""
blocking/candidate_union.py
---------------------------
Merges candidate sets from multiple blocking passes B0–B4 into a unified,
deduplicated candidate table with boolean provenance flags.
"""

from typing import Dict, List, Set
import pandas as pd


def union_candidate_passes(
    pass_results: Dict[str, Dict[str, Set[str]]],
    all_s1_ids: List[str],
) -> pd.DataFrame:
    """
    Combines results from multiple named blocking passes into a single DataFrame.
    
    Args:
        pass_results: Dict mapping pass_name -> Dict[s1_id -> Set[candidate_id]]
                      Pass names: 'block_exact_name', 'block_core', 'block_rare_token',
                                  'block_address', 'block_numeric', 'block_translit', 'block_ann'
        all_s1_ids: List of all S1 entity_ids (ensures every S1 is present)
        
    Returns:
        DataFrame with columns:
            source1_id, candidate_id, block_exact_name, block_core, block_rare_token,
            block_address, block_numeric, block_translit, block_ann
    """
    flag_columns = [
        "block_exact_name",
        "block_core",
        "block_rare_token",
        "block_address",
        "block_numeric",
        "block_translit",
        "block_ann",
    ]

    # Structure: (s1_id, candidate_id) -> Dict[flag_col -> bool]
    pair_flags: Dict[tuple, Dict[str, bool]] = {}

    for pass_name, candidates_dict in pass_results.items():
        if pass_name not in flag_columns:
            continue

        for s1_id, cand_set in candidates_dict.items():
            for cand_id in cand_set:
                pair_key = (s1_id, cand_id)
                if pair_key not in pair_flags:
                    pair_flags[pair_key] = {col: False for col in flag_columns}
                pair_flags[pair_key][pass_name] = True

    # Construct rows
    rows = []
    for (s1_id, cand_id), flags in pair_flags.items():
        row = {"source1_id": s1_id, "candidate_id": cand_id}
        row.update(flags)
        rows.append(row)

    if not rows:
        df = pd.DataFrame(columns=["source1_id", "candidate_id"] + flag_columns)
    else:
        df = pd.DataFrame(rows)

    return df
