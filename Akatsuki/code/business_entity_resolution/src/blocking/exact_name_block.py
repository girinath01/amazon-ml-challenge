"""
blocking/exact_name_block.py
-------------------------------
Blocking Pass B0/B1 — Exact name & core-name matching.

Builds inverted indexes:
    (country_norm, name_norm)  -> [S2/S3 ids]
    (country_norm, name_core)  -> [S2/S3 ids]

Applies a frequency cap to prevent common-name explosion.
"""

from collections import defaultdict
from typing import Dict, List, Set, Tuple
import pandas as pd

from preprocessing.name_normalizer import get_rare_tokens

_DEFAULT_FREQ_CAP = 200


def build_name_indexes(
    df_s2: pd.DataFrame,
    df_s3: pd.DataFrame,
    freq_cap: int = _DEFAULT_FREQ_CAP,
) -> Tuple[Dict[Tuple[str, str], Set[str]], Dict[Tuple[str, str], Set[str]]]:
    """
    Build inverted indexes from S2 and S3 DataFrames:
        exact_index: (country, name_norm) -> Set of entity_ids
        core_index:  (country, name_core) -> Set of entity_ids
    """
    exact_index: Dict[Tuple[str, str], Set[str]] = defaultdict(set)
    core_index: Dict[Tuple[str, str], Set[str]] = defaultdict(set)

    for df in [df_s2, df_s3]:
        if df.empty:
            continue
        for row in df[["entity_id", "country_norm", "name_norm", "name_core"]].itertuples(index=False):
            eid = row.entity_id
            country = getattr(row, "country_norm", "") or "UNKNOWN"
            name_n = getattr(row, "name_norm", "") or ""
            name_c = getattr(row, "name_core", "") or ""

            if name_n:
                exact_index[(country, name_n)].add(eid)
            if name_c:
                core_index[(country, name_c)].add(eid)

    return exact_index, core_index


def block_exact_and_core_name(
    df_s1: pd.DataFrame,
    exact_index: Dict[Tuple[str, str], Set[str]],
    core_index: Dict[Tuple[str, str], Set[str]],
    freq_cap: int = _DEFAULT_FREQ_CAP,
) -> Tuple[Dict[str, Set[str]], Dict[str, Set[str]]]:
    """
    Queries exact and core name indexes for all records in df_s1.
    
    Returns:
        cand_exact: Dict[s1_id -> Set of candidate_ids matching exact name]
        cand_core:  Dict[s1_id -> Set of candidate_ids matching core name]
    """
    cand_exact: Dict[str, Set[str]] = defaultdict(set)
    cand_core: Dict[str, Set[str]] = defaultdict(set)

    for row in df_s1[["entity_id", "country_norm", "name_norm", "name_core"]].itertuples(index=False):
        s1_id = row.entity_id
        country = getattr(row, "country_norm", "") or "UNKNOWN"
        name_n = getattr(row, "name_norm", "") or ""
        name_c = getattr(row, "name_core", "") or ""

        # Exact match query
        if name_n:
            exact_matches = exact_index.get((country, name_n))
            if exact_matches and len(exact_matches) <= freq_cap * 5:  # allow exact name up to 1000
                cand_exact[s1_id] = set(exact_matches)

        # Core match query
        if name_c and name_c != name_n:
            core_matches = core_index.get((country, name_c))
            if core_matches and len(core_matches) <= freq_cap:
                cand_core[s1_id] = set(core_matches)

    return cand_exact, cand_core
