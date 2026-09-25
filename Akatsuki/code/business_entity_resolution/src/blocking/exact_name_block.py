"""
blocking/exact_name_block.py
-------------------------------
Blocking Pass B0/B1 — Exact name, core-name, and word-order invariant matching.

Builds inverted indexes:
    (country_norm, name_norm)           -> [S2/S3 ids]
    (country_norm, name_core)           -> [S2/S3 ids]
    (country_norm, name_sorted_tokens)  -> [S2/S3 ids]  (Word-order invariant)
"""

from collections import defaultdict
from typing import Dict, List, Set, Tuple
import pandas as pd

from preprocessing.name_normalizer import get_rare_tokens

_DEFAULT_FREQ_CAP = 200


def get_sorted_tokens_key(name: str) -> str:
    """Helper to get a word-order invariant representation of a name."""
    if not name:
        return ""
    toks = [t for t in name.split() if len(t) >= 2]
    toks.sort()
    return " ".join(toks)


def build_name_indexes(
    df_s2: pd.DataFrame,
    df_s3: pd.DataFrame,
    freq_cap: int = _DEFAULT_FREQ_CAP,
) -> Tuple[Dict[Tuple[str, str], Set[str]], Dict[Tuple[str, str], Set[str]], Dict[Tuple[str, str], Set[str]]]:
    """
    Build inverted indexes from S2 and S3 DataFrames:
        exact_index:  (country, name_norm)          -> Set of entity_ids
        core_index:   (country, name_core)          -> Set of entity_ids
        sorted_index: (country, name_sorted_tokens) -> Set of entity_ids
    """
    exact_index: Dict[Tuple[str, str], Set[str]] = defaultdict(set)
    core_index: Dict[Tuple[str, str], Set[str]] = defaultdict(set)
    sorted_index: Dict[Tuple[str, str], Set[str]] = defaultdict(set)

    for df in [df_s2, df_s3]:
        if df.empty:
            continue
        for row in df[["entity_id", "country_norm", "name_norm", "name_core"]].itertuples(index=False):
            eid = row.entity_id
            country = getattr(row, "country_norm", "") or "UNKNOWN"
            name_n = getattr(row, "name_norm", "") or ""
            name_c = getattr(row, "name_core", "") or ""
            sorted_k = get_sorted_tokens_key(name_c or name_n)

            if name_n:
                exact_index[(country, name_n)].add(eid)
                exact_index[("UNKNOWN", name_n)].add(eid)
            if name_c:
                core_index[(country, name_c)].add(eid)
                core_index[("UNKNOWN", name_c)].add(eid)
            if sorted_k and len(sorted_k) >= 5:
                sorted_index[(country, sorted_k)].add(eid)
                sorted_index[("UNKNOWN", sorted_k)].add(eid)

    return exact_index, core_index, sorted_index


def block_exact_and_core_name(
    df_s1: pd.DataFrame,
    exact_index: Dict[Tuple[str, str], Set[str]],
    core_index: Dict[Tuple[str, str], Set[str]],
    sorted_index: Dict[Tuple[str, str], Set[str]] = None,
    freq_cap: int = _DEFAULT_FREQ_CAP,
) -> Tuple[Dict[str, Set[str]], Dict[str, Set[str]]]:
    """
    Queries exact, core, and word-order invariant name indexes for all records in df_s1.
    
    Returns:
        cand_exact: Dict[s1_id -> Set of candidate_ids matching exact name or sorted tokens]
        cand_core:  Dict[s1_id -> Set of candidate_ids matching core name]
    """
    cand_exact: Dict[str, Set[str]] = defaultdict(set)
    cand_core: Dict[str, Set[str]] = defaultdict(set)

    for row in df_s1[["entity_id", "country_norm", "name_norm", "name_core"]].itertuples(index=False):
        s1_id = row.entity_id
        country = getattr(row, "country_norm", "") or "UNKNOWN"
        name_n = getattr(row, "name_norm", "") or ""
        name_c = getattr(row, "name_core", "") or ""
        sorted_k = get_sorted_tokens_key(name_c or name_n)

        # 1. Exact match query (specific country + UNKNOWN fallback)
        if name_n:
            for c in [country, "UNKNOWN"]:
                exact_matches = exact_index.get((c, name_n))
                if exact_matches and len(exact_matches) <= freq_cap * 5:
                    cand_exact[s1_id].update(exact_matches)

        # 2. Word-order invariant query
        if sorted_index and sorted_k and len(sorted_k) >= 5:
            for c in [country, "UNKNOWN"]:
                sorted_matches = sorted_index.get((c, sorted_k))
                if sorted_matches and len(sorted_matches) <= freq_cap * 3:
                    cand_exact[s1_id].update(sorted_matches)

        # 3. Core match query
        if name_c and name_c != name_n:
            for c in [country, "UNKNOWN"]:
                core_matches = core_index.get((c, name_c))
                if core_matches and len(core_matches) <= freq_cap:
                    cand_core[s1_id].update(core_matches)

    return cand_exact, cand_core
