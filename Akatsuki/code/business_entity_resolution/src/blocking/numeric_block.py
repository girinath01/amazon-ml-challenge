"""
blocking/numeric_block.py
-------------------------
Blocking Pass B2 — Numeric & Locality Token Blocking.

Builds inverted index:
    (country_norm, numeric_token, rare_locality_token) -> [S2/S3 ids]

This captures entities with different business names but matching unique numeric identifiers 
(e.g., store numbers, street numbers, tax IDs embedded in strings) in the same region.
"""

from collections import Counter, defaultdict
from typing import Dict, List, Set, Tuple
import pandas as pd


def build_numeric_locality_index(
    df_s2: pd.DataFrame,
    df_s3: pd.DataFrame,
    max_locality_freq: int = 100,
) -> Tuple[Dict[Tuple[str, str, str], Set[str]], Set[Tuple[str, str]]]:
    """
    Builds inverted index mapping (country_norm, numeric_token, locality_token) -> Set[S2/S3 entity_ids].
    
    Numeric tokens are numbers extracted from address or name (e.g. '104', '4201').
    Locality tokens are rare words from business address (e.g. 'marathahalli', 'broadway').
    """
    locality_counts: Dict[str, Counter] = defaultdict(Counter)

    for df in [df_s2, df_s3]:
        if df.empty:
            continue
        for row in df.itertuples(index=False):
            country = getattr(row, "country_norm", "") or "UNKNOWN"
            addr_tokens = getattr(row, "address_tokens", []) or []
            for tok in set(addr_tokens):
                if not tok.isdigit() and len(tok) >= 4:
                    locality_counts[country][tok] += 1

    rare_localities = set()
    for country, counts in locality_counts.items():
        for tok, count in counts.items():
            if count <= max_locality_freq:
                rare_localities.add((country, tok))

    num_loc_index: Dict[Tuple[str, str, str], Set[str]] = defaultdict(set)

    for df in [df_s2, df_s3]:
        if df.empty:
            continue
        for row in df.itertuples(index=False):
            eid = row.entity_id
            country = getattr(row, "country_norm", "") or "UNKNOWN"
            nums = getattr(row, "address_numbers", []) or getattr(row, "name_numbers", []) or []
            addr_tokens = getattr(row, "address_tokens", []) or []

            locs = [t for t in addr_tokens if (country, t) in rare_localities]

            for n in set(nums):
                for l in set(locs):
                    num_loc_index[(country, n, l)].add(eid)

    return num_loc_index, rare_localities


def block_numeric_locality(
    df_s1: pd.DataFrame,
    num_loc_index: Dict[Tuple[str, str, str], Set[str]],
    max_candidates_per_s1: int = 100,
) -> Dict[str, Set[str]]:
    """
    Blocks S1 records against S2/S3 using numeric + rare locality tokens.
    
    Returns:
        candidates: Dict[s1_id -> Set of candidate_ids]
    """
    candidates: Dict[str, Set[str]] = defaultdict(set)

    for row in df_s1.itertuples(index=False):
        s1_id = row.entity_id
        country = getattr(row, "country_norm", "") or "UNKNOWN"
        nums = getattr(row, "address_numbers", []) or getattr(row, "name_numbers", []) or []
        addr_tokens = getattr(row, "address_tokens", []) or []

        s1_candidates = set()
        for n in set(nums):
            for l in set(addr_tokens):
                matches = num_loc_index.get((country, n, l))
                if matches:
                    s1_candidates.update(matches)
                    if len(s1_candidates) > max_candidates_per_s1 * 2:
                        break
            if len(s1_candidates) > max_candidates_per_s1 * 2:
                break

        if s1_candidates:
            candidates[s1_id] = s1_candidates

    return candidates
