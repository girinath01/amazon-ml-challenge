"""
blocking/address_block.py
---------------------------
Blocking Passes B2/B3 — Address + Numeric blocking.

Builds inverted indexes on:
    (country, postal_code)                         -> [ids]
    (country, house_number, rare_address_token)    -> [ids]
    (country, numeric_token, rare_address_token)   -> [ids]
    (country, rare_address_token)                  -> [ids]  (if doc freq is low)
"""

from collections import defaultdict
from typing import Dict, List, Set, Tuple
import pandas as pd

from preprocessing.address_normalizer import get_rare_address_tokens

_DEFAULT_FREQ_CAP = 200
_RARE_ADDR_TOKEN_CAP = 100


def build_address_indexes(
    df_s2: pd.DataFrame,
    df_s3: pd.DataFrame,
) -> Tuple[Dict, Dict, Dict, Dict]:
    """
    Build address-based inverted indexes across S2 and S3.
    """
    postcode_index: Dict[Tuple, Set[str]] = defaultdict(set)
    house_addr_index: Dict[Tuple, Set[str]] = defaultdict(set)
    num_addr_index: Dict[Tuple, Set[str]] = defaultdict(set)
    rare_addr_index: Dict[Tuple, Set[str]] = defaultdict(set)

    for df in [df_s2, df_s3]:
        if df.empty:
            continue
        cols = [c for c in ["entity_id", "country_norm", "address_norm", "house_number", "postal_code", "address_numbers", "address_missing"] if c in df.columns]

        for row in df.itertuples(index=False):
            eid = row.entity_id
            country = getattr(row, "country_norm", "") or "UNKNOWN"
            missing = getattr(row, "address_missing", False)

            if missing:
                continue

            addr_norm = getattr(row, "address_norm", "") or ""
            house_num = getattr(row, "house_number", "") or ""
            postal = getattr(row, "postal_code", "") or ""
            addr_nums = getattr(row, "address_numbers", []) or []

            rare_toks = get_rare_address_tokens(addr_norm)

            if postal:
                postcode_index[(country, postal)].add(eid)

            if house_num and rare_toks:
                for tok in rare_toks[:3]:
                    house_addr_index[(country, house_num, tok)].add(eid)

            for num in (addr_nums or [])[:3]:
                for tok in rare_toks[:2]:
                    num_addr_index[(country, num, tok)].add(eid)

            for tok in rare_toks[:2]:
                rare_addr_index[(country, tok)].add(eid)

    return postcode_index, house_addr_index, num_addr_index, rare_addr_index


def block_address_rare(
    df_s1: pd.DataFrame,
    df_s2: pd.DataFrame,
    df_s3: pd.DataFrame,
    max_candidates_per_s1: int = 50,
) -> Dict[str, Set[str]]:
    """
    Runs address blocking passes across df_s1 using S2/S3 address indexes.
    
    Returns:
        candidates: Dict[s1_id -> Set of candidate_ids]
    """
    postcode_idx, house_idx, num_idx, rare_idx = build_address_indexes(df_s2, df_s3)
    candidates: Dict[str, Set[str]] = defaultdict(set)

    for row in df_s1.itertuples(index=False):
        s1_id = row.entity_id
        country = getattr(row, "country_norm", "") or "UNKNOWN"
        missing = getattr(row, "address_missing", False)

        if missing:
            continue

        addr_norm = getattr(row, "address_norm", "") or ""
        house_num = getattr(row, "house_number", "") or ""
        postal = getattr(row, "postal_code", "") or ""
        addr_nums = getattr(row, "address_numbers", []) or []

        s1_candidates = set()
        rare_toks = get_rare_address_tokens(addr_norm)

        # 1. Postal code
        if postal:
            matches = postcode_idx.get((country, postal))
            if matches:
                s1_candidates.update(matches)

        # 2. House number + rare token
        if house_num and rare_toks:
            for tok in rare_toks[:3]:
                matches = house_idx.get((country, house_num, tok))
                if matches and len(matches) <= _DEFAULT_FREQ_CAP:
                    s1_candidates.update(matches)

        # 3. Numeric + rare token
        for num in (addr_nums or [])[:3]:
            for tok in rare_toks[:2]:
                matches = num_idx.get((country, num, tok))
                if matches and len(matches) <= _DEFAULT_FREQ_CAP:
                    s1_candidates.update(matches)

        # 4. Rare token alone
        for tok in rare_toks[:2]:
            matches = rare_idx.get((country, tok))
            if matches and len(matches) <= _RARE_ADDR_TOKEN_CAP:
                s1_candidates.update(matches)

        if s1_candidates:
            candidates[s1_id] = s1_candidates

    return candidates
