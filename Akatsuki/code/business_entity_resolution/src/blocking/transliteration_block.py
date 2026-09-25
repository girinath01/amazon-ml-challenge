"""
blocking/transliteration_block.py
-----------------------------------
Blocking Pass B3 — Transliteration-aware retrieval.

Specifically designed for Devanagari/Tamil cross-script cases:
Latin S1 names matched against Indic-script S2/S3 names.

Builds inverted indexes on name_translit (the Latin romanization).
"""

from collections import defaultdict
from typing import Dict, List, Set, Tuple
import pandas as pd

from preprocessing.name_normalizer import get_rare_tokens

_DEFAULT_FREQ_CAP = 200


def build_translit_index(
    df_s2: pd.DataFrame,
    df_s3: pd.DataFrame,
) -> Tuple[Dict[Tuple[str, str], Set[str]], Dict[Tuple[str, str], Set[str]]]:
    """
    Build transliteration-based inverted indexes across non-Latin S2 and S3 records.
    
    Returns:
        translit_exact_index: (country, name_translit) -> Set[entity_id]
        translit_token_index: (country, rare_translit_token) -> Set[entity_id]
    """
    translit_exact_index: Dict[Tuple[str, str], Set[str]] = defaultdict(set)
    translit_token_index: Dict[Tuple[str, str], Set[str]] = defaultdict(set)

    for df in [df_s2, df_s3]:
        if df.empty:
            continue
        cols = [c for c in ["entity_id", "country_norm", "name_translit", "name_is_non_latin"] if c in df.columns]

        for row in df[cols].itertuples(index=False):
            eid = row.entity_id
            country = getattr(row, "country_norm", "") or "UNKNOWN"
            translit = getattr(row, "name_translit", "") or ""
            is_non_lat = getattr(row, "name_is_non_latin", False)

            if not is_non_lat or not translit.strip():
                continue

            translit_exact_index[(country, translit)].add(eid)

            rare_toks = get_rare_tokens(translit)
            for tok in rare_toks:
                translit_token_index[(country, tok)].add(eid)

    return translit_exact_index, translit_token_index


def block_transliteration(
    df_s1: pd.DataFrame,
    translit_indexes: Tuple[Dict, Dict],
    max_candidates_per_s1: int = 50,
) -> Dict[str, Set[str]]:
    """
    Queries transliteration indexes for df_s1 records.
    
    Returns:
        candidates: Dict[s1_id -> Set of candidate_ids]
    """
    translit_exact_idx, translit_token_idx = translit_indexes
    candidates: Dict[str, Set[str]] = defaultdict(set)

    for row in df_s1.itertuples(index=False):
        s1_id = row.entity_id
        country = getattr(row, "country_norm", "") or "UNKNOWN"
        name_n = getattr(row, "name_norm", "") or ""
        translit = getattr(row, "name_translit", "") or ""

        s1_candidates = set()

        # Exact match of S1 name against S2/S3 transliteration
        if name_n:
            matches = translit_exact_idx.get((country, name_n))
            if matches:
                s1_candidates.update(matches)

        # Transliterated S1 name against S2/S3 transliteration
        if translit and translit != name_n:
            matches = translit_exact_idx.get((country, translit))
            if matches:
                s1_candidates.update(matches)

        # Token matches via transliteration
        rare_toks = get_rare_tokens(name_n)
        for tok in rare_toks:
            matches = translit_token_idx.get((country, tok))
            if matches and len(matches) <= _DEFAULT_FREQ_CAP:
                s1_candidates.update(matches)

        if s1_candidates:
            candidates[s1_id] = s1_candidates

    return candidates
