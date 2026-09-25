"""
blocking/token_block.py
-----------------------
Blocking Pass B1 — Rare Token Blocking.

Identifies rare name tokens (document frequency <= freq_cap per country, e.g. <= 50).
Builds inverted index: (country_norm, rare_token) -> [S2/S3 ids].
Blocks S1 records against S2/S3 matching on rare tokens.
"""

from collections import Counter, defaultdict
from typing import Dict, List, Set, Tuple
import pandas as pd


def build_rare_token_index(
    df_s2: pd.DataFrame,
    df_s3: pd.DataFrame,
    max_doc_freq: int = 50,
) -> Tuple[Dict[Tuple[str, str], Set[str]], Set[str]]:
    """
    Builds an inverted index mapping (country_norm, rare_token) -> Set of S2/S3 entity_ids.
    Tokens are identified as rare if their total occurrences within a country are <= max_doc_freq.
    
    Returns:
        rare_index: Dict[(country_norm, token), Set[entity_id]]
        rare_tokens_set: Set of (country_norm, token) tuples considered rare.
    """
    country_token_counts: Dict[str, Counter] = defaultdict(Counter)

    for df in [df_s2, df_s3]:
        if df.empty:
            continue
        for row in df[["entity_id", "country_norm", "name_tokens"]].itertuples(index=False):
            country = getattr(row, "country_norm", "") or "UNKNOWN"
            tokens = getattr(row, "name_tokens", []) or []
            # Count unique token per document
            for tok in set(tokens):
                if len(tok) >= 3:  # Skip 1-2 char tokens for rare indexing
                    country_token_counts[country][tok] += 1

    rare_tokens_set = set()
    for country, token_counts in country_token_counts.items():
        for tok, count in token_counts.items():
            if count <= max_doc_freq:
                rare_tokens_set.add((country, tok))

    rare_index: Dict[Tuple[str, str], Set[str]] = defaultdict(set)
    for df in [df_s2, df_s3]:
        if df.empty:
            continue
        for row in df[["entity_id", "country_norm", "name_tokens"]].itertuples(index=False):
            eid = row.entity_id
            country = getattr(row, "country_norm", "") or "UNKNOWN"
            tokens = getattr(row, "name_tokens", []) or []
            for tok in set(tokens):
                if (country, tok) in rare_tokens_set:
                    rare_index[(country, tok)].add(eid)

    return rare_index, rare_tokens_set


def block_rare_tokens(
    df_s1: pd.DataFrame,
    rare_index: Dict[Tuple[str, str], Set[str]],
    max_candidates_per_s1: int = 100,
) -> Dict[str, Set[str]]:
    """
    Blocks S1 records against S2/S3 using rare name tokens.
    
    Returns:
        candidates: Dict[s1_id -> Set of candidate_ids]
    """
    candidates: Dict[str, Set[str]] = defaultdict(set)

    for row in df_s1[["entity_id", "country_norm", "name_tokens"]].itertuples(index=False):
        s1_id = row.entity_id
        country = getattr(row, "country_norm", "") or "UNKNOWN"
        tokens = getattr(row, "name_tokens", []) or []

        s1_candidates = set()
        for tok in set(tokens):
            matches = rare_index.get((country, tok))
            if matches:
                s1_candidates.update(matches)
                if len(s1_candidates) > max_candidates_per_s1 * 2:
                    break

        if s1_candidates:
            candidates[s1_id] = s1_candidates

    return candidates
