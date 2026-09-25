"""
blocking/token_block.py
-----------------------
Blocking Pass B1 — Rare Token & Short Name Deletion-Hash Blocking.

1. Identifies rare name tokens (doc frequency <= max_doc_freq per country).
2. Generates 1-deletion variant signatures for short core names (len <= 12)
   to capture single-character typos (e.g., 'Acme' vs 'Akme', 'Nike' vs 'Nyke').
"""

from collections import Counter, defaultdict
from typing import Dict, List, Set, Tuple
import pandas as pd


def get_deletion_variants(name: str, min_len: int = 4, max_len: int = 12) -> List[str]:
    """
    Generates 1-deletion variant strings for short core names.
    E.g. 'nike' -> ['ike', 'nke', 'nie', 'nik']
    """
    clean = (name or "").strip().lower()
    if not clean or len(clean) < min_len or len(clean) > max_len:
        return []
    variants = []
    for i in range(len(clean)):
        var = clean[:i] + clean[i+1:]
        if len(var) >= 3:
            variants.append(var)
    return variants


def build_rare_token_index(
    df_s2: pd.DataFrame,
    df_s3: pd.DataFrame,
    max_doc_freq: int = 50,
) -> Tuple[Dict[Tuple[str, str], Set[str]], Set[str], Dict[Tuple[str, str], Set[str]]]:
    """
    Builds inverted indexes:
        rare_index:     (country_norm, rare_token)    -> Set[entity_ids]
        deletion_index: (country_norm, deletion_var)  -> Set[entity_ids]
    """
    country_token_counts: Dict[str, Counter] = defaultdict(Counter)

    for df in [df_s2, df_s3]:
        if df.empty:
            continue
        for row in df[["entity_id", "country_norm", "name_tokens"]].itertuples(index=False):
            country = getattr(row, "country_norm", "") or "UNKNOWN"
            tokens = getattr(row, "name_tokens", []) or []
            for tok in set(tokens):
                if len(tok) >= 3:
                    country_token_counts[country][tok] += 1

    rare_tokens_set = set()
    for country, token_counts in country_token_counts.items():
        for tok, count in token_counts.items():
            if count <= max_doc_freq:
                rare_tokens_set.add((country, tok))

    rare_index: Dict[Tuple[str, str], Set[str]] = defaultdict(set)
    deletion_index: Dict[Tuple[str, str], Set[str]] = defaultdict(set)

    for df in [df_s2, df_s3]:
        if df.empty:
            continue
        for row in df[["entity_id", "country_norm", "name_tokens", "name_core"]].itertuples(index=False):
            eid = row.entity_id
            country = getattr(row, "country_norm", "") or "UNKNOWN"
            tokens = getattr(row, "name_tokens", []) or []
            core = getattr(row, "name_core", "") or ""

            # Rare token indexing
            for tok in set(tokens):
                if (country, tok) in rare_tokens_set:
                    rare_index[(country, tok)].add(eid)
                    rare_index[("UNKNOWN", tok)].add(eid)

            # Deletion variant indexing for short core names
            del_vars = get_deletion_variants(core)
            for var in del_vars:
                deletion_index[(country, var)].add(eid)

    return rare_index, rare_tokens_set, deletion_index


def block_rare_tokens(
    df_s1: pd.DataFrame,
    rare_index: Dict[Tuple[str, str], Set[str]],
    deletion_index: Dict[Tuple[str, str], Set[str]] = None,
    max_candidates_per_s1: int = 100,
) -> Dict[str, Set[str]]:
    """
    Blocks S1 records against S2/S3 using rare tokens and short name deletion variants.
    
    Returns:
        candidates: Dict[s1_id -> Set of candidate_ids]
    """
    candidates: Dict[str, Set[str]] = defaultdict(set)

    for row in df_s1[["entity_id", "country_norm", "name_tokens", "name_core"]].itertuples(index=False):
        s1_id = row.entity_id
        country = getattr(row, "country_norm", "") or "UNKNOWN"
        tokens = getattr(row, "name_tokens", []) or []
        core = getattr(row, "name_core", "") or ""

        s1_candidates = set()

        # Rare token matches
        for tok in set(tokens):
            for c in [country, "UNKNOWN"]:
                matches = rare_index.get((c, tok))
                if matches:
                    s1_candidates.update(matches)

        # Deletion variant matches for short core names
        if deletion_index and core:
            del_vars = get_deletion_variants(core)
            for var in del_vars:
                for c in [country, "UNKNOWN"]:
                    matches = deletion_index.get((c, var))
                    if matches and len(matches) <= 30:  # Tight cap on deletion variants
                        s1_candidates.update(matches)

        if s1_candidates:
            candidates[s1_id] = s1_candidates

    return candidates
