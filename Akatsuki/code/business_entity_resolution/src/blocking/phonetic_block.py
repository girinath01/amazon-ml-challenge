"""
blocking/phonetic_block.py
--------------------------
Blocking Pass B7 — Supplemental Phonetic Token Blocking.

Uses Soundex encoding on business name tokens to capture phonetic spelling variations
(e.g., 'Snyder' vs 'Schneider', 'Pharma' vs 'Farma') within country partitions.

Applies tight frequency capping to prevent common phonetic bucket explosions.
"""

from collections import defaultdict
from typing import Dict, List, Set, Tuple
import pandas as pd


def soundex(token: str) -> str:
    """
    Computes standard Soundex code for a word token.
    """
    token = token.upper()
    if not token or not token.isalpha():
        return ""
    
    first = token[0]
    mapping = {
        'B': '1', 'F': '1', 'P': '1', 'V': '1',
        'C': '2', 'G': '2', 'J': '2', 'K': '2', 'Q': '2', 'S': '2', 'X': '2', 'Z': '2',
        'D': '3', 'T': '3',
        'L': '4',
        'M': '5', 'N': '5',
        'R': '6'
    }
    
    code = first
    prev = mapping.get(first, '')
    
    for char in token[1:]:
        digit = mapping.get(char, '')
        if digit and digit != prev:
            code += digit
            if len(code) == 4:
                break
        prev = digit if digit else prev
        
    return code.ljust(4, '0')


def build_phonetic_index(
    df_s2: pd.DataFrame,
    df_s3: pd.DataFrame,
    max_freq_cap: int = 100,
) -> Tuple[Dict[Tuple[str, str], Set[str]], Set[Tuple[str, str]]]:
    """
    Builds inverted index: (country_norm, soundex_code) -> Set[entity_ids]
    """
    phonetic_counts: Dict[Tuple[str, str], int] = defaultdict(int)

    for df in [df_s2, df_s3]:
        if df.empty:
            continue
        for row in df[["entity_id", "country_norm", "name_tokens"]].itertuples(index=False):
            country = getattr(row, "country_norm", "") or "UNKNOWN"
            tokens = getattr(row, "name_tokens", []) or []
            seen_codes = set()
            for tok in tokens:
                if len(tok) >= 3:
                    code = soundex(tok)
                    if code and code not in seen_codes:
                        seen_codes.add(code)
                        phonetic_counts[(country, code)] += 1

    valid_codes = {k for k, count in phonetic_counts.items() if count <= max_freq_cap}

    phonetic_index: Dict[Tuple[str, str], Set[str]] = defaultdict(set)

    for df in [df_s2, df_s3]:
        if df.empty:
            continue
        for row in df[["entity_id", "country_norm", "name_tokens"]].itertuples(index=False):
            eid = row.entity_id
            country = getattr(row, "country_norm", "") or "UNKNOWN"
            tokens = getattr(row, "name_tokens", []) or []
            for tok in tokens:
                if len(tok) >= 3:
                    code = soundex(tok)
                    if (country, code) in valid_codes:
                        phonetic_index[(country, code)].add(eid)
                        phonetic_index[("UNKNOWN", code)].add(eid)

    return phonetic_index, valid_codes


def block_phonetic(
    df_s1: pd.DataFrame,
    phonetic_index: Dict[Tuple[str, str], Set[str]],
    max_candidates_per_s1: int = 30,
) -> Dict[str, Set[str]]:
    """
    Blocks S1 records against S2/S3 using phonetic token codes.
    
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
            if len(tok) >= 3:
                code = soundex(tok)
                if code:
                    for c in [country, "UNKNOWN"]:
                        matches = phonetic_index.get((c, code))
                        if matches:
                            s1_candidates.update(matches)
                            if len(s1_candidates) > max_candidates_per_s1 * 2:
                                break

        if s1_candidates:
            candidates[s1_id] = s1_candidates

    return candidates
