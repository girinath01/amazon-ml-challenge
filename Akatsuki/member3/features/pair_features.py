"""
pair_features.py - Full pairwise feature generator for Member 3
Combines:
- 10 Name similarity features
- 8 Address similarity features
- 6 Context features
- 4 Interaction features
Total 28 engineered features per candidate pair.
"""

import math
from typing import Dict, Any, List, Optional
import pandas as pd

from .name_features import extract_name_features, TOKEN_PATTERN as NAME_TOKEN_PATTERN
from .address_features import extract_address_features, extract_numbers


def extract_context_features(
    name1: str, name2: str,
    addr1: str, addr2: str,
    country1: str, country2: str
) -> Dict[str, float]:
    """
    Extract 6 context features comparing record metadata.
    """
    # 1. Country match
    c1 = (country1 or "").strip().upper()
    c2 = (country2 or "").strip().upper()
    country_match = 1.0 if c1 and c2 and c1 == c2 else 0.0

    # 2. Name length ratio: min(len1, len2) / max(len1, len2)
    l_n1 = len((name1 or "").strip())
    l_n2 = len((name2 or "").strip())
    if l_n1 == 0 and l_n2 == 0:
        name_length_ratio = 1.0
    elif l_n1 == 0 or l_n2 == 0:
        name_length_ratio = 0.0
    else:
        name_length_ratio = float(min(l_n1, l_n2) / max(l_n1, l_n2))

    # 3. Address length ratio
    l_a1 = len((addr1 or "").strip())
    l_a2 = len((addr2 or "").strip())
    if l_a1 == 0 and l_a2 == 0:
        address_length_ratio = 1.0
    elif l_a1 == 0 or l_a2 == 0:
        address_length_ratio = 0.0
    else:
        address_length_ratio = float(min(l_a1, l_a2) / max(l_a1, l_a2))

    # 4. Name token count difference
    t_n1 = len(NAME_TOKEN_PATTERN.findall((name1 or "").lower()))
    t_n2 = len(NAME_TOKEN_PATTERN.findall((name2 or "").lower()))
    name_token_diff = float(abs(t_n1 - t_n2))

    # 5. Address token count difference
    t_a1 = len(NAME_TOKEN_PATTERN.findall((addr1 or "").lower()))
    t_a2 = len(NAME_TOKEN_PATTERN.findall((addr2 or "").lower()))
    addr_token_diff = float(abs(t_a1 - t_a2))

    # 6. Numeric conflict: both have digits, but no common digits
    d1 = set(extract_numbers(addr1 or ""))
    d2 = set(extract_numbers(addr2 or ""))
    if d1 and d2 and not (d1 & d2):
        numeric_conflict = 1.0
    else:
        numeric_conflict = 0.0

    return {
        "country_match": country_match,
        "name_length_ratio": name_length_ratio,
        "address_length_ratio": address_length_ratio,
        "name_token_count_difference": name_token_diff,
        "address_token_count_difference": addr_token_diff,
        "numeric_conflict": numeric_conflict
    }


def extract_interaction_features(
    name_feats: Dict[str, float],
    addr_feats: Dict[str, float],
    ctx_feats: Dict[str, float]
) -> Dict[str, float]:
    """
    Extract 4 interaction features based on name, address, and context signals.
    """
    name_jacc = name_feats.get("name_jaccard", 0.0)
    addr_jacc = addr_feats.get("address_jaccard", 0.0)
    num_conflict = ctx_feats.get("numeric_conflict", 0.0)

    # 1. Product of similarities
    sim_product = float(name_jacc * addr_jacc)

    # 2. High name similarity, low address similarity (branch/relocation/name collision)
    name_high_addr_low = 1.0 if (name_jacc >= 0.75 and addr_jacc <= 0.20) else 0.0

    # 3. High name similarity, high address similarity (strong match)
    name_high_addr_high = 1.0 if (name_jacc >= 0.75 and addr_jacc >= 0.75) else 0.0

    # 4. Name address conflict: High name match, but conflicting address digits
    name_addr_conflict = 1.0 if (name_jacc >= 0.80 and num_conflict == 1.0) else 0.0

    return {
        "name_address_similarity_product": sim_product,
        "name_high_address_low": name_high_addr_low,
        "name_high_address_high": name_high_addr_high,
        "name_address_conflict": name_addr_conflict
    }


def compute_pair_features(
    name1: str, name2: str,
    addr1: str, addr2: str,
    country1: str, country2: str
) -> Dict[str, float]:
    """
    Compute all 28 pairwise features for a single entity pair.
    """
    name_f = extract_name_features(name1, name2)
    addr_f = extract_address_features(addr1, addr2)
    ctx_f = extract_context_features(name1, name2, addr1, addr2, country1, country2)
    inter_f = extract_interaction_features(name_f, addr_f, ctx_f)

    # Combine in deterministic order
    features = {}
    features.update(name_f)
    features.update(addr_f)
    features.update(ctx_f)
    features.update(inter_f)
    return features


FEATURE_NAMES = [
    # Name features (10)
    "name_exact",
    "name_jaccard",
    "name_levenshtein",
    "name_jaro_winkler",
    "name_token_sort",
    "name_token_set",
    "name_char3_cosine",
    "name_char4_cosine",
    "name_core_similarity",
    "name_translit_similarity",
    # Address features (8)
    "address_jaccard",
    "address_levenshtein",
    "address_char3_cosine",
    "address_digit_overlap",
    "house_number_match",
    "postal_code_match",
    "address_exact",
    "address_missing",
    # Context features (6)
    "country_match",
    "name_length_ratio",
    "address_length_ratio",
    "name_token_count_difference",
    "address_token_count_difference",
    "numeric_conflict",
    # Interaction features (4)
    "name_address_similarity_product",
    "name_high_address_low",
    "name_high_address_high",
    "name_address_conflict"
]
