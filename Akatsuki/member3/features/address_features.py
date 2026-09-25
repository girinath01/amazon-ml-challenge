"""
address_features.py - Address similarity feature extraction for Member 3
Computes all required address similarity features between S1 and candidate S2/S3 records:
- address_jaccard
- address_levenshtein
- address_char3_cosine
- address_digit_overlap
- house_number_match
- postal_code_match
- address_exact
- address_missing
"""

import re
import math
import unicodedata
from collections import Counter
from typing import Dict, List, Optional, Set
from rapidfuzz import fuzz

# Precompiled regex patterns
TOKEN_PATTERN = re.compile(r'\w+')
DIGIT_PATTERN = re.compile(r'\d+')
POSTAL_CODE_PATTERN = re.compile(r'\b(?:\d{5}|\b[1-9]\d{5})\b')
LEADING_ZEROES = re.compile(r'^0+')

# Address street abbreviation mapping
STREET_ABBREVIATIONS = {
    'rd': 'road',
    'st': 'street',
    'ave': 'avenue',
    'av': 'avenue',
    'blvd': 'boulevard',
    'dr': 'drive',
    'ln': 'lane',
    'ct': 'court',
    'pl': 'place',
    'cir': 'circle',
    'ter': 'terrace',
    'ste': 'suite',
    'apt': 'apartment',
    'bldg': 'building',
    'fl': 'floor',
    'flr': 'floor',
    'hwy': 'highway',
    'pkwy': 'parkway',
    'pk': 'park',
    'sq': 'square',
    'w': 'west',
    'e': 'east',
    'n': 'north',
    's': 'south'
}


def clean_address(text: str) -> str:
    """Normalize whitespace, lowercase, expand street abbreviations."""
    if not text or not isinstance(text, str):
        return ""
    text = unicodedata.normalize('NFKD', text)
    tokens = TOKEN_PATTERN.findall(text.lower())
    canonical_tokens = [STREET_ABBREVIATIONS.get(t, t) for t in tokens]
    return " ".join(canonical_tokens)


def extract_numbers(text: str) -> List[str]:
    """Extract all numeric tokens, stripping leading zeroes for consistent comparison."""
    if not text:
        return []
    raw = DIGIT_PATTERN.findall(text)
    # Strip leading zeroes e.g. 01415 -> 1415, but preserve single '0'
    normalized = [t.lstrip('0') or '0' for t in raw]
    return normalized


def extract_house_number(text: str) -> Optional[str]:
    """Extract the first prominent building/house number token."""
    nums = extract_numbers(text)
    if nums:
        # First non-empty numeric token
        return nums[0]
    return None


def extract_postal_code(text: str) -> Optional[str]:
    """Extract 5-digit (US/France) or 6-digit (India) postal code."""
    if not text:
        return None
    matches = POSTAL_CODE_PATTERN.findall(text)
    if matches:
        return matches[-1] # Postal codes usually appear near end of address
    return None


def char_ngram_cosine(s1: str, s2: str, n: int = 3) -> float:
    """Compute character n-gram cosine similarity using term frequencies."""
    if not s1 and not s2:
        return 1.0
    if not s1 or not s2:
        return 0.0
    s1, s2 = s1.lower().strip(), s2.lower().strip()
    if len(s1) < n or len(s2) < n:
        return 1.0 if s1 == s2 else 0.0
    g1 = Counter(s1[i:i+n] for i in range(len(s1)-n+1))
    g2 = Counter(s2[i:i+n] for i in range(len(s2)-n+1))
    dot = sum(g1[k] * g2[k] for k in g1 if k in g2)
    norm1 = math.sqrt(sum(v*v for v in g1.values()))
    norm2 = math.sqrt(sum(v*v for v in g2.values()))
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return float(dot / (norm1 * norm2))


def extract_address_features(addr1: str, addr2: str) -> Dict[str, float]:
    """
    Extract all 8 pairwise address similarity features.
    
    Correctly handles missing addresses (address_missing = 1).
    """
    raw1 = (addr1 or "").strip()
    raw2 = (addr2 or "").strip()
    
    # Missing address flag: 1 if either is empty/missing
    address_missing = 1.0 if (not raw1 or not raw2) else 0.0
    
    if address_missing == 1.0:
        # Fallback values when address is missing in either record
        return {
            "address_jaccard": 0.0,
            "address_levenshtein": 0.0,
            "address_char3_cosine": 0.0,
            "address_digit_overlap": -1.0,  # Missing digit indicator
            "house_number_match": -1.0,     # Missing indicator
            "postal_code_match": -1.0,      # Missing indicator
            "address_exact": 0.0,
            "address_missing": 1.0
        }
    
    c1 = clean_address(raw1)
    c2 = clean_address(raw2)
    
    # 1. Address exact match
    address_exact = 1.0 if c1 == c2 else 0.0
    
    # 2. Token Jaccard
    t1 = set(c1.split())
    t2 = set(c2.split())
    address_jaccard = float(len(t1 & t2) / len(t1 | t2)) if (t1 or t2) else 1.0
    
    # 3. Levenshtein ratio
    address_levenshtein = float(fuzz.ratio(c1, c2) / 100.0)
    
    # 4. Character 3-gram cosine
    address_char3_cosine = char_ngram_cosine(c1, c2, n=3)
    
    # 5. Digit overlap
    nums1 = set(extract_numbers(raw1))
    nums2 = set(extract_numbers(raw2))
    if not nums1 and not nums2:
        address_digit_overlap = -1.0 # Neither has digits
    elif not nums1 or not nums2:
        address_digit_overlap = 0.0  # One has digits, other doesn't
    else:
        address_digit_overlap = float(len(nums1 & nums2) / len(nums1 | nums2))
        
    # 6. House number match
    h1 = extract_house_number(raw1)
    h2 = extract_house_number(raw2)
    if h1 is not None and h2 is not None:
        house_number_match = 1.0 if h1 == h2 else 0.0
    else:
        house_number_match = -1.0 # Missing indicator
        
    # 7. Postal code match
    p1 = extract_postal_code(raw1)
    p2 = extract_postal_code(raw2)
    if p1 is not None and p2 is not None:
        postal_code_match = 1.0 if p1 == p2 else 0.0
    else:
        postal_code_match = -1.0 # Missing indicator

    return {
        "address_jaccard": address_jaccard,
        "address_levenshtein": address_levenshtein,
        "address_char3_cosine": address_char3_cosine,
        "address_digit_overlap": address_digit_overlap,
        "house_number_match": house_number_match,
        "postal_code_match": postal_code_match,
        "address_exact": address_exact,
        "address_missing": 0.0
    }
