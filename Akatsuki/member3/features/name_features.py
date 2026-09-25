"""
name_features.py - Name similarity feature extraction for Member 3
Computes all required name similarity features between S1 and candidate S2/S3 records:
- name_exact
- name_jaccard
- name_levenshtein
- name_jaro_winkler
- name_token_sort
- name_token_set
- name_char3_cosine
- name_char4_cosine
- name_core_similarity
- name_translit_similarity
"""

import re
import math
import unicodedata
from collections import Counter
from typing import Dict, List, Set, Union
from rapidfuzz import fuzz, distance
from unidecode import unidecode


# Common legal suffixes to extract name_core
LEGAL_SUFFIXES_PATTERN = re.compile(
    r'\b(?:pvt|private|ltd|limited|llc|l\.l\.c\.|inc|incorporated|corp|corporation|llp|l\.l\.p\.|'
    r'co|company|enterprises|enterprise|holdings|holding|group|services|service|gmbh|sa|sarl|plc)\b',
    flags=re.IGNORECASE
)

# Precompiled token pattern
TOKEN_PATTERN = re.compile(r'\w+')


def clean_text(text: str) -> str:
    """Normalize whitespace and lowercase."""
    if not text or not isinstance(text, str):
        return ""
    # Unicode NFKD normalization
    text = unicodedata.normalize('NFKD', text)
    # Replace ampersand
    text = text.replace('&', ' and ')
    return " ".join(text.lower().split())


def extract_core_name(text: str) -> str:
    """Strip legal suffixes and surrounding punctuation to get core business name."""
    clean = clean_text(text)
    if not clean:
        return ""
    core = LEGAL_SUFFIXES_PATTERN.sub('', clean)
    core = re.sub(r'[^\w\s]', ' ', core)
    return " ".join(core.split())


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


def token_jaccard(s1: str, s2: str) -> float:
    """Compute token Jaccard similarity."""
    t1 = set(TOKEN_PATTERN.findall(s1.lower()))
    t2 = set(TOKEN_PATTERN.findall(s2.lower()))
    if not t1 and not t2:
        return 1.0
    if not t1 or not t2:
        return 0.0
    return float(len(t1 & t2) / len(t1 | t2))


def extract_name_features(name1: str, name2: str) -> Dict[str, float]:
    """
    Extract all 10 pairwise name similarity features.
    
    Safe for null, empty strings, and foreign characters.
    """
    c1 = clean_text(name1)
    c2 = clean_text(name2)
    
    # 1. Exact match
    name_exact = 1.0 if c1 and c2 and c1 == c2 else (1.0 if not c1 and not c2 else 0.0)
    
    # 2. Token Jaccard
    name_jaccard = token_jaccard(c1, c2)
    
    # 3. Levenshtein ratio (0.0 to 1.0)
    if not c1 and not c2:
        name_levenshtein = 1.0
    elif not c1 or not c2:
        name_levenshtein = 0.0
    else:
        name_levenshtein = float(fuzz.ratio(c1, c2) / 100.0)
        
    # 4. Jaro-Winkler similarity (0.0 to 1.0)
    if not c1 and not c2:
        name_jaro_winkler = 1.0
    elif not c1 or not c2:
        name_jaro_winkler = 0.0
    else:
        name_jaro_winkler = float(distance.JaroWinkler.similarity(c1, c2))
        
    # 5. Token sort ratio (0.0 to 1.0)
    if not c1 and not c2:
        name_token_sort = 1.0
    elif not c1 or not c2:
        name_token_sort = 0.0
    else:
        name_token_sort = float(fuzz.token_sort_ratio(c1, c2) / 100.0)
        
    # 6. Token set ratio (0.0 to 1.0)
    if not c1 and not c2:
        name_token_set = 1.0
    elif not c1 or not c2:
        name_token_set = 0.0
    else:
        name_token_set = float(fuzz.token_set_ratio(c1, c2) / 100.0)
        
    # 7. Character 3-gram cosine
    name_char3_cosine = char_ngram_cosine(c1, c2, n=3)
    
    # 8. Character 4-gram cosine
    name_char4_cosine = char_ngram_cosine(c1, c2, n=4)
    
    # 9. Core name similarity (stripped of legal suffixes)
    core1 = extract_core_name(c1)
    core2 = extract_core_name(c2)
    if not core1 and not core2:
        name_core_sim = 1.0
    elif not core1 or not core2:
        name_core_sim = 0.0
    else:
        name_core_sim = float(fuzz.token_sort_ratio(core1, core2) / 100.0)
        
    # 10. Transliterated similarity (Indic / Diacritics -> ASCII)
    tr1 = unidecode(c1)
    tr2 = unidecode(c2)
    if not tr1 and not tr2:
        name_translit_sim = 1.0
    elif not tr1 or not tr2:
        name_translit_sim = 0.0
    else:
        name_translit_sim = float(fuzz.token_set_ratio(tr1, tr2) / 100.0)

    return {
        "name_exact": name_exact,
        "name_jaccard": name_jaccard,
        "name_levenshtein": name_levenshtein,
        "name_jaro_winkler": name_jaro_winkler,
        "name_token_sort": name_token_sort,
        "name_token_set": name_token_set,
        "name_char3_cosine": name_char3_cosine,
        "name_char4_cosine": name_char4_cosine,
        "name_core_similarity": name_core_sim,
        "name_translit_similarity": name_translit_sim
    }
