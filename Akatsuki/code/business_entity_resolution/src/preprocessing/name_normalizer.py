"""
preprocessing/name_normalizer.py
---------------------------------
Normalizes business names into:
  name_norm      — lowercased, punctuation-cleaned, suffix-canonicalized
  name_core      — name_norm with legal suffixes removed
  name_tokens    — sorted token list from name_norm
  name_length    — character length of name_norm
  name_has_digits — boolean
"""

import re
import unicodedata

# ── Legal suffix canonical mapping ─────────────────────────────────────────
# All variants → one canonical token that stays in name_norm
_SUFFIX_MAP = {
    # Corporation variants
    "corporation": "corp", "corp.": "corp",
    # Incorporated variants
    "incorporated": "inc", "inc.": "inc",
    # Limited variants
    "limited": "ltd", "ltd.": "ltd",
    # Private variants
    "private": "pvt", "pvt.": "pvt",
    # LLC variants
    "l.l.c.": "llc", "l.l.c": "llc",
    # LLP variants
    "l.l.p.": "llp", "l.l.p": "llp",
    # Company variants
    "company": "co", "co.": "co",
    # Brothers variants
    "brothers": "bros", "bros.": "bros",
    # And variants
    "&": "and",
    # Associates variants
    "associates": "assoc", "assoc.": "assoc",
}

# Suffix tokens removed to create name_core
_LEGAL_SUFFIXES = frozenset({
    "corp", "inc", "ltd", "pvt", "llc", "llp",
    "co", "plc", "lp", "pc", "pa", "na",
    "bros", "assoc", "enterprises", "enterprise",
    "solutions", "services", "group", "holdings",
    "international", "global", "management",
})

_PUNCT_RE   = re.compile(r"[^\w\s]")
_SPACE_RE   = re.compile(r"\s+")
_DIGIT_RE   = re.compile(r"\d")


def unicode_normalize(text: str) -> str:
    """Apply NFKC normalization to collapse compatibility characters."""
    return unicodedata.normalize("NFKC", text)


def normalize_name(raw: str) -> dict:
    """
    Full name normalization pipeline.

    Returns a dict with:
        name_norm, name_core, name_tokens, name_length, name_has_digits
    """
    if not isinstance(raw, str) or not raw.strip():
        return {
            "name_norm": "",
            "name_core": "",
            "name_tokens": [],
            "name_length": 0,
            "name_has_digits": False,
        }

    # 1. Unicode normalization
    s = unicode_normalize(raw)

    # 2. Lowercase
    s = s.lower()

    # 3. Replace & → and (before punct removal)
    s = s.replace("&", " and ")

    # 4. Remove punctuation (keep alphanumeric + spaces)
    s = _PUNCT_RE.sub(" ", s)

    # 5. Collapse whitespace
    s = _SPACE_RE.sub(" ", s).strip()

    # 6. Tokenize
    tokens = s.split()

    # 7. Apply suffix map token-by-token (normalizes variant forms)
    tokens = [_SUFFIX_MAP.get(t, t) for t in tokens]

    # 8. name_norm — rejoined, suffix-canonicalized
    name_norm = " ".join(tokens)

    # 9. name_core — remove trailing legal suffix tokens
    core_tokens = tokens[:]
    while core_tokens and core_tokens[-1] in _LEGAL_SUFFIXES:
        core_tokens.pop()
    # Also strip leading legal suffix tokens
    while core_tokens and core_tokens[0] in _LEGAL_SUFFIXES:
        core_tokens.pop(0)
    name_core = " ".join(core_tokens) if core_tokens else name_norm

    return {
        "name_norm":      name_norm,
        "name_core":      name_core,
        "name_tokens":    sorted(set(tokens)),   # sorted unique tokens for indexing
        "name_length":    len(name_norm),
        "name_has_digits": bool(_DIGIT_RE.search(name_norm)),
    }


def get_rare_tokens(name_norm: str, stopwords: frozenset = frozenset()) -> list:
    """Return tokens from name_norm excluding stopwords and very short tokens."""
    tokens = name_norm.split()
    return [t for t in tokens if len(t) > 2 and t not in stopwords and t not in _LEGAL_SUFFIXES]
