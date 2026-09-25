"""
preprocessing/address_normalizer.py
-------------------------------------
Normalizes business addresses into:
  address_norm        — lowercased, abbrev-expanded, punctuation-cleaned
  address_tokens      — sorted token list
  address_numbers     — all numeric tokens found
  house_number        — first numeric token (likely house/plot/flat number)
  postal_code         — 5-or-6-digit postal code if found
  address_has_digits  — boolean
  address_missing     — boolean (True if raw was empty/NaN)
  address_length      — character length of address_norm
"""

import re
import unicodedata

# ── Address abbreviation expansion map ─────────────────────────────────────
_ABBREV_MAP = {
    r"\brd\b":   "road",
    r"\bst\b":   "street",
    r"\bave\b":  "avenue",
    r"\bav\b":   "avenue",
    r"\bblvd\b": "boulevard",
    r"\bdr\b":   "drive",
    r"\bln\b":   "lane",
    r"\bct\b":   "court",
    r"\bpl\b":   "place",
    r"\bpkwy\b": "parkway",
    r"\bhwy\b":  "highway",
    r"\bfwy\b":  "freeway",
    r"\bsq\b":   "square",
    r"\bste\b":  "suite",
    r"\bapt\b":  "apartment",
    r"\bflr\b":  "floor",
    r"\bfl\b":   "floor",
    r"\bno\b":   "number",
    r"\bno\.\b": "number",
    r"\bmt\b":   "mount",
    r"\bmtn\b":  "mountain",
    r"\bft\b":   "fort",
    r"\bn\b":    "north",
    r"\bs\b":    "south",
    r"\be\b":    "east",
    r"\bw\b":    "west",
    r"\bnw\b":   "northwest",
    r"\bne\b":   "northeast",
    r"\bsw\b":   "southwest",
    r"\bse\b":   "southeast",
}

_POSTAL_RE   = re.compile(r"\b(\d{5,6})\b")
_NUMBER_RE   = re.compile(r"\b(\d+)\b")
_PUNCT_RE    = re.compile(r"[^\w\s]")
_SPACE_RE    = re.compile(r"\s+")


def unicode_normalize(text: str) -> str:
    return unicodedata.normalize("NFKC", text)


def normalize_address(raw) -> dict:
    """
    Full address normalization pipeline.

    Returns a dict with:
        address_raw, address_norm, address_tokens, address_numbers,
        house_number, postal_code, address_has_digits,
        address_missing, address_length
    """
    # Handle missing
    if raw is None or (isinstance(raw, float)) or not str(raw).strip():
        return {
            "address_raw":      "",
            "address_norm":     "",
            "address_tokens":   [],
            "address_numbers":  [],
            "house_number":     "",
            "postal_code":      "",
            "address_has_digits": False,
            "address_missing":  True,
            "address_length":   0,
        }

    address_raw = str(raw).strip()
    s = unicode_normalize(address_raw)
    s = s.lower()

    # Expand abbreviations
    for pattern, replacement in _ABBREV_MAP.items():
        s = re.sub(pattern, replacement, s)

    # Remove punctuation
    s = _PUNCT_RE.sub(" ", s)

    # Collapse whitespace
    s = _SPACE_RE.sub(" ", s).strip()

    # Extract postal code (before tokenizing)
    postal_match = _POSTAL_RE.search(s)
    postal_code = postal_match.group(1) if postal_match else ""

    # Extract all numeric tokens
    numbers = _NUMBER_RE.findall(s)
    house_number = numbers[0] if numbers else ""

    # Tokenize
    tokens = s.split()

    return {
        "address_raw":      address_raw,
        "address_norm":     s,
        "address_tokens":   sorted(set(tokens)),
        "address_numbers":  numbers,
        "house_number":     house_number,
        "postal_code":      postal_code,
        "address_has_digits": len(numbers) > 0,
        "address_missing":  False,
        "address_length":   len(s),
    }


def get_rare_address_tokens(address_norm: str,
                            stopwords: frozenset = frozenset()) -> list:
    """Return non-numeric, informative address tokens for blocking."""
    _ADDR_STOP = frozenset({
        "road", "street", "avenue", "boulevard", "drive", "lane",
        "court", "place", "parkway", "highway", "suite", "apartment",
        "floor", "number", "north", "south", "east", "west",
        "northwest", "northeast", "southwest", "southeast",
        "building", "block", "near", "opp", "opposite", "main",
        "area", "colony", "nagar", "sector", "phase", "plot",
        "unit", "flat", "house", "door", "shop", "office",
        "the", "and", "of", "in", "at", "to", "for",
    }) | stopwords
    tokens = address_norm.split()
    return [t for t in tokens
            if len(t) > 2
            and not t.isdigit()
            and t not in _ADDR_STOP]
