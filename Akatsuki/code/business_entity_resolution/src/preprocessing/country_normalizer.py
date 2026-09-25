"""
preprocessing/country_normalizer.py
--------------------------------------
Normalizes country strings to a canonical open-set form.

Rules:
  - Lowercase and strip
  - Expand common abbreviations (USA → United States)
  - DO NOT hard-code an allowlist — must work for France or any new country
  - Return original normalized form if no mapping found
"""

_COUNTRY_MAP = {
    "usa":          "us",
    "united states": "us",
    "united states of america": "us",
    "u.s.a.":       "us",
    "u.s.":         "us",
    "america":      "us",
    "ind":          "india",
    "bharat":       "india",
    "republic of india": "india",
    "fr":           "france",
    "fra":          "france",
    "french republic": "france",
    "uk":           "united kingdom",
    "great britain": "united kingdom",
    "england":      "united kingdom",
}


def normalize_country(raw) -> str:
    """
    Return a normalized country string.
    Always returns a lowercase string.
    Returns empty string for null/missing.
    """
    if raw is None or (hasattr(raw, '__class__') and raw.__class__.__name__ == 'float'):
        return ""
    s = str(raw).strip().lower()
    if not s:
        return ""
    return _COUNTRY_MAP.get(s, s)
