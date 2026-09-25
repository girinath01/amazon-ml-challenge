"""
preprocessing/transliteration.py
----------------------------------
Produces name_translit: a Latin-script romanization of the input text.

Handles:
  - Devanagari (Hindi) → Latin
  - Tamil → Latin
  - Mixed Latin+Devanagari (partial transliteration)
  - Pure Latin text (pass-through)

Uses the `indic-transliteration` library (lightweight, no ML required).
Falls back gracefully if the library is not installed.
"""

import re
import unicodedata

# Script detection ranges
_DEVANAGARI_RE = re.compile(r"[\u0900-\u097F]")
_TAMIL_RE      = re.compile(r"[\u0B80-\u0BFF]")
_TELUGU_RE     = re.compile(r"[\u0C00-\u0C7F]")
_KANNADA_RE    = re.compile(r"[\u0C80-\u0CFF]")
_BENGALI_RE    = re.compile(r"[\u0980-\u09FF]")

try:
    from indic_transliteration import sanscript
    from indic_transliteration.sanscript import transliterate
    _HAS_INDIC = True
except ImportError:
    _HAS_INDIC = False


def detect_script(text: str) -> str:
    """Return dominant non-Latin script, or 'latin' if none found."""
    if _DEVANAGARI_RE.search(text):
        return "devanagari"
    if _TAMIL_RE.search(text):
        return "tamil"
    if _TELUGU_RE.search(text):
        return "telugu"
    if _KANNADA_RE.search(text):
        return "kannada"
    if _BENGALI_RE.search(text):
        return "bengali"
    return "latin"


def _transliterate_devanagari(text: str) -> str:
    """Romanize Devanagari to ITRANS/Harvard-Kyoto approximation."""
    if _HAS_INDIC:
        try:
            return transliterate(text, sanscript.DEVANAGARI, sanscript.IAST)
        except Exception:
            pass
    # Character-level fallback map for most common Devanagari consonants/vowels
    _DEVA_MAP = {
        "अ": "a", "आ": "aa", "इ": "i", "ई": "ii", "उ": "u", "ऊ": "uu",
        "ए": "e", "ऐ": "ai", "ओ": "o", "औ": "au",
        "क": "k", "ख": "kh", "ग": "g", "घ": "gh", "ङ": "ng",
        "च": "ch", "छ": "chh", "ज": "j", "झ": "jh", "ञ": "ny",
        "ट": "t", "ठ": "th", "ड": "d", "ढ": "dh", "ण": "n",
        "त": "t", "थ": "th", "द": "d", "ध": "dh", "न": "n",
        "प": "p", "फ": "ph", "ब": "b", "भ": "bh", "म": "m",
        "य": "y", "र": "r", "ल": "l", "व": "v",
        "श": "sh", "ष": "sh", "स": "s", "ह": "h",
        "क्ष": "ksh", "त्र": "tr", "ज्ञ": "gn",
        "ा": "a", "ि": "i", "ी": "i", "ु": "u", "ू": "u",
        "े": "e", "ै": "ai", "ो": "o", "ौ": "au",
        "ं": "n", "ः": "h", "्": "", "ँ": "n",
        "रॉ": "ro", "ॉ": "o",
    }
    result = text
    for src, tgt in _DEVA_MAP.items():
        result = result.replace(src, tgt)
    return result


def _transliterate_tamil(text: str) -> str:
    """Romanize Tamil to approximate Latin."""
    if _HAS_INDIC:
        try:
            return transliterate(text, sanscript.TAMIL, sanscript.IAST)
        except Exception:
            pass
    # Minimal Tamil vowel/consonant map
    _TAMIL_MAP = {
        "அ": "a", "ஆ": "aa", "இ": "i", "ஈ": "ii", "உ": "u", "ஊ": "uu",
        "எ": "e", "ஏ": "ee", "ஐ": "ai", "ஒ": "o", "ஓ": "oo", "ஔ": "au",
        "க": "k", "ங": "ng", "ச": "ch", "ஞ": "ny", "ட": "t", "ண": "n",
        "த": "th", "ந": "n", "ப": "p", "ம": "m", "ய": "y", "ர": "r",
        "ல": "l", "வ": "v", "ழ": "zh", "ள": "l", "ற": "r", "ன": "n",
        "ஜ": "j", "ஶ": "sh", "ஷ": "sh", "ஸ": "s", "ஹ": "h",
    }
    result = text
    for src, tgt in _TAMIL_MAP.items():
        result = result.replace(src, tgt)
    return result


def transliterate_name(text: str) -> str:
    """
    Produce a Latin romanization of text containing non-Latin scripts.

    - Pure Latin input → returned as-is (pass-through)
    - Devanagari/Tamil mixed with Latin → transliterate non-Latin portions
    - Result is always ASCII-compatible Latin
    """
    if not isinstance(text, str) or not text.strip():
        return ""

    script = detect_script(text)

    if script == "latin":
        return text.lower().strip()

    if script == "devanagari":
        result = _transliterate_devanagari(text)
    elif script == "tamil":
        result = _transliterate_tamil(text)
    else:
        # For Telugu/Kannada/Bengali: best effort with NFKD decomposition
        result = unicodedata.normalize("NFKD", text)
        result = result.encode("ascii", errors="ignore").decode("ascii")

    # Post-clean: lowercase, collapse whitespace
    result = re.sub(r"\s+", " ", result).strip().lower()
    return result


def is_non_latin(text: str) -> bool:
    """Return True if the text contains any non-Latin-script characters."""
    return detect_script(text) != "latin"
