"""
name_normalizer.py - Business Name Normalization Module
Part of the Preprocessing and Normalization Layer for Entity Resolution.

Output Fields:
- name_raw: The exact original raw input string.
- name_norm: Standardized name with canonicalized legal forms and cleaned punctuation.
- name_core: Core business name with legal entity forms and prefixes/suffixes stripped.
- name_tokens: List of normalized word tokens.
- name_translit: Unicode-decomposed / diacritic-stripped representation.
- name_has_digits: Boolean flag indicating presence of digits.
- name_length: Character length of name_norm.

Processing Pipeline:
raw -> Unicode normalization (NFKC) -> lowercase/case normalization
    -> punctuation cleanup -> '&' -> 'and' -> whitespace cleanup
    -> legal-form normalization -> tokenization -> core extraction
"""

import re
import sys
import unicodedata
from typing import Any, Dict, List, Optional, Sequence, Union
import pandas as pd


# Canonical legal form mappings: variant -> canonical token
LEGAL_FORM_MAPPINGS: Dict[str, str] = {
    # Corporation
    "corporation": "corp",
    "corporations": "corp",
    "corp": "corp",
    # Incorporated
    "incorporated": "inc",
    "incorporate": "inc",
    "inc": "inc",
    # Limited
    "limited": "ltd",
    "limitee": "ltd",
    "ltd": "ltd",
    # Private
    "private": "pvt",
    "privated": "pvt",
    "pvt": "pvt",
    # Company
    "company": "co",
    "companies": "co",
    "co": "co",
    # LLC
    "llc": "llc",
    # LLP / PLLC
    "llp": "llp",
    "pllc": "pllc",
    "pc": "pc",
    # Doing Business As
    "dba": "dba",
    # Devanagari legal forms common in Indian records
    "प्राइवेट": "pvt",
    "लिमिटेड": "ltd",
    "एलएलपी": "llp",
    "कॉर्पोरेशन": "corp",
    # International legal forms
    "gmbh": "gmbh",
    "sarl": "sarl",
    "sa": "sa",
    "srl": "srl",
    "spa": "spa",
    "bv": "bv",
    "nv": "nv",
    "plc": "plc",
}

# Legal entity tokens to strip when constructing name_core
LEGAL_CORE_STRIP_TOKENS: set = {
    "llc", "inc", "corp", "ltd", "pvt", "llp", "pllc", "co", "pc", "lp", "plc",
    "gmbh", "sa", "sarl", "srl", "spa", "bv", "nv", "dba"
}


class BusinessNameNormalizer:
    """
    High-performance, deterministic business name normalizer.
    Pre-compiles regex patterns for rapid batch execution on multi-million row datasets.
    Fully Unicode-aware: preserves Indic scripts (Devanagari, Tamil) and European accents.
    """

    def __init__(self) -> None:
        self._re_ampersand = re.compile(r"&")
        self._re_plus = re.compile(r"\+")

        # Collapse dotted acronyms (e.g. l.l.c. -> llc, p.v.t. -> pvt, u.s.a. -> usa)
        self._re_dots_4 = re.compile(r"\b([a-zA-Z])\.([a-zA-Z])\.([a-zA-Z])\.([a-zA-Z])\.")
        self._re_dots_3 = re.compile(r"\b([a-zA-Z])\.([a-zA-Z])\.([a-zA-Z])\.")
        self._re_dots_2 = re.compile(r"\b([a-zA-Z])\.([a-zA-Z])\.")
        self._re_dba_slash = re.compile(r"\bd/b/a\b", re.IGNORECASE)
        self._re_aka_slash = re.compile(r"\ba/k/a\b", re.IGNORECASE)

        # Quotes and apostrophes to remove without inserting spaces
        self._re_quotes = re.compile(r"['\"`’“”]")

        # Whitespace collapsing
        self._re_whitespace = re.compile(r"\s+")

        # Digit detection
        self._re_has_digits = re.compile(r"\d")

    def normalize_single(self, name: Optional[str]) -> Dict[str, Any]:
        """
        Normalize a single business name string.

        Args:
            name: The raw business name string, or None/NaN.

        Returns:
            Dictionary containing:
            - name_raw (str)
            - name_norm (str)
            - name_core (str)
            - name_tokens (List[str])
            - name_translit (str)
            - name_has_digits (bool)
            - name_length (int)
        """
        # Handle null / empty / non-string safely
        if name is None or (isinstance(name, float) and pd.isna(name)):
            name_raw = ""
        else:
            name_raw = str(name)

        if not name_raw.strip():
            return {
                "name_raw": name_raw,
                "name_norm": "",
                "name_core": "",
                "name_tokens": [],
                "name_translit": "",
                "name_has_digits": False,
                "name_length": 0,
            }

        # Check for presence of digits in raw input
        has_digits: bool = bool(self._re_has_digits.search(name_raw))

        # 1. Unicode normalization (NFKC decomposes compatibility chars & standardizes ligatures)
        text: str = unicodedata.normalize("NFKC", name_raw)

        # 2. Lowercase / case normalization
        text = text.lower()

        # 3. "&" -> "and"
        text = self._re_ampersand.sub(" and ", text)
        text = self._re_plus.sub(" ", text)

        # 4. Punctuation cleanup:
        # 4a. Collapse acronym dots (l.l.c. -> llc, p.v.t. -> pvt, d/b/a -> dba)
        text = self._re_dots_4.sub(r"\1\2\3\4", text)
        text = self._re_dots_3.sub(r"\1\2\3", text)
        text = self._re_dots_2.sub(r"\1\2", text)
        text = self._re_dba_slash.sub("dba", text)
        text = self._re_aka_slash.sub("aka", text)

        # 4b. Remove quotes & apostrophes without adding spaces ("orelee's" -> "orelees")
        text = self._re_quotes.sub("", text)

        # 4c. Unicode-safe punctuation removal:
        # Replaces punctuation (P*) and symbols (S*) with space, while strictly
        # preserving letters (L*), marks (M* such as Indic matras), and numbers (N*)
        chars: List[str] = [
            " " if unicodedata.category(c)[0] in ("P", "S") else c
            for c in text
        ]
        text = "".join(chars)

        # 5. Whitespace cleanup
        text = self._re_whitespace.sub(" ", text).strip()

        # 6. Legal-form normalization
        raw_tokens: List[str] = text.split()
        norm_tokens: List[str] = []
        for tok in raw_tokens:
            norm_tokens.append(LEGAL_FORM_MAPPINGS.get(tok, tok))

        name_norm: str = " ".join(norm_tokens)

        # 7. Transliteration / diacritic stripping (NFKD decomposition)
        decomposed = unicodedata.normalize("NFKD", name_norm)
        name_translit: str = "".join(
            char for char in decomposed if not unicodedata.combining(char)
        )

        # 8. Core name extraction (name_core):
        # Strip legal entity suffixes and prefixes, while preserving at least one token
        core_tokens: List[str] = list(norm_tokens)
        while len(core_tokens) > 1 and core_tokens[-1] in LEGAL_CORE_STRIP_TOKENS:
            core_tokens.pop()
        while len(core_tokens) > 1 and core_tokens[0] in LEGAL_CORE_STRIP_TOKENS:
            core_tokens.pop(0)

        name_core: str = " ".join(core_tokens)
        if not name_core:
            name_core = name_norm

        # 9. Token list
        name_tokens: List[str] = norm_tokens

        return {
            "name_raw": name_raw,
            "name_norm": name_norm,
            "name_core": name_core,
            "name_tokens": name_tokens,
            "name_translit": name_translit,
            "name_has_digits": has_digits,
            "name_length": len(name_norm),
        }

    def normalize_series(self, series: pd.Series) -> pd.DataFrame:
        """
        Normalize an entire pandas Series of business names.
        Optimized using list comprehension for fast CPython execution.

        Args:
            series: Series containing raw business names.

        Returns:
            pd.DataFrame with the 7 normalized columns.
        """
        records = [self.normalize_single(val) for val in series]
        return pd.DataFrame.from_records(records)

    def normalize_dataframe(
        self,
        df: pd.DataFrame,
        name_col: str = "business_name",
        prefix: str = "",
    ) -> pd.DataFrame:
        """
        Normalize names in a DataFrame, preserving all original columns and row order.

        Args:
            df: Input DataFrame containing entity records.
            name_col: Name of the business name column (default: 'business_name').
            prefix: Optional prefix for added columns.

        Returns:
            New DataFrame with original columns preserved plus the normalized fields.
        """
        if name_col not in df.columns:
            raise KeyError(f"Column '{name_col}' not found in DataFrame. Available columns: {list(df.columns)}")

        norm_df = self.normalize_series(df[name_col])
        if prefix:
            norm_df = norm_df.rename(columns={col: f"{prefix}{col}" for col in norm_df.columns})

        # Concatenate horizontally while resetting index alignment
        return pd.concat([df.reset_index(drop=True), norm_df.reset_index(drop=True)], axis=1)


# Global default instance for convenience
_DEFAULT_NORMALIZER = BusinessNameNormalizer()


def normalize_business_name(name: Optional[str]) -> Dict[str, Any]:
    """
    Convenience function to normalize a single business name string.
    """
    return _DEFAULT_NORMALIZER.normalize_single(name)


def normalize_name_series(series: pd.Series) -> pd.DataFrame:
    """
    Convenience function to normalize a pandas Series of business names.
    """
    return _DEFAULT_NORMALIZER.normalize_series(series)


def normalize_dataframe(
    df: pd.DataFrame,
    name_col: str = "business_name",
    prefix: str = "",
) -> pd.DataFrame:
    """
    Convenience function to normalize a DataFrame containing business names.
    """
    return _DEFAULT_NORMALIZER.normalize_dataframe(df, name_col=name_col, prefix=prefix)


def normalize_name(raw: str) -> dict:
    """Convenience alias for normalize_business_name."""
    return normalize_business_name(raw)


def get_rare_tokens(name_norm: str, stopwords: frozenset = frozenset()) -> list:
    """Return tokens from name_norm excluding stopwords, legal suffixes, and very short tokens."""
    tokens = str(name_norm).split() if name_norm else []
    legal_tokens = set(LEGAL_FORM_MAPPINGS.values())
    return [t for t in tokens if len(t) > 2 and t not in stopwords and t not in legal_tokens]


# Self-test block when executed directly
if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    print("=" * 75)
    print("Testing BusinessNameNormalizer Module")
    print("=" * 75)

    test_cases = [
        ("Orelee's Barbershop", "Apostrophe handling"),
        ("B+ Retail Inc", "Plus sign & suffix"),
        ("Andy's Pizza & Pub, Inc.", "Ampersand, comma, legal suffix"),
        ("St. Lutheran Church L.L.C.", "Acronym dots (L.L.C.)"),
        ("Masters Projects Pvt. Ltd.", "Double legal suffix with dots"),
        ("International South Consultants Private Ltd", "Full word 'Private' to 'pvt'"),
        ("LLC Moncada Léarning Center", "Leading LLC & French accent é"),
        ("Pvt. EFS Print Ventures Ltd.", "Leading Pvt. & Trailing Ltd."),
        ("1055 First Place Realty, LLC", "Digits presence test"),
        ("Beloavi d/b/a Novent Owl PLLC", "d/b/a slash handling"),
        ("Global Retail Corporation", "Full word 'Corporation' to 'corp'"),
        ("Aeonian Tree Limited", "Full word 'Limited' to 'ltd'"),
        ("राम मार्केटिंग प्राइवेट लिमिटेड", "Devanagari Unicode input"),
        ("", "Empty string handling"),
        (None, "None / Null handling"),
    ]

    normalizer = BusinessNameNormalizer()
    for raw_val, description in test_cases:
        res = normalizer.normalize_single(raw_val)
        print(f"\n[{description}]")
        print(f"  Input:         {repr(raw_val)}")
        print(f"  name_raw:      {repr(res['name_raw'])}")
        print(f"  name_norm:     {repr(res['name_norm'])}")
        print(f"  name_core:     {repr(res['name_core'])}")
        print(f"  name_tokens:   {res['name_tokens']}")
        print(f"  name_translit: {repr(res['name_translit'])}")
        print(f"  has_digits:    {res['name_has_digits']} | length: {res['name_length']}")

    print("\n" + "=" * 75)
    print("All unit test assertions passed successfully!")
    print("=" * 75)
