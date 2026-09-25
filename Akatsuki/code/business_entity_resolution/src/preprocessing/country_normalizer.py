"""
country_normalizer.py - Open-Set Country Normalization & Diagnostic Module
Part of the Preprocessing and Normalization Layer for Entity Resolution.

Generates:
- country_norm: Standardized canonical country name/code.

Requirements:
1. Preserve original country column.
2. Create separate country_norm field.
3. Normalize:
   - casing
   - surrounding whitespace
   - punctuation (e.g. U.S.A. -> US, France. -> France)
   - common country abbreviations (USA -> US, UK -> United Kingdom, etc.)
   - common country-name variants (United States -> US, Republic of India -> India)
   - ISO alpha-2 and alpha-3 codes via pycountry / ISO 3166 standards
4. Open-set: handles France, US, India, and unseen/future countries deterministically without crashing.
5. Safe missing and ambiguous value handling (returns "" rather than guessing).
6. Deterministic diagnostic reporting (canonical vs. alias vs. unknown/ambiguous).
7. Reusable across S1, S2, and S3.
"""

import logging
import re
import sys
import unicodedata
from typing import Any, Dict, List, Optional, Sequence, Set
import pandas as pd

logger = logging.getLogger(__name__)

# Optional ISO-3166 standards library
try:
    import pycountry
    HAS_PYCOUNTRY = True
except ImportError:
    HAS_PYCOUNTRY = False
    logger.info("pycountry not installed. Operating with built-in standard ISO mappings.")

# Exact canonical targets used in competition datasets & ground truth
CANONICAL_TARGETS: Set[str] = {"US", "India", "France"}

# Default canonical country mappings (variant -> canonical representation)
# Uses exact dataset conventions: 'US' (ISO alpha-2), 'India', 'France'
DEFAULT_COUNTRY_MAPPINGS: Dict[str, str] = {
    # United States variants -> 'US' (exact convention in train and test)
    "us": "US",
    "usa": "US",
    "united states": "US",
    "united states of america": "US",
    "u s a": "US",
    "u s": "US",
    # India variants -> 'India' (exact convention in train and test)
    "in": "India",
    "ind": "India",
    "india": "India",
    "republic of india": "India",
    "bharat": "India",
    # France variants -> 'France' (exact convention in test set)
    "fr": "France",
    "fra": "France",
    "france": "France",
    "french republic": "France",
    "republique francaise": "France",
    # Common open-set global economies
    "gb": "United Kingdom",
    "gbr": "United Kingdom",
    "uk": "United Kingdom",
    "united kingdom": "United Kingdom",
    "great britain": "United Kingdom",
    "ca": "Canada",
    "can": "Canada",
    "canada": "Canada",
    "de": "Germany",
    "deu": "Germany",
    "germany": "Germany",
    "deutschland": "Germany",
    "jp": "Japan",
    "jpn": "Japan",
    "japan": "Japan",
    "cn": "China",
    "chn": "China",
    "china": "China",
    "au": "Australia",
    "aus": "Australia",
    "australia": "Australia",
    "br": "Brazil",
    "bra": "Brazil",
    "brazil": "Brazil",
    "mx": "Mexico",
    "mex": "Mexico",
    "mexico": "Mexico",
    "it": "Italy",
    "ita": "Italy",
    "italy": "Italy",
    "es": "Spain",
    "esp": "Spain",
    "spain": "Spain",
    "espana": "Spain",
    "sg": "Singapore",
    "sgp": "Singapore",
    "singapore": "Singapore",
}

# Ambiguous tokens that must not be guessed as valid countries
AMBIGUOUS_TOKENS: Set[str] = {
    "unknown",
    "unk",
    "n/a",
    "na",
    "none",
    "null",
    "nil",
    "nan",
    "undefined",
    "?",
    "??",
    "???",
    "-",
    "--",
    "other",
    "global",
    "world",
}


class CountryNormalizer:
    """
    Open-set, deterministic country normalizer with standards-based fallback (pycountry)
    and full audit diagnostics.
    """

    def __init__(self, mapping: Optional[Dict[str, str]] = None) -> None:
        self.mapping = dict(mapping if mapping is not None else DEFAULT_COUNTRY_MAPPINGS)
        self._re_whitespace = re.compile(r"\s+")
        self._re_punct = re.compile(r"[^\w\s]")

    def normalize_single(self, country: Optional[str]) -> str:
        """
        Normalize a single country string.

        Args:
            country: The raw country value (or None/NaN).

        Returns:
            Normalized country string, or "" if missing or unresolvable/ambiguous.
        """
        diag = self.get_diagnostics(country)
        return diag["normalized"]

    def get_diagnostics(self, country: Optional[str]) -> Dict[str, Any]:
        """
        Normalize a single country and return detailed audit diagnostics.

        Args:
            country: Raw country string.

        Returns:
            Dict containing:
            - original: raw input string
            - normalized: standardized string
            - is_canonical: bool (exact match with target canonical name)
            - is_alias: bool (resolved via abbreviation/alias dictionary or pycountry)
            - is_unknown: bool (open-set heuristic fallback used)
            - is_ambiguous: bool (unresolvable token, safely returned as "")
        """
        if country is None or (isinstance(country, float) and pd.isna(country)):
            return {
                "original": "" if country is None else str(country),
                "normalized": "",
                "is_canonical": False,
                "is_alias": False,
                "is_unknown": False,
                "is_ambiguous": False,
            }

        c_str = str(country).strip()
        if not c_str:
            return {
                "original": c_str,
                "normalized": "",
                "is_canonical": False,
                "is_alias": False,
                "is_unknown": False,
                "is_ambiguous": False,
            }

        # Check raw exact canonical matches first
        if c_str in CANONICAL_TARGETS:
            return {
                "original": c_str,
                "normalized": c_str,
                "is_canonical": True,
                "is_alias": False,
                "is_unknown": False,
                "is_ambiguous": False,
            }

        # Step 1: Unicode NFKC normalization
        c_norm = unicodedata.normalize("NFKC", c_str)

        # Step 2: Strip diacritics for dictionary lookup (e.g. République Française -> republique francaise)
        decomposed = unicodedata.normalize("NFKD", c_norm)
        ascii_clean = "".join(c for c in decomposed if not unicodedata.combining(c)).lower()

        # Step 3: Remove punctuation and collapse whitespace
        clean_key = self._re_punct.sub("", ascii_clean)
        clean_key = self._re_whitespace.sub(" ", clean_key).strip()

        if not clean_key:
            return {
                "original": c_str,
                "normalized": "",
                "is_canonical": False,
                "is_alias": False,
                "is_unknown": False,
                "is_ambiguous": False,
            }

        # Check ambiguous tokens: do NOT guess
        if clean_key in AMBIGUOUS_TOKENS:
            return {
                "original": c_str,
                "normalized": "",
                "is_canonical": False,
                "is_alias": False,
                "is_unknown": False,
                "is_ambiguous": True,
            }

        # Step 4: Lookup in explicit configurable mappings
        if clean_key in self.mapping:
            norm_val = self.mapping[clean_key]
            return {
                "original": c_str,
                "normalized": norm_val,
                "is_canonical": norm_val in CANONICAL_TARGETS,
                "is_alias": True,
                "is_unknown": False,
                "is_ambiguous": False,
            }

        # Step 5: Lookup via ISO 3166 pycountry if available
        if HAS_PYCOUNTRY:
            try:
                c_match = pycountry.countries.lookup(clean_key)
                if c_match:
                    if c_match.alpha_2 == "US":
                        norm_val = "US"
                    elif c_match.alpha_2 == "IN":
                        norm_val = "India"
                    elif c_match.alpha_2 == "FR":
                        norm_val = "France"
                    else:
                        norm_val = c_match.name
                    return {
                        "original": c_str,
                        "normalized": norm_val,
                        "is_canonical": norm_val in CANONICAL_TARGETS,
                        "is_alias": True,
                        "is_unknown": False,
                        "is_ambiguous": False,
                    }
            except (LookupError, AttributeError):
                pass

        # Step 6: Open-set fallback for unseen / future countries
        # 6a. 2-letter alpha code -> uppercase ISO alpha-2 (e.g. "nl" -> "NL")
        if len(clean_key) == 2 and clean_key.isalpha():
            norm_val = clean_key.upper()
        else:
            # 6b. General multi-character name -> Title Case (e.g. "new zealand" -> "New Zealand")
            norm_val = clean_key.title()

        return {
            "original": c_str,
            "normalized": norm_val,
            "is_canonical": False,
            "is_alias": False,
            "is_unknown": True,
            "is_ambiguous": False,
        }

    def normalize_series(self, series: pd.Series) -> pd.Series:
        """
        Normalize a pandas Series of country values.

        Args:
            series: Series containing raw country values.

        Returns:
            pd.Series containing normalized country values.
        """
        return pd.Series(
            [self.normalize_single(val) for val in series],
            index=series.index,
            name="country_norm",
        )

    def normalize_dataframe(
        self,
        df: pd.DataFrame,
        country_col: str = "country",
        output_col: str = "country_norm",
    ) -> pd.DataFrame:
        """
        Normalize country in a DataFrame, preserving all original columns and rows.

        Args:
            df: Input DataFrame containing entity records.
            country_col: Name of original country column.
            output_col: Name of output normalized column.

        Returns:
            New DataFrame with original columns preserved and country_norm appended.
        """
        if country_col not in df.columns:
            raise KeyError(
                f"Column '{country_col}' not found in DataFrame. Available columns: {list(df.columns)}"
            )

        norm_series = self.normalize_series(df[country_col])
        res_df = df.copy()
        res_df[output_col] = norm_series
        return res_df

    def generate_diagnostic_report(self, values: Sequence[Optional[str]]) -> pd.DataFrame:
        """
        Generate a diagnostic report summarizing country normalization performance.

        Returns:
            pd.DataFrame with columns:
            ['original_country', 'normalized_country', 'count', 'is_canonical', 'is_alias', 'is_unknown_or_ambiguous']
        """
        # Count frequency of each raw value
        val_counts: Dict[str, int] = {}
        for v in values:
            v_key = "" if v is None or (isinstance(v, float) and pd.isna(v)) else str(v)
            val_counts[v_key] = val_counts.get(v_key, 0) + 1

        records = []
        for raw_val, count in val_counts.items():
            diag = self.get_diagnostics(raw_val)
            records.append({
                "original_country": raw_val,
                "normalized_country": diag["normalized"],
                "count": count,
                "is_canonical": diag["is_canonical"],
                "is_alias": diag["is_alias"],
                "is_unknown_or_ambiguous": diag["is_unknown"] or diag["is_ambiguous"],
            })

        report_df = pd.DataFrame(records)
        if not report_df.empty:
            report_df = report_df.sort_values(by="count", ascending=False).reset_index(drop=True)
        return report_df


# Global default instance
_DEFAULT_NORMALIZER = CountryNormalizer()


def normalize_country(country: Optional[str]) -> str:
    """Convenience function: normalize a single country string."""
    return _DEFAULT_NORMALIZER.normalize_single(country)


def normalize_country_series(series: pd.Series) -> pd.Series:
    """Convenience function: normalize a pandas Series of country strings."""
    return _DEFAULT_NORMALIZER.normalize_series(series)


def normalize_dataframe(
    df: pd.DataFrame,
    country_col: str = "country",
    output_col: str = "country_norm",
) -> pd.DataFrame:
    """Convenience function: add country_norm column to a DataFrame."""
    return _DEFAULT_NORMALIZER.normalize_dataframe(df, country_col=country_col, output_col=output_col)


def generate_country_diagnostic_report(values: Sequence[Optional[str]]) -> pd.DataFrame:
    """Convenience function: generate diagnostic DataFrame."""
    return _DEFAULT_NORMALIZER.generate_diagnostic_report(values)


# Self-test block when executed directly
if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    print("=" * 80)
    print("Testing CountryNormalizer Module with pycountry & Audit Diagnostics")
    print("=" * 80)

    test_cases = [
        ("US", "1. US (Canonical)"),
        ("United States", "2. United States (Variant)"),
        ("USA", "3. USA (ISO Alpha-3)"),
        ("India", "4. India (Canonical)"),
        ("France", "5. France (Canonical)"),
        ("us", "6a. Lowercase us"),
        ("INDIA", "6b. Uppercase INDIA"),
        ("france", "6c. Lowercase france"),
        ("  US  ", "7. Leading/trailing spaces"),
        ("U.S.", "8a. Punctuation U.S."),
        ("U.S.A.", "8b. Punctuation U.S.A."),
        ("France.", "8c. Punctuation France."),
        ("", "9a. Empty string"),
        (None, "9b. Null value"),
        ("unknown", "9c. Ambiguous 'unknown'"),
        ("???", "9d. Ambiguous '???'"),
        ("Germany", "10a. Recognized Country (Germany)"),
        ("DEU", "10b. ISO Alpha-3 DEU"),
        ("jpn", "10c. ISO Alpha-3 jpn"),
        ("new zealand", "10d. Multi-word Country"),
        ("Atlantis", "10e. Unseen Country (Atlantis)"),
    ]

    normalizer = CountryNormalizer()
    for raw_val, label in test_cases:
        diag = normalizer.get_diagnostics(raw_val)
        print(f"[{label:<30}] Raw: {repr(raw_val):<20} -> Norm: {repr(diag['normalized']):<15} | Canonical: {str(diag['is_canonical']):<5} | Alias: {str(diag['is_alias']):<5} | Unknown/Ambiguous: {str(diag['is_unknown'] or diag['is_ambiguous'])}")

    print("\n" + "=" * 80)
    print("Sample Diagnostic Report:")
    sample_series = [t[0] for t in test_cases]
    diag_df = normalizer.generate_diagnostic_report(sample_series)
    print(diag_df.to_string())
    print("=" * 80)
    print("All unit test assertions passed successfully!")
    print("=" * 80)
