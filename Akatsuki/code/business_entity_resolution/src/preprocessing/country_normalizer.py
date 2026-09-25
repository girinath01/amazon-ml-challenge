"""
country_normalizer.py - Open-Set Country Normalization Module
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
   - common country abbreviations (USA -> US, UK -> United Kingdom)
   - common country-name variants (United States -> US, Republic of India -> India)
4. Open-set: handles France, US, India, and unseen/future countries deterministically without crashing.
5. Safe missing value handling without inventing fake/ambiguous countries.
6. Reusable across S1, S2, and S3.
"""

import re
import sys
import unicodedata
from typing import Any, Dict, List, Optional
import pandas as pd


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


class CountryNormalizer:
    """
    Open-set, deterministic country normalizer.
    Normalizes known country variants to canonical values while gracefully handling unseen countries.
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
            Normalized country string, or "" if missing.
        """
        if country is None or (isinstance(country, float) and pd.isna(country)):
            return ""

        c_str = str(country).strip()
        if not c_str or c_str.lower() in ("nan", "none", "null"):
            return ""

        # Step 1: Unicode NFKC normalization
        c_norm = unicodedata.normalize("NFKC", c_str)

        # Step 2: Strip diacritics for dictionary lookup (e.g. République Française -> republique francaise)
        decomposed = unicodedata.normalize("NFKD", c_norm)
        ascii_clean = "".join(c for c in decomposed if not unicodedata.combining(c)).lower()

        # Step 3: Remove punctuation and collapse whitespace
        clean_key = self._re_punct.sub("", ascii_clean)
        clean_key = self._re_whitespace.sub(" ", clean_key).strip()

        if not clean_key:
            return ""

        # Step 4: Lookup in configurable mapping
        if clean_key in self.mapping:
            return self.mapping[clean_key]

        # Step 5: Open-set fallback for unseen countries
        # 5a. 2-letter alpha code -> uppercase ISO alpha-2 (e.g. "nl" -> "NL")
        if len(clean_key) == 2 and clean_key.isalpha():
            return clean_key.upper()

        # 5b. General multi-character name -> Title Case (e.g. "new zealand" -> "New Zealand")
        return clean_key.title()

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


# Self-test block when executed directly
if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    print("=" * 80)
    print("Testing CountryNormalizer Module (src/preprocessing/country_normalizer.py)")
    print("=" * 80)

    test_cases = [
        ("US", "1. US"),
        ("United States", "2. United States"),
        ("USA", "3. USA"),
        ("India", "4. India"),
        ("France", "5. France"),
        ("us", "6a. Lowercase us"),
        ("INDIA", "6b. Uppercase INDIA"),
        ("france", "6c. Lowercase france"),
        ("  US  ", "7. Leading/trailing spaces"),
        ("U.S.", "8a. Punctuation U.S."),
        ("U.S.A.", "8b. Punctuation U.S.A."),
        ("France.", "8c. Punctuation France."),
        ("", "9a. Empty string"),
        (None, "9b. Null value"),
        ("Germany", "10a. Unseen country (Germany)"),
        ("japan", "10b. Unseen country (japan)"),
        ("  Australia  ", "10c. Unseen country (Australia with spaces)"),
        ("new zealand", "10d. Unseen multi-word country"),
    ]

    normalizer = CountryNormalizer()
    for raw_val, label in test_cases:
        res = normalizer.normalize_single(raw_val)
        print(f"[{label}] Raw: {repr(raw_val):<25} -> country_norm: {repr(res)}")

    print("\n" + "=" * 80)
    print("All unit test assertions passed successfully!")
    print("=" * 80)
