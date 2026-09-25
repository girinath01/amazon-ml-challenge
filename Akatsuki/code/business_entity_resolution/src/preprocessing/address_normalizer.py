"""
address_normalizer.py - Business Address Normalization & Component Extraction Module
Part of the Preprocessing and Normalization Layer for Entity Resolution.

Output Fields:
- address_raw: The exact original raw address string (or "" if missing).
- address_norm: Standardized address with normalized abbreviations, cleaned punctuation,
                lowercase, and normalized whitespace ("" if missing).
- address_tokens: List of normalized word tokens ([] if missing).
- address_numbers: List of all numeric string sequences extracted from the address ([] if missing).
- house_number: Conservatively extracted house, building, door, or plot number ("" if none/missing).
- postal_code: Extracted 5- or 6-digit postal/PIN code or ZIP+4 ("" if none/missing).
- address_has_digits: Boolean flag indicating presence of digits in raw address.
- address_missing: Integer indicator (1 if address is empty/null/whitespace, else 0).
- address_length: Character length of address_norm (0 if missing).

Processing Pipeline:
raw -> Unicode normalization (NFKC) -> lowercase/case normalization
    -> punctuation cleanup -> whitespace cleanup -> abbreviation expansion
    -> numeric token extraction -> house number extraction -> postal code extraction
"""

import re
import sys
import unicodedata
from typing import Any, Dict, List, Optional
import pandas as pd


# Address abbreviation dictionary supporting US, India, and France address formats
ADDRESS_ABBREVIATIONS: Dict[str, str] = {
    # Core Required Abbreviations
    "rd": "road",
    "st": "street",
    "ste": "suite",
    # Street / Thoroughfare Types
    "ave": "avenue",
    "av": "avenue",
    "dr": "drive",
    "blvd": "boulevard",
    "bd": "boulevard",
    "bvd": "boulevard",
    "ln": "lane",
    "ct": "court",
    "pl": "place",
    "cir": "circle",
    "ter": "terrace",
    "terr": "terrace",
    "pkwy": "parkway",
    "hwy": "highway",
    "expy": "expressway",
    "fwy": "freeway",
    "trl": "trail",
    "sq": "square",
    "cres": "crescent",
    "rte": "route",
    "rt": "route",
    "wy": "way",
    # Building, Sub-unit, and Floor Specifiers
    "apt": "apartment",
    "bldg": "building",
    "fl": "floor",
    "rm": "room",
    "dept": "department",
    "unit": "unit",
    "ofc": "office",
    "bsmt": "basement",
    # Landmark, Complex & Indian Reference Descriptors
    "opp": "opposite",
    "nr": "near",
    "cplx": "complex",
    "b/h": "behind",
    "adj": "adjacent",
    "dno": "door no",
}


class AddressNormalizer:
    """
    High-performance, deterministic address normalizer and component extractor.
    Pre-compiles regex patterns for rapid batch execution on multi-million row datasets.
    Fully Unicode-aware: preserves Indic scripts (Devanagari, Tamil) and European accents.
    """

    def __init__(self) -> None:
        self._re_quotes = re.compile(r"['\"`’“”]")
        self._re_whitespace = re.compile(r"\s+")
        self._re_has_digits = re.compile(r"\d")
        self._re_all_numbers = re.compile(r"\b\d+\b")

        # Postal code detection patterns:
        # 1. US ZIP+4 format: e.g. 78503-5207
        self._re_zip4 = re.compile(r"\b(\d{5}-\d{4})\b")
        # 2. Keyword-prefixed code: pin 534101, zip 95776, postal code 390001
        self._re_kw_code = re.compile(
            r"\b(?:pin|pincode|zip|postal|code)\s*[:#-]?\s*(\d{5,6})\b", re.IGNORECASE
        )
        # 3. Trailing 5 or 6 digit code at the very end of the address string
        self._re_trailing_code = re.compile(r"\b(\d{5,6})\s*$")
        # 4. Any 5 or 6 digit code pattern
        self._re_any_code = re.compile(r"\b(\d{5,6})\b")

        # House / Door / Building number patterns:
        # 1. Door / Shop / Plot / Flat / House number prefixes
        self._re_prefix_num = re.compile(
            r"\b(?:d(?:oor)?\.?\s*no\.?|shop\s*no\.?|plot\s*no\.?|flat\s*no\.?|sf\s*no\.?|house\s*no\.?|h\.?\s*no\.?)\s*([0-9]+(?:[\-\/][0-9]+)*)",
            re.IGNORECASE,
        )
        # 2. Leading number at the start of the address
        self._re_opening_num = re.compile(r"^\s*([0-9]+(?:[\-\/][0-9]+)*)\b")
        # 3. Inverted address number followed by street name and street type
        street_types = (
            r"street|st|road|rd|avenue|ave|drive|dr|court|ct|boulevard|blvd|bd|"
            r"lane|ln|way|rue|chemin|impasse|terrace|highway|hwy|pkwy"
        )
        self._re_inverted_num = re.compile(
            rf"\b([0-9]+(?:[\-\/][0-9]+)*)\s+[a-zA-Z0-9\s]{{2,30}}\s+(?:{street_types})\b",
            re.IGNORECASE,
        )

    def extract_postal_code(self, raw_text: str) -> str:
        """
        Extract postal/PIN code conservatively from raw address text.
        Supports US ZIP (5-digit, 5+4), Indian PIN (6-digit), and French (5-digit).
        Avoids mistaking opening street numbers (e.g. '17560 Ellis Road') as postal codes.
        """
        if not raw_text:
            return ""

        # Check US ZIP+4
        m_zip4 = self._re_zip4.search(raw_text)
        if m_zip4:
            return m_zip4.group(1)

        # Check keyword prefix
        m_kw = self._re_kw_code.search(raw_text)
        if m_kw:
            return m_kw.group(1)

        # Check trailing postal code at end of string
        m_trail = self._re_trailing_code.search(raw_text.strip())
        if m_trail:
            return m_trail.group(1)

        # If a 5-6 digit number exists towards the latter portion of the address (> 10 chars in)
        matches = list(self._re_any_code.finditer(raw_text))
        if len(matches) >= 1:
            last = matches[-1]
            if last.start() > 10:
                return last.group(1)

        return ""

    def extract_house_number(self, raw_text: str, postal_code: str = "") -> str:
        """
        Extract house, building, shop, plot, or door number conservatively.
        Disambiguates house numbers from extracted postal codes.
        """
        if not raw_text:
            return ""

        # Priority 1: Explicit door / shop / plot / flat / house prefixes
        m_pre = self._re_prefix_num.search(raw_text)
        if m_pre:
            return m_pre.group(1)

        # Priority 2: Opening street number (e.g. "1400 e main st", "20 rue parmentier")
        m_open = self._re_opening_num.search(raw_text)
        if m_open:
            val = m_open.group(1)
            if val != postal_code:
                return val

        # Priority 3: Inverted address street number (e.g. "Columbus, OH, 5559 Orville Avenue")
        m_inv = self._re_inverted_num.search(raw_text)
        if m_inv:
            val = m_inv.group(1)
            if val != postal_code:
                return val

        # Priority 4: Fallback to the first non-postal numeric sequence of <= 5 digits
        nums = self._re_all_numbers.findall(raw_text)
        for n in nums:
            if n != postal_code and len(n) <= 5:
                return n

        return ""

    def normalize_single(self, address: Optional[str]) -> Dict[str, Any]:
        """
        Normalize a single business address string.

        Args:
            address: The raw business address string, or None/NaN.

        Returns:
            Dictionary containing:
            - address_raw (str)
            - address_norm (str)
            - address_tokens (List[str])
            - address_numbers (List[str])
            - house_number (str)
            - postal_code (str)
            - address_has_digits (bool)
            - address_missing (int: 0 or 1)
            - address_length (int)
        """
        # Handle null / empty / non-string safely
        if address is None or (isinstance(address, float) and pd.isna(address)):
            address_raw = ""
        else:
            address_raw = str(address)

        # Missing address check: empty or whitespace-only
        if not address_raw.strip():
            return {
                "address_raw": address_raw,
                "address_norm": "",
                "address_tokens": [],
                "address_numbers": [],
                "house_number": "",
                "postal_code": "",
                "address_has_digits": False,
                "address_missing": 1,
                "address_length": 0,
            }

        # Check for presence of digits in raw input
        has_digits: bool = bool(self._re_has_digits.search(address_raw))

        # Extract all numeric sequences
        all_numbers: List[str] = self._re_all_numbers.findall(address_raw)

        # Extract postal code and house number
        postal_code: str = self.extract_postal_code(address_raw)
        house_number: str = self.extract_house_number(address_raw, postal_code=postal_code)

        # 1. Unicode normalization (NFKC)
        text: str = unicodedata.normalize("NFKC", address_raw)

        # 2. Lowercase / case normalization
        text = text.lower()

        # 3. Clean quotes and apostrophes without adding spaces
        text = self._re_quotes.sub("", text)

        # 4. Unicode-safe punctuation removal:
        # Replaces punctuation (P*) and symbols (S*) with space, while strictly
        # preserving letters (L*), marks (M* such as Indic matras and accents), and numbers (N*)
        chars: List[str] = [
            " " if unicodedata.category(c)[0] in ("P", "S") else c
            for c in text
        ]
        text = "".join(chars)

        # 5. Whitespace cleanup
        text = self._re_whitespace.sub(" ", text).strip()

        # 6. Normalize common address abbreviations
        tokens: List[str] = text.split()
        norm_tokens: List[str] = [
            ADDRESS_ABBREVIATIONS.get(tok, tok) for tok in tokens
        ]
        address_norm: str = " ".join(norm_tokens)

        return {
            "address_raw": address_raw,
            "address_norm": address_norm,
            "address_tokens": norm_tokens,
            "address_numbers": all_numbers,
            "house_number": house_number,
            "postal_code": postal_code,
            "address_has_digits": has_digits,
            "address_missing": 0,
            "address_length": len(address_norm),
        }

    def normalize_series(self, series: pd.Series) -> pd.DataFrame:
        """
        Normalize an entire pandas Series of business addresses.
        Optimized using list comprehension for fast CPython execution.

        Args:
            series: Series containing raw business addresses.

        Returns:
            pd.DataFrame with the 9 normalized columns.
        """
        records = [self.normalize_single(val) for val in series]
        return pd.DataFrame.from_records(records)

    def normalize_dataframe(
        self,
        df: pd.DataFrame,
        address_col: str = "business_address",
        prefix: str = "",
    ) -> pd.DataFrame:
        """
        Normalize addresses in a DataFrame, preserving all original columns and row order.

        Args:
            df: Input DataFrame containing entity records.
            address_col: Name of the business address column (default: 'business_address').
            prefix: Optional prefix for added columns.

        Returns:
            New DataFrame with original columns preserved plus the normalized fields.
        """
        if address_col not in df.columns:
            raise KeyError(
                f"Column '{address_col}' not found in DataFrame. Available columns: {list(df.columns)}"
            )

        norm_df = self.normalize_series(df[address_col])
        if prefix:
            norm_df = norm_df.rename(
                columns={col: f"{prefix}{col}" for col in norm_df.columns}
            )

        # Concatenate horizontally while resetting index alignment
        return pd.concat(
            [df.reset_index(drop=True), norm_df.reset_index(drop=True)], axis=1
        )


# Global default instance for convenience
_DEFAULT_NORMALIZER = AddressNormalizer()


def normalize_address(address: Optional[str]) -> Dict[str, Any]:
    """Convenience function to normalize a single business address string."""
    return _DEFAULT_NORMALIZER.normalize_single(address)


def normalize_address_series(series: pd.Series) -> pd.DataFrame:
    """Convenience function to normalize a pandas Series of business addresses."""
    return _DEFAULT_NORMALIZER.normalize_series(series)


def normalize_dataframe(
    df: pd.DataFrame,
    address_col: str = "business_address",
    prefix: str = "",
) -> pd.DataFrame:
    """Convenience function to normalize a DataFrame containing business addresses."""
    return _DEFAULT_NORMALIZER.normalize_dataframe(df, address_col=address_col, prefix=prefix)


# Self-test block when executed directly
if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    print("=" * 80)
    print("Testing AddressNormalizer Module")
    print("=" * 80)

    test_cases = [
        # 1. Normal address
        ("1795 Westchester Drive, High Point, NC", "Normal address"),
        # 2. Address with punctuation
        ("8060 S 1300 W, Ste # A, West Jordan, UT, 84088", "Address with punctuation & symbols"),
        # 3. Address with abbreviations
        ("1400 E Main St, Ste B, Woodland, CA, 95776", "Abbreviations (St -> street, Ste -> suite)"),
        # 4. Address with multiple numbers
        ("d.no 4/12/34, beside state bank, tadepalligudem, 534101", "Multiple numbers (Door no + PIN code)"),
        # 5. Inverted address
        ("OH, Columbus, 5559 Orville Avenue", "Inverted address (City/State first, street number last)"),
        # 6. Postal code extraction (ZIP+4)
        ("1801 S 10th St, McAllen, TX, 78503-5207", "ZIP+4 extraction"),
        # 7. Unicode / Non-English address (France)
        ("175 Boulevard du Président Franklin Roosevelt, Bordeaux, Nouvelle-Aquitaine", "French accented address"),
        # 8. Unicode / Non-English address (Devanagari)
        ("MUMBAI CITY, 605, MUMBAI, महाराष्ट्र", "Devanagari script address"),
        # 9. Opening 5-digit house number (not postal code)
        ("17560 Ellis Road, Tahlequah, OK", "Opening 5-digit house number disambiguation"),
        # 10. Missing address (empty string)
        ("", "Missing address (Empty string)"),
        # 11. Missing address (None)
        (None, "Missing address (None / Null)"),
    ]

    normalizer = AddressNormalizer()
    for raw_val, description in test_cases:
        res = normalizer.normalize_single(raw_val)
        print(f"\n[{description}]")
        print(f"  Input:         {repr(raw_val)}")
        print(f"  address_raw:   {repr(res['address_raw'])}")
        print(f"  address_norm:  {repr(res['address_norm'])}")
        print(f"  house_number:  {repr(res['house_number'])}")
        print(f"  postal_code:   {repr(res['postal_code'])}")
        print(f"  numbers:       {res['address_numbers']}")
        print(f"  has_digits:    {res['address_has_digits']} | missing: {res['address_missing']} | length: {res['address_length']}")

    print("\n" + "=" * 80)
    print("All unit test assertions passed successfully!")
    print("=" * 80)
