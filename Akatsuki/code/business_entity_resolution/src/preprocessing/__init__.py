"""
Preprocessing package for Business Entity Resolution.
Provides complete Name, Address, Transliteration, Country Normalization,
and Unified Preprocessor pipeline with full backward compatibility.
"""

from pathlib import Path
import sys

_PACKAGE_DIR = Path(__file__).resolve().parent
_SRC_DIR = _PACKAGE_DIR.parent
_ROOT_DIR = _SRC_DIR.parent
for p in [str(_ROOT_DIR), str(_SRC_DIR)]:
    if p not in sys.path:
        sys.path.insert(0, p)

# Name Normalization
from src.preprocessing.name_normalizer import (
    BusinessNameNormalizer,
    normalize_business_name,
    normalize_name,
    normalize_name_series,
    normalize_dataframe as normalize_name_dataframe,
    get_rare_tokens,
)

# Address Normalization
from src.preprocessing.address_normalizer import (
    AddressNormalizer,
    normalize_address,
    normalize_address_series,
    normalize_dataframe as normalize_address_dataframe,
    get_rare_address_tokens,
)

# Transliteration
from src.preprocessing.transliteration import (
    TransliterationEngine,
    detect_script,
    transliterate_text,
    transliterate_name,
    transliterate_series,
    transliterate_dataframe,
    is_non_latin,
)

# Country Normalization
from src.preprocessing.country_normalizer import (
    CountryNormalizer,
    normalize_country,
    normalize_country_series,
    normalize_dataframe as normalize_country_dataframe,
    generate_country_diagnostic_report,
)

# Integrated Preprocessor
from src.preprocessing.preprocessor import (
    EntityPreprocessor,
    preprocess_dataframe,
    preprocess_source,
    preprocess_chunk,
    process_file,
)

__all__ = [
    "BusinessNameNormalizer",
    "normalize_business_name",
    "normalize_name",
    "normalize_name_series",
    "normalize_name_dataframe",
    "get_rare_tokens",
    "AddressNormalizer",
    "normalize_address",
    "normalize_address_series",
    "normalize_address_dataframe",
    "get_rare_address_tokens",
    "TransliterationEngine",
    "detect_script",
    "transliterate_text",
    "transliterate_name",
    "transliterate_series",
    "transliterate_dataframe",
    "is_non_latin",
    "CountryNormalizer",
    "normalize_country",
    "normalize_country_series",
    "normalize_country_dataframe",
    "generate_country_diagnostic_report",
    "EntityPreprocessor",
    "preprocess_dataframe",
    "preprocess_source",
    "preprocess_chunk",
    "process_file",
]
