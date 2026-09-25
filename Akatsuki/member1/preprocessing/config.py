"""
config.py - Preprocessing Configuration and Constants
Part of the Preprocessing and Normalization Layer for Entity Resolution.
"""

from pathlib import Path


def find_dataset_dir() -> Path:
    """Locate the student_resource/dataset directory dynamically."""
    curr = Path(__file__).resolve().parent
    for _ in range(6):
        cand = curr / "student_resource" / "dataset"
        if cand.exists():
            return cand
        curr = curr.parent
    return Path(r"D:\Akatsuki\student_resource\dataset")


# Base Paths
STUDENT_RESOURCE_DIR = find_dataset_dir()
WORKSPACE_ROOT = STUDENT_RESOURCE_DIR.parent.parent
TRAIN_DATA_DIR = STUDENT_RESOURCE_DIR / "train"
TEST_DATA_DIR = STUDENT_RESOURCE_DIR / "test"
REPORTS_DIR = WORKSPACE_ROOT / "reports"
OUTPUT_DIR = WORKSPACE_ROOT / "output"

# Standard Raw Column Names
ID_COL = "entity_id"
NAME_COL = "business_name"
ADDRESS_COL = "business_address"
COUNTRY_COL = "country"

RAW_COLUMNS = [ID_COL, NAME_COL, ADDRESS_COL, COUNTRY_COL]

# Standard Normalized Column Names
NAME_OUTPUT_COLS = [
    "name_raw",
    "name_norm",
    "name_core",
    "name_tokens",
    "name_translit",
    "name_phonetic",
    "name_has_digits",
    "name_length",
]

ADDRESS_OUTPUT_COLS = [
    "address_raw",
    "address_norm",
    "address_tokens",
    "address_numbers",
    "house_number",
    "postal_code",
    "address_has_digits",
    "address_missing",
    "address_length",
]

COUNTRY_OUTPUT_COLS = [
    "country_norm",
]

ALL_NORMALIZED_COLS = NAME_OUTPUT_COLS + ADDRESS_OUTPUT_COLS + COUNTRY_OUTPUT_COLS
FINAL_PROCESSED_COLS = RAW_COLUMNS + ALL_NORMALIZED_COLS

# Performance & Streaming Defaults
DEFAULT_CHUNK_SIZE = 100_000
BENCHMARK_SAMPLE_SIZE = 50_000
RANDOM_SEED = 42

# Challenge Reference Standard Countries
KNOWN_COUNTRIES = {"US", "India", "France"}
