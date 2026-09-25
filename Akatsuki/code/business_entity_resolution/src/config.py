"""
Configuration module for Akatsuki Business Entity Resolution Pipeline.
Defines hyper-parameters, path specifications, feature settings, and execution flags.
"""

from pathlib import Path

# Base Paths
BASE_DIR = Path(__file__).resolve().parent.parent.parent.parent
CODE_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "output"

# Input Data Paths (Student Resource / Workspace Default)
DATASET_DIR = BASE_DIR / "student_resource" / "dataset"
TRAIN_DIR = DATASET_DIR / "train"
TEST_DIR = DATASET_DIR / "test"

# Output Paths
MATCHING_RESULTS_PATH = OUTPUT_DIR / "matching_results.tsv"
CANDIDATE_PAIRS_PATH = OUTPUT_DIR / "candidate_pairs.tsv"

# Data Processing Settings
RANDOM_SEED = 42
OPEN_SET_COUNTRIES = {"US", "India", "France"}

# Blocking Agent Parameters
BLOCKING_NAME_PREFIX_LEN = 4
BLOCKING_MAX_CANDIDATES_PER_SOURCE1 = 150
BLOCKING_TFIDF_MIN_SIM = 0.25
BLOCKING_CHAR_NGRAM_RANGE = (3, 5)

# Feature Extraction Settings
SIMILARITY_METRICS = [
    "jaccard_name",
    "levenshtein_name",
    "jaro_winkler_name",
    "tfidf_cosine_name",
    "jaccard_address",
    "levenshtein_address",
    "digit_match_address",
    "country_match"
]

# Classifier Parameters
CLASSIFIER_MODEL_TYPE = "xgboost"  # Options: 'xgboost', 'random_forest'
CLASSIFIER_THRESHOLD = 0.65       # Optimized for F_0.5 macro score (precision weighted 2x)
F_BETA_VAL = 0.5
