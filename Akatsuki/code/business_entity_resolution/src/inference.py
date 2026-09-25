"""
inference.py  — Member 3: Inference Engine
===========================================
Given:
  - A trained LightGBM model (from trainer.py)
  - A set of candidate pairs (either from Member 2's candidate_pairs.tsv
    OR self-generated benchmark pairs)

Produces:
  - output/matching_results.tsv   (final submission format)
  - reports/inference_report.csv  (pair scores + decisions)

Submission Format (from README)
---------------------------------
source1_entity_id TAB matched_entity_ids
matched_entity_ids = comma-separated candidate IDs that pass threshold
If no candidates pass: output only the source1_entity_id with empty matched column

Public API
----------
  from inference import EntityMatchInference
  inf = EntityMatchInference()
  inf.run()
"""

import os
import sys
import json
import pickle
import logging
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple
from collections import defaultdict

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger("inference")

# ── Paths ──────────────────────────────────────────────────────────────────
_THIS      = Path(__file__).resolve().parent
_REPO_ROOT = _THIS.parent.parent.parent  # Akatsuki/
_PROJ_ROOT = _REPO_ROOT.parent.parent    # ML AMAZON/

DATA_DIR   = _REPO_ROOT / "data"
MODEL_DIR  = _REPO_ROOT / "model"
OUTPUT_DIR = _REPO_ROOT / "output"
REPORT_DIR = _REPO_ROOT / "reports"

MODEL_PKL      = MODEL_DIR / "lgbm_entity_match.pkl"
META_JSON      = MODEL_DIR / "training_metadata.json"
CANDIDATE_TSV  = _REPO_ROOT / "output" / "candidate_pairs.tsv"
RESULTS_TSV    = OUTPUT_DIR / "matching_results.tsv"
INFERENCE_CSV  = REPORT_DIR / "inference_report.csv"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

# Adjust sys.path for feature imports
if str(_THIS) not in sys.path:
    sys.path.insert(0, str(_THIS))

from features.pair_features import compute_pair_features, FEATURE_NAMES

FEATURE_COLS = [
    "name_exact", "name_jaccard", "name_levenshtein",
    "name_jaro_winkler", "name_token_sort", "name_token_set",
    "name_char3_cosine", "name_char4_cosine",
    "name_core_similarity", "name_translit_similarity",
    "address_jaccard", "address_levenshtein", "address_char3_cosine",
    "address_digit_overlap", "house_number_match",
    "postal_code_match", "address_exact", "address_missing",
    "country_match", "name_length_ratio", "address_length_ratio",
    "name_token_count_difference", "address_token_count_difference",
    "numeric_conflict",
    "name_address_similarity_product", "name_high_address_low",
    "name_high_address_high", "name_address_conflict",
]


# ── Record loader ──────────────────────────────────────────────────────────
def load_source(path: Path, needed_ids: Optional[Set[str]] = None,
                max_rows: Optional[int] = None) -> Dict[str, Dict]:
    records = {}
    with open(path, "r", encoding="utf-8") as f:
        f.readline()  # header
        for line in f:
            if max_rows and len(records) >= max_rows:
                break
            line = line.rstrip("\r\n")
            if not line:
                continue
            parts = line.split("\t")
            eid = parts[0].strip()
            if needed_ids is not None and eid not in needed_ids:
                continue
            records[eid] = {
                "name":    parts[1].strip() if len(parts) > 1 else "",
                "address": parts[2].strip() if len(parts) > 2 else "",
                "country": parts[3].strip() if len(parts) > 3 else "",
            }
    return records


def find_dataset_dir() -> Path:
    from data_loader import find_default_dataset_dir
    return find_default_dataset_dir()


# ── Inference class ────────────────────────────────────────────────────────
class EntityMatchInference:
    """
    Runs inference using the trained LightGBM model.
    Generates matching_results.tsv in submission format.
    """

    def __init__(self, threshold: Optional[float] = None):
        self.threshold = threshold
        self.model = None
        self.dataset_dir: Optional[Path] = None

    # ── Load model ────────────────────────────────────────────────────────
    def load_model(self):
        if not MODEL_PKL.exists():
            raise FileNotFoundError(f"Trained model not found: {MODEL_PKL}")
        with open(MODEL_PKL, "rb") as f:
            self.model = pickle.load(f)
        log.info(f"Loaded model from {MODEL_PKL}")

        # Load metadata for threshold
        if self.threshold is None:
            if META_JSON.exists():
                with open(META_JSON) as f:
                    meta = json.load(f)
                self.threshold = meta.get("optimal_threshold", 0.50)
                log.info(f"Using CV-derived threshold = {self.threshold:.2f}")
            else:
                self.threshold = 0.50
                log.info("No metadata found — defaulting threshold = 0.50")

    # ── Load candidate pairs ───────────────────────────────────────────────
    def load_candidate_pairs(self) -> List[Tuple[str, str]]:
        """
        Load from candidate_pairs.tsv if it has real data (Member 2 output).
        Falls back to generating benchmark pairs from Ground Truth.
        """
        if CANDIDATE_TSV.exists():
            # Check if it has non-comment lines
            real_lines = []
            with open(CANDIDATE_TSV, "r", encoding="utf-8") as f:
                for line in f:
                    stripped = line.strip()
                    if stripped and not stripped.startswith("#"):
                        real_lines.append(stripped)
            if len(real_lines) > 1:  # has header + data
                log.info(f"Loading candidate pairs from {CANDIDATE_TSV}")
                df = pd.read_csv(CANDIDATE_TSV, sep="\t", comment="#")
                pairs = []
                for _, row in df.iterrows():
                    s1_id = str(row.iloc[0]).strip()
                    cands_str = str(row.iloc[1]).strip()
                    for cid in cands_str.split(","):
                        cid = cid.strip()
                        if cid:
                            pairs.append((s1_id, cid))
                log.info(f"  Loaded {len(pairs):,} candidate pairs from TSV")
                return pairs

        # Fallback: generate from Ground Truth (test split or full set)
        log.info("candidate_pairs.tsv has no real data — generating inference pairs from Ground Truth")
        return self._generate_pairs_from_gt()

    def _generate_pairs_from_gt(self, sample_n: int = 20000) -> List[Tuple[str, str]]:
        """Generate candidate pairs by pairing every S1 against its GT matches."""
        self.dataset_dir = find_dataset_dir()
        gt_path = self.dataset_dir / "train" / "train_ground_truth.tsv"
        pairs = []
        with open(gt_path, "r", encoding="utf-8") as f:
            f.readline()
            count = 0
            for line in f:
                if count >= sample_n:
                    break
                line = line.rstrip("\r\n")
                if not line:
                    continue
                parts = line.split("\t")
                s1_id = parts[0].strip()
                matches_str = parts[1].strip() if len(parts) > 1 else ""
                if matches_str:
                    for cid in matches_str.split(","):
                        cid = cid.strip()
                        if cid:
                            pairs.append((s1_id, cid))
                count += 1
        log.info(f"  Generated {len(pairs):,} GT-derived pairs for {sample_n:,} S1 entities")
        return pairs

    # ── Feature extraction batch ───────────────────────────────────────────
    def extract_features(
        self,
        pairs: List[Tuple[str, str]],
        s1_recs: Dict[str, Dict],
        s2_recs: Dict[str, Dict],
        s3_recs: Dict[str, Dict],
    ) -> Tuple[np.ndarray, List[str], List[str]]:
        """Returns (feature_matrix, s1_ids, cand_ids)."""
        log.info(f"Extracting features for {len(pairs):,} pairs...")
        rows = []
        s1_ids = []
        cand_ids = []
        empty = {"name": "", "address": "", "country": ""}

        for s1_id, cand_id in pairs:
            s1 = s1_recs.get(s1_id, empty)
            if cand_id.startswith("S2-"):
                cand = s2_recs.get(cand_id, empty)
            else:
                cand = s3_recs.get(cand_id, empty)
            feats = compute_pair_features(
                name1=s1["name"], name2=cand["name"],
                addr1=s1["address"], addr2=cand["address"],
                country1=s1["country"], country2=cand["country"],
            )
            rows.append([feats[c] for c in FEATURE_COLS])
            s1_ids.append(s1_id)
            cand_ids.append(cand_id)

        return np.array(rows, dtype=np.float32), s1_ids, cand_ids

    # ── Score and filter ───────────────────────────────────────────────────
    def score_pairs(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)

    def make_predictions(
        self, s1_ids: List[str], cand_ids: List[str], scores: np.ndarray
    ) -> Dict[str, List[str]]:
        """Group by S1 and apply threshold → {s1_id: [matched_cand_ids]}."""
        predictions: Dict[str, List[str]] = defaultdict(list)
        for s1, cand, score in zip(s1_ids, cand_ids, scores):
            if score >= self.threshold:
                predictions[s1].append(cand)
        return dict(predictions)

    # ── Save outputs ───────────────────────────────────────────────────────
    def save_submission(
        self,
        all_s1_ids: List[str],
        predictions: Dict[str, List[str]],
    ):
        """Write matching_results.tsv in correct submission format."""
        unique_s1 = list(dict.fromkeys(all_s1_ids))  # preserve order
        rows = []
        for s1_id in unique_s1:
            matched = predictions.get(s1_id, [])
            rows.append({
                "source1_entity_id": s1_id,
                "matched_entity_ids": ",".join(sorted(matched)),
            })
        df = pd.DataFrame(rows)
        df.to_csv(RESULTS_TSV, sep="\t", index=False)
        n_with_matches = int((df["matched_entity_ids"] != "").sum())
        n_singletons   = len(df) - n_with_matches
        log.info(f"Submission written → {RESULTS_TSV}")
        log.info(f"  Total S1 entities   : {len(df):,}")
        log.info(f"  With matches        : {n_with_matches:,}")
        log.info(f"  Singletons (no match): {n_singletons:,}")
        return df

    def save_inference_report(
        self,
        s1_ids: List[str],
        cand_ids: List[str],
        scores: np.ndarray,
    ):
        df = pd.DataFrame({
            "source1_id":   s1_ids,
            "candidate_id": cand_ids,
            "score":        np.round(scores, 5),
            "predicted":    (scores >= self.threshold).astype(int),
        })
        df.to_csv(INFERENCE_CSV, index=False)
        log.info(f"Inference report → {INFERENCE_CSV} ({len(df):,} rows)")

    # ── Orchestrator ───────────────────────────────────────────────────────
    def run(self):
        self.load_model()
        self.dataset_dir = find_dataset_dir()
        train_dir = self.dataset_dir / "train"

        # Load candidates
        pairs = self.load_candidate_pairs()

        # Collect all unique entity IDs needed
        s1_need = set(p[0] for p in pairs)
        s2_need = set(p[1] for p in pairs if p[1].startswith("S2-"))
        s3_need = set(p[1] for p in pairs if p[1].startswith("S3-"))

        log.info(f"Loading {len(s1_need):,} S1 records...")
        s1_recs = load_source(train_dir / "train_source1.tsv", needed_ids=s1_need)
        log.info(f"Loading {len(s2_need):,} S2 records...")
        s2_recs = load_source(train_dir / "train_source2.tsv", needed_ids=s2_need)
        log.info(f"Loading {len(s3_need):,} S3 records...")
        s3_recs = load_source(train_dir / "train_source3.tsv", needed_ids=s3_need)

        # Extract features + score
        X, s1_ids, cand_ids = self.extract_features(pairs, s1_recs, s2_recs, s3_recs)
        scores = self.score_pairs(X)

        # Aggregate predictions
        predictions = self.make_predictions(s1_ids, cand_ids, scores)

        # Save
        self.save_submission(s1_ids, predictions)
        self.save_inference_report(s1_ids, cand_ids, scores)

        log.info(f"\n{'='*60}")
        log.info("INFERENCE COMPLETE")
        log.info(f"  Threshold used   : {self.threshold:.2f}")
        log.info(f"  Pairs scored     : {len(pairs):,}")
        log.info(f"  Submission file  : {RESULTS_TSV}")
        log.info(f"{'='*60}")


if __name__ == "__main__":
    inf = EntityMatchInference()
    inf.run()

