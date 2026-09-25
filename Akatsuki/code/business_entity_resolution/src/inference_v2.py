"""
inference_v2.py  — Member 3: Upgraded Inference Engine (v2)
=============================================================
Uses:
  - Upgraded LightGBM v2 model (lgbm_entity_match_v2.pkl)
  - 40 upgraded features (Soundex, Monge-Elkan, all-digits overlap, abbrev expansion, etc.)
  - Post-hoc Isotonic Probability Calibration
  - Source-specific adaptive thresholding (S2 vs S3)

Produces:
  - output/matching_results.tsv   (final submission format)
  - reports/inference_report_v2.csv (pair scores + decisions)

Public API:
  from inference_v2 import EntityMatchInferenceV2
  inf = EntityMatchInferenceV2()
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

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger("inference_v2")

# ── Paths ──────────────────────────────────────────────────────────────────
_THIS      = Path(__file__).resolve().parent
_REPO_ROOT = _THIS.parent.parent.parent  # Akatsuki/
_PROJ_ROOT = _REPO_ROOT.parent.parent    # ML AMAZON/

DATA_DIR   = _REPO_ROOT / "data"
MODEL_DIR  = _REPO_ROOT / "model"
OUTPUT_DIR = _REPO_ROOT / "output"
REPORT_DIR = _REPO_ROOT / "reports"

MODEL_PKL      = MODEL_DIR / "lgbm_entity_match_v2.pkl"
META_JSON      = MODEL_DIR / "training_metadata_v2.json"
CANDIDATE_TSV  = _REPO_ROOT / "output" / "candidate_pairs.tsv"
RESULTS_TSV    = OUTPUT_DIR / "matching_results.tsv"
INFERENCE_CSV  = REPORT_DIR / "inference_report_v2.csv"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

# Adjust sys.path for feature imports
if str(_THIS) not in sys.path:
    sys.path.insert(0, str(_THIS))

# LightGBM first before pandas
import lightgbm as lgb
import numpy as np
import pandas as pd

from features.pair_features import compute_pair_features
from features.upgrade_features import compute_upgraded_features, UPGRADE_FEATURE_NAMES

FEATURE_COLS = UPGRADE_FEATURE_NAMES


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
    candidates = [
        _PROJ_ROOT / "student_resource" / "dataset",
        Path(r"C:\Users\appu2\OneDrive\Desktop\ML AMAZON\student_resource\dataset"),
    ]
    for c in candidates:
        if c.exists():
            return c
    raise FileNotFoundError("Cannot locate student_resource/dataset/")


class EntityMatchInferenceV2:
    """Upgraded inference engine for entity matching."""

    def __init__(self):
        self.model = None
        self.calibrator = None
        self.threshold = 0.50
        self.s2_threshold = 0.50
        self.s3_threshold = 0.50
        self.feature_cols = FEATURE_COLS
        self.dataset_dir = None

    def load_model(self):
        if not MODEL_PKL.exists():
            raise FileNotFoundError(f"Model file not found: {MODEL_PKL}")
        with open(MODEL_PKL, "rb") as f:
            data = pickle.load(f)
            if isinstance(data, dict):
                self.model = data["model"]
                self.calibrator = data.get("calibrator")
                self.threshold = data.get("threshold", 0.50)
                self.s2_threshold = data.get("s2_threshold", self.threshold)
                self.s3_threshold = data.get("s3_threshold", self.threshold)
                self.feature_cols = data.get("feature_cols", FEATURE_COLS)
            else:
                self.model = data
        log.info(f"Loaded v2 model from {MODEL_PKL}")
        log.info(f"  Base Threshold : {self.threshold:.4f}")
        log.info(f"  S2 Threshold   : {self.s2_threshold:.4f}")
        log.info(f"  S3 Threshold   : {self.s3_threshold:.4f}")
        log.info(f"  Calibrator     : {'Isotonic' if self.calibrator else 'None'}")

    def load_candidate_pairs(self) -> List[Tuple[str, str]]:
        if CANDIDATE_TSV.exists():
            real_lines = []
            with open(CANDIDATE_TSV, "r", encoding="utf-8") as f:
                for line in f:
                    s = line.strip()
                    if s and not s.startswith("#"):
                        real_lines.append(s)
            if len(real_lines) > 1:
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

        log.info("Generating candidate pairs from Ground Truth (benchmark mode)")
        self.dataset_dir = find_dataset_dir()
        gt_path = self.dataset_dir / "train" / "train_ground_truth.tsv"
        pairs = []
        count = 0
        with open(gt_path, "r", encoding="utf-8") as f:
            f.readline()
            for line in f:
                if count >= 20000:
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
                else:
                    pairs.append((s1_id, "__SINGLETON__"))
                count += 1
        log.info(f"  Generated {len(pairs):,} pairs for {count:,} S1 entities")
        return pairs

    def extract_features(
        self,
        pairs: List[Tuple[str, str]],
        s1_recs: Dict[str, Dict],
        s2_recs: Dict[str, Dict],
        s3_recs: Dict[str, Dict],
    ) -> Tuple[np.ndarray, List[str], List[str]]:
        log.info(f"Extracting 40 upgraded features for {len(pairs):,} pairs...")
        rows = []
        s1_ids = []
        cand_ids = []
        empty = {"name": "", "address": "", "country": ""}

        for s1_id, cand_id in pairs:
            if cand_id == "__SINGLETON__":
                continue
            s1 = s1_recs.get(s1_id, empty)
            cand = s2_recs.get(cand_id, s3_recs.get(cand_id, empty))

            base = compute_pair_features(
                name1=s1["name"], name2=cand["name"],
                addr1=s1["address"], addr2=cand["address"],
                country1=s1["country"], country2=cand["country"],
            )
            upg = compute_upgraded_features(
                name1=s1["name"], name2=cand["name"],
                addr1=s1["address"], addr2=cand["address"],
                country1=s1["country"], country2=cand["country"],
                base_features=base,
            )
            rows.append([upg.get(c, 0.0) for c in self.feature_cols])
            s1_ids.append(s1_id)
            cand_ids.append(cand_id)

        return np.array(rows, dtype=np.float32), s1_ids, cand_ids

    def score_pairs(self, X: np.ndarray) -> np.ndarray:
        raw_scores = self.model.predict(X)
        if self.calibrator is not None:
            return self.calibrator.predict(raw_scores)
        return raw_scores

    def make_predictions(
        self, s1_ids: List[str], cand_ids: List[str], scores: np.ndarray
    ) -> Dict[str, List[str]]:
        predictions = defaultdict(list)
        for s1, cand, score in zip(s1_ids, cand_ids, scores):
            thr = self.s2_threshold if cand.startswith("S2-") else self.s3_threshold
            if score >= thr:
                predictions[s1].append(cand)
        return dict(predictions)

    def save_outputs(
        self,
        all_s1_ids: List[str],
        predictions: Dict[str, List[str]],
        s1_ids: List[str],
        cand_ids: List[str],
        scores: np.ndarray,
    ):
        unique_s1 = list(dict.fromkeys(all_s1_ids))
        rows = []
        for s1_id in unique_s1:
            matched = predictions.get(s1_id, [])
            rows.append({
                "source1_entity_id": s1_id,
                "matched_entity_ids": ",".join(sorted(matched)),
            })
        df = pd.DataFrame(rows)
        df.to_csv(RESULTS_TSV, sep="\t", index=False)
        n_matched = int((df["matched_entity_ids"] != "").sum())
        n_single = len(df) - n_matched
        log.info(f"Submission written to {RESULTS_TSV}")
        log.info(f"  Total S1 entities   : {len(df):,}")
        log.info(f"  With matches        : {n_matched:,}")
        log.info(f"  Singletons          : {n_single:,}")

        rep_df = pd.DataFrame({
            "source1_id": s1_ids,
            "candidate_id": cand_ids,
            "score": np.round(scores, 5),
            "threshold_used": [self.s2_threshold if c.startswith("S2-") else self.s3_threshold for c in cand_ids],
            "predicted": [int(s >= (self.s2_threshold if c.startswith("S2-") else self.s3_threshold)) for s, c in zip(scores, cand_ids)],
        })
        rep_df.to_csv(INFERENCE_CSV, index=False)
        log.info(f"Inference report written to {INFERENCE_CSV} ({len(rep_df):,} rows)")

    def run(self):
        self.load_model()
        self.dataset_dir = find_dataset_dir()
        train_dir = self.dataset_dir / "train"

        pairs = self.load_candidate_pairs()
        s1_need = set(p[0] for p in pairs)
        s2_need = set(p[1] for p in pairs if p[1].startswith("S2-"))
        s3_need = set(p[1] for p in pairs if p[1].startswith("S3-"))

        log.info(f"Loading {len(s1_need):,} S1 records...")
        s1_recs = load_source(train_dir / "train_source1.tsv", needed_ids=s1_need)
        log.info(f"Loading {len(s2_need):,} S2 records...")
        s2_recs = load_source(train_dir / "train_source2.tsv", needed_ids=s2_need)
        log.info(f"Loading {len(s3_need):,} S3 records...")
        s3_recs = load_source(train_dir / "train_source3.tsv", needed_ids=s3_need)

        X, s1_ids, cand_ids = self.extract_features(pairs, s1_recs, s2_recs, s3_recs)
        scores = self.score_pairs(X)
        predictions = self.make_predictions(s1_ids, cand_ids, scores)

        all_s1 = [p[0] for p in pairs]
        self.save_outputs(all_s1, predictions, s1_ids, cand_ids, scores)
        log.info("V2 INFERENCE COMPLETE")
