"""
pair_builder.py - Training Pair Dataset Builder for Member 3
Constructs pair-level training datasets:
- pair_id
- source1_id
- candidate_id
- label (1 = true match, 0 = non-match)

Supports:
- Loading candidates from candidate_pairs.tsv when available from Member 2
- Generating stratified training pairs (Ground Truth True Positives + Hard Negatives + In-Country Distractors)
- Full feature matrix generation and export to training_pair_features.parquet
"""

import os
import sys
import random
import re
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple, Optional, Iterator
import pandas as pd
import numpy as np

# Adjust sys.path to allow imports from features
src_dir = Path(__file__).resolve().parent
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from features.pair_features import compute_pair_features, FEATURE_NAMES


def _first_name_token(name: str) -> str:
    toks = (name or "").lower().split()
    return toks[0] if toks else ""


def _extract_house_number(address: str) -> str:
    if not address:
        return ""
    m = re.search(r"\b\d{1,8}\b", address)
    return m.group(0) if m else ""


class PairBuilder:
    """
    Constructs and validates pair-level datasets for feature extraction and model training.
    """

    def __init__(self, dataset_dir: Optional[Path] = None, seed: int = 42):
        self.seed = seed
        random.seed(seed)
        np.random.seed(seed)
        
        if dataset_dir is None:
            # Look in standard locations
            candidates = [
                Path("student_resource/dataset"),
                Path("../student_resource/dataset"),
                Path("../../student_resource/dataset"),
                Path(r"C:\Users\appu2\OneDrive\Desktop\ML AMAZON\student_resource\dataset")
            ]
            for c in candidates:
                if c.exists() and (c / "train").exists():
                    self.dataset_dir = c
                    break
            else:
                self.dataset_dir = Path("student_resource/dataset")
        else:
            self.dataset_dir = Path(dataset_dir)
            
        self.train_dir = self.dataset_dir / "train"

    def load_ground_truth_dict(self, nrows: Optional[int] = None) -> Dict[str, Set[str]]:
        """Load ground truth mapping: {s1_id: set(matched_s2_s3_ids)}."""
        gt_path = self.train_dir / "train_ground_truth.tsv"
        gt_dict = {}
        with open(gt_path, "r", encoding="utf-8") as f:
            f.readline() # Header
            count = 0
            for line in f:
                if nrows and count >= nrows:
                    break
                line = line.rstrip("\r\n")
                if not line:
                    continue
                parts = line.split("\t")
                s1_id = parts[0].strip()
                matches_str = parts[1].strip() if len(parts) > 1 else ""
                if matches_str:
                    matched_ids = {x.strip() for x in matches_str.split(",") if x.strip()}
                    gt_dict[s1_id] = matched_ids
                else:
                    gt_dict[s1_id] = set()
                count += 1
        return gt_dict

    def load_source_records(
        self,
        source_name: str,
        needed_ids: Optional[Set[str]] = None,
        max_records: Optional[int] = None
    ) -> Dict[str, Dict[str, str]]:
        """
        Load records from train_source1.tsv, train_source2.tsv, or train_source3.tsv.
        Returns: {entity_id: {'name': ..., 'address': ..., 'country': ...}}
        """
        fname = f"train_{source_name}.tsv"
        fpath = self.train_dir / fname
        records = {}
        with open(fpath, "r", encoding="utf-8") as f:
            f.readline() # Header
            count = 0
            for line in f:
                if max_records and count >= max_records:
                    break
                line = line.rstrip("\r\n")
                if not line:
                    continue
                parts = line.split("\t")
                e_id = parts[0].strip()
                if needed_ids is None or e_id in needed_ids:
                    records[e_id] = {
                        "name": parts[1].strip() if len(parts) > 1 else "",
                        "address": parts[2].strip() if len(parts) > 2 else "",
                        "country": parts[3].strip() if len(parts) > 3 else ""
                    }
                    count += 1
                    if needed_ids and len(records) == len(needed_ids):
                        break
        return records

    def build_benchmark_pairs(
        self,
        num_s1_entities: int = 15000,
        neg_to_pos_ratio: float = 2.0
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Construct a balanced, representative benchmark candidate pair dataset:
        1. True positive pairs from Ground Truth (label = 1)
        2. Hard negative pairs (name collisions / similar names across different businesses)
        3. Random in-country negative distractors (label = 0)
        
        Returns:
            pairs_df: DataFrame with [pair_id, source1_id, candidate_id, label, ...]
            stats: Validation dictionary
        """
        print(f"Loading Ground Truth for sample of {num_s1_entities} S1 entities...")
        gt_dict = {}
        sample_s1_ids = []
        
        # Stream ground truth and pick sample
        gt_path = self.train_dir / "train_ground_truth.tsv"
        with open(gt_path, "r", encoding="utf-8") as f:
            f.readline()
            for line in f:
                line = line.rstrip("\r\n")
                if not line: continue
                parts = line.split("\t")
                s1_id = parts[0].strip()
                matches_str = parts[1].strip() if len(parts) > 1 else ""
                matched_set = {x.strip() for x in matches_str.split(",") if x.strip()} if matches_str else set()
                gt_dict[s1_id] = matched_set
                sample_s1_ids.append(s1_id)
                if len(sample_s1_ids) >= num_s1_entities:
                    break

        # Collect all positive target IDs
        needed_s2_ids = set()
        needed_s3_ids = set()
        positive_pairs = []

        for s1_id in sample_s1_ids:
            matches = gt_dict.get(s1_id, set())
            for m_id in matches:
                positive_pairs.append((s1_id, m_id, 1))
                if m_id.startswith("S2-"):
                    needed_s2_ids.add(m_id)
                elif m_id.startswith("S3-"):
                    needed_s3_ids.add(m_id)

        print(f"Found {len(positive_pairs):,} true positive pairs. Loading source records...")
        
        # Load S1 records
        s1_records = self.load_source_records("source1", needed_ids=set(sample_s1_ids))
        
        # Load matching S2 and S3 records
        s2_records = self.load_source_records("source2", needed_ids=needed_s2_ids)
        s3_records = self.load_source_records("source3", needed_ids=needed_s3_ids)

        # Load distractor pool for hard-negative generation (20,000 extra records from S2 and S3)
        extra_s2 = self.load_source_records("source2", max_records=25000)
        extra_s3 = self.load_source_records("source3", max_records=25000)
        s2_records.update(extra_s2)
        s3_records.update(extra_s3)

        # Build open-set country pools for generating realistic negative candidates
        country_s2 = {}
        country_s3 = {}
        for e_id, rec in s2_records.items():
            c = rec.get("country", "")
            country_s2.setdefault(c, []).append(e_id)
        for e_id, rec in s3_records.items():
            c = rec.get("country", "")
            country_s3.setdefault(c, []).append(e_id)

        # Inverted indexes for hard-negative generation:
        #   1) first name token collisions
        #   2) same country + same house number collisions
        name_token_s2_index = {}
        addr_num_index = {}
        for e_id, rec in s2_records.items():
            first_tok = _first_name_token(rec.get("name", ""))
            if len(first_tok) >= 4:
                name_token_s2_index.setdefault(first_tok, []).append(e_id)
            country = rec.get("country", "")
            house_num = _extract_house_number(rec.get("address", ""))
            if country and house_num:
                addr_num_index.setdefault((country, house_num), []).append(e_id)
        for e_id, rec in s3_records.items():
            first_tok = _first_name_token(rec.get("name", ""))
            if len(first_tok) >= 4:
                name_token_s2_index.setdefault(first_tok, []).append(e_id)
            country = rec.get("country", "")
            house_num = _extract_house_number(rec.get("address", ""))
            if country and house_num:
                addr_num_index.setdefault((country, house_num), []).append(e_id)

        print("Constructing negative candidate pairs...")
        negative_pairs = []
        target_negatives = int(len(positive_pairs) * neg_to_pos_ratio)
        seen_pairs = set((s1, cand) for s1, cand, _ in positive_pairs)
        neg_type_counts = {
            "address_number_collision": 0,
            "name_token_collision": 0,
            "country_random": 0,
        }

        for s1_id in sample_s1_ids:
            if len(negative_pairs) >= target_negatives:
                break
            s1_rec = s1_records.get(s1_id)
            if not s1_rec:
                continue
            country = s1_rec.get("country", "US")
            true_matches = gt_dict.get(s1_id, set())

            # 1) Hard negative: same country + same house number (address collision)
            house_num = _extract_house_number(s1_rec.get("address", ""))
            if country and house_num:
                key = (country, house_num)
                for cand_id in random.sample(addr_num_index.get(key, []), min(5, len(addr_num_index.get(key, [])))):
                    if cand_id not in true_matches and (s1_id, cand_id) not in seen_pairs:
                        seen_pairs.add((s1_id, cand_id))
                        negative_pairs.append((s1_id, cand_id, 0))
                        neg_type_counts["address_number_collision"] += 1
                        break

            # 2) Hard negative: shared first name token
            first_tok = _first_name_token(s1_rec.get("name", ""))
            if first_tok in name_token_s2_index:
                for cand_id in random.sample(name_token_s2_index[first_tok], min(5, len(name_token_s2_index[first_tok]))):
                    if cand_id not in true_matches and (s1_id, cand_id) not in seen_pairs:
                        seen_pairs.add((s1_id, cand_id))
                        negative_pairs.append((s1_id, cand_id, 0))
                        neg_type_counts["name_token_collision"] += 1
                        break

            # 3) In-country random candidate (simulating blocking pass)
            pool = country_s2.get(country, []) + country_s3.get(country, [])
            if pool:
                cand_id = random.choice(pool)
                if cand_id not in true_matches and (s1_id, cand_id) not in seen_pairs:
                    seen_pairs.add((s1_id, cand_id))
                    negative_pairs.append((s1_id, cand_id, 0))
                    neg_type_counts["country_random"] += 1

        # Top-up pass in case target ratio is not met in first sweep
        if len(negative_pairs) < target_negatives and sample_s1_ids:
            while len(negative_pairs) < target_negatives:
                s1_id = random.choice(sample_s1_ids)
                s1_rec = s1_records.get(s1_id)
                if not s1_rec:
                    continue
                country = s1_rec.get("country", "US")
                true_matches = gt_dict.get(s1_id, set())
                pool = country_s2.get(country, []) + country_s3.get(country, [])
                if not pool:
                    continue
                cand_id = random.choice(pool)
                if cand_id in true_matches or (s1_id, cand_id) in seen_pairs:
                    continue
                seen_pairs.add((s1_id, cand_id))
                negative_pairs.append((s1_id, cand_id, 0))
                neg_type_counts["country_random"] += 1

        print(
            "Negative mix: "
            f"address+number={neg_type_counts['address_number_collision']:,}, "
            f"name-token={neg_type_counts['name_token_collision']:,}, "
            f"random-country={neg_type_counts['country_random']:,}"
        )

        all_pairs = positive_pairs + negative_pairs
        random.shuffle(all_pairs)

        pair_rows = []
        for idx, (s1_id, cand_id, lbl) in enumerate(all_pairs):
            pair_rows.append({
                "pair_id": f"P-{idx:07d}",
                "source1_id": s1_id,
                "candidate_id": cand_id,
                "label": int(lbl)
            })

        pairs_df = pd.DataFrame(pair_rows)

        # Validation statistics
        total_p = len(pairs_df)
        pos_p = int((pairs_df["label"] == 1).sum())
        neg_p = int((pairs_df["label"] == 0).sum())
        dup_count = int(pairs_df.duplicated(subset=["source1_id", "candidate_id"]).sum())
        missing_ids = int(pairs_df["source1_id"].isna().sum() + pairs_df["candidate_id"].isna().sum())

        stats = {
            "total_pairs": total_p,
            "positive_pairs": pos_p,
            "negative_pairs": neg_p,
            "positive_ratio": round(pos_p / total_p, 4) if total_p > 0 else 0.0,
            "duplicate_pairs": dup_count,
            "missing_ids": missing_ids,
            "num_s1_entities": len(sample_s1_ids)
        }

        return pairs_df, stats, s1_records, s2_records, s3_records

    def generate_feature_matrix(
        self,
        pairs_df: pd.DataFrame,
        s1_records: Dict[str, Dict[str, str]],
        s2_records: Dict[str, Dict[str, str]],
        s3_records: Dict[str, Dict[str, str]]
    ) -> pd.DataFrame:
        """
        Compute all 28 pairwise features for each pair in pairs_df.
        Returns a DataFrame containing pair metadata, label, and all 28 features.
        """
        print(f"Generating features for {len(pairs_df):,} pairs...")
        feature_rows = []

        for idx, row in pairs_df.iterrows():
            s1_id = row["source1_id"]
            cand_id = row["candidate_id"]
            s1_rec = s1_records.get(s1_id, {"name": "", "address": "", "country": ""})
            if cand_id.startswith("S2-"):
                cand_rec = s2_records.get(cand_id, {"name": "", "address": "", "country": ""})
            else:
                cand_rec = s3_records.get(cand_id, {"name": "", "address": "", "country": ""})

            feats = compute_pair_features(
                name1=s1_rec["name"],
                name2=cand_rec["name"],
                addr1=s1_rec["address"],
                addr2=cand_rec["address"],
                country1=s1_rec["country"],
                country2=cand_rec["country"]
            )
            
            # Combine pair identifiers + label + engineered features
            entry = {
                "pair_id": row["pair_id"],
                "source1_id": s1_id,
                "candidate_id": cand_id,
                "label": int(row["label"])
            }
            entry.update(feats)
            feature_rows.append(entry)

        feat_df = pd.DataFrame(feature_rows)
        return feat_df
