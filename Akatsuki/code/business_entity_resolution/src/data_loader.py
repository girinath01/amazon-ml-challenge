"""
data_loader.py - Reusable Data Ingestion Layer for Business Entity Resolution Challenge
Part of Member 1 Deliverables (Step 1: Data Ingestion & Quality).

Provides robust, memory-efficient loading of TSV datasets:
- train_source1.tsv, train_source2.tsv, train_source3.tsv, train_ground_truth.tsv
- test_source1.tsv, test_source2.tsv, test_source3.tsv

Uses explicit TSV parsing (`sep="\\t"`) as required by the challenge specification.
"""

import os
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Set, Union
import pandas as pd


# Default expected schema
SOURCE_COLUMNS = ["entity_id", "business_name", "business_address", "country"]
GROUND_TRUTH_COLUMNS = ["source1_entity_id", "matched_entity_ids"]


def find_default_dataset_dir() -> Path:
    """
    Search common relative and absolute paths to find the student_resource/dataset directory.
    """
    candidates = [
        Path(__file__).resolve().parents[4] / "student_resource" / "dataset",
        Path(__file__).resolve().parents[3] / "student_resource" / "dataset",
        Path.cwd() / "student_resource" / "dataset",
        Path.cwd().parent / "student_resource" / "dataset",
        Path(r"D:\Akatsuki\student_resource\dataset"),
    ]
    for p in candidates:
        if p.exists() and (p / "train").exists():
            return p
    return candidates[0]


class DataLoader:
    """
    Robust data loader for the ML Challenge 2026 Business Entity Resolution datasets.
    
    Supports:
    - Explicit tab-delimited parsing (`sep="\\t"`)
    - Streaming / chunked loading for large files (reducing peak RAM)
    - Full loading with explicit string dtypes (avoiding unwanted type conversions)
    - Ground-truth lookup structures for evaluation
    """

    def __init__(self, dataset_dir: Optional[Union[str, Path]] = None):
        if dataset_dir is None:
            self.dataset_dir = find_default_dataset_dir()
        else:
            self.dataset_dir = Path(dataset_dir)
            
        self.train_dir = self.dataset_dir / "train"
        self.test_dir = self.dataset_dir / "test"

    def get_source_path(self, source: str, split: str = "train") -> Path:
        """
        Get the path to a source file.
        
        Args:
            source: 'source1', 'source2', 'source3', 's1', 's2', 's3', '1', '2', '3'
            split: 'train' or 'test'
        """
        source_clean = source.lower().replace("source", "").replace("s", "")
        if source_clean not in {"1", "2", "3"}:
            raise ValueError(f"Invalid source '{source}'. Expected 1, 2, 3 or source1, source2, source3.")
        
        split_clean = split.lower().strip()
        if split_clean not in {"train", "test"}:
            raise ValueError(f"Invalid split '{split}'. Expected 'train' or 'test'.")
        
        folder = self.train_dir if split_clean == "train" else self.test_dir
        fname = f"{split_clean}_source{source_clean}.tsv"
        fpath = folder / fname
        
        if not fpath.exists():
            raise FileNotFoundError(f"File not found: {fpath}")
        return fpath

    def get_ground_truth_path(self, split: str = "train") -> Path:
        """Get the path to the ground truth TSV."""
        if split.lower() != "train":
            raise ValueError("Ground truth is only available for the 'train' split.")
        fpath = self.train_dir / "train_ground_truth.tsv"
        if not fpath.exists():
            raise FileNotFoundError(f"Ground truth file not found: {fpath}")
        return fpath

    def load_source(
        self,
        source: str,
        split: str = "train",
        nrows: Optional[int] = None,
        usecols: Optional[List[str]] = None,
        chunksize: Optional[int] = None,
        keep_default_na: bool = False,
    ) -> Union[pd.DataFrame, Iterator[pd.DataFrame]]:
        """
        Load a source file (source1, source2, or source3) with explicit TSV parsing.

        Args:
            source: Source identifier ('s1', 's2', 's3', etc.)
            split: 'train' or 'test'
            nrows: Optional row limit for fast sampling/prototyping
            usecols: Optional list of columns to load
            chunksize: If specified, returns an iterator of DataFrames
            keep_default_na: Whether to convert strings like 'NA', 'NULL' to NaN.
                             Default False preserves literal company names like 'NA Corp'.
                             
        Returns:
            pd.DataFrame or Iterator[pd.DataFrame]
        """
        fpath = self.get_source_path(source, split=split)
        
        # Explicit TSV parsing: sep="\t"
        # Always enforce string dtypes for entity resolution fields
        return pd.read_csv(
            fpath,
            sep="\t",
            dtype=str,
            nrows=nrows,
            usecols=usecols,
            chunksize=chunksize,
            keep_default_na=keep_default_na,
            na_values=[""] if not keep_default_na else None,
            encoding="utf-8",
            on_bad_lines="error",
        )

    def load_ground_truth(
        self,
        nrows: Optional[int] = None,
        keep_default_na: bool = False,
    ) -> pd.DataFrame:
        """
        Load train_ground_truth.tsv using explicit TSV parsing.

        Returns:
            DataFrame with columns ['source1_entity_id', 'matched_entity_ids']
        """
        fpath = self.get_ground_truth_path("train")
        return pd.read_csv(
            fpath,
            sep="\t",
            dtype=str,
            nrows=nrows,
            keep_default_na=keep_default_na,
            na_values=[""] if not keep_default_na else None,
            encoding="utf-8",
            on_bad_lines="error",
        )

    def load_ground_truth_dict(
        self, nrows: Optional[int] = None
    ) -> Dict[str, Set[str]]:
        """
        Load ground truth into a dictionary mapping source1_id -> Set of matched IDs.
        Singletons (0 matches) will map to an empty set.
        
        Optimized for O(1) evaluation lookup and low memory.
        """
        gt_path = self.get_ground_truth_path("train")
        gt_dict: Dict[str, Set[str]] = {}
        
        with open(gt_path, "r", encoding="utf-8", errors="replace") as f:
            header = f.readline().rstrip("\r\n").split("\t")
            count = 0
            for line in f:
                parts = line.rstrip("\r\n").split("\t")
                s1_id = parts[0]
                if len(parts) > 1 and parts[1].strip():
                    matched = set(parts[1].split(","))
                else:
                    matched = set()
                gt_dict[s1_id] = matched
                count += 1
                if nrows is not None and count >= nrows:
                    break
        return gt_dict

    def load_all_sources(
        self, split: str = "train", nrows: Optional[int] = None
    ) -> Dict[str, pd.DataFrame]:
        """
        Convenience method to load all 3 sources for a given split.
        NOTE: On full dataset (~12M rows), loading all simultaneously requires ~6-8GB RAM.
        Use chunked or individual loading if memory is constrained.
        """
        return {
            "s1": self.load_source("s1", split=split, nrows=nrows),
            "s2": self.load_source("s2", split=split, nrows=nrows),
            "s3": self.load_source("s3", split=split, nrows=nrows),
        }

    def stream_source_rows(
        self, source: str, split: str = "train"
    ) -> Iterator[Dict[str, str]]:
        """
        Memory-efficient generator yielding row dictionaries one-by-one.
        Ideal for large-scale streaming pipelines.
        """
        fpath = self.get_source_path(source, split=split)
        with open(fpath, "r", encoding="utf-8", errors="replace") as f:
            headers = f.readline().rstrip("\r\n").split("\t")
            for line in f:
                parts = line.rstrip("\r\n").split("\t")
                if len(parts) == len(headers):
                    yield dict(zip(headers, parts))


# Convenience functions for quick script usage
def load_source(source: str, split: str = "train", nrows: Optional[int] = None, dataset_dir: Optional[str] = None) -> pd.DataFrame:
    """Convenience wrapper to load a source DataFrame."""
    return DataLoader(dataset_dir).load_source(source, split=split, nrows=nrows)


def load_ground_truth(nrows: Optional[int] = None, dataset_dir: Optional[str] = None) -> pd.DataFrame:
    """Convenience wrapper to load ground truth DataFrame."""
    return DataLoader(dataset_dir).load_ground_truth(nrows=nrows)


if __name__ == "__main__":
    print("=" * 60)
    print("Member 1 - Data Loader Module Test")
    print("=" * 60)
    
    loader = DataLoader()
    print(f"Dataset root: {loader.dataset_dir}")
    
    # Test loading small sample of each source
    for split in ["train", "test"]:
        for src in ["s1", "s2", "s3"]:
            df_sample = loader.load_source(src, split=split, nrows=5)
            print(f"[{split.upper()} {src.upper()}] loaded successfully: shape={df_sample.shape}, cols={list(df_sample.columns)}")
            
    gt_sample = loader.load_ground_truth(nrows=5)
    print(f"[TRAIN GROUND TRUTH] loaded successfully: shape={gt_sample.shape}, cols={list(gt_sample.columns)}")
    
    gt_dict = loader.load_ground_truth_dict(nrows=5)
    print(f"GT Dict sample (first 2 entries):")
    for k in list(gt_dict.keys())[:2]:
        print(f"  {k} -> {gt_dict[k]}")
    print("\nData loader is ready and functioning perfectly!")
