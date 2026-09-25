"""
blocking/country_partition.py
--------------------------------
Partitions records by country_norm.
Returns per-country index dictionaries for S2 and S3.
"""

from collections import defaultdict
from typing import Dict, List, Tuple
import pandas as pd


def build_country_index(
    df: pd.DataFrame,
    id_col: str = "entity_id",
    country_col: str = "country_norm",
) -> Dict[str, List[str]]:
    """
    Build a mapping: country -> [entity_ids]
    """
    index: Dict[str, List[str]] = defaultdict(list)
    for row in df[[id_col, country_col]].itertuples(index=False):
        eid = getattr(row, id_col)
        ctry = getattr(row, country_col)
        index[ctry].append(eid)
    return dict(index)


def build_country_indexes(
    df_s2: pd.DataFrame,
    df_s3: pd.DataFrame,
) -> Tuple[Dict[str, List[str]], Dict[str, List[str]]]:
    """
    Builds per-country index dictionaries for both S2 and S3.
    """
    s2_idx = build_country_index(df_s2)
    s3_idx = build_country_index(df_s3)
    return s2_idx, s3_idx


def get_country_candidates(
    s1_country: str,
    s2_country_index: Dict[str, List[str]],
    s3_country_index: Dict[str, List[str]],
) -> Tuple[List[str], List[str]]:
    """
    Given an S1 country, return corresponding S2 and S3 entity ID lists.
    """
    s2_ids = s2_country_index.get(s1_country, [])
    s3_ids = s3_country_index.get(s1_country, [])
    return s2_ids, s3_ids


def filter_by_country(df: pd.DataFrame, country: str) -> pd.DataFrame:
    """
    Filters DataFrame to only rows matching the specified country.
    """
    return df[df["country_norm"] == country]
