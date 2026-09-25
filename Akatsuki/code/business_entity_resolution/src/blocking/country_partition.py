"""
blocking/country_partition.py
--------------------------------
Partitions records by country_norm with UNKNOWN cross-country fallback.
Returns per-country index dictionaries for S2 and S3.
"""

from collections import defaultdict
from typing import Dict, List, Set, Tuple
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
        ctry = getattr(row, country_col) or "UNKNOWN"
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


def get_query_countries(s1_country: str) -> List[str]:
    """
    Returns list of target country buckets to query for an S1 entity.
    Includes the specific country AND the 'UNKNOWN' fallback bucket.
    """
    ctry = (s1_country or "").strip()
    if not ctry or ctry == "UNKNOWN":
        return ["UNKNOWN"]
    return [ctry, "UNKNOWN"]


def get_country_candidates(
    s1_country: str,
    s2_country_index: Dict[str, List[str]],
    s3_country_index: Dict[str, List[str]],
) -> Tuple[List[str], List[str]]:
    """
    Given an S1 country, returns candidate S2 and S3 entity ID lists
    combining the specific country and the 'UNKNOWN' fallback bucket.
    """
    query_countries = get_query_countries(s1_country)
    
    s2_ids: Set[str] = set()
    s3_ids: Set[str] = set()
    
    for c in query_countries:
        if c in s2_country_index:
            s2_ids.update(s2_country_index[c])
        if c in s3_country_index:
            s3_ids.update(s3_country_index[c])
            
    return list(s2_ids), list(s3_ids)


def filter_by_country(df: pd.DataFrame, country: str, include_unknown: bool = True) -> pd.DataFrame:
    """
    Filters DataFrame to rows matching the specified country (or UNKNOWN if include_unknown=True).
    """
    if include_unknown and country != "UNKNOWN":
        return df[df["country_norm"].isin([country, "UNKNOWN"])]
    return df[df["country_norm"] == country]
