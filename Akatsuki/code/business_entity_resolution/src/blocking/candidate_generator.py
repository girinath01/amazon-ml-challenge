"""
blocking/candidate_generator.py
-------------------------------
Master Blocking Pipeline Generator for Business Entity Resolution Challenge.

Executes all candidate generation passes B0 through B5 sequentially,
generating:
  1. candidate_pairs_long.parquet (Long matrix with blocking provenance flags)
  2. candidate_pairs.tsv (Official candidate TSV format)

Supports split='train' or 'test'.
"""

import logging
import time
from typing import Dict, List, Optional, Tuple
import pandas as pd

from blocking.address_block import block_address_rare
from blocking.ann_name_retrieval import build_and_query_ann_per_country
from blocking.candidate_pruning import format_to_tsv, prune_candidate_matrix
from blocking.candidate_union import union_candidate_passes
from blocking.country_partition import build_country_indexes, filter_by_country
from blocking.exact_name_block import block_exact_and_core_name, build_name_indexes
from blocking.numeric_block import block_numeric_locality, build_numeric_locality_index
from blocking.token_block import block_rare_tokens, build_rare_token_index
from blocking.transliteration_block import block_transliteration, build_translit_index

logger = logging.getLogger(__name__)


def run_candidate_generation_pipeline(
    df_s1: pd.DataFrame,
    df_s2: pd.DataFrame,
    df_s3: pd.DataFrame,
    passes_to_run: Optional[List[str]] = None,
    max_total_per_s1: int = 200,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, float]]:
    """
    Runs the multi-pass blocking pipeline across preprocessed S1, S2, and S3 DataFrames.
    
    Args:
        df_s1: Preprocessed Source 1 DataFrame
        df_s2: Preprocessed Source 2 DataFrame
        df_s3: Preprocessed Source 3 DataFrame
        passes_to_run: List of pass names to execute. If None, runs all B0-B4 passes.
        max_total_per_s1: Maximum candidate budget per S1 entity
        
    Returns:
        df_long: Long-form candidates DataFrame with provenance flags
        df_tsv: Submission format TSV DataFrame
        timing_dict: Execution duration in seconds per pass
    """
    if passes_to_run is None:
        passes_to_run = ["b0", "b1", "b2", "b3", "b4"]

    timing_dict: Dict[str, float] = {}
    pass_results: Dict[str, Dict[str, set]] = {}

    all_s1_ids = df_s1["entity_id"].tolist()
    logger.info(f"Starting blocking pipeline for {len(all_s1_ids)} S1 records...")

    # Build country index
    t0 = time.time()
    s2_by_c, s3_by_c = build_country_indexes(df_s2, df_s3)
    timing_dict["country_partition"] = time.time() - t0

    # -------------------------------------------------------------
    # Pass B0 / B1 — Exact Name & Core Name Blocking
    # -------------------------------------------------------------
    if "b0" in passes_to_run or "b1" in passes_to_run:
        t0 = time.time()
        logger.info("Executing Pass B0/B1: Exact Name & Core Name Blocking...")
        exact_idx, core_idx = build_name_indexes(df_s2, df_s3)

        cand_exact, cand_core = block_exact_and_core_name(df_s1, exact_idx, core_idx)
        pass_results["block_exact_name"] = cand_exact
        pass_results["block_core"] = cand_core
        timing_dict["b0_b1_exact_name"] = time.time() - t0

    # -------------------------------------------------------------
    # Pass B1 — Rare Token Blocking
    # -------------------------------------------------------------
    if "b1" in passes_to_run:
        t0 = time.time()
        logger.info("Executing Pass B1: Rare Token Blocking...")
        rare_idx, _ = build_rare_token_index(df_s2, df_s3, max_doc_freq=50)
        cand_rare = block_rare_tokens(df_s1, rare_idx, max_candidates_per_s1=50)
        pass_results["block_rare_token"] = cand_rare
        timing_dict["b1_rare_token"] = time.time() - t0

    # -------------------------------------------------------------
    # Pass B2 — Address & Numeric Blocking
    # -------------------------------------------------------------
    if "b2" in passes_to_run:
        t0 = time.time()
        logger.info("Executing Pass B2: Address & Numeric Blocking...")
        cand_addr = block_address_rare(df_s1, df_s2, df_s3, max_candidates_per_s1=50)
        pass_results["block_address"] = cand_addr

        num_loc_idx, _ = build_numeric_locality_index(df_s2, df_s3, max_locality_freq=100)
        cand_num = block_numeric_locality(df_s1, num_loc_idx, max_candidates_per_s1=30)
        pass_results["block_numeric"] = cand_num
        timing_dict["b2_address_numeric"] = time.time() - t0

    # -------------------------------------------------------------
    # Pass B3 — Transliteration Blocking
    # -------------------------------------------------------------
    if "b3" in passes_to_run:
        t0 = time.time()
        logger.info("Executing Pass B3: Transliteration Blocking...")
        translit_idx = build_translit_index(df_s2, df_s3)
        cand_translit = block_transliteration(df_s1, translit_idx, max_candidates_per_s1=50)
        pass_results["block_translit"] = cand_translit
        timing_dict["b3_transliteration"] = time.time() - t0

    # -------------------------------------------------------------
    # Pass B4 — ANN Vector Retrieval (TF-IDF Cosine)
    # -------------------------------------------------------------
    if "b4" in passes_to_run:
        t0 = time.time()
        logger.info("Executing Pass B4: ANN Vector Retrieval...")
        cand_ann = build_and_query_ann_per_country(
            df_s1, df_s2, df_s3, top_k=30, min_similarity=0.55
        )
        pass_results["block_ann"] = cand_ann
        timing_dict["b4_ann"] = time.time() - t0

    # -------------------------------------------------------------
    # Pass B5 — Union & Controlled Pruning
    # -------------------------------------------------------------
    t0 = time.time()
    logger.info("Executing Pass B5: Candidate Union & Budget Pruning...")
    df_long = union_candidate_passes(pass_results, all_s1_ids)

    df_long_pruned = prune_candidate_matrix(
        df_long, max_total_per_s1=max_total_per_s1
    )

    df_tsv = format_to_tsv(df_long_pruned, all_s1_ids)
    timing_dict["b5_union_pruning"] = time.time() - t0

    logger.info(f"Pipeline complete! Generated {len(df_long_pruned)} total candidate pairs.")

    return df_long_pruned, df_tsv, timing_dict
