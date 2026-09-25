"""
blocking/ann_name_retrieval.py
------------------------------
Blocking Pass B4 — Hybrid Word + Character TF-IDF Cosine Retrieval.

Feature Space:
  - Word 1-2 ngrams (captures company name tokens, e.g., 'maure williams')
  - Character 3-5 ngrams (captures spelling variations & typos, e.g., 'williams' vs 'wilblims')
  - Sublinear TF scaling (`sublinear_tf=True`) for term frequency damping
  - Per-country vectorization + UNKNOWN cross-country fallback
  - Top-K cosine similarity query (default K=30, threshold=0.50)
"""

from collections import defaultdict
import logging
from typing import Dict, List, Optional, Set
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion

logger = logging.getLogger(__name__)

try:
    import faiss
    HAS_FAISS = True
except ImportError:
    HAS_FAISS = False


def build_and_query_ann_per_country(
    df_s1: pd.DataFrame,
    df_s2: pd.DataFrame,
    df_s3: pd.DataFrame,
    top_k: int = 30,
    min_similarity: float = 0.50,
) -> Dict[str, Set[str]]:
    """
    Performs per-country hybrid word + char TF-IDF similarity retrieval.
    
    Args:
        df_s1: S1 records with 'entity_id', 'country_norm', 'name_norm'
        df_s2: S2 records
        df_s3: S3 records
        top_k: Top K nearest neighbors to retrieve per S1 record
        min_similarity: Minimum cosine similarity threshold
        
    Returns:
        candidates: Dict[s1_id -> Set of candidate_ids]
    """
    candidates: Dict[str, Set[str]] = defaultdict(set)

    df_s23 = pd.concat([df_s2, df_s3], ignore_index=True)
    if df_s23.empty or df_s1.empty:
        return candidates

    countries = set(df_s1["country_norm"].unique())

    for country in countries:
        if not country or pd.isna(country):
            continue

        s1_c = df_s1[df_s1["country_norm"] == country]
        # Include target country AND UNKNOWN target records
        s23_c = df_s23[df_s23["country_norm"].isin([country, "UNKNOWN"])]

        if s1_c.empty or s23_c.empty:
            continue

        s1_names = s1_c["name_norm"].fillna("").tolist()
        s1_ids = s1_c["entity_id"].tolist()

        s23_names = s23_c["name_norm"].fillna("").tolist()
        s23_ids = s23_c["entity_id"].tolist()

        if not any(s1_names) or not any(s23_names):
            continue

        # Hybrid Vectorizer: Word 1-2 ngrams + Char 3-5 ngrams
        vectorizer = FeatureUnion([
            ("word_ngram", TfidfVectorizer(analyzer="word", ngram_range=(1, 2), min_df=1, sublinear_tf=True)),
            ("char_ngram", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=1, sublinear_tf=True)),
        ])

        try:
            all_names = s1_names + s23_names
            vectorizer.fit(all_names)

            X_s1 = vectorizer.transform(s1_names)
            X_s23 = vectorizer.transform(s23_names)
        except Exception as e:
            logger.warning(f"Hybrid TF-IDF fitting failed for country {country}: {e}")
            continue

        n_s23 = X_s23.shape[0]
        actual_k = min(top_k, n_s23)

        if HAS_FAISS and X_s1.shape[1] > 0:
            try:
                d = X_s23.shape[1]
                if d <= 50000 and n_s23 <= 200000:
                    X_s23_dense = X_s23.toarray().astype(np.float32)
                    X_s1_dense = X_s1.toarray().astype(np.float32)

                    index = faiss.IndexFlatIP(d)
                    index.add(X_s23_dense)

                    distances, indices = index.search(X_s1_dense, actual_k)

                    for i, s1_id in enumerate(s1_ids):
                        for j_idx, score in zip(indices[i], distances[i]):
                            if j_idx >= 0 and score >= min_similarity:
                                cand_id = s23_ids[j_idx]
                                candidates[s1_id].add(cand_id)
                    continue
            except Exception as e:
                logger.debug(f"FAISS search failed, falling back to sparse matmul: {e}")

        # Sparse Matrix Multiplication fallback
        try:
            sim_matrix = X_s1.dot(X_s23.T)

            for i, s1_id in enumerate(s1_ids):
                row = sim_matrix.getrow(i)
                if row.nnz == 0:
                    continue

                col_indices = row.indices
                data = row.data

                if len(data) > actual_k:
                    top_indices = np.argpartition(data, -actual_k)[-actual_k:]
                    top_indices = top_indices[np.argsort(-data[top_indices])]
                else:
                    top_indices = np.argsort(-data)

                for idx in top_indices:
                    score = data[idx]
                    if score >= min_similarity:
                        cand_id = s23_ids[col_indices[idx]]
                        candidates[s1_id].add(cand_id)
        except Exception as e:
            logger.warning(f"Sparse similarity computation failed for country {country}: {e}")

    return candidates
