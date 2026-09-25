"""
preprocessing/preprocessor.py
--------------------------------
Master preprocessor. Applies all normalization steps to a DataFrame
loaded from any source file (S1/S2/S3, train/test).

Usage:
    from preprocessing.preprocessor import preprocess_source
    df = preprocess_source(raw_df)

Output columns added (raw columns are preserved):
    name_norm, name_core, name_tokens,
    name_length, name_has_digits,
    name_translit, name_is_non_latin,
    address_norm, address_tokens, address_numbers,
    house_number, postal_code,
    address_has_digits, address_missing, address_length,
    country_norm
"""

import pandas as pd
from tqdm import tqdm
tqdm.pandas()

from .name_normalizer    import normalize_name
from .address_normalizer import normalize_address
from .transliteration    import transliterate_name, is_non_latin
from .country_normalizer import normalize_country


def preprocess_source(df: pd.DataFrame,
                      chunksize: int = 50_000,
                      verbose: bool = True) -> pd.DataFrame:
    """
    Apply all normalization steps to a source DataFrame in place.

    Parameters
    ----------
    df : pd.DataFrame
        Must have columns: entity_id, business_name, business_address, country
    chunksize : int
        Not used directly here (apply is row-wise), kept for API consistency.
    verbose : bool
        Show tqdm progress bars.

    Returns
    -------
    pd.DataFrame  — same df with additional normalized columns appended.
    """
    assert "entity_id"         in df.columns, "Missing entity_id column"
    assert "business_name"     in df.columns, "Missing business_name column"
    assert "business_address"  in df.columns, "Missing business_address column"
    assert "country"           in df.columns, "Missing country column"

    # ── Country normalization (vectorized) ─────────────────────────────────
    if verbose:
        print("  [1/4] Normalizing country...")
    df["country_norm"] = df["country"].apply(normalize_country)

    # ── Name normalization ─────────────────────────────────────────────────
    if verbose:
        print("  [2/4] Normalizing business names...")
    name_cols = df["business_name"].progress_apply(normalize_name)
    name_df   = pd.DataFrame(name_cols.tolist(), index=df.index)
    for col in name_df.columns:
        df[col] = name_df[col]

    # ── Transliteration ────────────────────────────────────────────────────
    if verbose:
        print("  [3/4] Applying transliteration...")
    df["name_is_non_latin"] = df["business_name"].progress_apply(is_non_latin)
    df["name_translit"]     = df["business_name"].progress_apply(transliterate_name)

    # ── Address normalization ──────────────────────────────────────────────
    if verbose:
        print("  [4/4] Normalizing business addresses...")
    addr_cols = df["business_address"].progress_apply(normalize_address)
    addr_df   = pd.DataFrame(addr_cols.tolist(), index=df.index)
    for col in addr_df.columns:
        df[col] = addr_df[col]

    return df


def preprocess_chunk(chunk: pd.DataFrame) -> pd.DataFrame:
    """Process a single chunk without verbose output (for parallel use)."""
    return preprocess_source(chunk, verbose=False)
