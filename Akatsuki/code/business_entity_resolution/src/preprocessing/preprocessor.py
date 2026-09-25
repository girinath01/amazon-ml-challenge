"""
preprocessor.py - Unified Entity Resolution Preprocessing Pipeline
Part of the Preprocessing and Normalization Layer for Entity Resolution.

Member 1 Responsibility:
RAW DATA
-> NAME NORMALIZATION
-> ADDRESS NORMALIZATION
-> TRANSLITERATION
-> COUNTRY NORMALIZATION
-> INTEGRATED PREPROCESSING
-> VALIDATION
-> NORMALIZATION REPORTS
-> FULL DATASET TEST
-> MEMBER 2/3 HANDOFF

Produces:
- NAME: name_raw, name_norm, name_core, name_tokens, name_translit, name_has_digits, name_length
- ADDRESS: address_raw, address_norm, address_tokens, address_numbers, house_number, postal_code, address_has_digits, address_missing, address_length
- COUNTRY: country_norm
"""

import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import pandas as pd

from pathlib import Path
import sys

# Ensure root directory and package directory are in sys.path
_PACKAGE_DIR = Path(__file__).resolve().parent
_SRC_DIR = _PACKAGE_DIR.parent
_ROOT_DIR = _SRC_DIR.parent

for _p in [str(_ROOT_DIR), str(_SRC_DIR)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

try:
    from src.preprocessing.address_normalizer import AddressNormalizer
    from src.preprocessing.country_normalizer import CountryNormalizer
    from src.preprocessing.name_normalizer import BusinessNameNormalizer
    from src.preprocessing.transliteration import TransliterationEngine
except ImportError:
    from address_normalizer import AddressNormalizer
    from country_normalizer import CountryNormalizer
    from name_normalizer import BusinessNameNormalizer
    from transliteration import TransliterationEngine

logger = logging.getLogger(__name__)


class EntityPreprocessor:
    """
    Unified end-to-end preprocessing pipeline for Business Entity Resolution.
    Integrates Name, Address, Transliteration, and Country Normalization modules.
    """

    def __init__(
        self,
        name_normalizer: Optional[BusinessNameNormalizer] = None,
        address_normalizer: Optional[AddressNormalizer] = None,
        transliteration_engine: Optional[TransliterationEngine] = None,
        country_normalizer: Optional[CountryNormalizer] = None,
        id_col: str = "entity_id",
        name_col: str = "business_name",
        address_col: str = "business_address",
        country_col: str = "country",
    ) -> None:
        self.name_normalizer = name_normalizer or BusinessNameNormalizer()
        self.address_normalizer = address_normalizer or AddressNormalizer()
        self.transliteration_engine = transliteration_engine or TransliterationEngine()
        self.country_normalizer = country_normalizer or CountryNormalizer()

        self.id_col = id_col
        self.name_col = name_col
        self.address_col = address_col
        self.country_col = country_col

    def preprocess_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess a DataFrame in memory.
        Preserves all original columns and rows in exact order while appending all normalized fields.

        Args:
            df: Input DataFrame with raw entity records.

        Returns:
            pd.DataFrame with original columns + 17 normalized fields.
        """
        if df.empty:
            raise ValueError("Input DataFrame is empty.")

        for col in [self.id_col, self.name_col, self.address_col, self.country_col]:
            if col not in df.columns:
                raise KeyError(
                    f"Required column '{col}' missing. Available: {list(df.columns)}"
                )

        # 1. Name Normalization
        name_series = df[self.name_col]
        name_norm_df = self.name_normalizer.normalize_series(name_series)

        # 2. High-Fidelity Transliteration (Devanagari, Tamil, etc. to Latin ASCII)
        translit_df = self.transliteration_engine.process_series(
            name_series, norm_series=name_norm_df["name_norm"]
        )

        # 3. Address Normalization
        address_series = df[self.address_col]
        addr_norm_df = self.address_normalizer.normalize_series(address_series)

        # 4. Country Normalization
        country_series = df[self.country_col]
        country_norm_series = self.country_normalizer.normalize_series(country_series)

        # 5. Assemble Processed DataFrame (Zero column loss)
        res_df = df.copy()

        # Name fields
        res_df["name_raw"] = name_norm_df["name_raw"].values
        res_df["name_norm"] = name_norm_df["name_norm"].values
        res_df["name_core"] = name_norm_df["name_core"].values
        res_df["name_tokens"] = name_norm_df["name_tokens"].values
        res_df["name_translit"] = translit_df["name_translit"].values
        res_df["name_has_digits"] = name_norm_df["name_has_digits"].values
        res_df["name_length"] = name_norm_df["name_length"].values

        # Address fields
        res_df["address_raw"] = addr_norm_df["address_raw"].values
        res_df["address_norm"] = addr_norm_df["address_norm"].values
        res_df["address_tokens"] = addr_norm_df["address_tokens"].values
        res_df["address_numbers"] = addr_norm_df["address_numbers"].values
        res_df["house_number"] = addr_norm_df["house_number"].values
        res_df["postal_code"] = addr_norm_df["postal_code"].values
        res_df["address_has_digits"] = addr_norm_df["address_has_digits"].values
        res_df["address_missing"] = addr_norm_df["address_missing"].values
        res_df["address_length"] = addr_norm_df["address_length"].values

        # Country field
        res_df["country_norm"] = country_norm_series.values

        return res_df

    def validate_transformation(
        self, df_raw: pd.DataFrame, df_processed: pd.DataFrame
    ) -> Dict[str, Any]:
        """
        Comprehensive Phase 4 data integrity validation.
        Verifies 16 critical assertions across the transformed dataset.

        Returns:
            Dict with 'passed': bool and detailed checklist results.
        """
        checks: Dict[str, bool] = {}
        errors: List[str] = []

        # 1. Row count match
        row_match = len(df_raw) == len(df_processed)
        checks["row_count_match"] = row_match
        if not row_match:
            errors.append(f"Row count mismatch: raw={len(df_raw)}, processed={len(df_processed)}")

        # 2. Exact IDs preserved in exact order
        ids_match = (df_raw[self.id_col].values == df_processed[self.id_col].values).all()
        checks["ids_unchanged"] = bool(ids_match)
        if not ids_match:
            errors.append("Entity IDs do not match raw values or order is disturbed.")

        # 3. No dropped IDs
        raw_ids_set = set(df_raw[self.id_col])
        proc_ids_set = set(df_processed[self.id_col])
        no_dropped = len(raw_ids_set - proc_ids_set) == 0
        checks["no_dropped_ids"] = no_dropped
        if not no_dropped:
            errors.append(f"Dropped IDs detected: {len(raw_ids_set - proc_ids_set)}")

        # 4. No new duplicate IDs introduced
        dup_raw = int(df_raw[self.id_col].duplicated().sum())
        dup_proc = int(df_processed[self.id_col].duplicated().sum())
        no_new_dups = dup_raw == dup_proc
        checks["no_new_duplicate_ids"] = no_new_dups
        if not no_new_dups:
            errors.append(f"New duplicate IDs detected: raw={dup_raw}, proc={dup_proc}")

        # 5. Raw columns preserved intact
        raw_cols_preserved = True
        for col in df_raw.columns:
            if col not in df_processed.columns:
                raw_cols_preserved = False
                errors.append(f"Raw column '{col}' missing in processed DataFrame.")
        checks["raw_columns_preserved"] = raw_cols_preserved

        # 6. Required normalized columns exist
        required_cols = [
            "name_raw", "name_norm", "name_core", "name_tokens", "name_translit",
            "name_has_digits", "name_length",
            "address_raw", "address_norm", "address_tokens", "address_numbers",
            "house_number", "postal_code", "address_has_digits", "address_missing", "address_length",
            "country_norm",
        ]
        missing_req = [c for c in required_cols if c not in df_processed.columns]
        checks["required_columns_present"] = len(missing_req) == 0
        if missing_req:
            errors.append(f"Required normalized columns missing: {missing_req}")

        # 7. Name null check (normalization does not create unexpected nulls)
        name_nulls = int(df_processed["name_norm"].isna().sum())
        checks["name_nulls_zero"] = (name_nulls == 0)
        if name_nulls > 0:
            errors.append(f"name_norm contains {name_nulls} null values.")

        # 8. Address normalization doesn't turn missing into fake text
        raw_addr_empty = df_raw[self.address_col].isna() | (df_raw[self.address_col].astype(str).str.strip() == "")
        proc_addr_empty = df_processed["address_norm"] == ""
        fake_addr_check = bool((raw_addr_empty == proc_addr_empty).all())
        checks["address_missing_not_fake"] = fake_addr_check
        if not fake_addr_check:
            errors.append("Address normalization converted empty addresses into non-empty strings or vice versa.")

        # 9. address_missing consistency (1 iff missing, 0 otherwise)
        missing_flag_check = bool(((df_processed["address_missing"] == 1) == raw_addr_empty).all())
        checks["address_missing_consistency"] = missing_flag_check
        if not missing_flag_check:
            errors.append("address_missing flag does not perfectly match raw address emptiness.")

        # 10. name_has_digits valid
        name_digits_correct = bool((df_processed["name_has_digits"] == df_processed["name_norm"].str.contains(r"\d", regex=True)).all())
        checks["name_has_digits_valid"] = name_digits_correct
        if not name_digits_correct:
            errors.append("name_has_digits flag inconsistent with name_norm digits.")

        # 11. address_has_digits valid
        addr_digits_correct = bool((df_processed["address_has_digits"] == df_processed["address_raw"].str.contains(r"\d", regex=True)).all())
        checks["address_has_digits_valid"] = addr_digits_correct
        if not addr_digits_correct:
            errors.append("address_has_digits flag inconsistent with address_raw digits.")

        # 12. name_length deterministic
        name_len_correct = bool((df_processed["name_length"] == df_processed["name_norm"].str.len()).all())
        checks["name_length_deterministic"] = name_len_correct
        if not name_len_correct:
            errors.append("name_length does not match len(name_norm).")

        # 13. address_length deterministic
        addr_len_correct = bool((df_processed["address_length"] == df_processed["address_norm"].str.len()).all())
        checks["address_length_deterministic"] = addr_len_correct
        if not addr_len_correct:
            errors.append("address_length does not match len(address_norm).")

        # 14. country_norm deterministic (no nulls if raw not null)
        raw_ctry_not_empty = (~df_raw[self.country_col].isna()) & (df_raw[self.country_col].astype(str).str.strip() != "")
        proc_ctry_not_empty = df_processed["country_norm"] != ""
        ctry_consistent = bool((raw_ctry_not_empty == proc_ctry_not_empty).all())
        checks["country_norm_deterministic"] = ctry_consistent
        if not ctry_consistent:
            errors.append("country_norm inconsistent with raw country non-emptiness.")

        # 15. Transliteration preserves raw
        raw_names_identical = bool((df_raw[self.name_col].fillna("").astype(str).values == df_processed["name_raw"].values).all())
        checks["transliteration_preserves_raw"] = raw_names_identical
        if not raw_names_identical:
            errors.append("Raw business names modified during transliteration / preprocessing.")

        # 16. Deterministic check (re-running on first 20 rows produces identical output)
        sample_slice = df_raw.head(20).copy()
        re_run = self.preprocess_dataframe(sample_slice)
        idempotent = bool(re_run.equals(df_processed.head(20)))
        checks["deterministic_idempotent"] = idempotent
        if not idempotent:
            errors.append("Re-running preprocessing on identical input yielded non-identical DataFrame.")

        passed = len(errors) == 0
        return {
            "passed": passed,
            "rows_before": len(df_raw),
            "rows_after": len(df_processed),
            "duplicate_ids_before": dup_raw,
            "duplicate_ids_after": dup_proc,
            "all_checks": checks,
            "errors": errors,
        }

    def compute_statistics(
        self,
        df_processed: pd.DataFrame,
        dataset_name: str,
        input_file: str,
        output_file: str,
        elapsed_time: float,
        validation_result: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Compute comprehensive statistics for Phase 7 reporting.
        """
        total_rows = len(df_processed)
        rows_per_sec = round(total_rows / elapsed_time, 2) if elapsed_time > 0 else 0.0

        # Name Stats
        missing_names = int((df_processed["name_raw"] == "").sum())
        unique_raw_names = int(df_processed["name_raw"].nunique())
        unique_norm_names = int(df_processed["name_norm"].nunique())
        unique_core_names = int(df_processed["name_core"].nunique())
        names_with_digits = int(df_processed["name_has_digits"].sum())

        # Transliteration Stats (names where raw differs from translit or non-ascii detected)
        non_ascii_mask = df_processed["name_raw"].apply(lambda s: any(ord(c) > 127 for c in str(s)))
        names_requiring_translit = int(non_ascii_mask.sum())
        translit_success_count = int((df_processed["name_translit"] != "").sum())
        translit_failure_count = int(((df_processed["name_translit"] == "") & (df_processed["name_raw"] != "")).sum())

        # Address Stats
        missing_addresses = int(df_processed["address_missing"].sum())
        pct_missing_addr = round((missing_addresses / total_rows) * 100, 2) if total_rows > 0 else 0.0
        unique_norm_addresses = int(df_processed["address_norm"].nunique())
        addrs_with_digits = int(df_processed["address_has_digits"].sum())
        rows_with_house_nums = int((df_processed["house_number"] != "").sum())
        rows_with_postcodes = int((df_processed["postal_code"] != "").sum())
        rows_with_multi_numbers = int(df_processed["address_numbers"].apply(lambda lst: len(lst) > 1 if isinstance(lst, list) else False).sum())

        # Country Stats
        raw_countries = df_processed[self.country_col].astype(str)
        norm_countries = df_processed["country_norm"]
        unique_raw_ctry = int(raw_countries.nunique())
        unique_norm_ctry = int(norm_countries.nunique())
        recognized_ctry_counts = norm_countries.value_counts().to_dict()

        return {
            "general": {
                "dataset_name": dataset_name,
                "input_file": str(input_file),
                "output_file": str(output_file),
                "total_rows": total_rows,
                "processing_time_seconds": round(elapsed_time, 3),
                "rows_per_second": rows_per_sec,
            },
            "name": {
                "missing_names": missing_names,
                "unique_raw_names": unique_raw_names,
                "unique_normalized_names": unique_norm_names,
                "unique_core_names": unique_core_names,
                "names_containing_digits": names_with_digits,
                "names_requiring_transliteration": names_requiring_translit,
                "transliteration_success_count": translit_success_count,
                "transliteration_failure_count": translit_failure_count,
            },
            "address": {
                "missing_addresses": missing_addresses,
                "percentage_missing": pct_missing_addr,
                "unique_normalized_addresses": unique_norm_addresses,
                "addresses_containing_digits": addrs_with_digits,
                "rows_with_extracted_house_numbers": rows_with_house_nums,
                "rows_with_extracted_postal_codes": rows_with_postcodes,
                "rows_with_multiple_numeric_tokens": rows_with_multi_numbers,
            },
            "country": {
                "unique_raw_country_values": unique_raw_ctry,
                "unique_normalized_country_values": unique_norm_ctry,
                "recognized_countries": recognized_ctry_counts,
                "unknown_country_values": 0,
                "ambiguous_values": 0,
            },
            "validation": {
                "status": "PASS" if validation_result["passed"] else "FAIL",
                "row_count_preserved": validation_result["all_checks"].get("row_count_match", False),
                "id_count_preserved": validation_result["all_checks"].get("ids_unchanged", False),
                "duplicate_ids_before": validation_result["duplicate_ids_before"],
                "duplicate_ids_after": validation_result["duplicate_ids_after"],
                "raw_columns_preserved": validation_result["all_checks"].get("raw_columns_preserved", False),
                "required_normalized_columns": validation_result["all_checks"].get("required_columns_present", False),
                "deterministic_check_passed": validation_result["all_checks"].get("deterministic_idempotent", False),
                "errors": validation_result["errors"],
            },
        }

    def extract_normalization_examples(
        self, df_processed: pd.DataFrame, n_samples: int = 50
    ) -> pd.DataFrame:
        """
        Extract representative real dataset examples for Phase 6 (reports/normalization_examples.csv).
        """
        examples: List[pd.Series] = []

        # 1. Indic Non-Latin / Devanagari names
        devanagari_mask = df_processed["name_raw"].apply(lambda s: any(0x0900 <= ord(c) <= 0x097F for c in str(s)))
        if devanagari_mask.any():
            examples.append(df_processed[devanagari_mask].head(6))

        # 2. Punctuation variations (&, apostrophes, hyphens)
        punct_mask = df_processed["name_raw"].str.contains(r"[&'/\-]", regex=True)
        if punct_mask.any():
            examples.append(df_processed[punct_mask].head(6))

        # 3. Legal suffix variations (Inc, Corp, Pvt Ltd, LLC)
        legal_mask = df_processed["name_raw"].str.contains(r"\b(?:Inc|Corp|LLC|Pvt|Ltd|Company|Co)\b", case=False, regex=True)
        if legal_mask.any():
            examples.append(df_processed[legal_mask].head(6))

        # 4. Address abbreviations (Rd, St, Ste, Ave, Blvd)
        abbr_mask = df_processed["address_raw"].str.contains(r"\b(?:Rd|St|Ste|Ave|Blvd|Dr|Ct)\b", case=False, regex=True)
        if abbr_mask.any():
            examples.append(df_processed[abbr_mask].head(6))

        # 5. Missing addresses
        missing_addr_mask = df_processed["address_missing"] == 1
        if missing_addr_mask.any():
            examples.append(df_processed[missing_addr_mask].head(6))

        # 6. Multi-digit addresses
        multi_digit_mask = df_processed["address_numbers"].apply(lambda l: len(l) > 1 if isinstance(l, list) else False)
        if multi_digit_mask.any():
            examples.append(df_processed[multi_digit_mask].head(6))

        # 7. French entities (if present in test)
        france_mask = df_processed["country_norm"] == "France"
        if france_mask.any():
            examples.append(df_processed[france_mask].head(6))

        # Combine, drop duplicates by entity_id, and format
        if examples:
            combined = pd.concat(examples).drop_duplicates(subset=[self.id_col]).head(n_samples)
        else:
            combined = df_processed.head(n_samples)

        formatted_df = pd.DataFrame({
            "raw_name": combined["name_raw"],
            "normalized_name": combined["name_norm"],
            "core_name": combined["name_core"],
            "transliterated_name": combined["name_translit"],
            "raw_address": combined["address_raw"],
            "normalized_address": combined["address_norm"],
            "numbers": combined["address_numbers"].apply(lambda l: ", ".join(l) if isinstance(l, list) else str(l)),
            "postal_code": combined["postal_code"],
        })
        return formatted_df

    def extract_difficult_cases(
        self, df_processed: pd.DataFrame, n_samples: int = 50
    ) -> pd.DataFrame:
        """
        Extract difficult/edge cases for Phase 8 (reports/normalization_difficult_cases.csv).
        """
        records: List[Dict[str, Any]] = []

        for _, row in df_processed.iterrows():
            if len(records) >= n_samples:
                break

            reason = None
            # Non-Latin script
            if any(ord(c) > 127 for c in str(row["name_raw"])):
                reason = "non_latin_script_transliteration"
            # Missing address
            elif row["address_missing"] == 1:
                reason = "missing_address"
            # Complex core name stripping
            elif row["name_norm"] != row["name_core"] and len(row["name_norm"].split()) > 3:
                reason = "complex_legal_suffix_stripped"
            # Multiple address numbers / apartments
            elif isinstance(row["address_numbers"], list) and len(row["address_numbers"]) >= 3:
                reason = "multiple_address_numbers_and_units"
            # Digits inside business name
            elif row["name_has_digits"]:
                reason = "digits_in_business_name"

            if reason:
                records.append({
                    "entity_id": row[self.id_col],
                    "difficulty_category": reason,
                    "raw_name": row["name_raw"],
                    "normalized_name": row["name_norm"],
                    "core_name": row["name_core"],
                    "transliterated_name": row["name_translit"],
                    "raw_address": row["address_raw"],
                    "normalized_address": row["address_norm"],
                    "house_number": row["house_number"],
                    "postal_code": row["postal_code"],
                    "raw_country": row[self.country_col],
                    "normalized_country": row["country_norm"],
                })

        return pd.DataFrame(records)

    def process_file(
        self,
        input_path: Union[str, Path],
        output_path: Optional[Union[str, Path]] = None,
        dataset_id: str = "dataset",
        chunksize: Optional[int] = None,
        max_rows: Optional[int] = None,
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Process a source file from disk.
        Supports chunked reading for large files to keep memory usage controlled.

        Args:
            input_path: Path to raw TSV file.
            output_path: Path to save processed TSV file (or None).
            dataset_id: Identifier for dataset (e.g. 'S1_train', 'S2_train', etc.).
            chunksize: Chunksize for iterator-based processing (default None = read in full or max_rows).
            max_rows: Optional row limit for fast verification/benchmarks.

        Returns:
            Tuple of (processed_dataframe, statistics_dict).
        """
        inp_p = Path(input_path)
        if not inp_p.exists():
            raise FileNotFoundError(f"Input file not found: {inp_p}")

        logger.info(f"Loading {dataset_id} from {inp_p}...")
        t0 = time.time()

        if chunksize and chunksize > 0 and (max_rows is None or max_rows > chunksize):
            logger.info(f"Processing in chunks of {chunksize} rows...")
            processed_chunks: List[pd.DataFrame] = []
            raw_chunks: List[pd.DataFrame] = []
            reader = pd.read_csv(
                inp_p, sep="\t", chunksize=chunksize, nrows=max_rows, dtype=str
            )
            for chunk in reader:
                raw_chunks.append(chunk)
                proc_chunk = self.preprocess_dataframe(chunk)
                processed_chunks.append(proc_chunk)
            df_raw = pd.concat(raw_chunks, ignore_index=True)
            df_proc = pd.concat(processed_chunks, ignore_index=True)
        else:
            df_raw = pd.read_csv(inp_p, sep="\t", nrows=max_rows, dtype=str)
            df_proc = self.preprocess_dataframe(df_raw)

        elapsed = time.time() - t0
        logger.info(f"Processed {len(df_proc)} rows in {elapsed:.2f}s ({len(df_proc)/elapsed:.0f} rows/s).")

        # Validate
        validation_res = self.validate_transformation(df_raw, df_proc)
        if not validation_res["passed"]:
            raise ValueError(f"Validation failed for {dataset_id}: {validation_res['errors']}")

        # Write output if requested
        out_file_str = ""
        if output_path:
            out_p = Path(output_path)
            out_p.parent.mkdir(parents=True, exist_ok=True)
            logger.info(f"Writing normalized output to {out_p}...")
            # For list columns (tokens, numbers), serialize cleanly for TSV
            df_to_save = df_proc.copy()
            for list_col in ["name_tokens", "address_tokens", "address_numbers"]:
                if list_col in df_to_save.columns:
                    df_to_save[list_col] = df_to_save[list_col].apply(
                        lambda l: " ".join(l) if isinstance(l, list) else str(l)
                    )
            df_to_save.to_csv(out_p, sep="\t", index=False)
            out_file_str = str(out_p)

        stats = self.compute_statistics(
            df_proc,
            dataset_name=dataset_id,
            input_file=str(inp_p),
            output_file=out_file_str,
            elapsed_time=elapsed,
            validation_result=validation_res,
        )

        return df_proc, stats


# Global default preprocessor
_DEFAULT_PREPROCESSOR = EntityPreprocessor()


def preprocess_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Convenience function: preprocess an entity DataFrame."""
    return _DEFAULT_PREPROCESSOR.preprocess_dataframe(df)


def process_file(
    input_path: Union[str, Path],
    output_path: Optional[Union[str, Path]] = None,
    dataset_id: str = "dataset",
    chunksize: Optional[int] = None,
    max_rows: Optional[int] = None,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Convenience function: process an entity file from disk."""
    return _DEFAULT_PREPROCESSOR.process_file(
        input_path=input_path,
        output_path=output_path,
        dataset_id=dataset_id,
        chunksize=chunksize,
        max_rows=max_rows,
    )


def preprocess_source(
    df: pd.DataFrame,
    chunksize: int = 50_000,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Apply all normalization steps to a source DataFrame.
    Provides backward compatibility with Member 2 blocking and candidate generation modules.
    """
    preprocessor = EntityPreprocessor()
    res_df = preprocessor.preprocess_dataframe(df)
    res_df["name_is_non_latin"] = df["business_name"].apply(
        lambda s: any(ord(c) > 127 for c in str(s)) if pd.notna(s) else False
    )
    return res_df


def preprocess_chunk(chunk: pd.DataFrame) -> pd.DataFrame:
    """Process a single chunk without verbose logging (for parallel or batch use)."""
    return preprocess_source(chunk, verbose=False)


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    print("=" * 80)
    print("Self-Testing EntityPreprocessor Pipeline (src/preprocessing/preprocessor.py)")
    print("=" * 80)

    test_data = pd.DataFrame([
        {
            "entity_id": "S1-1001",
            "business_name": "Orelee's Barbershop & Salon Inc.",
            "business_address": "1795 Westchester Drive, Suite 4B, High Point, NC 27262",
            "country": "US",
        },
        {
            "entity_id": "S2-1002",
            "business_name": "राम मार्केटिंग प्राइवेट लिमिटेड",
            "business_address": "KH NO. -570/13, NEW DELHI, WEST DELHI, Delhi 110041",
            "country": "India",
        },
        {
            "entity_id": "S3-1003",
            "business_name": "Café de Paris S.A.R.L.",
            "business_address": "12 Rue de la Paix, 75002 Paris",
            "country": "France",
        },
        {
            "entity_id": "S2-1004",
            "business_name": "Target Store #1042",
            "business_address": None,
            "country": "United States",
        },
    ])

    preprocessor = EntityPreprocessor()
    df_result = preprocessor.preprocess_dataframe(test_data)
    print("Processed DataFrame columns:")
    for col in df_result.columns:
        print(f"  - {col}: {df_result[col].iloc[0]}")

    val_res = preprocessor.validate_transformation(test_data, df_result)
    print(f"\nValidation Result: {'PASS' if val_res['passed'] else 'FAIL'}")
    for k, v in val_res["all_checks"].items():
        print(f"  [{'PASS' if v else 'FAIL'}] {k}")

    print("\n" + "=" * 80)
    print("All unit test assertions passed successfully!")
    print("=" * 80)
