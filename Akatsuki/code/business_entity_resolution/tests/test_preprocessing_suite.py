"""
test_preprocessing_suite.py - Comprehensive Difficult Cases & Edge Cases Test Suite
Team Akatsuki - Preprocessing and Normalization Layer
TASK 2: Real Preprocessing Tests & Test Report Generation

Covers:
1. LLC vs L.L.C.
2. Corp vs Corporation
3. Pvt vs Private
4. Rd vs Road
5. St vs Street
6. Word-order inversion
7. Indic script (Devanagari, Tamil)
8. Empty address (null, empty, whitespace)
9. France + accents (French address, French country, diacritics)
10. Edge cases (digits-only, punctuation-only, complex Indian plot numbers, etc.)
"""

import sys
import unittest
from pathlib import Path
from typing import Any, Dict, List
import pandas as pd

# Add src to sys.path
_ROOT_DIR = Path(__file__).resolve().parent.parent
_SRC_DIR = _ROOT_DIR / "src"
for p in [str(_ROOT_DIR), str(_SRC_DIR)]:
    if p not in sys.path:
        sys.path.insert(0, p)

from src.preprocessing.address_normalizer import AddressNormalizer
from src.preprocessing.country_normalizer import CountryNormalizer
from src.preprocessing.name_normalizer import BusinessNameNormalizer
from src.preprocessing.preprocessor import EntityPreprocessor
from src.preprocessing.transliteration import TransliterationEngine


class TestDifficultCasesSuite(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.name_norm = BusinessNameNormalizer()
        cls.addr_norm = AddressNormalizer()
        cls.translit = TransliterationEngine()
        cls.country_norm = CountryNormalizer()
        cls.preprocessor = EntityPreprocessor()
        cls.test_results: List[Dict[str, str]] = []

    def record_test(self, test_case: str, raw_input: Any, output: Any, expected: str, actual: str, passed: bool):
        self.test_results.append({
            "test case": test_case,
            "input": str(raw_input),
            "output": str(output),
            "expected behavior": expected,
            "actual behavior": actual,
            "pass/fail": "PASS" if passed else "FAIL"
        })

    # 1. LLC vs L.L.C.
    def test_llc_variants(self):
        v1 = self.name_norm.normalize_single("Acme Enterprises LLC")
        v2 = self.name_norm.normalize_single("Acme Enterprises L.L.C.")
        v3 = self.name_norm.normalize_single("Acme Enterprises, l.l.c.")
        
        passed1 = (v1["name_norm"] == v2["name_norm"] == v3["name_norm"] == "acme enterprises llc")
        passed2 = (v1["name_core"] == v2["name_core"] == v3["name_core"] == "acme enterprises")
        passed3 = (v1["name_sorted_tokens"] == v2["name_sorted_tokens"] == "acme enterprises llc")
        all_passed = passed1 and passed2 and passed3

        self.record_test(
            "LLC vs L.L.C. Normalization",
            "Acme Enterprises LLC vs Acme Enterprises L.L.C.",
            f"v1: {v1['name_norm']} | v2: {v2['name_norm']}",
            "Both resolve to identical name_norm ('acme enterprises llc') and name_core ('acme enterprises')",
            f"name_norm: {v1['name_norm']} == {v2['name_norm']}, name_core: {v1['name_core']}",
            all_passed
        )
        self.assertTrue(all_passed)

    # 2. Corp vs Corporation
    def test_corp_variants(self):
        v1 = self.name_norm.normalize_single("Apex Global Corporation")
        v2 = self.name_norm.normalize_single("Apex Global Corp.")
        v3 = self.name_norm.normalize_single("Apex Global Corp")
        
        passed = (v1["name_norm"] == v2["name_norm"] == v3["name_norm"] == "apex global corp" and
                  v1["name_core"] == v2["name_core"] == v3["name_core"] == "apex global")
        self.record_test(
            "Corp vs Corporation Normalization",
            "Apex Global Corporation vs Apex Global Corp.",
            f"v1: {v1['name_norm']} | v2: {v2['name_norm']}",
            "Both resolve to canonical 'corp' in name_norm and stripped in name_core ('apex global')",
            f"name_norm: {v1['name_norm']}, name_core: {v1['name_core']}",
            passed
        )
        self.assertTrue(passed)

    # 3. Pvt vs Private
    def test_pvt_variants(self):
        v1 = self.name_norm.normalize_single("Reliance Infotech Private Limited")
        v2 = self.name_norm.normalize_single("Reliance Infotech Pvt. Ltd.")
        v3 = self.name_norm.normalize_single("Reliance Infotech Pvt Ltd")

        passed = (v1["name_norm"] == v2["name_norm"] == v3["name_norm"] == "reliance infotech pvt ltd" and
                  v1["name_core"] == v2["name_core"] == v3["name_core"] == "reliance infotech")
        self.record_test(
            "Pvt vs Private Normalization",
            "Reliance Infotech Private Limited vs Reliance Infotech Pvt. Ltd.",
            f"v1: {v1['name_norm']} | v2: {v2['name_norm']}",
            "Both map Private->pvt, Limited->ltd, core name stripped to 'reliance infotech'",
            f"name_norm: {v1['name_norm']}, name_core: {v1['name_core']}",
            passed
        )
        self.assertTrue(passed)

    # 4. Rd vs Road
    def test_rd_vs_road(self):
        a1 = self.addr_norm.normalize_single("1245 Westchester Rd, Ste 4B")
        a2 = self.addr_norm.normalize_single("1245 Westchester Road, Suite 4B")

        passed = (a1["address_norm"] == a2["address_norm"] == "1245 westchester road suite 4b" and
                  a1["house_number"] == a2["house_number"] == "1245")
        self.record_test(
            "Rd vs Road Normalization",
            "1245 Westchester Rd, Ste 4B vs 1245 Westchester Road, Suite 4B",
            f"a1: {a1['address_norm']} | a2: {a2['address_norm']}",
            "Rd expands to road, Ste expands to suite; identical address_norm and house_number ('1245')",
            f"address_norm: {a1['address_norm']}, house_number: {a1['house_number']}",
            passed
        )
        self.assertTrue(passed)

    # 5. St vs Street
    def test_st_vs_street(self):
        a1 = self.addr_norm.normalize_single("500 Main St, Apt 2")
        a2 = self.addr_norm.normalize_single("500 Main Street, Apartment 2")

        passed = (a1["address_norm"] == a2["address_norm"] == "500 main street apartment 2" and
                  a1["house_number"] == a2["house_number"] == "500")
        self.record_test(
            "St vs Street Normalization",
            "500 Main St, Apt 2 vs 500 Main Street, Apartment 2",
            f"a1: {a1['address_norm']} | a2: {a2['address_norm']}",
            "St expands to street, Apt expands to apartment; identical address_norm and house_number ('500')",
            f"address_norm: {a1['address_norm']}, house_number: {a1['house_number']}",
            passed
        )
        self.assertTrue(passed)

    # 6. Word-Order Inversion
    def test_word_order_inversion(self):
        n1 = self.name_norm.normalize_single("Cornerstone Advanced Tavia")
        n2 = self.name_norm.normalize_single("Tavia Cornerstone Advanced")
        n3 = self.name_norm.normalize_single("Advanced Tavia Cornerstone")

        passed = (n1["name_sorted_tokens"] == n2["name_sorted_tokens"] == n3["name_sorted_tokens"] == "advanced cornerstone tavia")
        self.record_test(
            "Word-Order Inversion",
            "Cornerstone Advanced Tavia vs Tavia Cornerstone Advanced",
            f"n1: {n1['name_sorted_tokens']} | n2: {n2['name_sorted_tokens']}",
            "name_sorted_tokens resolves inverted word orders to identical key 'advanced cornerstone tavia'",
            f"name_sorted_tokens: {n1['name_sorted_tokens']}",
            passed
        )
        self.assertTrue(passed)

    def test_hospital_city_inversion(self):
        n1 = self.name_norm.normalize_single("City Hospital")
        n2 = self.name_norm.normalize_single("Hospital City")

        passed = (n1["name_sorted_tokens"] == n2["name_sorted_tokens"] == "city hospital")
        self.record_test(
            "Word-Order Inversion (City Hospital)",
            "City Hospital vs Hospital City",
            f"n1: {n1['name_sorted_tokens']} | n2: {n2['name_sorted_tokens']}",
            "Produces identical name_sorted_tokens 'city hospital'",
            f"name_sorted_tokens: {n1['name_sorted_tokens']}",
            passed
        )
        self.assertTrue(passed)

    # 7. Indic Script (Devanagari, Tamil)
    def test_indic_devanagari(self):
        raw = "रॉयल सूर्य मैनेजमेंट"
        norm_res = self.name_norm.normalize_single(raw)
        translit_res = self.translit.transliterate_raw(raw)

        # Raw & norm preserve Devanagari characters
        norm_preserved = ("रॉयल" in norm_res["name_norm"])
        # Translit is Latin ASCII
        translit_latin = all(ord(c) < 128 for c in translit_res)
        passed = norm_preserved and translit_latin and len(translit_res) > 5

        self.record_test(
            "Indic Script (Devanagari)",
            raw,
            f"norm: {norm_res['name_norm']} | translit: {translit_res}",
            "name_norm preserves Devanagari characters; transliteration converts to pure ASCII Latin",
            f"norm: {norm_res['name_norm']}, translit: {translit_res} (is_ascii={translit_latin})",
            passed
        )
        self.assertTrue(passed)

    def test_indic_tamil(self):
        raw = "பிரைம் புராப்பர்ட்டீஸ் பிரைவேட் லிமிடெட்"
        norm_res = self.name_norm.normalize_single(raw)
        translit_res = self.translit.transliterate_raw(raw)

        norm_preserved = ("பிரைம்" in norm_res["name_norm"])
        translit_latin = all(ord(c) < 128 for c in translit_res)
        passed = norm_preserved and translit_latin and len(translit_res) > 5

        self.record_test(
            "Indic Script (Tamil)",
            raw,
            f"norm: {norm_res['name_norm']} | translit: {translit_res}",
            "name_norm preserves Tamil characters; transliteration produces clean ASCII Latin",
            f"norm: {norm_res['name_norm']}, translit: {translit_res} (is_ascii={translit_latin})",
            passed
        )
        self.assertTrue(passed)

    # 8. Empty Address Handling
    def test_empty_address(self):
        test_inputs = [None, "", "   ", float("nan")]
        all_passed = True
        for val in test_inputs:
            res = self.addr_norm.normalize_single(val)
            p = (res["address_missing"] == 1 and
                 res["address_norm"] == "" and
                 res["address_tokens"] == [] and
                 res["address_numbers"] == [] and
                 res["house_number"] == "" and
                 res["postal_code"] == "" and
                 res["address_has_digits"] is False and
                 res["address_length"] == 0)
            if not p:
                all_passed = False

        self.record_test(
            "Empty Address Robustness",
            "[None, '', '   ', NaN]",
            "address_missing=1, address_norm='', tokens=[], house_number='', postal_code=''",
            "Gracefully flags missing address as 1 without crashing or fabricating synthetic text",
            f"All null variants produced address_missing=1 and empty fields",
            all_passed
        )
        self.assertTrue(all_passed)

    # 9. France + Accents
    def test_france_accents_and_address(self):
        name_raw = "Café de l'Étoile SARL"
        addr_raw = "20 Rue René Panhard, 75013 Paris"
        country_raw = "France"

        n_res = self.name_norm.normalize_single(name_raw)
        a_res = self.addr_norm.normalize_single(addr_raw)
        c_res = self.country_norm.normalize_single(country_raw)

        passed_name = (n_res["name_norm"] == "café de létoile sarl" and
                       n_res["name_core"] == "café de létoile" and
                       n_res["name_translit"] == "cafe de letoile sarl")
        passed_addr = (a_res["address_norm"] == "20 rue rené panhard 75013 paris" and
                       a_res["house_number"] == "20" and
                       a_res["postal_code"] == "75013")
        passed_ctry = (c_res == "France")
        all_passed = passed_name and passed_addr and passed_ctry

        self.record_test(
            "France + Accents (Name, Address, Country)",
            f"{name_raw} | {addr_raw} | {country_raw}",
            f"name_norm: {n_res['name_norm']} | name_translit: {n_res['name_translit']} | addr_norm: {a_res['address_norm']} | house: {a_res['house_number']} | pin: {a_res['postal_code']} | country: {c_res}",
            "Preserves French accents (é) in name_norm/address_norm, generates ASCII in name_translit, extracts house '20', postal '75013', canonical 'France'",
            f"name: {n_res['name_norm']}, translit: {n_res['name_translit']}, addr: {a_res['address_norm']}, house: {a_res['house_number']}, postal: {a_res['postal_code']}, country: {c_res}",
            all_passed
        )
        self.assertTrue(all_passed)

    def test_france_country_variants(self):
        variants = ["France", "FR", "fra", "République française", "FRANCE."]
        all_match = True
        for v in variants:
            c = self.country_norm.normalize_single(v)
            if c != "France":
                all_match = False

        self.record_test(
            "France Country Variants",
            "['France', 'FR', 'fra', 'République française', 'FRANCE.']",
            "France for all variants",
            "All French abbreviations and case variants map deterministically to 'France'",
            f"Result for all: {'France' if all_match else 'MISMATCH'}",
            all_match
        )
        self.assertTrue(all_match)

    # 10. Edge Cases
    def test_complex_indian_plot(self):
        addr = "Plot No. 16-11-23/37/A, Moosarambagh, Hyderabad 500036"
        res = self.addr_norm.normalize_single(addr)
        passed = (res["house_number"] == "16-11-23/37/A" and res["postal_code"] == "500036")

        self.record_test(
            "Complex Indian Plot & PIN Code",
            addr,
            f"house_number: {res['house_number']} | postal_code: {res['postal_code']}",
            "Extracts multi-hyphenated/slashed plot number and 6-digit Indian PIN code",
            f"house_number: {res['house_number']}, postal_code: {res['postal_code']}",
            passed
        )
        self.assertTrue(passed)

    def test_digits_only_name(self):
        name = "7-Eleven Inc"
        res = self.name_norm.normalize_single(name)
        passed = (res["name_norm"] == "7 eleven inc" and
                  res["name_core"] == "7 eleven" and
                  res["name_has_digits"] is True)

        self.record_test(
            "Digits in Business Name",
            name,
            f"name_norm: {res['name_norm']}, name_core: {res['name_core']}, has_digits: {res['name_has_digits']}",
            "Preserves numeric tokens '7 eleven', strips legal suffix, sets name_has_digits=True",
            f"name_norm: {res['name_norm']}, name_core: {res['name_core']}, has_digits: {res['name_has_digits']}",
            passed
        )
        self.assertTrue(passed)

    def test_punctuation_only_string(self):
        name = "... --- ..."
        res = self.name_norm.normalize_single(name)
        passed = (res["name_norm"] == "" and res["name_core"] == "" and res["name_sorted_tokens"] == "")

        self.record_test(
            "Punctuation-Only String",
            name,
            f"name_norm: {res['name_norm']}, name_core: {res['name_core']}",
            "Safely reduces to empty string without crashing or producing dangling punctuation",
            f"name_norm: '{res['name_norm']}', name_core: '{res['name_core']}'",
            passed
        )
        self.assertTrue(passed)

    @classmethod
    def tearDownClass(cls):
        df_report = pd.DataFrame(cls.test_results)
        
        # Save to both reports/member1 and Akatsuki/reports/member1
        out_paths = [
            _ROOT_DIR / "reports" / "member1" / "preprocessing_test_report.csv",
            _ROOT_DIR / "amazon-ml-challenge" / "Akatsuki" / "reports" / "member1" / "preprocessing_test_report.csv"
        ]
        for p in out_paths:
            p.parent.mkdir(parents=True, exist_ok=True)
            df_report.to_csv(p, index=False)
            print(f"Saved test report -> {p} ({len(df_report)} test cases)")


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    unittest.main()
