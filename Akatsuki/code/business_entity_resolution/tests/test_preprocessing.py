"""
test_preprocessing.py - Comprehensive Unit & Integration Test Suite for Member 1 Preprocessing
Tests Name Normalization, Address Normalization, Transliteration, Country Normalization,
and Unified Preprocessing Pipeline on synthetic test cases and real dataset examples.
"""

import sys
import unittest
from pathlib import Path

# Ensure paths
_TESTS_DIR = Path(__file__).resolve().parent
_ROOT_DIR = _TESTS_DIR.parent
if str(_ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(_ROOT_DIR))

import pandas as pd
from src.preprocessing.name_normalizer import BusinessNameNormalizer
from src.preprocessing.address_normalizer import AddressNormalizer
from src.preprocessing.transliteration import TransliterationEngine
from src.preprocessing.country_normalizer import CountryNormalizer
from src.preprocessing.preprocessor import EntityPreprocessor


class TestNameNormalization(unittest.TestCase):
    """Test cases for BusinessNameNormalizer."""

    def setUp(self):
        self.normalizer = BusinessNameNormalizer()

    def test_normal_latin_name(self):
        res = self.normalizer.normalize_single("Acme Industrial Solutions")
        self.assertEqual(res["name_norm"], "acme industrial solutions")
        self.assertEqual(res["name_core"], "acme industrial solutions")
        self.assertFalse(res["name_has_digits"])
        self.assertGreater(res["name_length"], 0)

    def test_punctuation_variation(self):
        res = self.normalizer.normalize_single("Orelee's Barbershop & Salon, Inc.")
        self.assertIn("orelees", res["name_norm"])
        self.assertIn("and", res["name_norm"])
        self.assertEqual(res["name_core"], "orelees barbershop and salon")
        self.assertFalse(res["name_has_digits"])

    def test_legal_suffix_variation(self):
        cases = [
            ("Apex Solutions Corporation", "apex solutions corp", "apex solutions"),
            ("Global Dynamics LLC", "global dynamics llc", "global dynamics"),
            ("Reliance Retail Private Limited", "reliance retail pvt ltd", "reliance retail"),
            ("Baker & McKenzie LLP", "baker and mckenzie llp", "baker and mckenzie"),
        ]
        for raw, expected_norm, expected_core in cases:
            res = self.normalizer.normalize_single(raw)
            self.assertEqual(res["name_norm"], expected_norm)
            self.assertEqual(res["name_core"], expected_core)

    def test_digits_in_name(self):
        res = self.normalizer.normalize_single("Store #1042 / 7-Eleven")
        self.assertTrue(res["name_has_digits"])
        self.assertIn("1042", res["name_norm"])
        self.assertIn("7 eleven", res["name_norm"])

    def test_unicode_text(self):
        res = self.normalizer.normalize_single("राम मार्केटिंग प्राइवेट लिमिटेड")
        # Should preserve Devanagari characters safely while mapping Devanagari legal suffix
        self.assertIn("राम", res["name_norm"])
        self.assertIn("pvt", res["name_norm"])
        self.assertIn("ltd", res["name_norm"])
        self.assertEqual(res["name_core"], "राम मार्केटिंग")

    def test_phonetic_soundex(self):
        res1 = self.normalizer.normalize_single("Smith Corp")
        res2 = self.normalizer.normalize_single("Smyth Inc")
        self.assertEqual(res1["name_phonetic"], "S530")
        self.assertEqual(res2["name_phonetic"], "S530")
        self.assertEqual(res1["name_phonetic"], res2["name_phonetic"])

    def test_empty_and_null_name(self):
        for empty_val in ["", "   ", None, float("nan")]:
            res = self.normalizer.normalize_single(empty_val)
            self.assertEqual(res["name_norm"], "")
            self.assertEqual(res["name_core"], "")
            self.assertEqual(res["name_tokens"], [])
            self.assertEqual(res["name_phonetic"], "")
            self.assertFalse(res["name_has_digits"])
            self.assertEqual(res["name_length"], 0)


class TestAddressNormalization(unittest.TestCase):
    """Test cases for AddressNormalizer."""

    def setUp(self):
        self.normalizer = AddressNormalizer()

    def test_ordinary_address(self):
        res = self.normalizer.normalize_single("123 Main Street, Austin, Texas")
        self.assertEqual(res["address_norm"], "123 main street austin texas")
        self.assertEqual(res["house_number"], "123")
        self.assertTrue(res["address_has_digits"])
        self.assertEqual(res["address_missing"], 0)

    def test_abbreviations(self):
        res = self.normalizer.normalize_single("1795 Westchester Rd, Ste 4B")
        self.assertIn("road", res["address_norm"])
        self.assertIn("suite", res["address_norm"])
        self.assertEqual(res["house_number"], "1795")

    def test_punctuation_and_formatting(self):
        res = self.normalizer.normalize_single("No. 10, Enkay Sq., Udyog Vihar Phase-V, Gurugram (HR)")
        self.assertIn("square", res["address_norm"])
        self.assertNotIn(".", res["address_norm"])
        self.assertNotIn("(", res["address_norm"])
        self.assertNotIn(")", res["address_norm"])

    def test_multiple_numeric_tokens(self):
        res = self.normalizer.normalize_single("1795 Westchester Drive, High Point, NC 27262")
        self.assertIn("1795", res["address_numbers"])
        self.assertIn("27262", res["address_numbers"])
        self.assertEqual(res["house_number"], "1795")
        self.assertEqual(res["postal_code"], "27262")

    def test_postal_code_extraction(self):
        # US ZIP
        res_us = self.normalizer.normalize_single("High Point, NC 27262")
        self.assertEqual(res_us["postal_code"], "27262")
        # India PIN code
        res_in = self.normalizer.normalize_single("NEW DELHI, WEST DELHI 110041")
        self.assertEqual(res_in["postal_code"], "110041")
        # France Postal code
        res_fr = self.normalizer.normalize_single("12 Rue de la Paix, 75002 Paris")
        self.assertEqual(res_fr["postal_code"], "75002")

    def test_missing_address(self):
        for empty_val in [None, "", "   ", float("nan")]:
            res = self.normalizer.normalize_single(empty_val)
            self.assertEqual(res["address_norm"], "")
            self.assertEqual(res["address_tokens"], [])
            self.assertEqual(res["address_numbers"], [])
            self.assertEqual(res["house_number"], "")
            self.assertEqual(res["postal_code"], "")
            self.assertFalse(res["address_has_digits"])
            self.assertEqual(res["address_missing"], 1)
            self.assertEqual(res["address_length"], 0)

    def test_unicode_address(self):
        res = self.normalizer.normalize_single("KH NO. -570/13, NEW DELHI, WEST DELHI, Delhi 110041")
        self.assertIn("570", res["address_numbers"])
        self.assertIn("13", res["address_numbers"])
        self.assertIn("110041", res["address_numbers"])
        self.assertEqual(res["postal_code"], "110041")


class TestTransliteration(unittest.TestCase):
    """Test cases for TransliterationEngine."""

    def setUp(self):
        self.engine = TransliterationEngine()

    def test_latin_script(self):
        res = self.engine.process_single("Zephay Labs Inc")
        self.assertEqual(res["detected_script"], "Latin")
        self.assertFalse(res["is_transliterated"])
        self.assertEqual(res["name_translit"], "zephay labs inc")

    def test_devanagari_transliteration(self):
        res = self.engine.process_single("राम मार्केटिंग प्राइवेट लिमिटेड")
        self.assertEqual(res["detected_script"], "Devanagari")
        self.assertTrue(res["is_transliterated"])
        # Should be pure Latin ASCII
        self.assertTrue(all(ord(c) < 128 for c in res["name_translit"]))
        self.assertIn("ram", res["name_translit"])

    def test_mixed_script(self):
        res = self.engine.process_single("ABC मॉडर्न स्टोर 123")
        self.assertTrue(res["is_transliterated"])
        self.assertTrue(all(ord(c) < 128 for c in res["name_translit"]))
        self.assertIn("abc", res["name_translit"])
        self.assertIn("123", res["name_translit"])

    def test_empty_and_null(self):
        for empty_val in [None, "", "   ", float("nan")]:
            res = self.engine.process_single(empty_val)
            self.assertEqual(res["name_translit"], "")
            self.assertFalse(res["is_transliterated"])


class TestCountryNormalization(unittest.TestCase):
    """Test cases for CountryNormalizer."""

    def setUp(self):
        self.normalizer = CountryNormalizer()

    def test_dataset_canonical_countries(self):
        self.assertEqual(self.normalizer.normalize_single("US"), "US")
        self.assertEqual(self.normalizer.normalize_single("India"), "India")
        self.assertEqual(self.normalizer.normalize_single("France"), "France")

    def test_us_variants(self):
        for var in ["USA", "United States", "united states of america", "U.S.", "U.S.A.", "us", "  US  "]:
            self.assertEqual(self.normalizer.normalize_single(var), "US")

    def test_india_variants(self):
        for var in ["IND", "Republic of India", "bharat", "INDIA", "in", "India."]:
            self.assertEqual(self.normalizer.normalize_single(var), "India")

    def test_france_variants(self):
        for var in ["FRA", "FR", "france", "French Republic", "republique francaise", "France."]:
            self.assertEqual(self.normalizer.normalize_single(var), "France")

    def test_open_set_recognized_countries(self):
        self.assertEqual(self.normalizer.normalize_single("Germany"), "Germany")
        self.assertEqual(self.normalizer.normalize_single("DEU"), "Germany")
        self.assertEqual(self.normalizer.normalize_single("jpn"), "Japan")
        self.assertEqual(self.normalizer.normalize_single("new zealand"), "New Zealand")

    def test_unseen_future_country(self):
        # Recognized ISO alpha-2 via pycountry resolves to official name
        self.assertEqual(self.normalizer.normalize_single("nl"), "Netherlands")
        # Truly unseen future / fictional country falls back cleanly
        self.assertEqual(self.normalizer.normalize_single("Atlantis"), "Atlantis")
        self.assertEqual(self.normalizer.normalize_single("xx"), "XX")

    def test_missing_country(self):
        for empty_val in [None, "", "   ", float("nan")]:
            self.assertEqual(self.normalizer.normalize_single(empty_val), "")

    def test_ambiguous_country(self):
        for ambig in ["unknown", "???", "n/a", "none"]:
            self.assertEqual(self.normalizer.normalize_single(ambig), "")


class TestUnifiedIntegration(unittest.TestCase):
    """End-to-end integration tests using real dataset slices."""

    def setUp(self):
        self.preprocessor = EntityPreprocessor()

    def test_full_pipeline_multi_record(self):
        raw_df = pd.DataFrame([
            {
                "entity_id": "S1-925783039",
                "business_name": "Orelee's Barbershop",
                "business_address": "1795 Westchester Drive, High Point, NC",
                "country": "US",
            },
            {
                "entity_id": "S2-166376419",
                "business_name": "राम मार्केटिंग प्राइवेट लिमिटेड",
                "business_address": "KH NO. -570/13, NEW DELHI, WEST DELHI, Delhi",
                "country": "India",
            },
            {
                "entity_id": "S3-202863386",
                "business_name": "wilfordhancock.com",
                "business_address": "Mack Rd, Haltom City, Texas",
                "country": "US",
            },
            {
                "entity_id": "S2-999999999",
                "business_name": "No Address Entity Inc",
                "business_address": None,
                "country": "USA",
            },
        ])

        processed = self.preprocessor.preprocess_dataframe(raw_df)
        val_res = self.preprocessor.validate_transformation(raw_df, processed)

        self.assertTrue(val_res["passed"], f"Validation failed: {val_res['errors']}")
        self.assertEqual(len(processed), len(raw_df))
        self.assertEqual(list(processed["entity_id"]), list(raw_df["entity_id"]))
        self.assertEqual(processed["address_missing"].tolist(), [0, 0, 0, 1])
        self.assertEqual(processed["country_norm"].tolist(), ["US", "India", "US", "US"])


if __name__ == "__main__":
    unittest.main()
