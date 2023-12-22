import unittest

import numpy as np
import pandas as pd
import pandas.testing as pdt

from address_utils import correct_spelling
from constants import MISSPELLINGS_MADRID, TRACK_CLASSES
from dataframe_utils import extract_and_store_patterns, get_unique_names
from text_utils import preprocess_text, remove_accents


class TestGetUniqueNames(unittest.TestCase):
    def setUp(self):
        self.data = pd.DataFrame(
            {
                "col1": [
                    "Alice",
                    "Bob",
                    "alice",
                    "Charlie",
                    np.NaN,
                    "Arroyomolinos",
                    "Molinos",
                ],
                "col2": [
                    "David",
                    "Eve",
                    "Frank",
                    "Bob",
                    "David",
                    "Los Molinos",
                    "Conde de Orgaz",
                ],
            }
        )

    def test_valid_input_two_columns(self):
        result = get_unique_names(self.data, "col1", "col2")
        expected = [
            "conde de orgaz",
            "arroyomolinos",
            "los molinos",
            "charlie",
            "molinos",
            "alice",
            "david",
            "frank",
            "bob",
            "eve",
        ]
        self.assertEqual(result, expected)

    def test_valid_input_one_column(self):
        result = get_unique_names(self.data, "col1")
        expected = ["arroyomolinos", "charlie", "molinos", "alice", "bob"]
        self.assertEqual(result, expected)

    def test_dataframe_with_nans(self):
        result = get_unique_names(self.data, "col1")
        self.assertIn("alice", result)

    def test_non_string_data(self):
        df = pd.DataFrame({"col1": [1, 2, 3, 2, 1], "col2": [4, 5, 6, 5, 4]})
        result = get_unique_names(df, "col1", "col2")
        expected = ["1", "2", "3", "4", "5", "6"]
        self.assertEqual(result, expected)

    def test_sorting_by_length(self):
        result = get_unique_names(self.data, "col1", "col2")
        lengths = [len(name) for name in result]
        self.assertEqual(lengths, sorted(lengths, reverse=True))

    def test_case_insensitivity(self):
        result = get_unique_names(self.data, "col1")
        self.assertIn("alice", result)
        self.assertNotIn("Alice", result)

    def test_nonexistent_columns(self):
        with self.assertRaises(KeyError):
            get_unique_names(self.data, "nonexistent_col")

    def test_single_item_dataframe(self):
        df = pd.DataFrame({"col1": ["Alice"], "col2": ["Bob"]})
        result = get_unique_names(df, "col1", "col2")
        self.assertEqual(result, ["alice", "bob"])


class TestRemoveAccents(unittest.TestCase):
    def test_empty_string(self):
        self.assertEqual(remove_accents(""), "")

    def test_no_accents(self):
        self.assertEqual(remove_accents("hello"), "hello")

    def test_accents(self):
        self.assertEqual(remove_accents("áéíóú"), "aeiou")

    def test_case_insensitive(self):
        self.assertEqual(remove_accents("ÁÉÍÓÚ"), "aeiou")

    # Test that special spanish characters are not removed
    def test_special_characters(self):
        self.assertEqual(remove_accents("ñü"), "ñü")

    # Test that the function works with composed characters
    def test_composed_characters(self):
        self.assertEqual(remove_accents("üéñ"), "üeñ")

    # Test a complex case combining all the previous ones, with many words
    def test_complex_case(self):
        self.assertEqual(remove_accents("ÁÉÍÓÚ üé ñü Pº p°"), "aeiou üe ñü pº p°")


class TestPreprocessText(unittest.TestCase):
    def setUp(self):
        self.ignore_preps = ["a", "en", "al", "de", "del", "la", "el", "los", "las"]

    def test_regular_case(self):
        self.assertEqual(
            preprocess_text("a la calle de la paloma", self.ignore_preps),
            "calle paloma",
        )

    def test_ignore_prepositions_none(self):
        self.assertEqual(
            preprocess_text("a la calle de la paloma", None),
            "a la calle de la paloma",
        )

    def test_empty_string(self):
        self.assertEqual(preprocess_text("", self.ignore_preps), "")

    def test_string_with_no_prepositions(self):
        self.assertEqual(
            preprocess_text("calle paloma", self.ignore_preps), "calle paloma"
        )

    def test_prepositions_not_included_in_list(self):
        self.assertEqual(
            preprocess_text(
                "ante la calle del martir bajo el puente y mediante la luz",
                self.ignore_preps,
            ),
            "ante calle martir bajo puente y mediante luz",
        )

    def test_repeated_prepositions(self):
        self.assertEqual(
            preprocess_text(
                "a la calle de la paloma del martes y los los jueves", self.ignore_preps
            ),
            "calle paloma martes y jueves",
        )

    def test_special_characters(self):
        self.assertEqual(
            [
                preprocess_text(x, self.ignore_preps)
                for x in [
                    "garci-nuñez",
                    "San blas - canillejas",
                    "c/ Dr. Martínez de la riva",
                    "s.a.r. borbon y battemberg",
                ]
            ],
            [
                "garci nuñez",
                "san blas canillejas",
                "c doctor martinez riva",
                "sar borbon y battemberg",
            ],
        )


class TestExtractAndStorePatterns(unittest.TestCase):
    def setUp(self):
        self.patterns_list = [
            "la corte de faraon",
            "puerto del suebe",
            "valle guerra",
            "doctor federico rubio y gali",
            "corregidor alonso de tobar",
            "pico de peña golosa",
            "isla sumatra",
            "alcalde sainz de baranda",
            "lopez grass",
            "pico beriain",
            "bueso pineda",
            "san blas - canillejas",
            "fuencarral - el pardo",
            "garci-nuño",
            "wad-ras",
            "antonio diaz-cañabate",
            "s.a.r. don juan de borbon y battemberg",
        ]
        self.ignore_preps = ["a", "en", "al", "de", "del", "la", "el", "los", "las"]
        self.patterns_dict = TRACK_CLASSES

    def test_empty_dataframe(self):
        df = pd.DataFrame({"ADDRESS": []})
        df = extract_and_store_patterns(df, self.patterns_list, "ADDRESS2", "ADDRESS")
        self.assertEqual(df["ADDRESS2"].tolist(), [])

    def test_no_matches(self):
        df = pd.DataFrame({"ADDRESS": ["corte del faraon", "valleguerra"]})
        df = extract_and_store_patterns(df, self.patterns_list, "ADDRESS2", "ADDRESS")

        expected = pd.Series([np.nan, np.nan], name="ADDRESS2")
        pdt.assert_series_equal(df["ADDRESS2"], expected, check_names=False)

    def test_regular_cases(self):
        df = pd.DataFrame(
            {
                "ADDRESS": [
                    "La Corte de Faraon",
                    "calle de la Corte de faraon, 465",
                    "Valleguerra",
                    "Valle de la Guerra",
                    "federico rubio y gali",
                    "pico de Peña Golosa",
                ]
            }
        )

        df = extract_and_store_patterns(df, self.patterns_list, "ADDRESS2", "ADDRESS")

        self.assertEqual(
            df["ADDRESS2"].tolist(),
            [
                "LA CORTE DE FARAON",
                "LA CORTE DE FARAON",
                np.nan,
                np.nan,
                np.nan,
                "PICO DE PEÑA GOLOSA",
            ],
        )

    def test_streets_with_accents(self):
        df = pd.DataFrame(
            {
                "ADDRESS": [
                    "avenida la Corte de faraón, 465",
                    "Doctor federico Rubio y galí",
                    "pasaje al Pico beriáin",
                ]
            }
        )

        df = extract_and_store_patterns(df, self.patterns_list, "ADDRESS2", "ADDRESS")

        self.assertEqual(
            df["ADDRESS2"].tolist(),
            [
                "LA CORTE DE FARAON",
                "DOCTOR FEDERICO RUBIO Y GALI",
                "PICO BERIAIN",
            ],
        )

    # Special characters should be ignored
    def test_street_with_special_characters(self):
        df = pd.DataFrame(
            {
                "ADDRESS": [
                    "san blas - canillejas",
                    "san blas-canillejas",
                    "san blas-canillejas, madrid",
                    "fuencarral - el pardo",
                    "garci - nuño, 26",
                    "garcinuño, 21",
                    "garci nuño, 49",
                    "antonio diaz cañabate",
                    "calle sar don juan de borbon y battemberg",
                ]
            }
        )

        df = extract_and_store_patterns(df, self.patterns_list, "ADDRESS2", "ADDRESS")

        self.assertEqual(
            df["ADDRESS2"].tolist(),
            [
                "SAN BLAS - CANILLEJAS",
                "SAN BLAS - CANILLEJAS",
                "SAN BLAS - CANILLEJAS",
                "FUENCARRAL - EL PARDO",
                "GARCI-NUÑO",
                np.nan,
                "GARCI-NUÑO",
                "ANTONIO DIAZ-CAÑABATE",
                "S.A.R. DON JUAN DE BORBON Y BATTEMBERG",
            ],
        )

    def test_composed_parameter(self):
        df = pd.DataFrame({"ADDRESS": ["san blas", "san blas, 36", "fuencarral"]})

        df = extract_and_store_patterns(
            df, self.patterns_list, "ADDRESS2", "ADDRESS", composed=True
        )

        self.assertEqual(
            df["ADDRESS2"].tolist(),
            [
                "SAN BLAS - CANILLEJAS",
                "SAN BLAS - CANILLEJAS",
                "FUENCARRAL - EL PARDO",
            ],
        )

    def test_ignore_prepositions(self):
        df = pd.DataFrame(
            {
                "ADDRESS": [
                    "corte de faraon",
                    "las corte del faraon",
                    "corte de el faraon",
                    "malecon de la corte de el faraon",
                    "la corte y el faraon",
                    "puerto de suebe",
                    "valle de la guerra",
                    "el doctor federico rubio y gali",
                    "doctor federico rubio gali",
                    "el pico de la peña golosa",
                    "calle isla de al sumatra",
                    "san blas y canillejas",
                    "fuencarral y el pardo",
                    "garci del nuño",
                    "wad las ras",
                ]
            }
        )

        df = extract_and_store_patterns(
            df,
            self.patterns_list,
            "ADDRESS2",
            "ADDRESS",
            ignore_prepositions=self.ignore_preps,
        )

        self.assertEqual(
            df["ADDRESS2"].tolist(),
            [
                "LA CORTE DE FARAON",
                "LA CORTE DE FARAON",
                "LA CORTE DE FARAON",
                "LA CORTE DE FARAON",
                np.nan,
                "PUERTO DEL SUEBE",
                "VALLE GUERRA",
                "DOCTOR FEDERICO RUBIO Y GALI",
                np.nan,
                "PICO DE PEÑA GOLOSA",
                "ISLA SUMATRA",
                np.nan,
                np.nan,
                "GARCI-NUÑO",
                "WAD-RAS",
            ],
        )

    def test_patterns_as_dictionary(self):
        df = pd.DataFrame(
            {
                "ADDRESS": [
                    "calle el aviador zorita",
                    "cl. hernando cortes",
                    " cl isla de sumatra",
                    "la cll. isla de sumatra",
                    "cal. el bambino",
                    "la corte del faraon",
                    "galr.",
                    "pico de Peña Golosa",
                    "Trav. del Norte",
                    "av. de la albufera",
                    "avda. de la albufera",
                    "la avda. de la albufera",
                    "Pasdz. Central",
                    "Pº de la Castellana",
                    "p/ de la Castellana",
                    "plza de la Castellana",
                    "cmno del Pardo",
                ]
            }
        )

        df = extract_and_store_patterns(
            df,
            self.patterns_dict,
            "ADDRESS2",
            "ADDRESS",
        )

        self.assertEqual(
            df["ADDRESS2"].tolist(),
            [
                "CALLE",
                "CALLE",
                "CALLE",
                "CALLE",
                "CALLE",
                np.nan,
                "GALERÍA",
                np.nan,
                "TRAVESÍA",
                "AVENIDA",
                "AVENIDA",
                "AVENIDA",
                "PASADIZO",
                "PASEO",
                "PASEO",
                "PLAZA",
                "CAMINO",
            ],
        )

    def test_pattern_priority(self):
        df = pd.DataFrame(
            {
                "ADDRESS": [
                    "valle guerra la corte de faraon",
                    "pico de peña golosa wad-ras",
                    "wad-ras pico de peña golosa",
                    "san blas fuencarral",
                ]
            }
        )

        df = extract_and_store_patterns(
            df,
            self.patterns_list,
            "ADDRESS2",
            "ADDRESS",
            ignore_prepositions=self.ignore_preps,
            composed=True,
        )

        self.assertEqual(
            df["ADDRESS2"].tolist(),
            [
                "VALLE GUERRA",
                "PICO DE PEÑA GOLOSA",
                "WAD-RAS",
                "SAN BLAS - CANILLEJAS",
            ],
        )


class TestCorrectSpelling(unittest.TestCase):
    misspellings = MISSPELLINGS_MADRID

    def test_basic_functionality(self):
        self.assertEqual(correct_spelling("alacala", MISSPELLINGS_MADRID), "alcala")

    def test_case_sensitivity(self):
        self.assertEqual(
            correct_spelling("LAS TABLAS", MISSPELLINGS_MADRID), "valverde"
        )

    def test_accent_handling(self):
        self.assertEqual(
            correct_spelling("espirítu sant", MISSPELLINGS_MADRID), "espiritu santo"
        )

    def test_multi_word_misspellings(self):
        self.assertEqual(
            correct_spelling("marques de silvea", MISSPELLINGS_MADRID),
            "marquesa de silvela",
        )

    def test_abbreviations_and_special_characters(self):
        self.assertEqual(
            correct_spelling("Dr. r dinamarca", MISSPELLINGS_MADRID),
            "doctor republica dinamarca",
        )

    def test_longest_match_priority(self):
        self.assertEqual(
            correct_spelling("dr ortega y gasset", MISSPELLINGS_MADRID),
            "doctor jose ortega y gasset",
        )

    def test_no_misspellings(self):
        self.assertEqual(correct_spelling("Gran Via", MISSPELLINGS_MADRID), "gran via")

    def test_empty_string(self):
        self.assertEqual(correct_spelling("", MISSPELLINGS_MADRID), "")
        self.assertEqual(correct_spelling("", MISSPELLINGS_MADRID), "")
