import re
import unicodedata


def preprocess_text(text: str, ignore_prepositions: list = None) -> str:
    """
    Preprocess the text by normalizing special characters and removing prepositions.
    """
    text = remove_accents(text)
    # Known abbreviations handling
    known_abbreviations = {"s.a.r.": "sar", "dr.": "doctor", "r.": "republica"}
    for abbreviation, replacement in known_abbreviations.items():
        text = text.replace(abbreviation, replacement)
    text = re.sub(r"[-./]", " ", text)
    text = re.sub(r"\s+", " ", text)
    if ignore_prepositions:
        words = text.split()
        text = " ".join(
            word for word in words if word.lower() not in ignore_prepositions
        )

    return text.strip()


def remove_accents(input_str: str, protected_chars: list = ["ñ", "ü", "°", "º"]) -> str:
    """
    Remove accents from a string while preserving special characters.

    Parameters:
    - input_str (str): The string from which to remove accents.
    - protected_chars (list): A list of special characters to preserve.
    """
    input_str = input_str.lower()
    placeholders = []
    for i, char in enumerate(protected_chars):
        placeholder = f"PLACEHOLDER_{i}"
        input_str = input_str.replace(char, placeholder)
        placeholders.append((placeholder, char))
    nfkd_form = unicodedata.normalize("NFKD", input_str)
    only_ascii = nfkd_form.encode("ASCII", "ignore").decode("utf-8")
    for placeholder, char in placeholders:
        only_ascii = only_ascii.replace(placeholder, char)
    return only_ascii
