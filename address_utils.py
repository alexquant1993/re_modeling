import re

from text_utils import remove_accents


def add_num_to_address(address):
    if not re.search(r"\d", address):
        address_split = address.split(",")
        address_split[0] = address_split[0] + " 1"
        address = ",".join(address_split)
    return address


def correct_spelling(street_name: str, misspellings: dict) -> str:
    """
    Correct the spelling of a street name based on a dictionary of misspellings.

    This function takes a street name and a dictionary of misspellings and corrections.
    It replaces any misspelled words or phrases in the street name with the correct
    spelling.
    The function handles multi-word misspellings and ensures that the longest matches are
    replaced first.

    Parameters:
    - street_name (str): The street name to correct.
    - misspellings (dict): A dictionary where the keys are misspelled words or phrases
        and the values are the correct spellings.

    Returns:
    str: The corrected street name.
    """
    # Normalize the street name by converting to lowercase and removing accents
    street_name = street_name.lower()
    street_name = street_name.replace("´", "'").replace("`", "'")
    street_name = remove_accents(street_name)

    # Sort the misspellings by length of keys in descending order
    misspellings = sorted(
        misspellings.items(), key=lambda item: len(item[0]), reverse=True
    )

    # Split the street name into words, considering special cases for abbreviations
    # Include hyphens as part of a word
    words = re.findall(r"\b[\w\'-]+|\.", street_name)
    corrected_words = []
    i = 0

    # Loop through each word in the street name
    while i < len(words):
        found = False
        for misspelled, correct in misspellings:
            misspelled_len = len(re.findall(r"\b[\w\'-]+|\.", misspelled))
            # Check if the next few words match a misspelled phrase
            phrase_to_check = " ".join(words[i : i + misspelled_len]).replace(" .", ".")
            if i + misspelled_len <= len(words) and phrase_to_check == misspelled:
                corrected_words.append(correct)
                i += misspelled_len
                found = True
                break

        if not found:
            corrected_words.append(words[i])
            i += 1

    return " ".join(corrected_words)
