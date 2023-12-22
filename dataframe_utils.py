import re
from typing import List

import numpy as np
import pandas as pd

from text_utils import preprocess_text


def get_unique_names(
    df: pd.DataFrame, target_var: str, target_var2: str = None
) -> list:
    """
    This function extracts unique names from two columns of a DataFrame, merges them,
    removes duplicates, and sorts them by length in descending order.

    Parameters:
    df (pandas.DataFrame): The DataFrame to process.
    target_var (str): The name of the first column from which to extract unique names.
    target_var2 (str): The name of the second column from which to extract unique names.

    Returns:
    unique_names (list): The list of unique names sorted by length in descending order.
    """
    if target_var not in df.columns:
        raise KeyError(f"Column not found: {target_var}")
    if target_var2 is not None and target_var2 not in df.columns:
        raise KeyError(f"Column not found: {target_var2}")

    # Convert to string and get unique names from the first target variable
    names = pd.Series(df[target_var].dropna().astype(str).unique()).str.lower().tolist()

    # If target_var2 is provided and exists, get unique names from it
    if target_var2:
        names2 = (
            pd.Series(df[target_var2].dropna().astype(str).unique())
            .str.lower()
            .tolist()
        )
        names.extend(names2)

    # Remove duplicates and sort by length
    unique_names = sorted(set(names), key=lambda s: (-len(s), s))

    return unique_names


def extract_and_store_patterns(
    df: pd.DataFrame,
    patterns: list or dict,
    target_var: str,
    base_var: str,
    composed: bool = False,
    ignore_prepositions: list = None,
):
    """
    Extracts patterns from a base column of a DataFrame and stores them in a target column.

    This function takes a DataFrame and a set of patterns, and extracts the patterns from a
    specified base column of the DataFrame. The extracted patterns are then stored in a
    specified target column of the DataFrame.

    Parameters:
    - df (pandas.DataFrame): The DataFrame from which to extract patterns.
    - patterns (list or dict): The patterns to extract. If a dictionary is provided, the keys
        are the patterns and the values are the full names to use in the target column.
    - target_var (str): The name of the column in which to store the extracted patterns.
    - base_var (str): The name of the column from which to extract patterns.
    - composed (bool, optional): If True, the function will split the patterns on " - " and
        only use the first part for matching. Defaults to False.
    - ignore_prepositions (list, optional): A list of prepositions to ignore when processing
        the text. Defaults to None.

    Returns:
    pandas.DataFrame: The DataFrame with the extracted patterns stored in the target column.
    """
    # Preprocess and compile regex patterns
    if isinstance(patterns, dict):
        pattern_to_full_name = pair_values_to_key(patterns, sort_keys=True)
    else:
        pattern_to_full_name = {pattern.lower(): pattern for pattern in patterns}

    compiled_patterns = {}
    for pattern, full_name in pattern_to_full_name.items():
        pattern_part = pattern.split(" - ")[0] if composed else pattern
        processed_pattern = preprocess_text(pattern_part, ignore_prepositions)
        regex = re.compile(
            r"\b{}\b".format(re.escape(processed_pattern)), flags=re.IGNORECASE
        )
        compiled_patterns[regex] = full_name

    def find_earliest_pattern(text):
        if pd.isna(text):
            return np.nan

        processed_text = preprocess_text(text, ignore_prepositions)
        earliest_match = None
        earliest_match_name = None

        for regex, full_name in compiled_patterns.items():
            match = regex.search(processed_text)
            if match:
                if earliest_match is None or match.start() < earliest_match.start():
                    earliest_match = match
                    earliest_match_name = full_name

        return earliest_match_name if earliest_match_name is not None else np.nan

    # Process dataframe
    df = df.copy()
    df[target_var] = df[base_var].apply(find_earliest_pattern)
    df[target_var] = df[target_var].apply(
        lambda x: x.upper() if isinstance(x, str) else x
    )

    return df


def pair_values_to_key(input_dict, sort_keys=False):
    """
    This function takes a dictionary and returns a new dictionary where each value from
    the original dictionary (including the key itself) is paired with its corresponding
    key. If sort_keys is True, the keys in the output dictionary are sorted by their
    length in descending order.

    Parameters:
    - input_dict (dict): A dictionary where the keys are the main items and the values
        are lists of related items.
    - sort_keys (bool): A flag to indicate whether to sort the keys in the output
        dictionary by their length in descending order.

    Returns:
    output_dict (dict): A dictionary where each related item (including the main item)
    is paired with the main item.
    """
    output_dict = {}
    for key in input_dict.keys():
        values = input_dict[key]
        output_dict[key.lower()] = key
        for value in values:
            output_dict[value.lower()] = key

    if sort_keys:
        output_dict = {
            k: output_dict[k] for k in sorted(output_dict, key=len, reverse=True)
        }

    return output_dict


def fill_na_df(
    df: pd.DataFrame,
    df2: pd.DataFrame,
    target_columns: List[str],
    filler_columns: List[str] = None,
) -> pd.DataFrame:
    """
    Fills missing values in columns of DataFrame df using values from corresponding
    columns in DataFrame df2.
    If filler_columns are not provided, it is assumed that they are the same as
    target_columns.

    Parameters:
    - df (pd.DataFrame): The DataFrame with missing values to fill.
    - df2 (pd.DataFrame): The DataFrame providing the filling information.
    - target_columns (list of str): List of column names in df with missing values to be
    filled.
    - filler_columns (list of str, optional): List of column names in df2 that correspond
    to target_columns.
    If None, it's assumed to be the same as target_columns.

    Returns:
    pd.DataFrame: DataFrame with missing values filled.
    """
    if filler_columns is None:
        filler_columns = target_columns

    if len(target_columns) != len(filler_columns):
        raise ValueError(
            "The length of target_columns and filler_columns must be the same."
        )

    for target_col, filler_col in zip(target_columns, filler_columns):
        if target_col not in df.columns or filler_col not in df2.columns:
            raise ValueError(
                f"Column not found in respective DataFrame: {target_col} or {filler_col}"
            )

        df[target_col] = df.apply(
            lambda row: row[target_col]
            if pd.notna(row[target_col])
            else df2.loc[row.name, filler_col],
            axis=1,
        )

    return df
