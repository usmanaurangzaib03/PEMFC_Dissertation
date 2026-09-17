"""
Utility functions for the PEMFC degradation prediction project.

This module contains reusable helper functions for:

- creating output directories;
- validating file and directory paths;
- producing safe filenames;
- saving pandas DataFrames;
- reporting saved outputs.

Keeping these operations in one module avoids repeating the same code
throughout the project notebooks.
"""

from pathlib import Path
import re

import pandas as pd


def ensure_directory(directory_path):
    """
    Create a directory when it does not already exist.

    Parameters
    ----------
    directory_path : str or pathlib.Path
        Directory to create.

    Returns
    -------
    pathlib.Path
        Validated directory path.
    """

    directory = Path(directory_path)

    directory.mkdir(
        parents=True,
        exist_ok=True
    )

    return directory


def validate_file(file_path):
    """
    Confirm that a required file exists.

    Parameters
    ----------
    file_path : str or pathlib.Path
        File path to validate.

    Returns
    -------
    pathlib.Path
        Validated file path.

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    """

    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(
            f"Required file was not found:\n{file_path}"
        )

    if not file_path.is_file():
        raise ValueError(
            f"The supplied path is not a file:\n{file_path}"
        )

    return file_path


def validate_directory(directory_path):
    """
    Confirm that a required directory exists.

    Parameters
    ----------
    directory_path : str or pathlib.Path
        Directory path to validate.

    Returns
    -------
    pathlib.Path
        Validated directory path.

    Raises
    ------
    FileNotFoundError
        If the directory does not exist.
    """

    directory = Path(directory_path)

    if not directory.exists():
        raise FileNotFoundError(
            f"Required directory was not found:\n{directory}"
        )

    if not directory.is_dir():
        raise NotADirectoryError(
            f"The supplied path is not a directory:\n{directory}"
        )

    return directory


def create_safe_filename(text):
    """
    Convert text into a safe filename.

    Spaces and unsupported characters are replaced with underscores.

    Parameters
    ----------
    text : str
        Text to convert.

    Returns
    -------
    str
        Filesystem-safe filename.
    """

    safe_text = re.sub(
        pattern=r"[^A-Za-z0-9_.-]+",
        repl="_",
        string=str(text).strip()
    )

    safe_text = safe_text.strip("_")

    return safe_text or "output"


def save_dataframe(
    dataframe,
    output_path,
    index=False
):
    """
    Save a pandas DataFrame as a CSV file.

    The parent directory is created automatically when necessary.

    Parameters
    ----------
    dataframe : pandas.DataFrame
        DataFrame to save.

    output_path : str or pathlib.Path
        Destination CSV path.

    index : bool, default=False
        Whether to include the DataFrame index.

    Returns
    -------
    pathlib.Path
        Final saved file path.

    Raises
    ------
    TypeError
        If the supplied object is not a pandas DataFrame.
    """

    if not isinstance(dataframe, pd.DataFrame):
        raise TypeError(
            "The supplied object must be a pandas DataFrame."
        )

    output_path = Path(output_path)

    ensure_directory(output_path.parent)

    dataframe.to_csv(
        output_path,
        index=index
    )

    return output_path


def format_file_size(size_bytes):
    """
    Convert a file size in bytes into a readable unit.

    Parameters
    ----------
    size_bytes : int or float
        File size in bytes.

    Returns
    -------
    str
        Human-readable file size.
    """

    size_bytes = float(size_bytes)

    units = [
        "bytes",
        "KB",
        "MB",
        "GB",
        "TB"
    ]

    unit_index = 0

    while size_bytes >= 1024 and unit_index < len(units) - 1:
        size_bytes /= 1024
        unit_index += 1

    return f"{size_bytes:.2f} {units[unit_index]}"


def report_saved_file(file_path):
    """
    Print confirmation information for a saved file.

    Parameters
    ----------
    file_path : str or pathlib.Path
        Path of the saved file.
    """

    file_path = validate_file(file_path)

    file_size = format_file_size(
        file_path.stat().st_size
    )

    print("File saved successfully.")
    print(f"Location: {file_path}")
    print(f"File size: {file_size}")