"""
Includes a small bundle of helpful assertion functions.
"""

from pathlib import Path

import requests


def assert_non_empty_str(value: str, name: str) -> None:
    """
    Assert that a value is a non-empty string.

    Args:
        value (str): The value to check.
        name (str): The name of the value (for error messages).

    Raises:
        ValueError: If the value is not a non-empty string.
    """
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty string")


def assert_int_ge(value: int, name: str, min_value: int) -> None:
    """
    Assert that a value is an integer greater than or equal to a minimum value.

    Args:
        value (int): The value to check.
        name (str): The name of the value (for error messages).
        min_value (int): The minimum value.

    Raises:
        ValueError: If the value is not an integer or is less than the minimum
            value.
    """
    if not isinstance(value, int):
        raise ValueError(f"{name} must be an int")
    if value < min_value:
        raise ValueError(f"{name} must be >= {min_value}")


def assert_file_exists(path: str, name: str = "file_path") -> Path:
    """
    Assert that a file exists and is a file.

    Args:
        path (str): The path to the file.
        name (str): The name of the path (for error messages).

    Returns:
        Path: The path to the file.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the path is not a file.
    """
    assert_non_empty_str(path, name)
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"File not found: {path}")
    if not p.is_file():
        raise ValueError(f"Not a file: {path}")
    return p


def raise_for_status(resp: requests.Response) -> None:
    """
    Raise an exception if the response status code is not 200.

    Args:
        resp (requests.Response): The response to check.

    Raises:
        RuntimeError: If the response status code is not 200.
    """
    try:
        resp.raise_for_status()
    except requests.HTTPError as e:
        raise RuntimeError(
            f"Error accessing webpage {resp.url}: {resp.status_code}"
        ) from e
