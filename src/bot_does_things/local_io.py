import json
import os
import re
from pathlib import Path
from typing import Iterable
import pandas as pd
from langchain_community.document_loaders import PyPDFLoader

from .assertions import (
    assert_file_exists,
    assert_int_ge,
    assert_non_empty_str,
)

DOWNLOAD_DIR = "./data/downloads"


def read_file(file_path: str) -> str:
    """
    Read a file and return its contents as a string.

    Args:
        file_path (str): The path to the file.

    Returns:
        str: The contents of the file.
    """
    assert_non_empty_str(file_path, "file_path")
    with open(file_path, "r", encoding="utf-8", errors="replace") as f:
        return f.read()


def read_file_range(
    file_path: str,
    start_line: int = 1,
    end_line: int = 200,
    max_chars: int = 20000,
) -> str:
    """
    Read a range of lines from a file and return them as a string.

    Args:
        file_path (str): The path to the file.
        start_line (int): The line number to start reading from (1-indexed).
        end_line (int): The line number to stop reading at (1-indexed).
        max_chars (int): The maximum number of characters to return.

    Returns:
        str: The contents of the file.
    """
    assert_non_empty_str(file_path, "file_path")
    assert_int_ge(start_line, "start_line", 1)
    assert_int_ge(end_line, "end_line", start_line)
    assert_int_ge(max_chars, "max_chars", 1)

    out_lines: list[str] = []
    out_len = 0
    with open(file_path, "r", encoding="utf-8", errors="replace") as f:
        for i, line in enumerate(f, start=1):
            if i < start_line:
                continue
            if i > end_line:
                break
            if out_len + len(line) > max_chars:
                remaining = max_chars - out_len
                if remaining > 0:
                    out_lines.append(line[:remaining])
                break
            out_lines.append(line)
            out_len += len(line)

    return "".join(out_lines)


def read_json(file_path: str, max_chars: int = 20000) -> str:
    """
    Read a JSON file and return its contents as a string.

    Args:
        file_path (str): The path to the JSON file.
        max_chars (int): The maximum number of characters to return.

    Returns:
        str: The contents of the JSON file.
    """
    assert_int_ge(max_chars, "max_chars", 1)
    p = assert_file_exists(file_path)

    with p.open("r", encoding="utf-8") as f:
        data = json.load(f)

    text = json.dumps(data, indent=2, ensure_ascii=False)
    if len(text) > max_chars:
        text = text[:max_chars] + "\n... (truncated)"
    return text


def write_file(file_path: str, content: str) -> str:
    """
    Write content to a file.

    Args:
        file_path (str): The path to the file.
        content (str): The content to write to the file.

    Returns:
        str: A message indicating the file was written successfully.
    """
    assert_non_empty_str(file_path, "file_path")
    assert_non_empty_str(content, "content")
    if os.path.exists(file_path):
        raise FileExistsError(f"File {file_path} already exists.")
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(content)
    return f"File {file_path} written successfully."


def read_table(file_path: str, max_rows: int = 25) -> str:
    """
    Read a table from a file and return it as a string.

    Args:
        file_path (str): The path to the file.
        max_rows (int): The maximum number of rows to return.

    Returns:
        str: The contents of the table.
    """

    assert_non_empty_str(file_path, "file_path")
    assert_int_ge(max_rows, "max_rows", 1)

    if file_path.endswith(".xlsx"):
        df = pd.read_excel(file_path)
    elif file_path.endswith(".csv"):
        df = pd.read_csv(file_path)
    elif file_path.endswith(".tsv"):
        df = pd.read_table(file_path, sep="\t")
    elif file_path.endswith(".json"):
        df = pd.read_json(file_path)
    elif file_path.endswith(".parquet"):
        df = pd.read_parquet(file_path)
    else:
        raise ValueError("File must be either an Excel or CSV file.")
    return df.to_string(max_rows=max_rows)


def list_files_tree(
    directory: str = ".",
    max_depth: int = 2,
    max_entries_per_dir: int = 10,
    show_hidden: bool = False,
) -> str:
    """
    List files in a directory tree.

    Args:
        directory (str): The directory to list files in.
        max_depth (int): The maximum depth to traverse.
        max_entries_per_dir (int): The maximum number of entries to show per directory.
        show_hidden (bool): Whether to show hidden files.

    Returns:
        str: A string representation of the files in the directory.
    """
    assert_non_empty_str(directory, "directory")
    if max_depth < 0:
        raise ValueError("max_depth must be >= 0")
    assert_int_ge(max_entries_per_dir, "max_entries_per_dir", 1)

    root = Path(directory)
    if not root.exists():
        raise FileNotFoundError(f"Directory not found: {directory}")
    if not root.is_dir():
        raise NotADirectoryError(f"Not a directory: {directory}")

    def _iter_entries(p: Path) -> list[Path]:
        entries = list(p.iterdir())
        if not show_hidden:
            entries = [e for e in entries if not e.name.startswith(".")]
        entries.sort(key=lambda x: (not x.is_dir(), x.name.lower()))
        return entries

    def _tree_lines(p: Path, prefix: str, depth: int) -> Iterable[str]:
        if depth > max_depth:
            return

        entries = _iter_entries(p)
        shown = entries[:max_entries_per_dir]
        remaining = len(entries) - len(shown)

        for i, child in enumerate(shown):
            is_last = (i == len(shown) - 1) and remaining == 0
            branch = "`-- " if is_last else "|-- "
            yield f"{prefix}{branch}{child.name}{'/' if child.is_dir() else ''}"

            if child.is_dir() and depth < max_depth:
                extension = "    " if is_last else "|   "
                yield from _tree_lines(child, prefix + extension, depth + 1)

        if remaining > 0:
            yield f"{prefix}|-- ... (+{remaining} more)"

    lines = [str(root)]
    lines.extend(_tree_lines(root, prefix="", depth=1) or [])
    return "\n".join(lines)


def search_files(
    directory: str = ".",
    file_regex: str = r".*",
    content_query: str | None = None,
    max_matches: int = 100,
    ignore_case: bool = True,
    max_file_size_bytes: int = 1_000_000,
    exclude_dirs: list[str] | None = None,
) -> list[str] | str:
    assert_non_empty_str(directory, "directory")
    assert_non_empty_str(file_regex, "file_regex")
    if content_query is not None:
        assert_non_empty_str(content_query, "content_query")
        assert_int_ge(max_matches, "max_matches", 1)
        assert_int_ge(max_file_size_bytes, "max_file_size_bytes", 1)

    root = Path(directory)
    if not root.exists():
        raise FileNotFoundError(f"Directory not found: {directory}")

    file_pattern = re.compile(file_regex)
    exclude = set(
        exclude_dirs or [".git", ".venv", "__pycache__", ".mypy_cache"]
    )

    if content_query is None:
        paths: list[str] = []
        for dirpath, _dirnames, filenames in os.walk(root):
            for fn in filenames:
                if file_pattern.search(fn):
                    paths.append(str(Path(dirpath) / fn))
        paths.sort()
        return paths

    flags = re.IGNORECASE if ignore_case else 0
    pattern = re.compile(content_query, flags)
    matches: list[str] = []
    truncated = False

    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in exclude]
        for fn in filenames:
            if not file_pattern.search(fn):
                continue
            path = Path(dirpath) / fn
            try:
                if path.stat().st_size > max_file_size_bytes:
                    continue
            except OSError:
                continue

            try:
                with open(path, "r", encoding="utf-8", errors="replace") as f:
                    for line_no, line in enumerate(f, start=1):
                        if pattern.search(line):
                            matches.append(f"{path}:{line_no}: {line.rstrip()}")
                            if len(matches) >= max_matches:
                                truncated = True
                                break
                if truncated:
                    break
            except (OSError, UnicodeError):
                continue
        if truncated:
            break

    if not matches:
        return ""
    if truncated:
        return "\n".join(matches) + "\n... (truncated)"
    return "\n".join(matches)


def load_pdf(file_path: str) -> str:
    assert_non_empty_str(file_path, "file_path")
    if not file_path.lower().endswith(".pdf"):
        raise ValueError("file_path must point to a .pdf file")

    docs = PyPDFLoader(file_path).load()
    return "\n\n".join(d.page_content for d in docs if d.page_content)
