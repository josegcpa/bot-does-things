import base64
import hashlib
import mimetypes
import os
import re
import urllib.parse
import logging
from pathlib import Path
from typing import Iterable

import requests
from langchain_community.chat_models import ChatOpenAI
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.utilities import GoogleSerperAPIWrapper
from langchain_core.messages import HumanMessage
from html_to_markdown import convert_with_metadata, convert


OLLAMA_BASE_URL = os.environ.get(
    "OLLAMA_BASE_URL", "http://localhost:11434/v1"
).rstrip("/")
USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/120.0.0.0 Safari/537.36"
)
DOWNLOAD_DIR = "./data/downloads"
SERPPER_API_KEY = os.environ.get("SERPER_API_KEY", None)
LOGGING_LEVEL = os.environ.get("LOGGING_LEVEL", "INFO")

logger = logging.getLogger(__name__)
logger.handlers = []
_handler = logging.StreamHandler()
_handler.setFormatter(logging.Formatter("%(levelname)s:%(name)s:%(message)s"))
logger.addHandler(_handler)
logger.setLevel(LOGGING_LEVEL)

env_vars = {
    "OLLAMA_BASE_URL": OLLAMA_BASE_URL,
    "USER_AGENT": USER_AGENT,
    "DOWNLOAD_DIR": DOWNLOAD_DIR,
    "LOGGING_LEVEL": LOGGING_LEVEL,
}

for key, value in env_vars.items():
    logger.info(f"{key}: {value}")

if not SERPPER_API_KEY:
    logger.warning(
        "SERPER_API_KEY environment variable is not set but is required for web search"
    )


def _safe_raise_for_status(resp: requests.Response) -> str:
    try:
        resp.raise_for_status()
    except:
        return f"Error accessing webpage {resp.url}: {resp.status_code}"


def read_file(file_path: str) -> str:
    """
    Reads a text file and returns its contents.

    Args:
        file_path (str): The path to the file to read.

    Returns:
        str: The contents of the file.
    """
    with open(file_path, "r") as f:
        return f.read()


def write_file(file_path: str, content: str) -> str:
    """
    Writes content to a text file.

    Args:
        file_path (str): The path to the file to write.
        content (str): The content to write to the file.

    Returns:
        str: A message indicating the result of the operation.
    """
    if os.path.exists(file_path):
        raise FileExistsError(f"File {file_path} already exists.")
    with open(file_path, "w") as f:
        f.write(content)
    return f"File {file_path} written successfully."


def list_files(directory: str = ".", max_depth: int = 2) -> str:
    """
    Lists files in a directory using a simple tree structure.

    If a directory has more than 10 entries, it shows the first 10 (sorted) and
    summarizes the remainder.

    Args:
        directory: Directory to list.
        max_depth: Maximum recursion depth.

    Returns:
        A tree-like string.
    """
    root = Path(directory)
    if not root.exists():
        raise FileNotFoundError(f"Directory not found: {directory}")
    if not root.is_dir():
        raise NotADirectoryError(f"Not a directory: {directory}")

    def _iter_entries(p: Path) -> list[Path]:
        entries = list(p.iterdir())
        entries.sort(key=lambda x: (not x.is_dir(), x.name.lower()))
        return entries

    def _tree_lines(p: Path, prefix: str, depth: int) -> Iterable[str]:
        if depth > max_depth:
            return

        entries = _iter_entries(p)
        shown = entries[:10]
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


def search_files(directory: str = ".", regex: str = r"\.csv$") -> list[str]:
    """
    Search files by matching a regex against the filename.

    This is intended for filtering by termination/extension.

    Args:
        directory: Directory root to search.
        regex: Regex applied to filenames (e.g. r"\.md$", r"\.csv$").

    Returns:
        List of matching file paths.
    """
    root = Path(directory)
    if not root.exists():
        raise FileNotFoundError(f"Directory not found: {directory}")

    pattern = re.compile(regex)
    matches: list[str] = []
    for dirpath, _dirnames, filenames in os.walk(root):
        for fn in filenames:
            if pattern.search(fn):
                matches.append(str(Path(dirpath) / fn))
    matches.sort()
    return matches


def interpret_image(image_path: str, query: str) -> str:
    """
    Uses Gemma3 to interpret an image and answer a question about it.

    Args:
        image_path (str): The path to the image to interpret.
        query (str): The question to answer about the image.

    Returns:
        str: The answer to the question.
    """
    base_url = OLLAMA_BASE_URL
    model = os.environ.get("OLLAMA_IMAGE_INTERPRETATION_MODEL", "gemma3:4b")

    mime_type, _ = mimetypes.guess_type(image_path)
    if not mime_type:
        mime_type = "application/octet-stream"

    with open(image_path, "rb") as f:
        image_b64 = base64.b64encode(f.read()).decode("utf-8")

    llm = ChatOpenAI(
        model=model,
        openai_api_base=base_url,
        openai_api_key=os.environ.get("OPENAI_API_KEY", "ollama"),
    )

    msg = HumanMessage(
        content=[
            {"type": "text", "text": query},
            {
                "type": "image_url",
                "image_url": {"url": f"data:{mime_type};base64,{image_b64}"},
            },
        ]
    )

    result = llm.invoke([msg])
    return result.content


def read_table(file_path: str, max_rows: int = 25) -> str:
    """
    Reads a table from a file and returns its contents.

    Args:
        file_path (str): The path to the file to read.
        max_rows (int, optional): The maximum number of rows to return.
            Defaults to 25.

    Returns:
        str: The contents of the table.
    """
    import pandas as pd

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


def download_and_read_pdf(pdf_url: str) -> str:
    """
    Downloads a PDF from a URL and returns its extracted text.

    Args:
        pdf_url (str): The URL of the PDF to download.

    Returns:
        str: The extracted text from the PDF.
    """

    if not isinstance(pdf_url, str) or not pdf_url:
        raise ValueError("pdf_url must be a non-empty string")

    out_dir = Path(DOWNLOAD_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    parsed = urllib.parse.urlparse(pdf_url)
    filename = Path(parsed.path).name
    if not filename:
        digest = hashlib.sha256(pdf_url.encode("utf-8")).hexdigest()[:16]
        filename = f"download_{digest}.pdf"
    if not filename.lower().endswith(".pdf"):
        filename = f"{filename}.pdf"
    dest = out_dir / filename

    resp = requests.get(
        pdf_url,
        headers={"User-Agent": USER_AGENT},
        timeout=120,
        stream=True,
    )
    if error := _safe_raise_for_status(resp):
        return error

    content_type = resp.headers.get("Content-Type", "")
    if "pdf" not in content_type.lower() and not pdf_url.lower().endswith(
        ".pdf"
    ):
        raise RuntimeError(
            f"URL did not look like a PDF (Content-Type={content_type}): {pdf_url}"
        )

    with dest.open("wb") as f:
        for chunk in resp.iter_content(chunk_size=1024 * 64):
            if chunk:
                f.write(chunk)

    return load_pdf(str(dest))


def load_pdf(file_path: str) -> str:
    """
    Loads a PDF file and returns its extracted text.

    Args:
        file_path: Path to a local PDF file.

    Returns:
        Extracted text content of the PDF.
    """
    if not file_path.lower().endswith(".pdf"):
        raise ValueError("file_path must point to a .pdf file")

    docs = PyPDFLoader(file_path).load()
    return "\n\n".join(d.page_content for d in docs if d.page_content)


def web_search(query: str, max_results: int = 10) -> list[dict]:
    """
    Searches the web using Serper and returns structured results.

    Args:
        query: The search query.
        max_results: Maximum number of results to request.

    Returns:
        A list of result dicts (standard Serper organic results entries).
    """
    if not SERPPER_API_KEY:
        raise RuntimeError("SERPER_API_KEY is not set")
    data = GoogleSerperAPIWrapper(k=max_results).results(query)
    organic = data.get("organic", []) or []
    if not isinstance(organic, list):
        raise RuntimeError(f"Unexpected Serper response shape: {data}")
    return organic


def retrieve_webpage(url: str, only_text: bool = True) -> str:
    """
    Fetches a webpage and converts it to Markdown.

    Images are downloaded and stored in DOWNLOAD_DIR and image references are
    rewritten to the local downloaded paths. If only_text is True, the function
    returns only the plain text content. This is helpful if no formatting, images
    or further navigation is needed.

    Args:
        url: Webpage URL.
        only_text: If True, return only plain text (no links/images/etc.).

    Returns:
        Markdown content.
    """
    resp = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=60)
    if error := _safe_raise_for_status(resp):
        return error
    html = resp.text

    # extract main content to reduce noise to LLM
    main_match = re.search(
        r"<main\b[^>]*>(?P<content>.*?)</main>",
        html,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if main_match:
        html = main_match.group("content")

    if only_text:
        markdown = convert(html)
        text = markdown
        text = re.sub(r"!\[[^\]]*\]\([^)]*\)", "", text)
        text = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", text)
        text = re.sub(r"\[[^\]]+\]:\s*\S+.*", "", text)
        text = re.sub(r"`{3}.*?`{3}", "", text, flags=re.DOTALL)
        text = re.sub(r"`([^`]+)`", r"\1", text)
        text = re.sub(r"^\s{0,3}#{1,6}\s*", "", text, flags=re.MULTILINE)
        text = re.sub(r"^\s*([-*+]\s+|\d+\.\s+)", "", text, flags=re.MULTILINE)
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()

    markdown, metadata = convert_with_metadata(html)

    out_dir = Path(DOWNLOAD_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    def _download_image(img_url: str) -> str | None:
        try:
            img_resp = requests.get(
                img_url,
                headers={"User-Agent": USER_AGENT},
                timeout=60,
                stream=True,
            )
            if error := _safe_raise_for_status(img_resp):
                return None
        except requests.RequestException:
            return None

        content_type = img_resp.headers.get("Content-Type", "")
        ext = mimetypes.guess_extension(content_type.split(";")[0].strip())
        if not ext:
            ext = Path(urllib.parse.urlparse(img_url).path).suffix or ".bin"

        digest = hashlib.sha256(img_url.encode("utf-8")).hexdigest()[:16]
        filename = f"image_{digest}{ext}"
        dest = out_dir / filename
        if not dest.exists():
            with dest.open("wb") as f:
                for chunk in img_resp.iter_content(chunk_size=1024 * 64):
                    if chunk:
                        f.write(chunk)
        return str(dest)

    images = metadata.get("images", []) if isinstance(metadata, dict) else []
    if isinstance(images, list):
        for img in images:
            if not isinstance(img, dict):
                continue
            src = img.get("src") or img.get("url")
            if not src or not isinstance(src, str):
                continue
            if src.startswith("data:"):
                continue

            abs_src = urllib.parse.urljoin(url, src)
            local_path = _download_image(abs_src)
            if not local_path:
                continue

            markdown = markdown.replace(f"({src})", f"({local_path})")

    return markdown


TOOLS = {
    "read_file": read_file,
    "write_file": write_file,
    "list_files": list_files,
    "search_files": search_files,
    "interpret_image": interpret_image,
    "read_table": read_table,
    "load_pdf": load_pdf,
    "web_search": web_search,
    "retrieve_webpage": retrieve_webpage,
}

if __name__ == "__main__":

    out_dir = Path(DOWNLOAD_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    sample_txt = out_dir / "_sample.txt"
    sample_txt.write_text("hello\n", encoding="utf-8")

    sample_csv = out_dir / "_sample.csv"
    sample_csv.write_text("a,b\n1,2\n", encoding="utf-8")

    sample_pdf = out_dir / "_sample.pdf"
    if not sample_pdf.exists():
        r = requests.get(
            "https://www.w3.org/WAI/ER/tests/xhtml/testfiles/resources/pdf/dummy.pdf",
            headers={"User-Agent": USER_AGENT},
            timeout=60,
        )
        r.raise_for_status()
        sample_pdf.write_bytes(r.content)

    sample_png = out_dir / "_sample.png"
    if not sample_png.exists():
        r = requests.get(
            "https://www.w3.org/Graphics/PNG/nurbcup2si.png",
            headers={"User-Agent": USER_AGENT},
            timeout=60,
        )
        r.raise_for_status()
        sample_png.write_bytes(r.content)

    sample_written = out_dir / "_sample_written.txt"
    if sample_written.exists():
        os.remove(str(sample_written))

    test_args = {
        "read_file": [str(sample_txt)],
        "write_file": [str(out_dir / "_sample_written.txt"), "hello"],
        "list_files": [str(out_dir)],
        "search_files": [str(out_dir), r"\.csv$"],
        "read_table": [str(sample_csv)],
        "load_pdf": [str(sample_pdf)],
        "web_search": ["Example Domain"],
        "retrieve_webpage": ["https://www.wikipedia.org/"],
        "interpret_image": (str(sample_png), "What is in this image?"),
    }

    for name, fn in TOOLS.items():
        out = fn(*test_args[name])

        preview = str(out)
        print(f"{name}: OK ({preview[:200]})")
