"""
Includes tools to interact with the web.
"""

import hashlib
import mimetypes
import re
import urllib.parse
from pathlib import Path

import requests
from html_to_markdown import convert, convert_with_metadata
from langchain_community.utilities import GoogleSerperAPIWrapper

from bot_does_things.assertions import (
    assert_int_ge,
    assert_non_empty_str,
    raise_for_status,
)
from bot_does_things.config import (
    DOWNLOAD_DIR,
    SERPPER_API_KEY,
    USER_AGENT,
)
from bot_does_things.tool_wrapper import tool_wrapper


def _web_search(query: str, max_results: int = 10) -> list[dict]:
    """
    Search the web using Serper API.

    Args:
        query (str): The search query.
        max_results (int): The maximum number of results to return. Defaults to 10.

    Returns:
        list[dict]: The search results.
    """
    assert_non_empty_str(query, "query")
    assert_int_ge(max_results, "max_results", 1)
    if not SERPPER_API_KEY:
        raise RuntimeError("SERPER_API_KEY is not set")

    data = GoogleSerperAPIWrapper(k=max_results).results(query)
    organic = data.get("organic", []) or []
    if not isinstance(organic, list):
        raise RuntimeError(f"Unexpected Serper response shape: {data}")
    return organic


@tool_wrapper
def search_web(
    query: str,
    max_results: int = 10,
    site: str | None = None,
    filetype: str | None = None,
    intitle: str | None = None,
    inurl: str | None = None,
) -> list[dict]:
    """
    Search the web using Serper API with optional filters.

    Args:
        query (str): The search query.
        max_results (int): The maximum number of results to return. Defaults to 10.
        site (str | None): The site to search in. Defaults to None.
        filetype (str | None): The file type to search for. Defaults to None.
        intitle (str | None): The title to search for. Defaults to None.
        inurl (str | None): The URL to search for. Defaults to None.

    Returns:
        list[dict]: The search results.
    """
    assert_non_empty_str(query, "query")

    q = query
    if site:
        q = f"site:{site} {q}"
    if filetype:
        q = f"filetype:{filetype} {q}"
    if intitle:
        q = f"intitle:{intitle} {q}"
    if inurl:
        q = f"inurl:{inurl} {q}"
    return _web_search(q, max_results=max_results)


@tool_wrapper
def download_file(
    url: str,
    dest_dir: str = DOWNLOAD_DIR,
    filename: str | None = None,
    overwrite: bool = False,
    timeout: int = 60,
) -> str:
    """
    Download a file from a URL and save it to a local directory.

    Args:
        url (str): The URL to download from.
        dest_dir (str): The directory to save the file to. Defaults to DOWNLOAD_DIR.
        filename (str | None): The name to save the file as. Defaults to None.
        overwrite (bool): Whether to overwrite an existing file. Defaults to False.
        timeout (int): The timeout for the request. Defaults to 60.

    Returns:
        str: The path to the downloaded file.
    """
    assert_non_empty_str(url, "url")
    assert_non_empty_str(dest_dir, "dest_dir")
    assert_int_ge(timeout, "timeout", 1)

    out_dir = Path(dest_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        resp = requests.get(
            url,
            headers={"User-Agent": USER_AGENT},
            timeout=timeout,
            stream=True,
        )
    except requests.RequestException as e:
        raise RuntimeError(f"Error downloading URL {url}: {e}") from e
    raise_for_status(resp)

    chosen_name = filename
    if not chosen_name:
        cd = resp.headers.get("Content-Disposition", "")
        m = re.search(r"filename\*?=(?:UTF-8''|\")?([^;\"\n]+)", cd)
        if m:
            chosen_name = m.group(1).strip().strip('"')
        if not chosen_name:
            chosen_name = Path(urllib.parse.urlparse(url).path).name

    if not chosen_name:
        digest = hashlib.sha256(url.encode("utf-8")).hexdigest()[:16]
        chosen_name = f"download_{digest}"

    chosen_name = Path(chosen_name).name

    if "." not in chosen_name:
        content_type = (
            resp.headers.get("Content-Type", "").split(";")[0].strip()
        )
        ext = mimetypes.guess_extension(content_type) if content_type else None
        if ext:
            chosen_name = f"{chosen_name}{ext}"

    dest = out_dir / chosen_name
    if dest.exists() and not overwrite:
        raise FileExistsError(f"File already exists: {dest}")

    with dest.open("wb") as f:
        for chunk in resp.iter_content(chunk_size=1024 * 64):
            if chunk:
                f.write(chunk)

    return str(dest)


@tool_wrapper
def fetch_url(
    url: str,
    timeout: int = 60,
    max_chars: int | None = None,
    headers: dict[str, str] | None = None,
) -> dict:
    """
    Retrieves the content of a URL.

    Args:
        url (str): The URL to retrieve.
        timeout (int): The timeout for the request. Defaults to 60.
        max_chars (int | None): The maximum number of characters to return.
            Defaults to None.
        headers (dict[str, str] | None): The headers to send with the request.
            Defaults to None.

    Returns:
        dict: The content of the URL.
    """
    assert_non_empty_str(url, "url")
    assert_int_ge(timeout, "timeout", 1)
    if max_chars is not None:
        assert_int_ge(max_chars, "max_chars", 1)

    req_headers = {"User-Agent": USER_AGENT}
    if headers:
        req_headers.update(headers)

    try:
        resp = requests.get(url, headers=req_headers, timeout=timeout)
    except requests.RequestException as e:
        raise RuntimeError(f"Error fetching URL {url}: {e}") from e

    raise_for_status(resp)

    if not resp.encoding:
        resp.encoding = resp.apparent_encoding or "utf-8"

    text = resp.text
    truncated = False
    if max_chars is not None and len(text) > max_chars:
        text = text[:max_chars]
        truncated = True

    return {
        "url": resp.url,
        "status_code": resp.status_code,
        "content_type": resp.headers.get("Content-Type", ""),
        "text": text,
        "truncated": truncated,
    }


@tool_wrapper
def extract_main_content(html: str) -> str:
    """
    Extracts the main content from an HTML string.

    Args:
        html (str): The HTML string to extract the main content from.

    Returns:
        str: The main content of the HTML string.
    """
    assert_non_empty_str(html, "html")

    for pat in [
        r"<main\b[^>]*>(?P<content>.*?)</main>",
        r"<article\b[^>]*>(?P<content>.*?)</article>",
        r"<[^>]+\brole=['\"]main['\"][^>]*>(?P<content>.*?)</[^>]+>",
    ]:
        m = re.search(pat, html, flags=re.IGNORECASE | re.DOTALL)
        if m:
            return m.group("content")

    body_match = re.search(
        r"<body\b[^>]*>(?P<content>.*?)</body>",
        html,
        flags=re.IGNORECASE | re.DOTALL,
    )
    content = body_match.group("content") if body_match else html

    content = re.sub(
        r"<(script|style|noscript)\b[^>]*>.*?</\1>",
        "",
        content,
        flags=re.IGNORECASE | re.DOTALL,
    )
    content = re.sub(
        r"<(header|footer|nav|aside)\b[^>]*>.*?</\1>",
        "",
        content,
        flags=re.IGNORECASE | re.DOTALL,
    )
    return content


@tool_wrapper
def retrieve_webpage(
    url: str, only_text: bool = True, main_content: bool = True
) -> str:
    """
    Retrieves the content of a URL and returns it as a string.

    Args:
        url (str): The URL to retrieve.
        only_text (bool): Whether to return only the text content.
            Defaults to True.
        main_content (bool): Whether to return only the main content.
            Defaults to True.

    Returns:
        str: The content of the URL.
    """
    fetched = fetch_url(url, timeout=60)
    html = str(fetched.get("text", ""))
    if main_content:
        html = extract_main_content(html)
    if len(html) > 400000:
        html = html[:400000]

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
            raise_for_status(img_resp)
        except (requests.RequestException, RuntimeError):
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
