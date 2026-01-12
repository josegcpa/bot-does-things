import hashlib
import mimetypes
import re
import urllib.parse
from pathlib import Path

import requests
from html_to_markdown import convert, convert_with_metadata
from langchain_community.utilities import GoogleSerperAPIWrapper

from .assertions import assert_int_ge, assert_non_empty_str, raise_for_status
from .config import DOWNLOAD_DIR, SERPPER_API_KEY, USER_AGENT


def web_search(query: str, max_results: int = 10) -> list[dict]:
    assert_non_empty_str(query, "query")
    assert_int_ge(max_results, "max_results", 1)
    if not SERPPER_API_KEY:
        raise RuntimeError("SERPER_API_KEY is not set")

    data = GoogleSerperAPIWrapper(k=max_results).results(query)
    organic = data.get("organic", []) or []
    if not isinstance(organic, list):
        raise RuntimeError(f"Unexpected Serper response shape: {data}")
    return organic


def search_web(
    query: str,
    max_results: int = 10,
    site: str | None = None,
    filetype: str | None = None,
    intitle: str | None = None,
    inurl: str | None = None,
) -> list[dict]:
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
    return web_search(q, max_results=max_results)


def download_file(
    url: str,
    dest_dir: str = DOWNLOAD_DIR,
    filename: str | None = None,
    overwrite: bool = False,
    timeout: int = 60,
) -> str:
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


def download_and_read_pdf(pdf_url: str) -> str:
    from .local_io import load_pdf

    assert_non_empty_str(pdf_url, "pdf_url")

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

    try:
        resp = requests.get(
            pdf_url,
            headers={"User-Agent": USER_AGENT},
            timeout=120,
            stream=True,
        )
    except requests.RequestException as e:
        raise RuntimeError(f"Error downloading URL {pdf_url}: {e}") from e
    raise_for_status(resp)

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


def fetch_url(
    url: str,
    timeout: int = 60,
    max_chars: int | None = None,
    headers: dict[str, str] | None = None,
) -> dict:
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


def extract_main_content(html: str) -> str:
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


def retrieve_webpage(
    url: str, only_text: bool = True, main_content: bool = True
) -> str:
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
