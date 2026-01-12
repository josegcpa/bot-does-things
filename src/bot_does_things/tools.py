import logging
import os

from bot_does_things.config import (
    DOWNLOAD_DIR,
    LOGGING_LEVEL,
    OLLAMA_BASE_URL,
    SERPPER_API_KEY,
    USER_AGENT,
)
from bot_does_things.local_io import (
    list_files_tree,
    load_pdf,
    read_file,
    read_file_range,
    read_json,
    read_table,
    search_files,
    write_file,
)
from bot_does_things.web_tools import (
    download_file,
    extract_main_content,
    fetch_url,
    retrieve_webpage,
    search_web,
)
from bot_does_things.image_tools import interpret_image, ocr_image
from bot_does_things.general_tools import (
    now,
    cache_get,
    cache_set,
    calculator,
    generate_random_integer,
    generate_random_float,
)


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


TOOLS = {
    "read_file": read_file,
    "read_json": read_json,
    "write_file": write_file,
    "download_file": download_file,
    "list_files_tree": list_files_tree,
    "search_files": search_files,
    "interpret_image": interpret_image,
    "read_table": read_table,
    "load_pdf": load_pdf,
    "ocr_image": ocr_image,
    "now": now,
    "cache_get": cache_get,
    "cache_set": cache_set,
    "calculator": calculator,
    "generate_random_integer": generate_random_integer,
    "generate_random_float": generate_random_float,
    "search_web": search_web,
    "fetch_url": fetch_url,
    "extract_main_content": extract_main_content,
    "retrieve_webpage": retrieve_webpage,
    "read_file_range": read_file_range,
}

if __name__ == "__main__":

    import requests
    from pathlib import Path

    out_dir = Path(DOWNLOAD_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    sample_txt = out_dir / "_sample.txt"
    sample_txt.write_text("hello\n", encoding="utf-8")

    sample_csv = out_dir / "_sample.csv"
    sample_csv.write_text("a,b\n1,2\n", encoding="utf-8")

    sample_json = out_dir / "_sample.json"
    sample_json.write_text('{"a": 1, "b": [1, 2, 3]}\n', encoding="utf-8")

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

    sample_downloaded = out_dir / "_downloaded.png"
    if sample_downloaded.exists():
        os.remove(str(sample_downloaded))

    sample_written = out_dir / "_sample_written.txt"
    if sample_written.exists():
        os.remove(str(sample_written))

    test_args = {
        "read_file": [str(sample_txt)],
        "read_json": [str(sample_json)],
        "read_file_range": [str(sample_txt), 1, 5],
        "write_file": [str(out_dir / "_sample_written.txt"), "hello"],
        "download_file": [
            "https://www.w3.org/Graphics/PNG/nurbcup2si.png",
            str(out_dir),
            "_downloaded.png",
            True,
        ],
        "list_files_tree": [str(out_dir)],
        "search_files": [str(out_dir), r"_sample.*\.txt$", "hello"],
        "read_table": [str(sample_csv)],
        "load_pdf": [str(sample_pdf)],
        "ocr_image": [str(sample_png)],
        "now": [],
        "cache_set": ["k", "v", 60],
        "cache_get": ["k"],
        "calculator": ["2 + 2"],
        "generate_random_integer": [1, 10],
        "generate_random_float": [0.0, 1.0],
        "search_web": ["Example Domain"],
        "fetch_url": ["https://www.wikipedia.org/"],
        "extract_main_content": [
            "<html><body><main><p>Hi</p></main></body></html>"
        ],
        "retrieve_webpage": ["https://www.wikipedia.org/"],
        "interpret_image": (str(sample_png), "What is in this image?"),
    }

    for name, fn in TOOLS.items():
        try:
            out = fn(*test_args[name])
            preview = str(out)
            print(f"{name}: OK ({preview[:200]})")
        except Exception as e:
            print(f"{name}: ERROR ({e})")
