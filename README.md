# `bot_does_things`

A collection of useful Python functions that can be used as tools for LangChain/LangGraph agents. The focus of this package is on tools which can be run locally - all functionalities relating to running models locally (i.e. image interpretation) are set up to require a local LLM server like Ollama, but this can be altered to use other OpenAI API-compatible providers.

## Installation

This repo is set up as a package for `uv`.

```bash
uv sync
```

## Configuration

### Serper (web search)

`web_search()` uses Serper via LangChain’s `GoogleSerperAPIWrapper`. Requires setting the `SERPER_API_KEY` environment variable.

```bash
export SERPER_API_KEY="..."
```

### Ollama (image interpretation)

`interpret_image()` uses an OpenAI-compatible endpoint (defaults to Ollama).

```bash
export OLLAMA_BASE_URL="http://localhost:11434/v1"
export OLLAMA_IMAGE_INTERPRETATION_MODEL="gemma3:4b"
```

`OPENAI_API_KEY` is read but defaults to `ollama`.

### Download directory

Some tools save files under:

```bash
export DOWNLOAD_DIR = "./data/downloads"
```

## Available tools

All tools are plain Python callables and can be imported from `bot_does_things.tools`.

### Local I/O

**read_file(file_path: str) -> str**  
Read a file and return its contents as a string.  
```python
from bot_does_things.tools import read_file
text = read_file("./notes.txt")
```

**read_file_range(file_path: str, start_line: int = 1, end_line: int = 200, max_chars: int = 20000) -> str**  
Read a range of lines from a file and return them as a string.  
```python
from bot_does_things.tools import read_file_range
text = read_file_range("./notes.txt", 1, 50)
```

**read_json(file_path: str, max_chars: int = 20000) -> str**  
Read a JSON file and return its contents as a formatted string.  
```python
from bot_does_things.tools import read_json
data = read_json("./config.json")
```

**write_file(file_path: str, content: str) -> str**  
Write content to a file. Returns success message.  
```python
from bot_does_things.tools import write_file
result = write_file("./output.txt", "Hello world")
```

**read_table(file_path: str, max_rows: int = 25) -> str**  
Read a table file and return it as a string. Supports .xlsx, .csv, .tsv, .json, .parquet.  
```python
from bot_does_things.tools import read_table
table = read_table("./data.csv")
```

**list_files_tree(directory: str = ".", max_depth: int = 2, max_entries_per_dir: int = 10, show_hidden: bool = False) -> str**  
List files in a directory tree with configurable depth and limits.  
```python
from bot_does_things.tools import list_files_tree
tree = list_files_tree("./src", max_depth=3)
```

**search_files(directory: str = ".", file_regex: str = r".*", content_query: str | None = None, max_matches: int = 100, ignore_case: bool = True, max_file_size_bytes: int = 1000000, exclude_dirs: list[str] | None = None) -> list[str] | str**  
Search for files by name pattern and optionally by content.  
```python
from bot_does_things.tools import search_files
matches = search_files("./src", r"\.py$", "import")
```

**load_pdf(file_path: str) -> str**  
Load a PDF file and return its text content.  
```python
from bot_does_things.tools import load_pdf
text = load_pdf("./document.pdf")
```

### Web Tools

**search_web(query: str, max_results: int = 10, site: str | None = None, filetype: str | None = None, intitle: str | None = None, inurl: str | None = None) -> list[dict]**  
Search the web using Serper API with optional filters. Returns list of search results.  
```python
from bot_does_things.tools import search_web
results = search_web("Python tutorials", max_results=5)
```

**download_file(url: str, dest_dir: str = "./data/downloads", filename: str | None = None, overwrite: bool = False, timeout: int = 60) -> str**  
Download a file from URL and save to local directory. Returns path to downloaded file.  
```python
from bot_does_things.tools import download_file
path = download_file("https://example.com/file.pdf")
```

**fetch_url(url: str, timeout: int = 60, max_chars: int | None = None, headers: dict[str, str] | None = None) -> dict**  
Retrieve URL content with metadata. Returns dict with url, status_code, content_type, text, and truncated flag.  
```python
from bot_does_things.tools import fetch_url
data = fetch_url("https://api.example.com/data")
```

**extract_main_content(html: str) -> str**  
Extract main content from HTML string by finding <main>, <article>, or role="main" elements.  
```python
from bot_does_things.tools import extract_main_content
content = extract_main_content("<html><main>...</main></html>")
```

**retrieve_webpage(url: str, only_text: bool = True, main_content: bool = True) -> str**  
Retrieve webpage content and convert to text or markdown. Downloads images if only_text=False.  
```python
from bot_does_things.tools import retrieve_webpage
text = retrieve_webpage("https://example.com", only_text=True)
```

### Image Tools

**interpret_image(image_path: str, query: str) -> str**  
Use LLM to interpret an image and answer questions about it. Requires Ollama or OpenAI-compatible endpoint.  
```python
from bot_does_things.tools import interpret_image
answer = interpret_image("./image.png", "What is in this image?")
```

**ocr_image(image_path: str) -> str**  
Extract text from an image using OCR. Requires Pillow and pytesseract.  
```python
from bot_does_things.tools import ocr_image
text = ocr_image("./scan.png")
```

### General Tools

**now() -> str**  
Get current UTC time in ISO format.  
```python
from bot_does_things.tools import now
timestamp = now()
```

**cache_set(key: str, value: str, ttl_seconds: int | None = None) -> str**  
Store a value in local cache with optional TTL. Returns cache file path.  
```python
from bot_does_things.tools import cache_set
path = cache_set("mykey", "myvalue", 3600)
```

**cache_get(key: str) -> str | None**  
Retrieve a value from local cache. Returns None if not found or expired.  
```python
from bot_does_things.tools import cache_get
value = cache_get("mykey")
```

**calculator(expression: str) -> int | float**  
Evaluate mathematical expressions. Supports sin, cos, tan, asin, acos, atan, sinh, cosh, tanh, pi, e, inf, log, exp, sqrt, abs.  
```python
from bot_does_things.tools import calculator
result = calculator("sin(pi/2) + sqrt(4)")
```

**generate_random_integer(min: int, max: int) -> int**  
Generate random integer between min and max (inclusive).  
```python
from bot_does_things.tools import generate_random_integer
num = generate_random_integer(1, 100)
```

**generate_random_float(min: float, max: float) -> float**  
Generate random float between min and max (inclusive).  
```python
from bot_does_things.tools import generate_random_float
num = generate_random_float(0.0, 1.0)
```

## Quick sanity check

`bot_does_things.tools` contains a minimal `__main__` runner that calls each tool once.

```bash
uv run python -m bot_does_things.tools