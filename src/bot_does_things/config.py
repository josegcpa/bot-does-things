import os

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
CACHE_DIR = "./data/cache"
