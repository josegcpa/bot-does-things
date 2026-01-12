import datetime
import hashlib
import json
from pathlib import Path

from bot_does_things.config import CACHE_DIR
from bot_does_things.assertions import assert_int_ge, assert_non_empty_str


def now() -> str:
    """
    Get the current UTC time.

    Returns:
        str: The current UTC time in ISO format.
    """
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def cache_set(
    key: str,
    value: str,
    ttl_seconds: int | None = None,
) -> str:
    assert_non_empty_str(key, "key")
    if not isinstance(value, str):
        raise ValueError("value must be a string")
    if ttl_seconds is not None:
        assert_int_ge(ttl_seconds, "ttl_seconds", 1)

    out_dir = Path(CACHE_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()
    p = out_dir / f"{digest}.json"

    expires_at: str | None = None
    if ttl_seconds is not None:
        expires = datetime.datetime.now(
            datetime.timezone.utc
        ) + datetime.timedelta(seconds=ttl_seconds)
        expires_at = expires.isoformat()

    payload = {"key": key, "value": value, "expires_at": expires_at}
    p.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    return str(p)


def cache_get(key: str) -> str | None:
    assert_non_empty_str(key, "key")

    p = (
        Path(CACHE_DIR)
        / f"{hashlib.sha256(key.encode('utf-8')).hexdigest()}.json"
    )
    if not p.exists():
        return None

    try:
        payload = json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None

    if not isinstance(payload, dict):
        return None

    expires_at = payload.get("expires_at")
    if isinstance(expires_at, str) and expires_at:
        try:
            exp = datetime.datetime.fromisoformat(expires_at)
        except ValueError:
            exp = None
        if exp is not None:
            if exp.tzinfo is None:
                exp = exp.replace(tzinfo=datetime.timezone.utc)
            if datetime.datetime.now(datetime.timezone.utc) >= exp:
                try:
                    p.unlink()
                except OSError:
                    pass
                return None

    val = payload.get("value")
    return val if isinstance(val, str) else None
