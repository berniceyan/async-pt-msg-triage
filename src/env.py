"""Read `.env` into the process. One reader, imported by every entry point.
"""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ENV_PATH = ROOT / ".env"

KEY_VARS = ("OPENROUTER_API_KEY", "DISPO_API_KEY")


def load_env(path: Path = ENV_PATH) -> None:
    """Set each `.env` variable that is not already in the environment.

    setdefault, so an exported value always wins over the file. Missing file
    is fine: --replay and --no-model paths need no key and run without one.
    """
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def api_key() -> str | None:
    """The key judge() will use, or None. Reads the environment only."""
    for var in KEY_VARS:
        if os.environ.get(var):
            return os.environ[var]
    return None


def require_key() -> str:
    key = api_key()
    if not key:
        raise RuntimeError(
            "no API key: set OPENROUTER_API_KEY in .env (copy .env.example) "
            "or export it. A run without a key cannot judge anything, and "
            "would record every uncached message as MODEL_FAILURE.")
    return key
