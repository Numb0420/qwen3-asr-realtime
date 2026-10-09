"""Repo root vs green-pack root. No logger/config imports (loaded very early)."""
from __future__ import annotations

import os
import sys
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parent
_CODE_ROOT = SRC_DIR.parent  # repo root, or app/ inside a green pack


def is_green_pack() -> bool:
    """True when running copied sources under app/ next to runtime/python.exe."""
    pack = _CODE_ROOT.parent
    return (_CODE_ROOT / "run.py").exists() and (pack / "runtime" / "python.exe").is_file()


def is_frozen() -> bool:
    return bool(getattr(sys, "frozen", False))


def pack_or_repo_root() -> Path:
    """Directory that contains .env, models/, logs/."""
    if is_frozen():
        return Path(sys.executable).resolve().parent
    if is_green_pack():
        return _CODE_ROOT.parent
    return _CODE_ROOT


ROOT_DIR = pack_or_repo_root()
# vendor/ always sits next to src/ (repo/vendor or pack/app/vendor).
VENDOR_DIR = _CODE_ROOT / "vendor"


def default_log_dir() -> Path:
    return ROOT_DIR / "logs"


def resolve_log_dir() -> Path:
    """Directory for log files. Honors LOG_DIR (absolute or relative to ROOT_DIR)."""
    raw = os.getenv("LOG_DIR", "").strip()
    if not raw:
        return default_log_dir()
    p = Path(raw).expanduser()
    if not p.is_absolute():
        p = (ROOT_DIR / p).resolve()
    return p
