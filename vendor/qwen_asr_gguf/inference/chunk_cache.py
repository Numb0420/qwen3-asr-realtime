# coding=utf-8
"""P3-A / P3-A+ session cache helpers.

encoder_cache[i] is a dict:
  embd       ndarray   full configured window embedding (P3-A)
  text       str       frozen window text (P3-A+)
  committed  bool      text is frozen and must not be rewritten
"""
from __future__ import annotations


def cache_entry(cache, idx):
    if cache is None:
        return None
    return cache.get(idx)


def cache_embd(entry):
    if entry is None:
        return None
    if isinstance(entry, dict):
        return entry.get("embd")
    return entry


def committed_text(entry):
    if not isinstance(entry, dict) or not entry.get("committed"):
        return None
    if "text" not in entry:
        return None
    return entry.get("text") or ""


def set_cache_embd(cache, idx, embd):
    if cache is None or embd is None:
        return None
    entry = cache.get(idx)
    if not isinstance(entry, dict):
        entry = {}
        cache[idx] = entry
    entry["embd"] = embd
    return entry


def chunk_commit_ready(
    chunk_idx: int,
    *,
    chunk_size_sec: float,
    total_duration: float,
    margin_sec: float,
    was_last: bool,
    full_window: bool,
) -> bool:
    """Commit only a complete non-last window after the safety margin.

    Full 8s is not enough: that decode still has is_last_chunk=True and flushes
    rollback tokens. Wait until audio has crossed into the next window by
    ``margin_sec`` so chunk_idx was decoded the same way later finals will.
    """
    if not full_window or was_last:
        return False
    margin = max(0.0, float(margin_sec))
    closed_at = (int(chunk_idx) + 1) * float(chunk_size_sec) + margin
    return float(total_duration) + 1e-9 >= closed_at


def should_skip_decode(cache, idx, *, was_last: bool) -> tuple:
    """Skip decode of a frozen non-last window. Last window always decodes."""
    if was_last:
        return False, None
    entry = cache_entry(cache, idx)
    text = committed_text(entry)
    if text is None or cache_embd(entry) is None:
        return False, None
    return True, text


def commit_chunk_text(cache, idx, text) -> bool:
    """Freeze window text. Returns True if this call stored a new commit."""
    if cache is None:
        return False
    entry = cache.get(idx)
    if not isinstance(entry, dict):
        entry = {} if entry is None else {"embd": entry}
        cache[idx] = entry
    if entry.get("committed"):
        return False
    entry["text"] = text if text is not None else ""
    entry["committed"] = True
    return True
