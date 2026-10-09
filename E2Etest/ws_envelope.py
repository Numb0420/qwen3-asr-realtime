"""JSON data-envelope helpers for /realtime/stream clients."""
from __future__ import annotations

import base64
import json

FORMAT_16K = "audio/L16;rate=16000"
FORMAT_8K = "audio/L16;rate=8000"


def ws_audio_msg(
    status: int,
    pcm: bytes | None = None,
    fmt: str = FORMAT_16K,
    eos: int | None = None,
) -> str:
    data: dict = {"status": int(status)}
    if status == 0:
        data["format"] = fmt
        if eos is not None:
            data["eos"] = int(eos)
    if pcm:
        data["audio"] = base64.b64encode(pcm).decode("ascii")
    return json.dumps({"data": data})
