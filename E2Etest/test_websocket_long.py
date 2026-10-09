"""Stream PCM until the 60s session cap; expect final + closed.

    conda activate lingting
    python E2Etest/test_websocket_long.py
"""
from __future__ import annotations

import asyncio
import json
import sys
from math import gcd
from pathlib import Path

import httpx
import numpy as np
import soundfile as sf
from scipy.signal import resample_poly
import websockets

WAV_PATH = r"C:\Users\Numb\Desktop\11s.wav"
LANGUAGE = "auto"
BASE_URL = "http://127.0.0.1:8765"
WS_URL = "ws://127.0.0.1:8765/realtime/stream"
CHUNK_MS = 200
TARGET_SR = 16000
# Server VAD off so looping speech is not cut by in-file silence before 60s.
USE_VAD = False


def load_pcm_s16le(path: Path) -> tuple[bytes, int, float]:
    audio, sr = sf.read(str(path), always_2d=False)
    if getattr(audio, "ndim", 1) > 1:
        audio = audio.mean(axis=1)
    audio = np.asarray(audio, dtype=np.float32)
    if int(sr) != TARGET_SR:
        g = gcd(int(sr), TARGET_SR)
        audio = resample_poly(audio, TARGET_SR // g, int(sr) // g).astype(np.float32)
        sr = TARGET_SR
    peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    if peak > 1.0:
        audio = audio / peak
    pcm = np.clip(audio * 32767.0, -32768, 32767).astype(np.int16).tobytes()
    duration = len(audio) / float(sr)
    return pcm, int(sr), duration


async def run() -> int:
    wav = Path(WAV_PATH).expanduser()
    if not wav.is_file():
        print(f"wav not found: {wav}", file=sys.stderr)
        return 1

    try:
        health = httpx.get(f"{BASE_URL}/health", timeout=5.0)
        health.raise_for_status()
    except Exception as exc:
        print(f"server unreachable at {BASE_URL}: {exc}", file=sys.stderr)
        return 1
    print("health:", health.json())
    cap = float((health.json() or {}).get("max_audio_seconds") or 60)

    pcm, sr, duration = load_pcm_s16le(wav)
    chunk_bytes = max(2, int(sr * CHUNK_MS / 1000) * 2)
    print(f"wav={wav} duration={duration:.2f}s loop until {cap:.1f}s vad={USE_VAD}")

    url = f"{WS_URL}?sample_rate={sr}&use_server_vad={'true' if USE_VAD else 'false'}"
    finals: list[dict] = []
    closed: dict | None = None

    async with websockets.connect(url, max_size=None, open_timeout=10) as ws:
        hello = json.loads(await ws.recv())
        print("connected:", hello)
        if hello.get("status") != "connected":
            return 1

        await ws.send(json.dumps({"action": "config", "language": LANGUAGE, "use_server_vad": USE_VAD}))
        print("configured:", json.loads(await ws.recv()))

        got_closed = asyncio.Event()

        async def reader():
            nonlocal closed
            try:
                while True:
                    raw = await ws.recv()
                    if isinstance(raw, bytes):
                        raw = raw.decode("utf-8")
                    msg = json.loads(raw)
                    print(msg)
                    if msg.get("type") == "final" or msg.get("is_final"):
                        finals.append(msg)
                    if msg.get("status") == "closed":
                        closed = msg
                        got_closed.set()
            except websockets.ConnectionClosed:
                got_closed.set()

        reader_task = asyncio.create_task(reader())
        try:
            sent = 0
            target = int(cap * sr) * 2  # bytes at 16-bit
            while sent < target and not got_closed.is_set():
                offset = sent % len(pcm)
                take = min(chunk_bytes, len(pcm) - offset)
                await ws.send(pcm[offset : offset + take])
                sent += take
                await asyncio.sleep(CHUNK_MS / 1000.0)
            await asyncio.wait_for(got_closed.wait(), timeout=120.0)
        except asyncio.TimeoutError:
            print("timeout waiting for 60s closed", file=sys.stderr)
            return 1
        finally:
            reader_task.cancel()
            try:
                await reader_task
            except asyncio.CancelledError:
                pass

    if not finals:
        print("no final received", file=sys.stderr)
        return 1
    last = finals[-1]
    reason = (closed or {}).get("reason") or last.get("eos_reason")
    print("final_text:", last.get("text"))
    print("eos_reason:", reason)
    if reason != "max_duration":
        print(f"expected max_duration, got {reason!r}", file=sys.stderr)
        return 1
    return 0


def main() -> int:
    try:
        return asyncio.run(run())
    except KeyboardInterrupt:
        print("interrupted", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
