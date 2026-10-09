# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Windows-local FastAPI server for Qwen3-ASR using **ONNX Encoder (DirectML) + GGUF Decoder (llama.cpp / Vulkan)**. Real-time WebSocket only. No Docker/PyTorch in the supported runtime.

## Environment

Use conda env **`lingting`** (`D:\anaconda3\envs\lingting`, Python 3.10). Do not run with `base` / other envs.

```bash
conda activate lingting
```

## Common Commands

```bash
conda activate lingting

# Install (no torch)
pip install -r requirements.txt

# Start locally
python run.py

# Green pack (do not zip this git repo): packaging/build.ps1 → dist/ASR-REALTIME/
# See packaging/README.md

# Health check
curl http://127.0.0.1:8765/health
```

Place llama.cpp Vulkan DLLs in `vendor/qwen_asr_gguf/inference/bin/` and models in `models/` (`qwen3_asr_llm.q5_k.gguf` + int8 encoder ONNX + `silero_vad.onnx`). ForcedAligner is not loaded.

### Manual clients (E2Etest)

Server must be running. Set `WAV_PATH` at the top of each script, then:

```bash
conda activate lingting
python E2Etest/test_ws_tune.py
python E2Etest/test_websocket.py
python E2Etest/test_websocket_long.py
```

## Architecture

### File Organization

- `src/server.py` — FastAPI + WebSocket real-time transcription
- `src/vad.py` — Streaming Silero VAD / backend endpoint (EOS)
- `src/ws_overlap.py` — Pre-roll helpers
- `src/inference_queue.py` — Single-thread GPU infer queue (partial coalesce)
- `src/logger.py` — Loguru-based structured logging
- `src/schemas.py` — Pydantic models for Swagger UI
- `web/index.html` — Mic start/stop page at `/`
- `E2Etest/` — live-server E2E: `test_websocket.py`, `test_websocket_long.py`

### API Endpoints

| Endpoint | Method | Purpose |
|---|---|---|
| `/health` | GET | Health check, model status, EOS/max-audio config |
| `/admin/warmup` | POST | Load the ASR model |
| `/admin/shutdown` | POST | Stop the server |
| `/realtime/stream` | WebSocket | Real-time streaming; JSON `{data:{status,format,audio}}` |

### Concurrency Model

**PriorityInferQueue**:
- Single dedicated ThreadPoolExecutor (`max_workers=1`) for all GPU inference
- Pending partials with the same coalesce key are dropped when a newer partial/final arrives
- All inference runs via `run_in_executor()` to keep the async event loop unblocked

### Model Lifecycle

- Model loads at startup (and again on first request if needed)
- `_unload_watchdog()` unloads after `UNLOAD_GRACE_S` seconds of idle (default 0 = keep loaded)
- One live WebSocket session at a time

### WebSocket Real-Time Transcription (`/realtime/stream`)

- Accepts JSON `{ "data": { "status", "format", "eos?", "audio" } }`. `audio` is base64 s16le PCM (8 kHz or 16 kHz)
- `status`: `0` first frame (`format` required), `1` middle, `2` last (ends session; audio optional)
- One connection = one utterance. Partials are cumulative (client replaces, never appends)
- Final has **no `chars`**. Payload: `{text, type, start, end, eos_reason}`
- Three equivalent end paths, all `final` then `{status:"closed", reason}` then disconnect:
  - **silence EOS**: backend endpoint, silence ≥ session `eos` (default `WS_EOS_SILENCE_MS`)
  - **stop**: client `{ "data": { "status": 2 } }`
  - **max_duration**: accumulated 16 kHz PCM ≥ `WS_MAX_AUDIO_SECONDS` (max 60)
- No binary PCM, no `action` / `use_server_vad`. Server VAD is always on
- VAD skip: silent frames before first speech are not buffered; pre-roll keeps ~400ms of onset

## Key Environment Variables

| Variable | Default | Purpose |
|---|---|---|
| `WS_MAX_AUDIO_SECONDS` | `60` | Session PCM cap; then final + close (must be ≤ 60) |
| `WS_EOS_SILENCE_MS` | `600` | Backend endpoint silence after speech |
| `WS_EARLY_SILENCE_MS` | `1000` | Longer silence required before ~0.8s of speech |
| `WS_PARTIAL_SECONDS` | `1.5` | Partial emit interval |
| `WS_VAD_PRE_ROLL_MS` | `400` | Onset pre-roll |
| `ASR_USE_SERVER_VAD` | `true` | Unused for WS; server VAD is always on |
| `UNLOAD_GRACE_S` | `0` | Seconds before model unloads (0 = keep loaded) |
| `REQUEST_TIMEOUT` | `300` | Max inference time per request |
| `REALTIME_CHUNK_SIZE_SEC` | `8` | Encoder window; 60s audio is split internally |

## Docs

- `docs/ARCHITECTURE.md` — current runtime, WS protocol, queue, 8s cache
- `docs/STREAMING_TODO.md` — streaming backlog / P0–P3对照
- `CHANGELOG.md` — Version history
- `LEARNING_LOG.md` — Technical learnings and decisions
