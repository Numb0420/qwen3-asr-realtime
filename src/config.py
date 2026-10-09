"""Runtime configuration for the local GGUF ASR service."""
from __future__ import annotations

import os
import sys
from pathlib import Path
from logger import log
from paths import ROOT_DIR, resolve_log_dir


def _safe_float(name: str, default: str) -> float:
    raw = os.getenv(name, default)
    try:
        return float(raw)
    except ValueError:
        log.error("Config error: {} must be a float, got '{}' — using default {}", name, raw, default)
        return float(default)


def _safe_int(name: str, default: str) -> int:
    raw = os.getenv(name, default)
    try:
        return int(raw)
    except ValueError:
        log.error("Config error: {} must be an integer, got '{}' — using default {}", name, raw, default)
        return int(default)


def _safe_bool(name: str, default: str) -> bool:
    return os.getenv(name, default).strip().lower() in ("1", "true", "yes", "on")


_VALID_LOG_LEVELS = {"TRACE", "DEBUG", "INFO", "WARNING", "WARN", "ERROR", "CRITICAL", "FATAL"}
_LOG_LEVEL_ALIASES = {"WARN": "WARNING", "FATAL": "CRITICAL"}
_VALID_ONNX_PROVIDERS = {"DML", "CPU", "CUDA", "TRT", "TENSORRT"}

MODEL_DIR = Path(os.getenv("MODEL_DIR", str(ROOT_DIR / "models"))).expanduser()
if not MODEL_DIR.is_absolute():
    MODEL_DIR = (ROOT_DIR / MODEL_DIR).resolve()

LLM_FN = os.getenv("LLM_FN", "qwen3_asr_llm.q5_k.gguf")
ENCODER_FRONTEND_FN = os.getenv("ENCODER_FRONTEND_FN", "qwen3_asr_encoder_frontend.int8.onnx")
ENCODER_BACKEND_FN = os.getenv("ENCODER_BACKEND_FN", "qwen3_asr_encoder_backend.int8.onnx")
ONNX_PROVIDER = os.getenv("ONNX_PROVIDER", "DML").upper()
LLM_USE_GPU = _safe_bool("LLM_USE_GPU", "true")
# llama.cpp Vulkan reads this from the process env (not a Python API).
GGML_VK_DISABLE_F16 = _safe_bool("GGML_VK_DISABLE_F16", "false")
ASR_TEMPERATURE = _safe_float("ASR_TEMPERATURE", "0.0")  # legacy alias of WS_ASR_TEMPERATURE
WS_ASR_TEMPERATURE = _safe_float("WS_ASR_TEMPERATURE", "0.0")


def apply_llm_device_env() -> None:
    """Push decoder device flags into the env before llama.cpp loads DLLs."""
    if LLM_USE_GPU:
        os.environ.setdefault("GGML_VULKAN", "1")
        if GGML_VK_DISABLE_F16:
            os.environ["GGML_VK_DISABLE_F16"] = "1"
        else:
            os.environ.pop("GGML_VK_DISABLE_F16", None)
    else:
        os.environ["GGML_VULKAN"] = "0"


apply_llm_device_env()

WS_PARTIAL_SECONDS = _safe_float("WS_PARTIAL_SECONDS", "1.5")
# One WebSocket session accepts at most this many seconds of 16 kHz PCM, then final+close.
WS_MAX_AUDIO_SECONDS = _safe_float("WS_MAX_AUDIO_SECONDS", os.getenv("WS_MAX_UTTERANCE_SECONDS", "60"))
WS_VAD_PRE_ROLL_MS = _safe_int("WS_VAD_PRE_ROLL_MS", "400")
# Backend endpoint: silence this long after speech → audio ended.
WS_EOS_SILENCE_MS = _safe_int("WS_EOS_SILENCE_MS", os.getenv("WS_FLUSH_SILENCE_MS", "600"))
# Short utterances need a slightly longer silence before EOS (avoids cutting on 啊).
WS_EARLY_SILENCE_MS = _safe_int("WS_EARLY_SILENCE_MS", "1000")
WS_COMMITTED_SPEECH_SECONDS = _safe_float("WS_COMMITTED_SPEECH_SECONDS", "0.8")
# After committed speech, start speculative final this many ms into silence (0 = off).
# Real EOS is still WS_EOS_SILENCE_MS; speculative is cancelled if speech resumes.
WS_SPECULATIVE_FINAL_MS = _safe_int("WS_SPECULATIVE_FINAL_MS", "300")
# P3-A+: freeze a full 8s window's text only after audio has gone this far past the
# window end (so that window was decoded as non-last, matching later finals).
WS_CHUNK_COMMIT_MARGIN_MS = _safe_int("WS_CHUNK_COMMIT_MARGIN_MS", "400")
REALTIME_CHUNK_SIZE_SEC = _safe_float("REALTIME_CHUNK_SIZE_SEC", "8")
# Seconds of idle before unloading. 0 = keep weights loaded (desktop default).
UNLOAD_GRACE_S = _safe_int("UNLOAD_GRACE_S", "0")
REQUEST_TIMEOUT = _safe_int("REQUEST_TIMEOUT", "300")
PORT = _safe_int("PORT", "8765")
HOST = os.getenv("HOST", "127.0.0.1")
ASR_USE_SERVER_VAD = _safe_bool("ASR_USE_SERVER_VAD", "true")
FORMAT_NUM = _safe_bool("FORMAT_NUM", "true")


def _under_model_dir(env_name: str, default_fn: str) -> Path:
    """Resolve a model file: absolute as-is, otherwise relative to MODEL_DIR.

    Old values like ``./models/silero_vad.onnx`` still work (leading ``models/`` is stripped).
    """
    raw = os.getenv(env_name, default_fn).strip() or default_fn
    path = Path(raw).expanduser()
    if path.is_absolute():
        return path
    parts = list(path.parts)
    if parts and parts[0] == ".":
        parts = parts[1:]
    if parts and parts[0] == "models":
        parts = parts[1:]
    if not parts:
        parts = [default_fn]
    return (MODEL_DIR.joinpath(*parts)).resolve()


VAD_MODEL_PATH = _under_model_dir("VAD_MODEL_PATH", "silero_vad.onnx")
TARGET_SR = 16000

# -----------------
# Hotword correction (phoneme-based, after ITN)
# -----------------
HOTWORD_ENABLED = _safe_bool("HOTWORD_ENABLED", "true")
HOTWORDS_PATH = Path(os.getenv("HOTWORDS_PATH", str(ROOT_DIR / "hotwords.txt"))).expanduser()
if not HOTWORDS_PATH.is_absolute():
    HOTWORDS_PATH = (ROOT_DIR / HOTWORDS_PATH).resolve()
# Match threshold 0~1, higher = stricter. 0.85 balances recall/precision for CJK.
HOTWORD_THRESHOLD = _safe_float("HOTWORD_THRESHOLD", "0.85")

DML_PAD_TO = _safe_int("DML_PAD_TO", str(int(REALTIME_CHUNK_SIZE_SEC)))

# Mirrors logger.py (same resolve_log_dir / LOG_DIR env).
LOG_DIR = resolve_log_dir()


def validate_env() -> None:
    """Validate critical environment variables at startup. Exits on invalid config."""
    errors = []

    if ONNX_PROVIDER not in _VALID_ONNX_PROVIDERS:
        errors.append(f"ONNX_PROVIDER must be one of {_VALID_ONNX_PROVIDERS}, got '{ONNX_PROVIDER}'")

    if REQUEST_TIMEOUT <= 0:
        errors.append(f"REQUEST_TIMEOUT must be positive, got {REQUEST_TIMEOUT}")
    if UNLOAD_GRACE_S < 0:
        errors.append(f"UNLOAD_GRACE_S must be non-negative, got {UNLOAD_GRACE_S}")
    if WS_PARTIAL_SECONDS <= 0:
        errors.append(f"WS_PARTIAL_SECONDS must be positive, got {WS_PARTIAL_SECONDS}")
    if WS_MAX_AUDIO_SECONDS <= 0:
        errors.append(f"WS_MAX_AUDIO_SECONDS must be positive, got {WS_MAX_AUDIO_SECONDS}")
    if WS_MAX_AUDIO_SECONDS > 60:
        errors.append(f"WS_MAX_AUDIO_SECONDS must be <= 60, got {WS_MAX_AUDIO_SECONDS}")
    if WS_VAD_PRE_ROLL_MS < 0:
        errors.append(f"WS_VAD_PRE_ROLL_MS must be non-negative, got {WS_VAD_PRE_ROLL_MS}")
    if WS_EOS_SILENCE_MS <= 0:
        errors.append(f"WS_EOS_SILENCE_MS must be positive, got {WS_EOS_SILENCE_MS}")
    if WS_EARLY_SILENCE_MS <= 0:
        errors.append(f"WS_EARLY_SILENCE_MS must be positive, got {WS_EARLY_SILENCE_MS}")
    if WS_COMMITTED_SPEECH_SECONDS <= 0:
        errors.append(f"WS_COMMITTED_SPEECH_SECONDS must be positive, got {WS_COMMITTED_SPEECH_SECONDS}")
    if WS_SPECULATIVE_FINAL_MS < 0:
        errors.append(f"WS_SPECULATIVE_FINAL_MS must be non-negative, got {WS_SPECULATIVE_FINAL_MS}")
    if WS_CHUNK_COMMIT_MARGIN_MS < 0:
        errors.append(f"WS_CHUNK_COMMIT_MARGIN_MS must be non-negative, got {WS_CHUNK_COMMIT_MARGIN_MS}")
    if REALTIME_CHUNK_SIZE_SEC <= 0:
        errors.append(f"REALTIME_CHUNK_SIZE_SEC must be positive, got {REALTIME_CHUNK_SIZE_SEC}")

    log_level = os.getenv("LOG_LEVEL", "info").upper()
    log_level = _LOG_LEVEL_ALIASES.get(log_level, log_level)
    if log_level not in _VALID_LOG_LEVELS:
        errors.append(f"LOG_LEVEL must be one of {_VALID_LOG_LEVELS}, got '{log_level}'")

    required_files = [
        MODEL_DIR / LLM_FN,
        MODEL_DIR / ENCODER_FRONTEND_FN,
        MODEL_DIR / ENCODER_BACKEND_FN,
        VAD_MODEL_PATH,
    ]
    for path in required_files:
        if not path.exists():
            errors.append(f"Missing model file: {path}")

    if errors:
        for err in errors:
            log.error("Config validation failed: {}", err)
        sys.exit(1)

    log.info("Config validation passed | model_dir={} provider={}", MODEL_DIR, ONNX_PROVIDER)
