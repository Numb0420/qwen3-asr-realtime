from __future__ import annotations

import gc
import os
import sys
import threading
import numpy as np
from scipy.signal import resample_poly

from backend.base import ASRBackend, ASRResult, BackendState
from config import (
    DML_PAD_TO,
    ENCODER_BACKEND_FN,
    ENCODER_FRONTEND_FN,
    GGML_VK_DISABLE_F16,
    LLM_FN,
    LLM_USE_GPU,
    MODEL_DIR,
    ONNX_PROVIDER,
    REALTIME_CHUNK_SIZE_SEC,
    WS_ASR_TEMPERATURE,
    WS_CHUNK_COMMIT_MARGIN_MS,
    TARGET_SR,
    apply_llm_device_env,
)
from logger import log
from paths import VENDOR_DIR

_ABORT_MARK = "====解码有误，强制熔断===="

_LANG_MAP = {
    "zh": "Chinese",
    "zh-cn": "Chinese",
    "zh-hans": "Chinese",
    "cn": "Chinese",
    "en": "English",
    "yue": "Cantonese",
    "zh-hk": "Cantonese",
    "zh-yue": "Cantonese",
    "ja": "Japanese",
    "ko": "Korean",
    "th": "Thai",
    "hi": "Hindi",
    "ar": "Arabic",
    "de": "German",
    "fr": "French",
    "es": "Spanish",
    "pt": "Portuguese",
    "id": "Indonesian",
    "it": "Italian",
    "ru": "Russian",
    "vi": "Vietnamese",
    "tr": "Turkish",
    "ms": "Malay",
    "nl": "Dutch",
    "sv": "Swedish",
    "da": "Danish",
    "fi": "Finnish",
    "pl": "Polish",
    "cs": "Czech",
    "el": "Greek",
    "ro": "Romanian",
    "hu": "Hungarian",
}


def _ensure_vendor_path() -> None:
    vendor = str(VENDOR_DIR)
    if vendor not in sys.path:
        sys.path.insert(0, vendor)


def map_language(language: str | None) -> str | None:
    if language is None:
        return None
    raw = str(language).strip()
    if not raw or raw.lower() == "auto":
        return None
    mapped = _LANG_MAP.get(raw.lower())
    if mapped:
        return mapped
    return raw[:1].upper() + raw[1:].lower() if raw else None


def public_language(internal: str | None, requested: str | None) -> str:
    if requested and requested.lower() not in ("auto", ""):
        return requested.lower() if len(requested) <= 8 else requested
    if internal == "Chinese":
        return "zh"
    if internal == "English":
        return "en"
    if internal:
        return internal
    return "auto"


def _to_16k_mono(audio: np.ndarray, sample_rate: int) -> np.ndarray:
    wav = np.asarray(audio, dtype=np.float32)
    if wav.ndim > 1:
        wav = wav.mean(axis=1).astype(np.float32)
    if sample_rate != TARGET_SR:
        from math import gcd
        g = gcd(int(sample_rate), TARGET_SR)
        wav = resample_poly(wav, TARGET_SR // g, int(sample_rate) // g).astype(np.float32)
    return wav


def _strip_abort(text: str) -> str:
    if not text:
        return ""
    return text.replace(_ABORT_MARK, "").strip()


class QwenGGUFBackend(ASRBackend):
    def __init__(self):
        self._lock = threading.Lock()
        self._state = BackendState.UNLOADED
        self._engine = None
        self._enc_cache: dict[str, dict[int, dict]] = {}

    def encoder_cache_for(self, session_id: str | None) -> dict | None:
        if not session_id:
            return None
        return self._enc_cache.setdefault(session_id, {})

    def clear_encoder_cache(self, session_id: str | None = None) -> None:
        if session_id is None:
            self._enc_cache.clear()
            return
        self._enc_cache.pop(session_id, None)

    @property
    def state(self) -> BackendState:
        return self._state

    def load(self) -> None:
        with self._lock:
            if self._state == BackendState.READY and self._engine is not None:
                return
            if self._state in (BackendState.LOADING, BackendState.UNLOADING):
                raise RuntimeError(f"Cannot load while backend is {self._state.value}")
            self._state = BackendState.LOADING
        try:
            apply_llm_device_env()
            _ensure_vendor_path()
            from qwen_asr_gguf.inference import ASREngineConfig, QwenASREngine

            config = ASREngineConfig(
                model_dir=str(MODEL_DIR),
                encoder_frontend_fn=ENCODER_FRONTEND_FN,
                encoder_backend_fn=ENCODER_BACKEND_FN,
                llm_fn=LLM_FN,
                onnx_provider=ONNX_PROVIDER,
                llm_use_gpu=LLM_USE_GPU,
                enable_aligner=False,
                align_config=None,
                verbose=False,
                dml_pad_to=DML_PAD_TO,
            )
            log.info(
                "Loading GGUF engine | dir={} llm={} provider={} vulkan={} "
                "llm_gpu={} vk_disable_f16={} aligner=false",
                MODEL_DIR,
                LLM_FN,
                ONNX_PROVIDER,
                os.environ.get("GGML_VULKAN"),
                LLM_USE_GPU,
                GGML_VK_DISABLE_F16,
            )
            engine = QwenASREngine(config)
            with self._lock:
                self._engine = engine
                self._state = BackendState.READY
            log.info("GGUF engine ready")
        except Exception:
            with self._lock:
                self._engine = None
                self._state = BackendState.UNLOADED
            raise

    def _require_ready(self):
        if self._state != BackendState.READY or self._engine is None:
            raise RuntimeError(f"Backend not ready ({self._state.value})")

    def _run(
        self,
        audio,
        sample_rate: int,
        language: str | None,
        chunk_size_sec: float,
        is_final: bool,
        temperature: float = WS_ASR_TEMPERATURE,
        cancel_event=None,
        encoder_cache=None,
        text_commit_margin_sec: float | None = None,
    ) -> ASRResult:
        with self._lock:
            self._require_ready()
            engine = self._engine
        wav = _to_16k_mono(audio, sample_rate)
        internal_lang = map_language(language)
        try:
            result = engine.asr(
                audio=wav,
                context="",
                language=internal_lang,
                chunk_size_sec=chunk_size_sec,
                temperature=temperature,
                streaming=False,
                do_align=False,
                prefix_text="",
                cancel_event=cancel_event,
                encoder_cache=encoder_cache,
                text_commit_margin_sec=(
                    WS_CHUNK_COMMIT_MARGIN_MS / 1000.0
                    if text_commit_margin_sec is None
                    else float(text_commit_margin_sec)
                ),
            )
        except Exception:
            log.exception("ASR decode failed; reload after ASR failure")
            try:
                self.unload()
                self.load()
            except Exception:
                log.exception("Engine reload after ASR failure also failed")
            raise
        text = _strip_abort(getattr(result, "text", "") or "")
        perf = getattr(result, "performance", None) or {}
        n_gen = int(perf.get("decode_tokens", 0) or 0)
        abort_stage = perf.get("abort_stage")
        chunks = perf.get("encode_chunks") or []
        chunk_s = ",".join(
            f"{c.get('i')}:{c.get('ms')}:{'hit' if c.get('hit') else 'miss'}"
            for c in chunks
        ) or "-"
        decode_chunks = perf.get("decode_chunks") or []
        decode_s = ",".join(
            f"{c.get('i')}:skip" if c.get("skipped") else f"{c.get('i')}:{c.get('ms')}"
            for c in decode_chunks
        ) or "-"
        view = {
            "encode_ms": round(float(perf.get("encode_time", 0) or 0) * 1000.0),
            "prefill_ms": round(float(perf.get("prefill_time", 0) or 0) * 1000.0),
            "decode_ms": round(float(perf.get("decode_time", 0) or 0) * 1000.0),
            "n_chunks": int(perf.get("n_chunks") or len(chunks) or 0),
            "cache_hits": int(perf.get("cache_hits") or 0),
            "decode_skipped": int(perf.get("decode_skipped") or 0),
            "committed_chunks": int(perf.get("committed_chunks") or 0),
            "encode_chunks": chunk_s,
            "decode_chunks": decode_s,
            "mel_ms": round(sum(float(c.get("mel_ms") or 0) for c in chunks), 1),
            "frontend_ms": round(sum(float(c.get("frontend_ms") or 0) for c in chunks), 1),
            "backend_ms": round(sum(float(c.get("backend_ms") or 0) for c in chunks), 1),
        }
        return ASRResult(
            text=text,
            language=public_language(internal_lang, language),
            is_final=is_final,
            words=None,
            n_generate=n_gen,
            cancelled=bool(abort_stage),
            abort_stage=abort_stage,
            perf=view,
        )

    def transcribe_realtime(
        self,
        audio,
        sample_rate: int,
        language: str | None = None,
        is_final: bool = False,
        prefix_text: str = "",
        cancel_event=None,
        session_id: str | None = None,
    ) -> ASRResult:
        # Always 8s encoder windows; a 60s utterance is split inside the engine.
        del prefix_text
        return self._run(
            audio,
            sample_rate,
            language,
            REALTIME_CHUNK_SIZE_SEC,
            is_final=is_final,
            temperature=WS_ASR_TEMPERATURE,
            cancel_event=cancel_event,
            encoder_cache=self.encoder_cache_for(session_id),
        )

    def unload(self) -> None:
        with self._lock:
            if self._state == BackendState.UNLOADED and self._engine is None:
                return
            if self._state == BackendState.LOADING:
                raise RuntimeError("Cannot unload while LOADING")
            self._state = BackendState.UNLOADING
            engine = self._engine
            self._engine = None
            self._enc_cache.clear()
        try:
            if engine is not None:
                engine.close()
        finally:
            gc.collect()
            with self._lock:
                self._state = BackendState.UNLOADED
            log.info("GGUF engine unloaded")
