from __future__ import annotations

import asyncio
import json
import sys
import time
import uuid as _uuid_module
from contextlib import asynccontextmanager
from math import gcd
from pathlib import Path

import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from scipy.signal import butter, resample_poly, sosfilt

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
VENDOR = ROOT.parent / "vendor"
if str(VENDOR) not in sys.path:
    sys.path.insert(0, str(VENDOR))

from backend import QwenGGUFBackend
from backend.base import BackendState
from config import (
    ASR_USE_SERVER_VAD,
    FORMAT_NUM,
    HOST,
    LLM_FN,
    LLM_USE_GPU,
    ONNX_PROVIDER,
    PORT,
    REQUEST_TIMEOUT,
    TARGET_SR,
    UNLOAD_GRACE_S,
    apply_llm_device_env,
    WS_EARLY_SILENCE_MS,
    WS_EOS_SILENCE_MS,
    WS_MAX_AUDIO_SECONDS,
    WS_PARTIAL_SECONDS,
    WS_SPECULATIVE_FINAL_MS,
    WS_VAD_PRE_ROLL_MS,
    validate_env,
)
from hotword_correct import apply_hotword
from inference_queue import infer_queue
from itn import apply_chinese_itn
from logger import log, reset_request_id, set_request_id
from repetition import collapse_repetitions
from schemas import (
    API_DESCRIPTION,
    API_TAGS,
    ErrorResponse,
    HealthResponse,
)
from vad import RealtimeVad
from ws_overlap import append_pre_roll, unused_pre_roll

backend = QwenGGUFBackend()
_last_used = 0.0
_active_ws: WebSocket | None = None
_ws_lock = asyncio.Lock()


def _touch() -> None:
    global _last_used
    _last_used = time.time()


def _telephony_bandpass(audio: np.ndarray, sr: int) -> np.ndarray:
    sos = butter(4, [300, 3400], btype="bandpass", fs=sr, output="sos")
    return sosfilt(sos, audio).astype(np.float32)


def _resample_pcm_bytes(pcm_bytes: bytes, orig_sr: int) -> bytes:
    if orig_sr == TARGET_SR:
        return pcm_bytes
    samples = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32)
    g = gcd(orig_sr, TARGET_SR)
    resampled = resample_poly(samples, TARGET_SR // g, orig_sr // g)
    return resampled.astype(np.int16).tobytes()


def detect_and_fix_repetitions(text: str, max_repeats: int = 2) -> str:
    return collapse_repetitions(text)


def _ws_preview(text: str, limit: int = 48) -> str:
    raw = (text or "").replace("\n", " ")
    if len(raw) <= limit:
        return f"n={len(raw)} {raw!r}"
    return f"n={len(raw)} {raw[:28]!r}…{raw[-16:]!r}"


def _pcm_to_float(pcm: bytes) -> np.ndarray:
    return np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0


async def _ensure_model_loaded() -> None:
    deadline = time.time() + 60.0
    waited = False
    while True:
        if backend.loaded:
            _touch()
            return
        if backend.state in (BackendState.UNLOADING, BackendState.LOADING):
            if time.time() >= deadline:
                raise RuntimeError(
                    f"Timed out waiting for backend {backend.state.value}"
                )
            if not waited:
                log.info("Waiting for backend {} before load", backend.state.value)
                waited = True
            await asyncio.sleep(0.05)
            continue
        break
    await infer_queue.run_blocking(backend.load)
    _touch()


async def _unload_model(
    *,
    reason: str = "unspecified",
    force: bool = False,
    ignore_ws: bool = False,
) -> None:
    await infer_queue.wait_idle()
    if not force:
        if not ignore_ws and _active_ws is not None:
            log.info("Skip unload | reason={} blocked_by=live WebSocket", reason)
            return
    log.info("Unloading model | reason={} force={}", reason, force)
    if backend.loaded or backend.state != BackendState.UNLOADED:
        await infer_queue.run_blocking(backend.unload)


async def _run_asr(
    audio: np.ndarray,
    language: str | None,
    *,
    is_final: bool,
    kind: str,
    session_id: str,
    eos_reason: str | None = None,
):
    await _ensure_model_loaded()
    _touch()
    duration = float(audio.size) / float(TARGET_SR) if audio.size else 0.0
    priority = 0 if kind in ("final", "speculative") else 1

    def _job(cancel_event):
        return backend.transcribe_realtime(
            audio,
            TARGET_SR,
            language,
            is_final=is_final,
            cancel_event=cancel_event,
            session_id=session_id,
        )

    ticket = await asyncio.wait_for(
        infer_queue.submit(_job, priority=priority, kind=kind, coalesce_key=session_id),
        timeout=REQUEST_TIMEOUT,
    )
    perf = getattr(ticket.value, "perf", None) or {}
    log.info(
        "WS infer_done | job_id={} session_id={} kind={} audio={:.3f}s "
        "enqueue_ts={:.3f} start_ts={:.3f} finish_ts={:.3f} "
        "queue_wait_ms={:.0f} infer_ms={:.0f} cancelled={} abort_stage={} eos_reason={} "
        "encode_ms={} prefill_ms={} decode_ms={} n_chunks={} cache_hits={} "
        "committed_chunks={} decode_skipped={} decode_chunks={} "
        "mel_ms={} frontend_ms={} backend_ms={} encode_chunks={}",
        ticket.job_id,
        session_id,
        kind,
        duration,
        ticket.enqueue_ts,
        ticket.start_ts,
        ticket.finish_ts,
        ticket.queue_wait_ms,
        ticket.infer_ms,
        int(ticket.cancelled),
        ticket.abort_stage,
        eos_reason,
        perf.get("encode_ms", 0),
        perf.get("prefill_ms", 0),
        perf.get("decode_ms", 0),
        perf.get("n_chunks", 0),
        perf.get("cache_hits", 0),
        perf.get("committed_chunks", 0),
        perf.get("decode_skipped", 0),
        perf.get("decode_chunks", "-"),
        perf.get("mel_ms", 0),
        perf.get("frontend_ms", 0),
        perf.get("backend_ms", 0),
        perf.get("encode_chunks", "-"),
    )
    result = ticket.value
    if ticket.cancelled or result is None:
        return None
    result.text = detect_and_fix_repetitions(result.text)
    return result


async def _unload_watchdog() -> None:
    while True:
        await asyncio.sleep(15)
        if UNLOAD_GRACE_S <= 0:
            continue
        if _active_ws is not None:
            continue
        if not backend.loaded:
            continue
        if time.time() - _last_used <= UNLOAD_GRACE_S:
            continue
        log.info("Unload grace reached ({}s), unloading model", UNLOAD_GRACE_S)
        try:
            await _unload_model(reason="idle_grace")
        except Exception as e:
            log.error("Idle unload failed: {}", e)


@asynccontextmanager
async def lifespan(_app):
    apply_llm_device_env()
    validate_env()
    infer_queue.start()
    try:
        await _ensure_model_loaded()
    except Exception as e:
        log.error("Startup model load failed: {}", e)
    watchdog = asyncio.create_task(_unload_watchdog())
    yield
    watchdog.cancel()
    try:
        await _unload_model(reason="shutdown", force=True)
    except Exception:
        pass
    infer_queue.stop()


_uvicorn_server = None


def bind_uvicorn_server(server) -> None:
    """Allow /admin/shutdown to stop the uvicorn Server started from run.py."""
    global _uvicorn_server
    _uvicorn_server = server


app = FastAPI(
    title="Qwen3-ASR",
    version="1.0.0",
    description=API_DESCRIPTION,
    openapi_tags=API_TAGS,
    lifespan=lifespan,
    responses={
        422: {"model": ErrorResponse, "description": "Audio decode or validation error"},
        504: {"model": ErrorResponse, "description": "Inference timed out"},
    },
)


@app.middleware("http")
async def _request_id_middleware(request, call_next):
    req_id = request.headers.get("x-request-id") or str(_uuid_module.uuid4())
    token = set_request_id(req_id)
    try:
        response = await call_next(request)
        response.headers["X-Request-ID"] = req_id
        return response
    finally:
        reset_request_id(token)


WEB_DIR = ROOT.parent / "web"


@app.get("/", include_in_schema=False)
async def web_index():
    page = WEB_DIR / "index.html"
    if not page.is_file():
        return {"status": "ok", "message": "web/index.html not found"}
    return FileResponse(page, media_type="text/html; charset=utf-8")


@app.get("/health", response_model=HealthResponse, tags=["System"], summary="Health check")
async def health():
    return {
        "status": "ok",
        "model_loaded": backend.loaded,
        "backend": "gguf",
        "state": backend.state.value,
        "model_id": LLM_FN,
        "onnx_provider": ONNX_PROVIDER,
        "llm_gpu": "vulkan" if LLM_USE_GPU else "cpu",
        "format_num": FORMAT_NUM,
        "max_audio_seconds": WS_MAX_AUDIO_SECONDS,
        "eos_silence_ms": WS_EOS_SILENCE_MS,
    }


@app.post("/admin/warmup", tags=["System"], summary="Load the ASR model")
async def admin_warmup():
    await _ensure_model_loaded()
    return {
        "ok": True,
        "state": backend.state.value,
        "model_loaded": backend.loaded,
    }


@app.post("/admin/shutdown", tags=["System"], summary="Stop the ASR server")
async def admin_shutdown():
    if _uvicorn_server is not None:
        _uvicorn_server.should_exit = True
        _uvicorn_server.force_exit = True
    return {"ok": True}


@app.websocket("/realtime/stream")
async def websocket_transcribe(websocket: WebSocket):
    global _active_ws

    async with _ws_lock:
        if _active_ws is not None:
            await websocket.accept()
            await websocket.send_json({
                "code": "WS_BUSY",
                "message": "Only one WebSocket transcription session is allowed",
                "statusCode": 409,
            })
            await websocket.close(code=1008)
            return
        await websocket.accept()
        _active_ws = websocket

    ws_req_id = websocket.query_params.get("request_id") or str(_uuid_module.uuid4())
    token = set_request_id(ws_req_id)

    use_vad = ASR_USE_SERVER_VAD
    vad_param = websocket.query_params.get("use_server_vad")
    if vad_param is not None:
        use_vad = vad_param.lower() in ("true", "1", "yes")

    client_sr = int(websocket.query_params.get("sample_rate", str(TARGET_SR)))
    if client_sr not in (8000, 16000):
        await websocket.send_json({
            "code": "UNSUPPORTED_SAMPLE_RATE",
            "message": f"sample_rate must be 8000 or 16000, got {client_sr}",
            "statusCode": 400,
        })
        await websocket.close()
        async with _ws_lock:
            if _active_ws is websocket:
                _active_ws = None
        reset_request_id(token)
        return

    log.info(
        "[WS] Client connected | request_id={} use_vad={} sample_rate={}",
        ws_req_id,
        use_vad,
        client_sr,
    )
    await infer_queue.open_session(ws_req_id)

    lang_code: str | None = None
    accept_pcm = True
    session_phase = "listening"
    speech_buf = np.zeros(0, dtype=np.float32)
    pre_roll = np.zeros(0, dtype=np.float32)
    last_partial_at = 0
    stream_samples = 0
    buffer_audio_start = 0.0
    send_lock = asyncio.Lock()
    vad = RealtimeVad()
    partial_samples = int(WS_PARTIAL_SECONDS * TARGET_SR)
    max_samples = int(WS_MAX_AUDIO_SECONDS * TARGET_SR)
    pre_roll_samples = int(round(WS_VAD_PRE_ROLL_MS * TARGET_SR / 1000.0))
    speculative_task: asyncio.Task | None = None

    async def _send(payload: dict) -> None:
        async with send_lock:
            await websocket.send_json(payload)

    def _spawn_partial() -> None:
        if session_phase != "listening":
            return
        asyncio.create_task(_emit_partial())

    async def _cancel_speculative(reason: str) -> None:
        nonlocal speculative_task
        if speculative_task is None:
            return
        await infer_queue.cancel_jobs(ws_req_id, ("speculative",))
        log.info("WS speculative_cancel | session_id={} reason={}", ws_req_id, reason)
        speculative_task = None

    def _spawn_speculative() -> None:
        nonlocal speculative_task
        if session_phase != "listening" or speech_buf.size == 0:
            return
        if speculative_task is not None:
            return
        speculative_task = asyncio.create_task(_run_speculative())

    async def _run_speculative():
        if session_phase != "listening" or speech_buf.size == 0:
            return None
        audio = speech_buf.copy()
        if client_sr == 8000:
            audio = _telephony_bandpass(audio, TARGET_SR)
        duration = float(audio.size) / float(TARGET_SR)
        log.info(
            "WS speculative_submit | session_id={} audio={:.3f}s",
            ws_req_id,
            duration,
        )
        return await _run_asr(
            audio,
            lang_code,
            is_final=True,
            kind="speculative",
            session_id=ws_req_id,
            eos_reason="speculative",
        )

    async def _emit_partial() -> None:
        if session_phase != "listening" or speech_buf.size == 0:
            return
        audio = speech_buf.copy()
        if client_sr == 8000:
            audio = _telephony_bandpass(audio, TARGET_SR)
        if session_phase != "listening":
            return
        result = await _run_asr(
            audio,
            lang_code,
            is_final=False,
            kind="partial",
            session_id=ws_req_id,
        )
        if result is None or session_phase != "listening":
            return
        text = (result.text or "").strip()
        if not text:
            return
        log.info("WS emit | type=partial session_id={} out={}", ws_req_id, _ws_preview(text))
        await _send({"text": text, "type": "partial"})

    def _finalize_text(text: str) -> str:
        itn_text, _ = apply_chinese_itn(text, None)
        if itn_text != text:
            log.info("WS itn | raw={} out={}", _ws_preview(text), _ws_preview(itn_text))
            text = itn_text
        hw_text, _ = apply_hotword(text, None)
        if hw_text != text:
            log.info("WS hotword | raw={} out={}", _ws_preview(text), _ws_preview(hw_text))
            text = hw_text
        return text

    async def _send_final(text: str, reason: str, start: float, end: float) -> None:
        text = _finalize_text(text)
        log.info(
            "WS emit | type=final session_id={} reason={} start={:.2f} end={:.2f} out={}",
            ws_req_id,
            reason,
            start,
            end,
            _ws_preview(text),
        )
        await _send({
            "text": text,
            "type": "final",
            "start": round(start, 2),
            "end": round(end, 2),
            "eos_reason": reason,
        })

    async def _emit_final(reason: str) -> None:
        nonlocal speculative_task
        t_end = round(stream_samples / TARGET_SR, 2)
        if speech_buf.size == 0:
            log.info("WS emit | type=final reason={} empty_buf end={:.2f}", reason, t_end)
            await _send({
                "text": "",
                "type": "final",
                "start": t_end,
                "end": t_end,
                "eos_reason": reason,
            })
            return
        start = float(buffer_audio_start)
        duration = float(speech_buf.size) / float(TARGET_SR)
        end = start + duration
        if speculative_task is not None:
            log.info(
                "WS speculative_wait | session_id={} reason={} audio={:.3f}s",
                ws_req_id,
                reason,
                duration,
            )
            try:
                result = await speculative_task
            except Exception:
                result = None
            speculative_task = None
            if result is not None:
                await _send_final(result.text or "", reason, start, end)
                return
        audio = speech_buf.copy()
        if client_sr == 8000:
            audio = _telephony_bandpass(audio, TARGET_SR)
        log.info(
            "WS final_submit | session_id={} reason={} audio={:.3f}s",
            ws_req_id,
            reason,
            duration,
        )
        result = await _run_asr(
            audio,
            lang_code,
            is_final=True,
            kind="final",
            session_id=ws_req_id,
            eos_reason=reason,
        )
        text = (result.text or "") if result is not None else ""
        await _send_final(text, reason, start, end)

    async def _finish_session(reason: str) -> None:
        global _active_ws
        nonlocal accept_pcm, session_phase
        if session_phase != "listening":
            return
        session_phase = "closing"
        accept_pcm = False
        log.info(
            "WS session_closing | session_id={} reason={} buf={:.3f}s",
            ws_req_id,
            reason,
            speech_buf.size / TARGET_SR if speech_buf.size else 0.0,
        )
        await infer_queue.close_session(ws_req_id)
        await _emit_final(reason)
        session_phase = "idle"
        log.info("WS session closed | session_id={} reason={}", ws_req_id, reason)
        async with _ws_lock:
            if _active_ws is websocket:
                _active_ws = None
        await _send({"status": "closed", "reason": reason})
        try:
            await websocket.close()
        except Exception:
            pass

    try:
        await _ensure_model_loaded()
        await websocket.send_json({
            "status": "connected",
            "session_id": ws_req_id,
            "sample_rate": client_sr,
            "format": "pcm_s16le",
            "partial_seconds": WS_PARTIAL_SECONDS,
            "max_audio_seconds": WS_MAX_AUDIO_SECONDS,
            "eos_silence_ms": WS_EOS_SILENCE_MS,
            "early_silence_ms": WS_EARLY_SILENCE_MS,
            "vad_pre_roll_ms": WS_VAD_PRE_ROLL_MS,
            "use_server_vad": use_vad,
        })

        while True:
            if session_phase != "listening":
                break
            data = await websocket.receive()
            if data.get("type") == "websocket.disconnect":
                break
            if session_phase != "listening":
                break

            if "text" in data:
                try:
                    msg = json.loads(data["text"])
                except json.JSONDecodeError:
                    await _send({
                        "code": "INVALID_JSON",
                        "message": "Invalid JSON command",
                        "statusCode": 400,
                    })
                    continue
                action = msg.get("action", "")
                if action == "reset":
                    log.info(
                        "WS reset | cursor={:.3f} buf={:.3f}s",
                        stream_samples / TARGET_SR,
                        speech_buf.size / TARGET_SR if speech_buf.size else 0.0,
                    )
                    speech_buf = np.zeros(0, dtype=np.float32)
                    pre_roll = np.zeros(0, dtype=np.float32)
                    last_partial_at = 0
                    vad.reset()
                    buffer_audio_start = stream_samples / TARGET_SR
                    await _cancel_speculative("reset")
                    backend.clear_encoder_cache(ws_req_id)
                    await _send({"status": "buffer_reset"})
                elif action == "config":
                    new_lang = msg.get("language")
                    if new_lang == "auto":
                        lang_code = None
                    elif new_lang:
                        lang_code = new_lang
                    if "use_server_vad" in msg:
                        use_vad = bool(msg["use_server_vad"])
                    await _send({
                        "status": "configured",
                        "language": lang_code or "auto",
                        "use_server_vad": use_vad,
                    })
                elif action == "stop":
                    await _finish_session("stop")
                    break
                else:
                    await _send({
                        "code": "UNKNOWN_ACTION",
                        "message": f"Unknown action: {action!r}",
                        "statusCode": 400,
                    })
                continue

            if "bytes" not in data or not accept_pcm or session_phase != "listening":
                continue

            incoming = data["bytes"]
            if client_sr != TARGET_SR:
                incoming = _resample_pcm_bytes(incoming, client_sr)
            chunk = _pcm_to_float(incoming)
            remaining = max_samples - stream_samples
            if remaining <= 0:
                await _finish_session("max_duration")
                break
            if chunk.size > remaining:
                chunk = chunk[:remaining]

            pre_roll = append_pre_roll(pre_roll, chunk, pre_roll_samples)
            if use_vad:
                event = vad.accept_audio(chunk)
            else:
                event = type("E", (), {"is_speech": True, "speech_ended": False, "silence_ms": 0.0, "committed": False})()
            stream_samples += int(chunk.size)

            if event.is_speech or speech_buf.size > 0:
                if speech_buf.size == 0:
                    unused = unused_pre_roll(pre_roll, chunk)
                    if unused.size:
                        speech_buf = unused
                    onset_n = int(speech_buf.size) + int(chunk.size)
                    buffer_audio_start = (stream_samples - onset_n) / TARGET_SR
                    log.info(
                        "WS onset | cursor={:.3f} pre_roll={:.3f}s audio_start={:.3f}",
                        stream_samples / TARGET_SR,
                        unused.size / TARGET_SR,
                        buffer_audio_start,
                    )
                speech_buf = np.concatenate([speech_buf, chunk]) if speech_buf.size else chunk.copy()

            if stream_samples >= max_samples:
                await _finish_session("max_duration")
                break

            silence_ms = float(getattr(event, "silence_ms", 0.0) or 0.0)
            committed = bool(getattr(event, "committed", False))
            if use_vad and silence_ms <= 0.0 and speculative_task is not None:
                await _cancel_speculative("speech")
            if (
                use_vad
                and WS_SPECULATIVE_FINAL_MS > 0
                and committed
                and silence_ms >= float(WS_SPECULATIVE_FINAL_MS)
                and not event.speech_ended
            ):
                _spawn_speculative()

            if use_vad and event.speech_ended and speech_buf.size > 0:
                log.info(
                    "WS eos | reason=silence cursor={:.3f} buf={:.3f}s silence_ms={:.0f}",
                    stream_samples / TARGET_SR,
                    speech_buf.size / TARGET_SR,
                    silence_ms,
                )
                await _finish_session("silence")
                break

            if (
                session_phase == "listening"
                and silence_ms < float(WS_SPECULATIVE_FINAL_MS or 10**9)
                and speech_buf.size >= last_partial_at + partial_samples
                and speech_buf.size >= partial_samples
            ):
                last_partial_at = speech_buf.size
                _spawn_partial()

    except WebSocketDisconnect:
        log.info("[WS] Client disconnected")
    except Exception as e:
        log.error("WebSocket error: {}", e)
        try:
            await websocket.send_json({"code": "WEBSOCKET_ERROR", "message": str(e), "statusCode": 500})
        except Exception:
            pass
    finally:
        try:
            if session_phase == "listening":
                await _finish_session("disconnect")
        except Exception:
            pass
        backend.clear_encoder_cache(ws_req_id)
        reset_request_id(token)
        async with _ws_lock:
            if _active_ws is websocket:
                _active_ws = None
        _touch()
        try:
            await websocket.close()
        except Exception:
            pass


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host=HOST, port=PORT)
