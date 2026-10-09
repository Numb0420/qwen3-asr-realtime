"""No-GPU checks for WS VAD timing, pre-roll, and queue coalesce.

    conda activate lingting
    python E2Etest/test_ws_tune.py
"""
from __future__ import annotations

import asyncio
import importlib.util
import inspect
import math
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import numpy as np

from inference_queue import PriorityInferQueue, QueueResult
from vad import RealtimeVad, silence_threshold_samples
from ws_overlap import append_pre_roll, unused_pre_roll

_CC_SPEC = importlib.util.spec_from_file_location(
    "asr_chunk_cache",
    ROOT / "vendor" / "qwen_asr_gguf" / "inference" / "chunk_cache.py",
)
_cc = importlib.util.module_from_spec(_CC_SPEC)
assert _CC_SPEC.loader is not None
_CC_SPEC.loader.exec_module(_cc)


def _cancelled_result(stage: str = "encoder"):
    return type("ASR", (), {"cancelled": True, "abort_stage": stage})()


def _job(name: str, delay: float, poll: float = 0.01):
    def _fn(cancel_event):
        deadline = time.perf_counter() + delay
        while time.perf_counter() < deadline:
            if cancel_event.is_set():
                return _cancelled_result("encoder")
            time.sleep(min(poll, max(0.0, deadline - time.perf_counter())))
        return name
    return _fn


def _assert(cond: bool, msg: str) -> None:
    if not cond:
        raise AssertionError(msg)


def test_two_tier_silence() -> None:
    sr = 16000
    early = silence_threshold_samples(int(0.2 * sr), sr, eos_ms=600, early_ms=1000, committed_s=0.8)
    committed = silence_threshold_samples(int(0.9 * sr), sr, eos_ms=600, early_ms=1000, committed_s=0.8)
    _assert(early == int(sr * 1.0), str(early))
    _assert(committed == int(sr * 0.6), str(committed))


def test_no_rms_fallback() -> None:
    src = inspect.getsource(RealtimeVad.accept_audio)
    _assert("0.01" not in src, "RMS fallback should be removed")
    _assert("sqrt" not in src, src)


async def _test_final_drops_pending_partial() -> None:
    q = PriorityInferQueue()
    q.start()
    try:
        t1 = asyncio.create_task(q.submit(_job("p1", 0.20), priority=1, kind="partial", coalesce_key="ws"))
        await asyncio.sleep(0.03)
        t2 = asyncio.create_task(q.submit(_job("p2", 0.0), priority=1, kind="partial", coalesce_key="ws"))
        t3 = asyncio.create_task(q.submit(_job("f", 0.0), priority=0, kind="final", coalesce_key="ws"))
        r1, r2, r3 = await asyncio.gather(t1, t2, t3)
        _assert(isinstance(r1, QueueResult) and isinstance(r2, QueueResult) and isinstance(r3, QueueResult), "QueueResult")
        _assert(not r3.cancelled and r3.value == "f", f"final should run, got {r3!r}")
        _assert(r2.cancelled and r2.abort_stage in ("pending", "after_final"), f"pending partial should be dropped, got {r2!r}")
        _assert(r1.cancelled and r1.abort_stage in ("encoder", "pending"), f"inflight partial should abort, got {r1!r}")
        t4 = await q.submit(_job("p3", 0.0), priority=1, kind="partial", coalesce_key="ws")
        _assert(t4.cancelled and t4.abort_stage == "after_final", f"late partial after final, got {t4!r}")
        await q.open_session("ws")
        t5 = await q.submit(_job("p4", 0.0), priority=1, kind="partial", coalesce_key="ws")
        _assert(not t5.cancelled and t5.value == "p4", f"new session must accept partials, got {t5!r}")
    finally:
        q.stop()


async def _test_inflight_chunk_abort() -> None:
    q = PriorityInferQueue()
    q.start()

    def chunked(n: int, dt: float):
        def _fn(cancel_event):
            ran = 0
            for _ in range(n):
                if cancel_event.is_set():
                    obj = _cancelled_result("encoder")
                    obj.chunks_ran = ran
                    return obj
                time.sleep(dt)
                ran += 1
            return f"ran{ran}"
        return _fn

    try:
        t0 = time.perf_counter()
        t1 = asyncio.create_task(q.submit(chunked(8, 0.04), priority=1, kind="partial", coalesce_key="sess"))
        for _ in range(50):
            if q._running_job is not None and q._running_job.kind == "partial":
                break
            await asyncio.sleep(0.01)
        _assert(q._running_job is not None, "partial should be inflight before close_session")
        await q.close_session("sess")
        t2 = asyncio.create_task(q.submit(_job("f", 0.0), priority=0, kind="final", coalesce_key="sess"))
        r1, r2 = await asyncio.gather(t1, t2)
        elapsed = time.perf_counter() - t0
        _assert(not r2.cancelled and r2.value == "f", f"final should run, got {r2!r}")
        _assert(r1.cancelled and r1.abort_stage == "encoder", f"chunk abort, got {r1!r}")
        _assert(elapsed < 0.20, f"abort should skip remaining chunks, elapsed={elapsed:.3f}s")
    finally:
        q.stop()


def test_queue_final_drops_partial() -> None:
    asyncio.run(_test_final_drops_pending_partial())


def test_inflight_chunk_abort() -> None:
    asyncio.run(_test_inflight_chunk_abort())


async def _test_speculative_does_not_close() -> None:
    q = PriorityInferQueue()
    q.start()
    try:
        r1 = await q.submit(_job("s", 0.0), priority=0, kind="speculative", coalesce_key="ws")
        r2 = await q.submit(_job("p", 0.0), priority=1, kind="partial", coalesce_key="ws")
        _assert(not r1.cancelled and r1.value == "s", f"speculative should run, got {r1!r}")
        _assert(not r2.cancelled and r2.value == "p", f"speculative must not close session, got {r2!r}")
    finally:
        q.stop()


async def _test_cancel_speculative() -> None:
    q = PriorityInferQueue()
    q.start()

    def chunked(n: int, dt: float):
        def _fn(cancel_event):
            for _ in range(n):
                if cancel_event.is_set():
                    return _cancelled_result("encoder")
                time.sleep(dt)
            return "spec"
        return _fn

    try:
        t1 = asyncio.create_task(
            q.submit(chunked(8, 0.04), priority=0, kind="speculative", coalesce_key="sess")
        )
        for _ in range(50):
            if q._running_job is not None and q._running_job.kind == "speculative":
                break
            await asyncio.sleep(0.01)
        n = await q.cancel_jobs("sess", ("speculative",))
        r1 = await t1
        _assert(n >= 1, "cancel_jobs should hit inflight speculative")
        _assert(r1.cancelled, f"speculative should abort, got {r1!r}")
    finally:
        q.stop()


def test_speculative_does_not_close() -> None:
    asyncio.run(_test_speculative_does_not_close())


def test_cancel_speculative() -> None:
    asyncio.run(_test_cancel_speculative())


def test_pre_roll_unused_excludes_current_packet() -> None:
    max_n = 6400
    ring = np.zeros(0, dtype=np.float32)
    p1 = np.ones(1000, dtype=np.float32)
    ring = append_pre_roll(ring, p1, max_n)
    p2 = np.full(1000, 2.0, dtype=np.float32)
    ring = append_pre_roll(ring, p2, max_n)
    unused = unused_pre_roll(ring, p2)
    _assert(unused.size == 1000, str(unused.size))
    _assert(float(unused[0]) == 1.0, str(unused[0]))
    big = np.ones(8000, dtype=np.float32)
    ring = append_pre_roll(np.zeros(0, dtype=np.float32), big, max_n)
    _assert(unused_pre_roll(ring, big).size == 0, "current packet already covers the ring")


def test_session_finish_paths() -> None:
    src = (Path(__file__).resolve().parents[1] / "src" / "server.py").read_text(encoding="utf-8")
    _assert("async def _finish_session" in src, "unified finish helper missing")
    _assert('_finish_session("silence")' in src, "silence EOS must finish the session")
    _assert('_finish_session("stop")' in src, "client stop must finish the session")
    _assert('_finish_session("max_duration")' in src, "60s cap must finish the session")
    _assert('"eos_reason"' in src, "final must include eos_reason")
    _assert("action == \"flush\"" not in src, "flush must be removed")
    _assert('"chars"' not in src, "final must not emit chars")
    finish = src.split("async def _finish_session", 1)[1].split("try:", 1)[0]
    _assert("global _active_ws" in finish, "nested _finish_session must declare global _active_ws")
    _assert("_drain" not in finish, "CLOSING must not drain inflight partials before final")
    _assert('session_phase = "closing"' in finish, "EOS must enter CLOSING")
    _assert("close_session" in finish, "EOS must preempt the queue before final submit")
    _assert("open_session(ws_req_id)" in src, "new WS must reopen queue for reused session_id")
    _assert('session_phase != "listening"' in src.split("def _spawn_partial", 1)[1][:200], "CLOSING must not spawn partials")
    _assert('priority = 0 if kind in ("final", "speculative") else 1' in src, "final/speculative must outrank partial")
    _assert("WS speculative_submit" in src, "P2 speculative final")
    _assert('"session_id": ws_req_id' in src, "connected handshake includes session_id")
    _assert("encoder_cache" in src or "session_id=session_id" in src, "encoder cache session id")
    _assert("encode_ms" in src, "profiling encode_ms")
    _assert("WS final_submit" in src, "log EOS → final_submit")
    queue = (Path(__file__).resolve().parents[1] / "src" / "inference_queue.py").read_text(encoding="utf-8")
    _assert("WS final_start" in queue, "log final_submit → final_start")
    _assert("cancel_event" in src, "queue cancel must reach ASR")
    backend = (Path(__file__).resolve().parents[1] / "src" / "backend" / "gguf_backend.py").read_text(encoding="utf-8")
    _assert("enable_aligner=False" in backend, "aligner must stay unloaded")
    _assert("do_align=False" in backend, "realtime must not align")
    _assert("REALTIME_CHUNK_SIZE_SEC" in backend.split("def transcribe_realtime")[1], backend)
    _assert("cancel_event=cancel_event" in backend, "backend must forward cancel_event")
    asr = (
        Path(__file__).resolve().parents[1]
        / "vendor"
        / "qwen_asr_gguf"
        / "inference"
        / "asr.py"
    ).read_text(encoding="utf-8")
    _assert('stats["abort_stage"] = "encoder"' in asr, "encoder chunk loop must abort")
    _assert("encoder_cache" in asr, "P3-A encoder cache")
    _assert("configured_spc" in asr, "cache only complete configured 8s windows, not shrunk short utterances")
    _assert("cache_hits" in asr, "cache hit stats")
    _assert("encode_chunks" in asr, "per-chunk encode stats")
    _assert("should_skip_decode" in asr, "P3-A+ skip committed window decode")
    _assert("text_commit_margin_sec" in asr, "P3-A+ safety margin")
    _assert("decode_skipped" in asr, "P3-A+ decode_skipped stats")
    _assert("committed_chunks" in src, "infer_done logs committed_chunks")
    _assert("decode_chunks" in src, "infer_done logs decode_chunks")
    _assert("WS_CHUNK_COMMIT_MARGIN_MS" in backend, "backend forwards commit margin")


def test_eval_cer_and_seq_stem() -> None:
    spec = importlib.util.spec_from_file_location(
        "eval_ws_excel",
        Path(__file__).resolve().parent / "eval_ws_excel.py",
    )
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    _assert(mod.seq_stem(1) == "1", mod.seq_stem(1))
    _assert(mod.seq_stem(1.0) == "1", mod.seq_stem(1.0))
    _assert(mod.cer("你好。", "你好", True) == 0.0, "punct-insensitive")
    _assert(mod.cer("测试。", "测试。", False) == 0.0, "exact")
    _assert(mod.cer("你好世界", "你好", True) == 0.5, str(mod.cer("你好世界", "你好", True)))
    _assert(hasattr(mod, "acquire_eval_lock"), "eval must refuse a second concurrent process")
    _assert(hasattr(mod, "resolve_server_log"), "eval must join daily-rotated server_YYYY-MM-DD.log")


def test_log_daily_rotation() -> None:
    src = (Path(__file__).resolve().parents[1] / "src" / "logger.py").read_text(encoding="utf-8")
    _assert('rotation="00:00"' in src, "server log must rotate at midnight")
    _assert("server_{time:YYYY-MM-DD}.log" in src, "daily log filename")
    _assert("retention=\"14 days\"" in src, "keep two weeks of daily logs")


def test_chunk_commit_ready_margin() -> None:
    ready = _cc.chunk_commit_ready
    kw = dict(chunk_size_sec=8.0, margin_sec=0.4, full_window=True)
    _assert(not ready(0, total_duration=7.7, was_last=True, **kw), "7.7s last window")
    _assert(not ready(0, total_duration=8.0, was_last=True, **kw), "8.0s last flush must not freeze")
    _assert(not ready(0, total_duration=8.2, was_last=False, **kw), "8.2s still inside safety margin")
    _assert(ready(0, total_duration=8.4, was_last=False, **kw), "8.4s non-last may freeze chunk0")
    _assert(ready(0, total_duration=8.5, was_last=False, **kw), "8.5s non-last may freeze chunk0")
    _assert(not ready(1, total_duration=15.7, was_last=False, **kw), "15.7s chunk1 still inside margin")
    _assert(not ready(1, total_duration=16.0, was_last=True, **kw), "16.0s last must not freeze chunk1")
    _assert(not ready(1, total_duration=16.3, was_last=False, **kw), "16.3s with 0.4 margin is not ready")
    _assert(ready(1, total_duration=16.4, was_last=False, **kw), "16.4s non-last may freeze chunk1")
    _assert(not ready(0, total_duration=20.0, was_last=False, full_window=False, chunk_size_sec=8.0, margin_sec=0.4), "tail never commits")


def test_text_freeze_and_skip() -> None:
    cache: dict = {}
    _cc.set_cache_embd(cache, 0, "e0")
    _assert(_cc.commit_chunk_text(cache, 0, "会议纪"), "first freeze")
    _assert(not _cc.commit_chunk_text(cache, 0, "会议纪要"), "frozen text must not change")
    skip, text = _cc.should_skip_decode(cache, 0, was_last=False)
    _assert(skip and text == "会议纪", text)
    skip_last, _ = _cc.should_skip_decode(cache, 0, was_last=True)
    _assert(not skip_last, "last window must still decode / flush rollback")


def test_p3a_plus_replay_concat() -> None:
    """Growing-buffer: freeze non-last text0, later requests skip decode of chunk0."""
    cache: dict = {}
    decoded: list[tuple] = []
    prefixes: list[tuple] = []
    chunk = 8.0
    margin = 0.4

    def run(dur: float) -> str:
        num = 1 if dur <= chunk + 0.05 else int(math.ceil(dur / chunk))
        parts: list[str] = []
        memory: list[tuple] = []
        for i in range(num):
            start = i * chunk
            end = min((i + 1) * chunk, dur)
            full_window = abs((end - start) - chunk) < 1e-9
            was_last = i == num - 1
            if full_window:
                _cc.set_cache_embd(cache, i, f"e{i}")
            skip, frozen = _cc.should_skip_decode(cache, i, was_last=was_last)
            if skip:
                parts.append(frozen)
                memory.append((f"e{i}", frozen))
                continue
            prefix = "".join(m[1] for m in memory)
            prefixes.append((dur, i, prefix, [m[0] for m in memory]))
            # last-flush analogue: extra 'L' only when this window is last
            text = f"C{i}" + ("L" if was_last else "")
            decoded.append((dur, i, was_last))
            parts.append(text)
            memory.append((f"e{i}", text))
            if _cc.chunk_commit_ready(
                i,
                chunk_size_sec=chunk,
                total_duration=dur,
                margin_sec=margin,
                was_last=was_last,
                full_window=full_window,
            ):
                _cc.commit_chunk_text(cache, i, text)
        return "".join(parts)

    _assert(run(5.0) == "C0L", "short utterance unchanged")
    _assert(run(8.0) == "C0L", "exactly 8s still last-flush, not frozen")
    _assert(run(8.2) == "C0C1L", "8.2s decodes chunk0 as non-last but does not freeze")
    _assert(run(8.5) == "C0C1L", "8.5s freezes non-last C0")
    _assert(run(14.0) == "C0C1L", "14s concatenates frozen C0 + tail")
    _assert((14.0, 0, False) not in decoded, "14s must not re-decode committed chunk0")
    _assert(any(row[0] == 14.0 and row[1] == 1 for row in decoded), "14s still decodes tail")
    tail_prefix = [row for row in prefixes if row[0] == 14.0 and row[1] == 1]
    _assert(tail_prefix and tail_prefix[0][2] == "C0", tail_prefix)
    _assert(tail_prefix[0][3] == ["e0"], "tail decode must keep previous window embedding")


if __name__ == "__main__":
    test_two_tier_silence()
    print("ok test_two_tier_silence")
    test_no_rms_fallback()
    print("ok test_no_rms_fallback")
    test_queue_final_drops_partial()
    print("ok test_queue_final_drops_partial")
    test_inflight_chunk_abort()
    print("ok test_inflight_chunk_abort")
    test_speculative_does_not_close()
    print("ok test_speculative_does_not_close")
    test_cancel_speculative()
    print("ok test_cancel_speculative")
    test_pre_roll_unused_excludes_current_packet()
    print("ok test_pre_roll_unused_excludes_current_packet")
    test_session_finish_paths()
    print("ok test_session_finish_paths")
    test_chunk_commit_ready_margin()
    print("ok test_chunk_commit_ready_margin")
    test_text_freeze_and_skip()
    print("ok test_text_freeze_and_skip")
    test_p3a_plus_replay_concat()
    print("ok test_p3a_plus_replay_concat")
    test_eval_cer_and_seq_stem()
    print("ok test_eval_cer_and_seq_stem")
    test_log_daily_rotation()
    print("ok test_log_daily_rotation")
    print("ws_tune tests passed")
