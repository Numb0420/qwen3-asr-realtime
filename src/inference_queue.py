from __future__ import annotations

import asyncio
import concurrent.futures
import dataclasses
import heapq
import threading
import time
from typing import Any, Callable, Literal

from logger import log

JobKind = Literal["normal", "partial", "final", "speculative"]

_infer_executor = concurrent.futures.ThreadPoolExecutor(
    max_workers=1,
    thread_name_prefix="qwen3-asr-infer",
)


@dataclasses.dataclass
class QueueResult:
    value: Any = None
    job_id: int = 0
    kind: JobKind = "normal"
    cancelled: bool = False
    abort_stage: str | None = None
    queue_wait_ms: float = 0.0
    infer_ms: float = 0.0
    enqueue_ts: float = 0.0
    start_ts: float = 0.0
    finish_ts: float = 0.0


@dataclasses.dataclass(order=True)
class _InferJob:
    priority: int
    submit_time: float
    seq: int
    future: asyncio.Future = dataclasses.field(compare=False)
    fn: Callable = dataclasses.field(compare=False)
    kind: JobKind = dataclasses.field(default="normal", compare=False)
    coalesce_key: str | None = dataclasses.field(default=None, compare=False)
    cancel: threading.Event = dataclasses.field(default_factory=threading.Event, compare=False)
    enqueue_ts: float = dataclasses.field(default=0.0, compare=False)


class PriorityInferQueue:
    """Single-worker queue. final (0) preempts pending partial (1)."""

    def __init__(self):
        self._heap: list[_InferJob] = []
        self._lock = asyncio.Lock()
        self._has_work = asyncio.Event()
        self._worker_task: asyncio.Task | None = None
        self._running = 0
        self._running_job: _InferJob | None = None
        self._seq = 0
        self._idle = asyncio.Event()
        self._idle.set()
        self._closed_keys: set[str] = set()

    def start(self):
        if self._worker_task is None or self._worker_task.done():
            self._worker_task = asyncio.create_task(self._worker())

    def stop(self):
        if self._worker_task:
            self._worker_task.cancel()
            self._worker_task = None

    @property
    def busy(self) -> bool:
        return self._running > 0 or bool(self._heap)

    async def wait_idle(self) -> None:
        await self._idle.wait()

    async def run_blocking(self, fn: Callable, priority: int = 0) -> Any:
        def _wrapped(_cancel: threading.Event):
            return fn()

        ticket = await self.submit(_wrapped, priority=priority, kind="final")
        if ticket.cancelled:
            return None
        return ticket.value

    def _ticket(
        self,
        job: _InferJob,
        *,
        value: Any = None,
        cancelled: bool = False,
        abort_stage: str | None = None,
        start_ts: float = 0.0,
        finish_ts: float | None = None,
        infer_ms: float = 0.0,
    ) -> QueueResult:
        done = time.perf_counter() if finish_ts is None else finish_ts
        started = start_ts or job.enqueue_ts
        return QueueResult(
            value=value,
            job_id=job.seq,
            kind=job.kind,
            cancelled=cancelled,
            abort_stage=abort_stage,
            queue_wait_ms=max(0.0, (started - job.enqueue_ts) * 1000.0),
            infer_ms=infer_ms,
            enqueue_ts=job.enqueue_ts,
            start_ts=started,
            finish_ts=done,
        )

    async def _worker(self):
        loop = asyncio.get_event_loop()
        while True:
            await self._has_work.wait()
            async with self._lock:
                if not self._heap:
                    self._has_work.clear()
                    if self._running == 0:
                        self._idle.set()
                    continue
                job = heapq.heappop(self._heap)
                self._running += 1
                self._running_job = job
                self._idle.clear()
            start_ts = time.perf_counter()
            queue_wait_ms = max(0.0, (start_ts - job.enqueue_ts) * 1000.0)
            log.info(
                "WS infer_start | job_id={} kind={} queue_wait_ms={:.0f}",
                job.seq,
                job.kind,
                queue_wait_ms,
            )
            if job.kind == "final":
                log.info(
                    "WS final_start | job_id={} queue_wait_ms={:.0f}",
                    job.seq,
                    queue_wait_ms,
                )
            elif job.kind == "speculative":
                log.info(
                    "WS speculative_start | job_id={} queue_wait_ms={:.0f}",
                    job.seq,
                    queue_wait_ms,
                )
            try:
                if job.cancel.is_set():
                    result = self._ticket(
                        job,
                        cancelled=True,
                        abort_stage="pending",
                        start_ts=start_ts,
                        infer_ms=0.0,
                    )
                else:

                    def _run():
                        return job.fn(job.cancel)

                    value = await loop.run_in_executor(_infer_executor, _run)
                    finish_ts = time.perf_counter()
                    abort_stage = None
                    cancelled = False
                    if isinstance(value, QueueResult):
                        result = value
                    else:
                        if getattr(value, "cancelled", False):
                            cancelled = True
                            abort_stage = getattr(value, "abort_stage", None) or "encoder"
                        result = self._ticket(
                            job,
                            value=value,
                            cancelled=cancelled,
                            abort_stage=abort_stage,
                            start_ts=start_ts,
                            finish_ts=finish_ts,
                            infer_ms=(finish_ts - start_ts) * 1000.0,
                        )
                if not job.future.done():
                    job.future.set_result(result)
            except Exception as e:
                if not job.future.done():
                    job.future.set_exception(e)
            finally:
                async with self._lock:
                    self._running -= 1
                    if self._running_job is job:
                        self._running_job = None
                    if self._running == 0 and not self._heap:
                        self._idle.set()
                        self._has_work.clear()

    async def submit(
        self,
        fn: Callable,
        priority: int = 1,
        kind: JobKind = "normal",
        coalesce_key: str | None = None,
    ) -> QueueResult:
        loop = asyncio.get_event_loop()
        future = loop.create_future()
        async with self._lock:
            if kind == "final" and coalesce_key:
                self._closed_keys.add(coalesce_key)
            if kind in ("partial", "speculative") and coalesce_key and coalesce_key in self._closed_keys:
                self._seq += 1
                now = time.perf_counter()
                stub = _InferJob(
                    priority=priority,
                    submit_time=time.time(),
                    seq=self._seq,
                    future=future,
                    fn=fn,
                    kind=kind,
                    coalesce_key=coalesce_key,
                    enqueue_ts=now,
                )
                log.info(
                    "WS infer_cancel | job_id={} kind={} stage=after_final coalesce={}",
                    stub.seq,
                    kind,
                    coalesce_key,
                )
                return self._ticket(
                    stub,
                    cancelled=True,
                    abort_stage="after_final",
                    start_ts=now,
                    finish_ts=now,
                    infer_ms=0.0,
                )
            dropped = 0
            if coalesce_key and kind in ("partial", "final", "speculative"):
                dropped = self._drop_pending_partials(coalesce_key)
            if coalesce_key and kind in ("final", "speculative"):
                running = self._running_job
                if (
                    running is not None
                    and running.kind == "partial"
                    and running.coalesce_key == coalesce_key
                    and not running.cancel.is_set()
                ):
                    running.cancel.set()
                    log.info(
                        "WS infer_cancel | job_id={} kind=partial stage=inflight coalesce={}",
                        running.seq,
                        coalesce_key,
                    )
            self._seq += 1
            now = time.perf_counter()
            job = _InferJob(
                priority=priority,
                submit_time=time.time(),
                seq=self._seq,
                future=future,
                fn=fn,
                kind=kind,
                coalesce_key=coalesce_key,
                enqueue_ts=now,
            )
            if dropped:
                log.info(
                    "WS infer_drop_pending | dropped={} kind={} job_id={} coalesce={}",
                    dropped,
                    kind,
                    job.seq,
                    coalesce_key,
                )
            log.info(
                "WS infer_enqueue | job_id={} kind={} priority={} coalesce={}",
                job.seq,
                kind,
                priority,
                coalesce_key,
            )
            heapq.heappush(self._heap, job)
            self._idle.clear()
            self._has_work.set()
        return await future

    def _drop_pending_partials(self, coalesce_key: str) -> int:
        kept: list[_InferJob] = []
        dropped = 0
        now = time.perf_counter()
        for job in self._heap:
            if job.kind == "partial" and job.coalesce_key == coalesce_key and not job.future.done():
                job.cancel.set()
                job.future.set_result(
                    self._ticket(
                        job,
                        cancelled=True,
                        abort_stage="pending",
                        start_ts=now,
                        finish_ts=now,
                        infer_ms=0.0,
                    )
                )
                dropped += 1
                log.info(
                    "WS infer_cancel | job_id={} kind=partial stage=pending coalesce={}",
                    job.seq,
                    coalesce_key,
                )
                continue
            kept.append(job)
        self._heap = kept
        heapq.heapify(self._heap)
        return dropped

    async def open_session(self, coalesce_key: str) -> None:
        """New WebSocket: allow partials again even if this session_id was used before."""
        if not coalesce_key:
            return
        async with self._lock:
            self._closed_keys.discard(coalesce_key)

    async def close_session(self, coalesce_key: str) -> None:
        """EOS/CLOSING: forbid new partials, drop pending, cancel inflight encoder chunks."""
        if not coalesce_key:
            return
        async with self._lock:
            self._closed_keys.add(coalesce_key)
            dropped = self._drop_pending_partials(coalesce_key)
            running = self._running_job
            inflight = (
                running is not None
                and running.kind == "partial"
                and running.coalesce_key == coalesce_key
            )
            if inflight and not running.cancel.is_set():
                running.cancel.set()
                log.info(
                    "WS infer_cancel | job_id={} kind=partial stage=inflight coalesce={}",
                    running.seq,
                    coalesce_key,
                )
            log.info(
                "WS session_preempt | coalesce={} dropped={} inflight={}",
                coalesce_key,
                dropped,
                int(bool(inflight)),
            )

    async def cancel_jobs(self, coalesce_key: str, kinds: tuple[str, ...]) -> int:
        """Cancel pending+inflight jobs of the given kinds (speech resume vs speculative)."""
        if not coalesce_key or not kinds:
            return 0
        async with self._lock:
            kept: list[_InferJob] = []
            dropped = 0
            now = time.perf_counter()
            for job in self._heap:
                if job.kind in kinds and job.coalesce_key == coalesce_key and not job.future.done():
                    job.cancel.set()
                    job.future.set_result(
                        self._ticket(
                            job,
                            cancelled=True,
                            abort_stage="resume",
                            start_ts=now,
                            finish_ts=now,
                            infer_ms=0.0,
                        )
                    )
                    dropped += 1
                    log.info(
                        "WS infer_cancel | job_id={} kind={} stage=pending coalesce={}",
                        job.seq,
                        job.kind,
                        coalesce_key,
                    )
                    continue
                kept.append(job)
            self._heap = kept
            heapq.heapify(self._heap)
            running = self._running_job
            if (
                running is not None
                and running.kind in kinds
                and running.coalesce_key == coalesce_key
                and not running.cancel.is_set()
            ):
                running.cancel.set()
                dropped += 1
                log.info(
                    "WS infer_cancel | job_id={} kind={} stage=inflight coalesce={}",
                    running.seq,
                    running.kind,
                    coalesce_key,
                )
            return dropped


infer_queue = PriorityInferQueue()
