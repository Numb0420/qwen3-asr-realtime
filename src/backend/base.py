from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class BackendState(str, Enum):
    UNLOADED = "UNLOADED"
    LOADING = "LOADING"
    READY = "READY"
    UNLOADING = "UNLOADING"


@dataclass
class WordTimestamp:
    word: str
    start: float
    end: float


@dataclass
class ASRResult:
    text: str
    language: str
    is_final: bool = True
    words: list[WordTimestamp] | None = None
    n_generate: int = 0  # decode token count (0 = likely first-token EOS)
    cancelled: bool = False
    abort_stage: str | None = None
    perf: dict | None = None


class ASRBackend:
    def load(self) -> None:
        raise NotImplementedError

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
        raise NotImplementedError

    def unload(self) -> None:
        raise NotImplementedError

    @property
    def state(self) -> BackendState:
        raise NotImplementedError

    @property
    def loaded(self) -> bool:
        return self.state == BackendState.READY
