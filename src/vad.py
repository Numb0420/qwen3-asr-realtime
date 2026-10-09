from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from config import (
    TARGET_SR,
    VAD_MODEL_PATH,
    WS_COMMITTED_SPEECH_SECONDS,
    WS_EARLY_SILENCE_MS,
    WS_EOS_SILENCE_MS,
)
from logger import log


@dataclass
class VadEvent:
    is_speech: bool
    speech_ended: bool
    silence_ms: float = 0.0
    committed: bool = False


def silence_threshold_samples(
    speech_samples: int,
    sample_rate: int,
    eos_ms: int = WS_EOS_SILENCE_MS,
    early_ms: int = WS_EARLY_SILENCE_MS,
    committed_s: float = WS_COMMITTED_SPEECH_SECONDS,
) -> int:
    """Longer silence is required before the utterance has ~0.8s of speech."""
    committed = int(committed_s * sample_rate)
    ms = eos_ms if speech_samples >= committed else early_ms
    return int(sample_rate * (ms / 1000.0))


class RealtimeVad:
    """Streaming Silero VAD via sherpa-onnx. One instance per WS session."""

    def __init__(
        self,
        model_path: Path | None = None,
        sample_rate: int = TARGET_SR,
        eos_ms: int | None = None,
    ):
        self.sample_rate = sample_rate
        self.model_path = Path(model_path or VAD_MODEL_PATH)
        self.eos_ms = int(eos_ms if eos_ms is not None else WS_EOS_SILENCE_MS)
        self._vad = None
        self._window = 512
        self._pending = np.zeros(0, dtype=np.float32)
        self._enabled = False
        self._speech = False
        self._speech_samples = 0
        self._silence_samples = 0
        self._init()

    def _init(self) -> None:
        if not self.model_path.exists():
            log.warning("VAD model missing at {}, falling back to always-speech", self.model_path)
            return
        try:
            import sherpa_onnx

            cfg = sherpa_onnx.VadModelConfig()
            cfg.silero_vad.model = str(self.model_path)
            cfg.silero_vad.threshold = 0.5
            cfg.silero_vad.min_silence_duration = WS_EOS_SILENCE_MS / 1000.0
            cfg.silero_vad.min_speech_duration = 0.15
            cfg.sample_rate = self.sample_rate
            self._vad = sherpa_onnx.VoiceActivityDetector(cfg, buffer_size_in_seconds=60)
            window = getattr(cfg.silero_vad, "window_size", None)
            if window:
                self._window = int(window)
            self._enabled = True
            log.info("RealtimeVad ready | model={} eos_ms={}", self.model_path, self.eos_ms)
        except Exception as e:
            log.error("RealtimeVad init failed: {} — fallback always-speech", e)
            self._vad = None
            self._enabled = False

    def accept_audio(self, pcm: np.ndarray) -> VadEvent:
        samples = np.asarray(pcm, dtype=np.float32).reshape(-1)
        if samples.size == 0:
            return VadEvent(is_speech=self._speech, speech_ended=False)

        if not self._enabled or self._vad is None:
            self._speech = True
            return VadEvent(is_speech=True, speech_ended=False)

        self._pending = np.concatenate([self._pending, samples])
        speech_this = False
        while self._pending.size >= self._window:
            chunk = self._pending[: self._window]
            self._pending = self._pending[self._window :]
            self._vad.accept_waveform(chunk)
            detected = False
            if hasattr(self._vad, "is_speech_detected"):
                try:
                    detected = bool(self._vad.is_speech_detected())
                except Exception:
                    detected = False
            if detected:
                speech_this = True

        if speech_this:
            if not self._speech:
                self._speech_samples = 0
            self._speech = True
            self._speech_samples += int(samples.size)
            self._silence_samples = 0
            return VadEvent(is_speech=True, speech_ended=False, silence_ms=0.0)

        if self._speech:
            self._silence_samples += int(samples.size)
            sil_ms = self._silence_samples / float(self.sample_rate) * 1000.0
            needed = silence_threshold_samples(
                self._speech_samples,
                self.sample_rate,
                self.eos_ms,
                WS_EARLY_SILENCE_MS,
                WS_COMMITTED_SPEECH_SECONDS,
            )
            if self._silence_samples >= needed:
                self._speech = False
                committed = self._speech_samples >= int(
                    WS_COMMITTED_SPEECH_SECONDS * self.sample_rate
                )
                self._speech_samples = 0
                self._silence_samples = 0
                return VadEvent(
                    is_speech=False,
                    speech_ended=True,
                    silence_ms=sil_ms,
                    committed=committed,
                )
            committed = self._speech_samples >= int(
                WS_COMMITTED_SPEECH_SECONDS * self.sample_rate
            )
            return VadEvent(
                is_speech=True,
                speech_ended=False,
                silence_ms=sil_ms,
                committed=committed,
            )

        return VadEvent(is_speech=False, speech_ended=False)

    def reset(self) -> None:
        self._pending = np.zeros(0, dtype=np.float32)
        self._speech = False
        self._speech_samples = 0
        self._silence_samples = 0
        if self._vad is not None and hasattr(self._vad, "reset"):
            try:
                self._vad.reset()
            except Exception:
                pass
