from pydantic import BaseModel, Field
from typing import Optional


class ErrorResponse(BaseModel):
    code: str = Field(..., description="Machine-readable error identifier")
    message: str = Field(..., description="Human-readable error description")
    context: Optional[dict] = Field(None, description="Debug data")
    statusCode: int = Field(..., description="HTTP status code")


class HealthResponse(BaseModel):
    status: str = Field(..., description="Status of the service", examples=["ok"])
    model_loaded: bool = Field(..., description="Whether the GGUF engine is loaded")
    backend: str = Field(..., description="Inference backend", examples=["gguf"])
    state: str = Field(..., description="Backend lifecycle state")
    model_id: Optional[str] = Field(None, description="Decoder filename")
    onnx_provider: Optional[str] = Field(None, description="ONNX EP", examples=["DML"])
    llm_gpu: Optional[str] = Field(None, description="Decoder accelerator", examples=["vulkan"])
    format_num: bool = Field(False, description="Whether Chinese number ITN is enabled")
    max_audio_seconds: float = Field(60, description="Max PCM seconds per WebSocket session")
    eos_silence_ms: int = Field(600, description="Backend endpoint silence in milliseconds")


API_TAGS = [
    {
        "name": "Streaming",
        "description": "Single-connection WebSocket real-time transcription.",
    },
    {
        "name": "System",
        "description": "Health checks and model status.",
    },
]

API_DESCRIPTION = """\
Local real-time speech-to-text using Qwen3-ASR (ONNX Encoder + GGUF Decoder).

## Features
- Single-connection WebSocket `/realtime/stream`
- One session, one utterance: client `data.status=2`, silence EOS, or 60s max → final then close
- Windows local runtime (DirectML + Vulkan), no Docker / PyTorch

## Audio
Inbound JSON envelope `{ "data": { "status", "format", "eos?", "audio" } }`.
`format` is `audio/L16;rate=16000` or `audio/L16;rate=8000`. `audio` is base64 s16le PCM.
"""
