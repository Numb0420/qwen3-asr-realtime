# Qwen3-ASR Realtime

Windows 本地实时语音转写服务：[Qwen3-ASR](https://github.com/QwenLM/Qwen3-ASR) 的 **ONNX Encoder（DirectML）+ GGUF Decoder（llama.cpp / Vulkan）**。

只提供实时 WebSocket。一次连接转一句，没有离线 HTTP 转写，也没有 Docker / PyTorch 运行时。

仓库：<https://github.com/Numb0420/qwen3-asr-realtime>

## 功能

- **WebSocket 实时转写** — `WS /realtime/stream`，原始 PCM（s16le，单声道，8 kHz 或 16 kHz）
- **一连接一句** — 静音 EOS、客户端 `{"action":"stop"}`、或累计 16 kHz PCM 满 `WS_MAX_AUDIO_SECONDS`（≤ 60）→ `final` → `closed` → 断开
- **Partial 整段替换** — 客户端覆盖显示，不要拼接。`final` 只有 `text` / `start` / `end` / `eos_reason`，没有 `chars`
- **服务端 VAD** — Silero：起音前静音不缓冲，pre-roll 保留约 400ms；说话后静音达到阈值结束
- **单路并发** — 同时只允许一路 WebSocket；第二路返回 `WS_BUSY`（409）。GPU 推理单线程排队，final 可抢占 pending/inflight partial
- **8s 窗 + encoder cache** — 长句按 8 秒切开；满窗 embedding 可复用，满窗文本可冻结后只解尾巴
- **Final 后处理** — 中文数字 ITN、音素热词纠错（partial 不做）
- **本机网页** — `GET /` 麦克风页：开始推流，停止发 `stop`
- **绿色包** — `packaging/build.ps1` → `dist/ASR-REALTIME/`，托盘 `ASRRealtime.exe`

## 环境要求

- Windows + 支持 DirectML 的 GPU（解码走 Vulkan）
- conda 环境 **`lingting`**（Python 3.10），不要用 `base`
- 模型放到 `MODEL_DIR`（默认 `./models/`）：
  - `qwen3_asr_llm.q5_k.gguf`
  - `qwen3_asr_encoder_frontend.int8.onnx`
  - `qwen3_asr_encoder_backend.int8.onnx`
  - `silero_vad.onnx`
- llama.cpp Windows Vulkan DLL 放到 `vendor/qwen_asr_gguf/inference/bin/`（版本见 [`vendor/VENDOR.md`](vendor/VENDOR.md)）

不加载 ForcedAligner。

## 快速开始

```powershell
conda activate lingting
pip install -r requirements.txt
# 按本机路径改 .env（HOST / PORT / MODEL_DIR / HOTWORDS_PATH）
python run.py
```

默认代码监听 `127.0.0.1:8765`。仓库里的 `.env` 当前是 `0.0.0.0:8766`，以 `.env` 为准。

```powershell
curl http://127.0.0.1:8766/health
```

浏览器打开同一地址的 `/` 即可用麦克风试转写。

手工客户端（服务已启动，脚本顶部设 `WAV_PATH`）：

```powershell
conda activate lingting
python E2Etest/test_ws_tune.py
python E2Etest/test_websocket.py
python E2Etest/test_websocket_long.py
```

## 接口

| 接口 | 方法 | 说明 |
|---|---|---|
| `/` | GET | 麦克风测试页 |
| `/health` | GET | 模型状态、EOS / 最大时长 |
| `/admin/warmup` | POST | 加载模型 |
| `/admin/shutdown` | POST | 停止进程 |
| `/realtime/stream` | WebSocket | 实时转写 |
| `/docs` | GET | Swagger |

没有 `/v1/audio/transcriptions`、SSE、字幕接口。

### `GET /health`

```json
{
  "status": "ok",
  "model_loaded": true,
  "backend": "gguf",
  "state": "ready",
  "model_id": "qwen3_asr_llm.q5_k.gguf",
  "onnx_provider": "DML",
  "llm_gpu": "vulkan",
  "format_num": true,
  "max_audio_seconds": 60.0,
  "eos_silence_ms": 600
}
```

### `WS /realtime/stream`

Query：

| 参数 | 说明 |
|---|---|
| `sample_rate` | `8000` 或 `16000` |
| `use_server_vad` | 覆盖 `ASR_USE_SERVER_VAD` |
| `request_id` | 会话 id / 日志关联；缺省生成 UUID |

二进制帧 = PCM。JSON 控制：`config`（语言 / VAD）、`reset`（清缓冲、保持连接）、`stop`。

握手后服务端先发 `connected`，然后在说话过程中发 `type=partial`。三种结束路径等价：先 `type=final`，再 `{ "status": "closed", "reason": "..." }`，然后断开。

`eos_reason` / `closed.reason`：`silence` | `stop` | `max_duration` | `disconnect`。

60 秒上限计的是**累计 16 kHz PCM**，不是连接墙钟。只连不送音频不会因满 60 秒断开。

## 配置（`.env`）

`run.py` 启动时读取根目录 `.env`。代码默认与当前仓库 `.env` 不一致时，以 `.env` 为准。

| 变量 | 代码默认 | 含义 |
|---|---|---|
| `HOST` / `PORT` | `127.0.0.1` / `8765` | 监听地址 |
| `MODEL_DIR` | `./models` | GGUF / ONNX / VAD |
| `LLM_FN` | `qwen3_asr_llm.q5_k.gguf` | 解码器 |
| `ONNX_PROVIDER` | `DML` | Encoder EP |
| `LLM_USE_GPU` | `true` | Vulkan 解码 |
| `WS_MAX_AUDIO_SECONDS` | `60` | 句长上限（必须 ≤ 60） |
| `WS_EOS_SILENCE_MS` | `600` | 后端点静音 |
| `WS_EARLY_SILENCE_MS` | `1000` | 短句（未 committed）更长静音 |
| `WS_PARTIAL_SECONDS` | `1.5` | partial 间隔（`.env` 现为 `0.8`） |
| `WS_SPECULATIVE_FINAL_MS` | `300` | 投机跑 final；`0`=关 |
| `WS_CHUNK_COMMIT_MARGIN_MS` | `400` | 满 8s 窗文本冻结边距 |
| `WS_VAD_PRE_ROLL_MS` | `400` | 起音 pre-roll |
| `ASR_USE_SERVER_VAD` | `true` | 可被连接 query / `config` 覆盖 |
| `UNLOAD_GRACE_S` | `0` | `0`=模型常驻 |
| `REALTIME_CHUNK_SIZE_SEC` | `8` | 编码器窗，不要改默认 |
| `LOG_DIR` | `./logs` | 按天滚动 |
| `REQUEST_TIMEOUT` | `300` | 单次推理超时秒 |
| `HOTWORD_ENABLED` | `true` | final 热词纠错 |
| `HOTWORDS_PATH` | `./hotwords.txt` | 热词文件 |

更细的状态机、队列、cache 见 [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md)。未做项见 [`docs/STREAMING_TODO.md`](docs/STREAMING_TODO.md)。

## 目录

```
stt-qwen3-realtime/
├── run.py                 # 读 .env、单实例、启动 uvicorn
├── .env                   # 本机配置（克隆后按路径改 MODEL_DIR / HOTWORDS_PATH）
├── requirements.txt
├── src/
│   ├── server.py          # FastAPI + WebSocket 会话
│   ├── inference_queue.py # 单线程 GPU 队列
│   ├── vad.py             # Silero 后端点
│   ├── backend/           # GGUF / ONNX 加载
│   └── tray_app.py        # 绿色包托盘
├── vendor/qwen_asr_gguf/  # 8s 分窗 ASR + encoder cache（不含 DLL）
├── web/index.html
├── E2Etest/
├── packaging/             # 绿色包构建
├── docs/
└── models/                # 本地模型，不进 git
```

## 打包

不要直接压缩本仓库给人用。开发机：

```powershell
conda activate lingting
pip install -r packaging/requirements-build.txt
powershell -ExecutionPolicy Bypass -File packaging/build.ps1
```

产物：`dist/ASR-REALTIME/`（不含模型）。对方把 GGUF / ONNX / VAD 放进包内 `models/`，双击 `ASRRealtime.exe`。说明见 [`packaging/README.md`](packaging/README.md)。

## 致谢

- [Qwen3-ASR](https://github.com/QwenLM/Qwen3-ASR)
- [Qwen3-ASR-GGUF](https://github.com/HaujetZhao/Qwen3-ASR-GGUF)
- [llama.cpp](https://github.com/ggml-org/llama.cpp)
