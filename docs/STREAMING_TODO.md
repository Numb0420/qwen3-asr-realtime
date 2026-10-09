# Streaming ASR 待办（对照用）

> 2026-09-16。P0/P1 已落地。本文记录后续分层，避免把「调度」和「模型重算」再混在一起。

## 已经成立的结论

P0/P1（`final` 抢占 + pending drop + inflight 在 encoder 8s 边界 abort + `CLOSING`）已经把调度浪费基本清掉。

对照数据（重启后）：

| 场景 | EOS → emit | 其中排队 | 其中推理 |
|---|---|---|---|
| ≤5s 短句 | 0.45–0.9s | 0–350ms（当前窗收尾） | ~0.45–0.53s |
| 15s | 1.67s | 272ms | 1.39s |
| 19s | 2.87s（改前同量级约 5.7s） | 453ms | 2.41s |

`queue_wait≈453ms` 正好对应当前 encoder chunk 收尾，没有再跑后续旧 partial。

8s 是明显计算台阶：每次 partial/final 都从 0 秒重新跑整段 growing-buffer。

```text
<8s   1 个 encoder window    7.7s → 718ms
8~16s 2 个 window            8.5s → 1324ms
16~24s 3 个 window           19s  → ~2.4s
```

真正浪费已经不是 queue，而是：用户已经看到大部分 partial 之后，EOS 时 Qwen 又从 0 秒 encode + decode。

单次 19s `asr()` **内部已经分窗**（0–8、8–16、16–19 pad 到 8s），且每次 decode 都 `clear_kv_cache()`。重复发生在 **跨请求**：8s partial、12s partial、19s final 各自再跑一遍整条链。未满 8s 的尾窗每次都被 pad 成整窗，**不能缓存**。

`encode_time / prefill_time / decode_time` 引擎里已有，此前没有打进 WS 日志，所以还不能凭总 `infer_ms` 押 encoder vs decoder。

---

## 落地顺序（严格按此对照）

| # | 项 | 状态 | 说明 |
|---|---|---|---|
| ① | Profiling：encode / prefill / decode / per-chunk / cache_hit | **已做** | `infer_done` 已带 encode_ms/prefill_ms/decode_ms/encode_chunks |
| ② | P2 speculative final | **已做** | `WS_SPECULATIVE_FINAL_MS=300`；committed 后开跑；开口 cancel |
| ③ | P3-A 完整 8s encoder 窗缓存 | **已做** | session 内复用满窗 embedding；尾窗不缓存；看 `cache_hits` |
| ④ | P3-A+ 满窗 text 提交，final 只跑尾窗 | **已做** | 安全余量 400ms；skip 已冻结窗 decode；尾窗仍吃上一窗 embd+prefix |
| ⑤ | 4s / 6s / 8s chunk A/B | 未做 | 有 per-chunk 数再测；**不改默认 8s** |
| ⑥ | P3-B decoder KV reuse | 明确不做 | 每次 decode 都清 KV；audio embedding 一变旧 KV 失效 |

产品节奏：P0/P1 + 当前整段 final 已接近 3–10s 指令可用。15–60s 靠 ③ embedding cache + ④ 满窗 text 冻结。不要为少数长句先上完整 streaming KV。不改默认 8s chunk。

---

## ① Profiling（必做）

`infer_done` 增加：

```text
encode_ms  prefill_ms  decode_ms
n_chunks  cache_hits
encode_chunks   # 例 0:520:hit,1:510:hit,2:280:miss
mel_ms / frontend_ms / backend_ms  按窗累计或写在 encode_chunks 里
```

看 19s final 的 2410ms 怎么拆。encoder 过半 → ③ 是下一刀；decoder 过半 → 再讨论 ④/⑥。

## ② P2 speculative final

```text
committed 语音（≥ ~0.8s）
silence ≥ 300ms  → 提交 speculative（priority=0，不进入 CLOSING）
silence ≥ 600ms  → EOS，复用已在跑/已完成的 speculative
用户又开口      → cancel speculative，继续 LISTENING
短句（未 committed）→ 仍走 WS_EARLY_SILENCE_MS=1000，不开投机
```

预期：短句 EOS→出字藏掉约 300ms；19s 长句只藏 300ms，2.4s 重算还在。假 final 占 GPU 时必须能被开口抢占（依赖 P1 `cancel_event`）。

环境变量：`WS_SPECULATIVE_FINAL_MS`（默认 300，0=关闭）。

## ③ P3-A encoder cache

```text
session cache:
  chunk0 = encode(0~8)    满窗，可复用
  chunk1 = encode(8~16)   满窗，可复用
  tail   = 永远不缓存     pad 形态每次都变
```

满窗判定必须用配置的 `chunk_size_sec`（8s），**不能**用短句那条 `samples_per_chunk = total_len` 的收缩。否则 0.8s partial 会被当成 chunk0 写入，后续 1.6s/3s/6s 全部 `cache hit` 复用「测试」那一段 embedding。

## ④ P3-A+（已做）

满窗 text 冻结后，后续 partial/final 不再 decode 该窗，只跑尾窗。尾窗 decode **仍走现有 memory**：上一窗 embedding 拼进 `combined_audio`，committed text 作为 `prefix_text`，再 `committed + tail` 对外拼接。不要做成「只听尾窗音频再字符串拼接」。

```text
session cache[i]:
  embd       满配置窗 embedding（P3-A）
  text       冻结文本
  committed  一旦 True 不再改写

commit 条件（同时满足）：
  full_window（configured 8s，不是短句收缩窗）
  该窗不是当前请求的 last（避开 is_last_chunk flush rollback）
  current_audio >= chunk_end + WS_CHUNK_COMMIT_MARGIN_MS（默认 400ms）

skip：非 last 且已 committed 且 embd 在 → 不 decode，把 (embd, text) 写入 asr_memory
```

短句 <8s：零触发，路径与 ③ 相同。

对照日志：`committed_chunks` / `decode_skipped` / `decode_chunks`（例 `0:skip,1:412`）。14s 成功时 decode 应接近单窗，而不是再 ~900ms 两窗之和。

边界验收（刻意跨窗词）：7.7 / 8.0 / 8.2 / 8.5 / 15.7 / 16.0 / 16.3s。看漏词、重复、标点、上下文断裂。不改默认 chunk size。

## ⑤ chunk 尺寸 A/B（暂缓）

P1 残留 270–450ms 来自当前 8s 窗。`DML_PAD_TO` 跟着 chunk 走，改 4s 可能 abort 快、总 infer 更差。测 5s/15s/30s × 4/6/8s 后再决定。

## ⑥ P3-B KV（不做）

引擎已有 `rollback_num=5`，那是单窗内部稳定输出，不是跨窗 KV。

---

## 理想架构（④ 之后才接近）

```text
④ 之后：
已冻结窗 text+embd ████████████
新增尾窗                         ███
只 encode/decode 尾窗（prompt 仍带上一窗 embd+prefix）
```

对照日志时先看：`WS speculative_submit` → `WS eos` → `final_start queue_wait` → `infer_done encode_ms/prefill_ms/decode_ms/cache_hits/committed_chunks/decode_skipped/decode_chunks`。
