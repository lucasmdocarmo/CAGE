# KV Cache Format Specification

**Version:** cage-kv v1.0
**Last Updated:** 2026-04-08

## 1. Overview

This document specifies a binary format for storing and exchanging pre-computed KV (Key-Value) cache states. This format is designed for potential future use in CAGE for cache persistence and cross-node transfer experiments.

**Current status:** Specification only. Phase 1 uses vLLM's built-in prefix caching, which manages KV blocks internally. This format would be used if CAGE needs to persist or transfer KV states outside vLLM.

## 2. File Structure

```
┌──────────────────────────────────────────┐
│           FILE HEADER (64 bytes)          │
├──────────────────────────────────────────┤
│         LAYER HEADERS (variable)          │
├──────────────────────────────────────────┤
│          KV DATA BLOCKS (variable)        │
└──────────────────────────────────────────┘
```

## 3. File Header (64 bytes)

| Offset | Size | Type | Field | Description |
|---|---|---|---|---|
| 0 | 4 | char[4] | magic | "CAGE" |
| 4 | 2 | uint16 | version_major | 1 |
| 6 | 2 | uint16 | version_minor | 0 |
| 8 | 4 | uint32 | num_layers | Transformer layers |
| 12 | 4 | uint32 | num_heads | Attention heads per layer |
| 16 | 4 | uint32 | head_dim | Dimension per head |
| 20 | 4 | uint32 | hidden_size | num_heads × head_dim |
| 24 | 4 | uint32 | max_seq_length | Max cached sequence |
| 28 | 4 | uint32 | actual_seq_length | Actual cached length |
| 32 | 1 | uint8 | dtype | 0=fp32, 1=fp16, 2=bf16, 3=int8 |
| 33 | 1 | uint8 | endianness | 0=little, 1=big |
| 34 | 2 | uint16 | flags | Reserved |
| 36 | 20 | char[20] | model_hash | Model identifier prefix |
| 56 | 8 | uint64 | checksum | CRC64 of data blocks |

## 4. KV Data Layout

Per layer:
```
Key tensor:   [num_kv_heads, seq_length, head_dim]  (row-major)
Value tensor: [num_kv_heads, seq_length, head_dim]  (row-major)
```

## 5. Model-Specific Parameters

| Model | Layers | KV Heads | Head Dim | KV Size per 4K ctx (fp16) |
|---|---|---|---|---|
| Qwen3-4B | 32 | 8 | 128 | 512 MB |
| Qwen3-8B | 40 | 8 | 128 | 640 MB |
| Qwen3-14B | 40 | 8 | 128 | 640 MB |
| Llama-3.1-8B | 32 | 8 | 128 | 512 MB |

## 6. Sharding Convention

For distributed caches:
```
cache_prefix_shard{N}of{M}.cage
```

Each shard is self-contained with its own header.
