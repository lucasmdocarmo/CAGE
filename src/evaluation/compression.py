"""Compression metrics for the CAGE compression axis.

Two families:
- Text compression (compressed_rag): compression_ratio from token counts (recorded by
  src/orchestration/compression.ContextCompressor).
- KV-cache footprint (compressed_cag / general): estimate bytes held in the KV cache for a
  given prompt length, so fp8 vs bf16 (and MLA's low-rank KV) can be compared. This is an
  ANALYTICAL estimate; the empirical footprint comes from GPUMetricsTracker on real hardware.
"""

from __future__ import annotations

from typing import Optional

# Bytes per element by KV cache dtype.
_DTYPE_BYTES = {
    "auto": 2.0, "bf16": 2.0, "float16": 2.0, "fp16": 2.0,
    "fp8": 1.0, "fp8_e4m3": 1.0, "fp8_e5m2": 1.0,
    "int8": 1.0, "fp4": 0.5,
}


def kv_cache_bytes(
    num_tokens: int,
    *,
    num_layers: int,
    num_kv_heads: int,
    head_dim: int,
    dtype: str = "bf16",
    mla_latent_dim: Optional[int] = None,
) -> float:
    """Estimate KV-cache bytes for `num_tokens` tokens.

    Standard (MHA/GQA): 2 (K and V) * layers * kv_heads * head_dim * tokens * bytes/elem.
    MLA (DeepSeek): a single low-rank latent of size `mla_latent_dim` per token per layer
    replaces the per-head K/V, so bytes = layers * latent_dim * tokens * bytes/elem.
    """
    b = _DTYPE_BYTES.get(str(dtype).lower(), 2.0)
    if mla_latent_dim:
        return float(num_layers) * float(mla_latent_dim) * float(num_tokens) * b
    return 2.0 * float(num_layers) * float(num_kv_heads) * float(head_dim) * float(num_tokens) * b


def kv_bytes_per_token(
    *,
    num_layers: int,
    num_kv_heads: int,
    head_dim: int,
    dtype: str = "bf16",
    mla_latent_dim: Optional[int] = None,
) -> float:
    return kv_cache_bytes(
        1, num_layers=num_layers, num_kv_heads=num_kv_heads, head_dim=head_dim,
        dtype=dtype, mla_latent_dim=mla_latent_dim,
    )


def compression_ratio(original_tokens: int, compressed_tokens: int) -> Optional[float]:
    """compressed / original (1.0 = none, <1.0 = compressed). None if undefined."""
    if not original_tokens:
        return None
    return float(compressed_tokens) / float(original_tokens)


def transfer_bytes_for(
    kv_bytes: float, *, num_nodes: int, fraction_remote: Optional[float] = None
) -> float:
    """Bytes that must move cross-node for a sharded KV of size `kv_bytes`.

    Default fraction_remote = (num_nodes-1)/num_nodes (each node holds 1/N locally).
    NOTE: this is the analytical model; the *measured* transfer bytes come from the real
    vLLM KV connector (LMCache/NIXL) once wired — see DEV_BACKLOG #6.
    """
    if num_nodes <= 1:
        return 0.0
    fr = fraction_remote if fraction_remote is not None else (num_nodes - 1) / num_nodes
    return float(kv_bytes) * float(fr)
