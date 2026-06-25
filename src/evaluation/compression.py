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


def analytical_kv_footprint(
    model_name: str,
    num_tokens: int,
    *,
    dtype: str = "bf16",
    baseline_dtype: str = "bf16",
) -> Optional[dict]:
    """Best-effort analytical KV-cache footprint for a HF model at ``num_tokens``.

    Bridges the model architecture (layers, KV heads, head dim) read from the HF config
    into :func:`kv_cache_bytes`, so the compression axis carries an analytical estimate
    alongside the empirical footprint from GPUMetricsTracker. Compares the configured
    KV dtype (e.g. ``fp8`` for ``compressed_cag``) against a ``bf16`` baseline.

    Returns ``None`` if the config cannot be read (never raises), so it is safe to call
    from the run summary on any host.
    """
    try:
        from transformers import AutoConfig

        cfg = AutoConfig.from_pretrained(model_name)
        num_layers = getattr(cfg, "num_hidden_layers", None)
        num_heads = getattr(cfg, "num_attention_heads", None)
        num_kv_heads = getattr(cfg, "num_key_value_heads", None) or num_heads
        hidden = getattr(cfg, "hidden_size", None)
        head_dim = getattr(cfg, "head_dim", None) or (
            int(hidden // num_heads) if hidden and num_heads else None
        )
        if not (num_layers and num_kv_heads and head_dim and num_tokens):
            return None

        comp = kv_cache_bytes(
            num_tokens, num_layers=num_layers, num_kv_heads=num_kv_heads,
            head_dim=head_dim, dtype=dtype,
        )
        base = kv_cache_bytes(
            num_tokens, num_layers=num_layers, num_kv_heads=num_kv_heads,
            head_dim=head_dim, dtype=baseline_dtype,
        )
        return {
            "model": model_name,
            "num_tokens": int(num_tokens),
            "kv_cache_dtype": dtype,
            "num_layers": int(num_layers),
            "num_kv_heads": int(num_kv_heads),
            "head_dim": int(head_dim),
            "kv_bytes_per_token": kv_bytes_per_token(
                num_layers=num_layers, num_kv_heads=num_kv_heads,
                head_dim=head_dim, dtype=dtype,
            ),
            "kv_cache_bytes": comp,
            "kv_cache_bytes_baseline_bf16": base,
            "kv_compression_ratio": (comp / base) if base else None,
        }
    except Exception:
        return None
