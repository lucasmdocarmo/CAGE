"""Compression metrics for the CAGE compression axis.

Two families:
- Text compression (compressed_rag): compression_ratio from token counts (recorded by
  src/orchestration/compression.ContextCompressor).
- KV-cache footprint (compressed_cag / general): estimate bytes held in the KV cache for a
  given prompt length, so fp8 vs bf16 (and MLA's low-rank KV) can be compared. This is an
  ANALYTICAL estimate; the empirical footprint comes from GPUMetricsTracker on real hardware.
"""

from __future__ import annotations

import os
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


def _model_config(model_name: str) -> dict:
    """The model's config.json as a dict: the locally cached file through
    huggingface_hub (about 0.05 s), else ``AutoConfig``. S0F-88 audit (U1):
    importing transformers in the window tail held every vLLM window 25 to
    32 s after its last completion on S0 attempt 4."""
    try:
        import json as _json

        from huggingface_hub import try_to_load_from_cache

        path = try_to_load_from_cache(model_name, "config.json")
        if isinstance(path, str) and os.path.isfile(path):
            with open(path, encoding="utf-8") as handle:
                return dict(_json.load(handle))
    except Exception:  # noqa: BLE001 -- no package, a local path, a bad repo id: AutoConfig decides
        pass
    from transformers import AutoConfig

    cfg = AutoConfig.from_pretrained(model_name)
    return cfg.to_dict() if callable(getattr(cfg, "to_dict", None)) else dict(vars(cfg))


def analytical_kv_footprint(
    model_name: str,
    num_tokens: int,
    *,
    dtype: str = "bf16",
    baseline_dtype: str = "bf16",
    mla_latent_dim: Optional[int] = None,
) -> Optional[dict]:
    """Best-effort analytical KV-cache footprint for a HF model at ``num_tokens``.

    Bridges the model architecture (layers, KV heads, head dim) read from the HF config
    into :func:`kv_cache_bytes`, so the compression axis carries an analytical estimate
    alongside the empirical footprint from GPUMetricsTracker. Compares the configured
    KV dtype (e.g. ``fp8`` for ``compressed_cag``) against a ``bf16`` baseline.

    MLA models (DeepSeek-V2/V3) cache a compressed KV latent (``kv_lora_rank``) PLUS a
    decoupled RoPE key shared across heads (``qk_rope_head_dim``) per token per layer,
    replacing the per-head K/V that ``kv_cache_bytes`` assumes by default. Unless the
    caller passes an explicit ``mla_latent_dim`` override, it is auto-derived from the
    HF config as ``kv_lora_rank + qk_rope_head_dim`` when ``kv_lora_rank`` is present
    (e.g. DeepSeek-V3: 512 + 64 = 576 -> 68.6 KiB/token). Omitting the decoupled RoPE
    key would silently undercount MLA models by ~11%.

    Returns ``None`` if the config cannot be read (never raises), so it is safe to call
    from the run summary on any host.
    """
    try:
        cfg = _model_config(model_name)
        num_layers = cfg.get("num_hidden_layers")
        num_heads = cfg.get("num_attention_heads")
        num_kv_heads = cfg.get("num_key_value_heads") or num_heads
        hidden = cfg.get("hidden_size")
        head_dim = cfg.get("head_dim") or (
            int(hidden // num_heads) if hidden and num_heads else None
        )
        if mla_latent_dim is None:
            kv_lora_rank = cfg.get("kv_lora_rank")
            if kv_lora_rank:
                qk_rope_head_dim = cfg.get("qk_rope_head_dim", 0) or 0
                mla_latent_dim = int(kv_lora_rank) + int(qk_rope_head_dim)
        if not (num_layers and num_kv_heads and head_dim and num_tokens):
            return None

        comp = kv_cache_bytes(
            num_tokens, num_layers=num_layers, num_kv_heads=num_kv_heads,
            head_dim=head_dim, dtype=dtype, mla_latent_dim=mla_latent_dim,
        )
        base = kv_cache_bytes(
            num_tokens, num_layers=num_layers, num_kv_heads=num_kv_heads,
            head_dim=head_dim, dtype=baseline_dtype, mla_latent_dim=mla_latent_dim,
        )
        return {
            "model": model_name,
            "num_tokens": int(num_tokens),
            "kv_cache_dtype": dtype,
            "num_layers": int(num_layers),
            "num_kv_heads": int(num_kv_heads),
            "head_dim": int(head_dim),
            "mla_latent_dim": mla_latent_dim,
            "kv_bytes_per_token": kv_bytes_per_token(
                num_layers=num_layers, num_kv_heads=num_kv_heads,
                head_dim=head_dim, dtype=dtype, mla_latent_dim=mla_latent_dim,
            ),
            "kv_cache_bytes": comp,
            "kv_cache_bytes_baseline_bf16": base,
            "kv_compression_ratio": (comp / base) if base else None,
        }
    except Exception:
        return None
