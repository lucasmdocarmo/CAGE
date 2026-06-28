# Phase 3 — Heavier Models & Speculative‑Method Coverage (decision doc)

> Research synthesis for choosing model(s) for a **next‑phase** CAGE run that (a) induces
> **heavier memory pressure** than Phase 2's Qwen3‑8B on a single L4, and (b) unlocks
> **speculative‑decoding methods** Qwen3‑8B couldn't run. All availability **verified on
> Hugging Face** + against vLLM 0.11.0's installed supported‑method list. Generated 2026‑06‑27.
> `docs/` is gitignored — this stays local. **Decision deferred — for later.**

## Context
- **Phase 2 (done/running):** Qwen3‑8B, single NVIDIA L4 (24 GB), `--enforce-eager`,
  `max_model_len` capped. Speculative methods usable here: **ngram** (no head) + **EAGLE‑3**
  (head `AngelSlim/Qwen3-8B_eagle3`). Everything else was blocked (see below).
- **Phase 3 (future):** multi‑node / bigger GPUs + real cross‑node KV transfer. The models
  below are the candidates to stress that, and to broaden the speculative comparison.

## Key finding (upfront)
**No single model unlocks all of {ngram, EAGLE‑3, MTP, Medusa, SAM, R‑SD}.**
- **SAM‑decoding & R‑SD** → *not implemented in vLLM 0.11.0 for any model.* They are
  research‑paper methods; using them would require custom code. **Out of scope.**
- **MTP** → requires a model with **native multi‑token‑prediction modules** (Qwen3 has none).
  Unlocked by DeepSeek‑V2/V3, MiMo, GLM‑4.5, Ernie‑4.5 (vLLM: `deepseek_mtp` / `mimo_mtp` /
  `glm` / `ernie_mtp`). **This is the one advanced method that genuinely needs a model swap.**
- **Medusa** → only **dated base models** (Vicuna, Llama‑2) have published heads; no Qwen3
  head. Largely superseded by EAGLE‑3. **Not recommended.**
- **EAGLE‑3** → scale up within the **Qwen3 family** (heads exist 1.7B→32B).

## Candidate models (verified on HF)

| Model | Memory pressure | Speculative unlocked | GPU / phase fit |
|---|---|---|---|
| [Qwen3‑14B](https://hf.co/Qwen/Qwen3-14B) | ~28 GB (dense) | **EAGLE‑3** ✅ [AngelSlim/Qwen3‑14B_eagle3](https://hf.co/AngelSlim/Qwen3-14B_eagle3) | 1×A100‑40 or 2×L4 |
| [Qwen3‑32B](https://hf.co/Qwen/Qwen3-32B) | ~64 GB (dense) | **EAGLE‑3** ✅ [AngelSlim](https://hf.co/AngelSlim/Qwen3-32B_eagle3), [RedHatAI](https://hf.co/RedHatAI/Qwen3-32B-speculator.eagle3) | multi‑GPU / **Phase 3** |
| [DeepSeek‑V2‑Lite](https://hf.co/deepseek-ai/DeepSeek-V2-Lite) | 16B MoE (~2.4B active), ~31 GB + MLA | **MTP** ✅ native (`deepseek_mtp`) | 1×A100 or 2×L4 |
| [DeepSeek‑Coder‑V2‑Lite](https://hf.co/deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct) | 16B MoE + MLA | **MTP** ✅ native | 1×A100 or 2×L4 |
| [MiMo‑7B‑RL](https://hf.co/XiaomiMiMo/MiMo-7B-RL) | 7B dense (light) | **MTP** ✅ native (`mimo_mtp`) | 1×L4 |
| [GLM‑4.5‑Air](https://hf.co/zai-org/GLM-4.5-Air) / [GLM‑4.5](https://hf.co/zai-org/GLM-4.5) | 106B / 355B MoE | **MTP** ✅ native (`glm`) | multi‑node Phase 3 (very heavy) |
| Llama‑3.3‑70B | ~140 GB | EAGLE heads exist | multi‑node Phase 3 (extreme) |

## Recommendations
1. **Best "heavier + new method" single swap → DeepSeek‑V2‑Lite.** MoE + MLA gives more
   memory pressure than Qwen3‑8B *and* unlocks **MTP** → enables a `ngram + MTP` comparison
   impossible on Qwen3. (MLA itself is also a compression‑axis story — ties to the 2×2.)
2. **Maximal memory pressure (true Phase‑3 distributed) → Qwen3‑32B** (EAGLE‑3, family‑consistent
   with Phase 2) **or GLM‑4.5‑Air** (MoE + MTP). These genuinely stress multi‑node KV.
3. **Full speculative coverage needs two families:** Qwen3 (ngram + EAGLE‑3) **+**
   DeepSeek‑V2‑Lite or MiMo (ngram + MTP) → spans all three *viable* advanced methods.
   Dissertation framing: *"speculative methods are architecture‑gated — EAGLE‑3 via trained
   heads, MTP via native modules — and CAGE measures both under one quality‑efficiency lens."*

## Notes / caveats
- All these models exceed a single 24 GB L4 at bf16 (except MiMo‑7B), so they imply **Phase 3
  hardware** (multi‑GPU or A100‑40/80). Cost scales accordingly — budget per the per‑hour GPU rate.
- Speculative decoding is **output‑lossless**: across methods, *quality is identical*; the axis
  varies serving speed (acceptance / TTFT / throughput) only. So model choice for the speculative
  study is about which *method* you can run, not about quality.
- vLLM 0.11.0 verified speculative methods: `ngram, draft_model, eagle, eagle3, medusa,
  mlp_speculator, mtp, deepseek_mtp, ernie_mtp, mimo_mtp, longcat_flash_mtp`.
