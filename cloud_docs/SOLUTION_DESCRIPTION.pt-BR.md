# Framework CAGE - Descrição da Solução

**Última atualização:** 28/06/2026 · **Status:** ATUAL (Fases 1 e 2 concluídas; Fase 3 a seguir)
**Autor:** Lucas Mariano do Carmo · **Instituição:** Pontifícia Universidade Católica de Minas Gerais (PUC Minas) · **Contato:** lucas.mariano.carmo@gmail.com

> Detalhamento técnico complementar: [`TECHNICAL_ARCHITECTURE.pt-BR.md`](TECHNICAL_ARCHITECTURE.pt-BR.md).
> Referências autoritativas: [`KNOWLEDGE_BASE.md`](KNOWLEDGE_BASE.md), [`RUNBOOK.md`](RUNBOOK.md).
> Resultados da Fase 2: `phase2_archive/PHASE2_ANALYSIS.md`. Versão em inglês: [`SOLUTION_DESCRIPTION.md`](SOLUTION_DESCRIPTION.md).

---

## 1. O que é o CAGE?

O CAGE (Cache-Augmented Generation Evaluation) é um framework de avaliação que, pela primeira
vez, mede de forma **conjunta** as duas dimensões que as demais ferramentas medem isoladamente:
a **eficiência de serviço** de LLMs (latência, vazão, reuso de cache KV) e a **qualidade da
resposta** (fundamentação, fidelidade). Ele existe para responder, com rigor, a uma pergunta:
*quando o reuso de estado de cache chave-valor (KV) pré-computado, ou seja, a Geração Aumentada
por Cache (CAG), supera a Geração Aumentada por Recuperação (RAG), e a que custo de qualidade?*

A lacuna que ele preenche: frameworks de avaliação de RAG (RAGAS, ARES) medem apenas qualidade;
sistemas de serviço (vLLM, SGLang) medem apenas latência. Ninguém mede o compromisso entre as
duas, por consulta, com significância estatística. O CAGE faz isso, sobre uma taxonomia de 9
famílias de baselines em uma pilha de inferência real, reportando testes de Wilcoxon por consulta
com correção de Holm.

**Tese central:** um cache de prefixo contextual distribuído é superior ao RAG para bases de
conhecimento estáticas ou semi-estáticas, alcançando menor tempo até o primeiro token (TTFT) e
mantendo ou melhorando a fundamentação.

---

## 2. Arquitetura

O CAGE é a camada de orquestração e avaliação. O **vLLM** é o motor de inferência. A comunicação
ocorre exclusivamente por HTTP (a API compatível com OpenAI `/v1/completions`), de modo que o
CAGE nunca importa internamente o vLLM e pode avaliar qualquer servidor compatível.

```
Carga (CLI)  →  Orquestrador CAGE   →  Servidor(es) vLLM  →  Telemetria + Avaliador de Qualidade
run_experiment.py  baselines.py / ir.py    API HTTP            vllm_telemetry.py + quality.py
                   compression.py        (+ roteador no         performance.py
                                          modo distribuído)
```

Componentes principais (consulte `TECHNICAL_ARCHITECTURE.pt-BR.md` para o detalhamento por módulo):
- `src/data/loader.py` - carregamento de datasets (SQuAD v2, HotpotQA, TriviaQA, NQ, MuSiQue) em objetos `CAGExample`.
- `src/inference/vllm_adapter.py` - cliente HTTP com medição de TTFT por streaming e telemetria de uso. (Há também `gemini_adapter.py` e `ollama_adapter.py` para backends alternativos.)
- `src/orchestration/baselines.py` - definição das famílias de baseline e configuração por baseline.
- `src/orchestration/ir.py` - recuperação densa com FAISS (e5-large-v2) + reranker BGE para os baselines da família RAG.
- `src/orchestration/compression.py` - compressão de prompt LLMLingua-2 no lado do cliente (o arm `compressed_rag`).
- `src/orchestration/redis_cache.py` - cache de artefatos de recuperação em Redis.
- `src/orchestration/router.py` - roteador FastAPI por hash de prefixo para o cluster distribuído.
- `src/orchestration/cache_manager.py` - políticas de distribuição do cache KV (replicada agora; fragmentada/offload na Fase 3).
- `src/evaluation/quality.py` - fundamentação LettuceDetect (primária), fidelidade NLI, F1/EM, ROUGE-L, relevância de contexto.
- `src/evaluation/performance.py` - latência/vazão/percentis + agregação de telemetria de cache.
- `src/monitoring/vllm_telemetry.py` - amostrador contínuo de telemetria de GPU/KV/serviço (cage-stats).
- `scripts/run_experiment.py` - executor principal de experimentos (ponto de entrada da CLI).

---

## 3. Taxonomia de baselines (9 famílias + eixo de compressão 2×2)

O CAGE define **nove famílias de baseline** em dois eixos, a fonte do contexto (gold vs
recuperado) e a política de reuso, mais as extensões ortogonais de **compressão** e
**decodificação especulativa**.

**Famílias principais:**
1. **no_cache** - passagem gold, sem cache de prefixo. Prefill completo a cada requisição (controle de pior caso).
2. **prefix_cache** - passagem gold, cache de prefixo do vLLM ativo. Prefixos compartilhados reutilizam blocos KV.
3. **rag** - recuperação FAISS + rerank BGE, sem usar a passagem gold; sem cache de prefixo.
4. **redis** - RAG com artefatos de recuperação em cache no Redis (frio/quente).
5. **hybrid** - recuperação + cache de prefixo (+ Redis), variantes fria e quente.
6. **distributed** - N réplicas vLLM atrás de um roteador por hash de prefixo (transferência real de KV entre nós na Fase 3).
7. **speculative** - decodificação especulativa (ngram, EAGLE-3 e, na Fase 3, MTP) sobre uma estratégia de contexto.
8. **compressed_rag** - RAG + compressão de prompt LLMLingua-2 (lado do cliente, ~2× menos tokens de prompt).
9. **compressed_cag** - cache de prefixo + quantização FP8 do cache KV (alavanca no lançamento do servidor, ~2× menor KV).

**O eixo de compressão 2×2** (leia na VERTICAL para CAG vs RAG, na HORIZONTAL para full vs comprimido):

| | full | comprimido |
|---|---|---|
| **CAG** (contexto gold) | prefix_cache / cag_full | compressed_cag (FP8 KV) |
| **RAG** (contexto recuperado) | rag / rag_full | compressed_rag (LLMLingua-2) |

---

## 4. Métricas

**Qualidade (PRIMÁRIA: fundamentação LettuceDetect):**
- **Fundamentação (LettuceDetect, primária):** detecção de alucinação em nível de token/trecho via modelo ModernBERT. `grounding_score = 1 − razão_de_trechos_alucinados`.
- **Fidelidade (secundária):** entailment NLI por afirmação, da resposta contra o contexto.
- **F1 / Exact Match:** correção padrão de QA.
- **ROUGE-L:** F1 da maior subsequência comum.
- **Relevância de contexto:** similaridade de embeddings pergunta/contexto (apenas diagnóstica).
- **BERTScore: descontinuada** (não discriminativa entre baselines; mantida apenas como controle negativo).

**Serviço (via cage-stats + pynvml):** QPS, tokens/s, TTFT, TPOT, latência fim a fim (média + p50/p95/p99), razão de acerto de prefixo, utilização do cache KV, memória/potência/temperatura de GPU e (quando especulativo está ativo) taxa de aceitação.

**Camada estatística (`scripts/statistical_tests.py`):** testes de **Wilcoxon** por consulta contra um baseline de referência, correção de comparações múltiplas de **Holm**, tamanho de efeito **delta de Cliff** e intervalos de confiança por **bootstrap**.

---

## 5. Resultados

### Fase 1 (CPU, validação de protocolo, apenas relativa)
Configuração: Qwen3-4B, SQuAD v2, CPU Apple M4 Pro, 50 consultas × 3 trials × 7 baselines. As
latências absolutas em CPU não são generalizáveis; os rankings são. O prefix cache vence (−37,4%
de latência, −65,7% de TTFT, qualidade igual); o RAG é o mais lento e perde fidelidade; o
distribuído mostra espalhamento de cauda p95/p50 de 7,6×; o BERTScore é não discriminativo.

### Fase 2 (GPU NVIDIA L4 única, o baseline relevante para produção) - CONCLUÍDA em 27/06/2026
Configuração: **Qwen3-8B, vLLM 0.11.0, SQuAD v2, L4 única (24 GB), 100 consultas × 1 trial**, 14
conjuntos de resultados em 8 das 9 famílias (distribuído adiado para a Fase 3). Toda significância
contra `no_cache`, com correção de Holm. Métrica primária = fundamentação.

| Achado | Serviço | Qualidade | Veredito |
|---|---|---|---|
| **Cache de prefixo** | TTFT −3,3% (p=1,2e-11) | fundamentação idêntica (0,938) | sem perda |
| **FP8 KV (compressed_cag)** | KV pela metade | fundamentação 0,936 vs 0,938 (n.s.) | sem perda |
| **RAG** vs CAG gold | TTFT +87% (p=5e-17) | fidelidade −24,7% (p=8,8e-05); fundamentação 0,66 | custa nos dois eixos aqui |
| **Especulativo EAGLE-3** | TPOT −41% (54→32 ms), latência −32,5% (p=5e-11) | fundamentação inalterada | sem perda, maior ganho |

Ressalvas honestas: no SQuAD v2 a passagem gold é o contexto ideal, então o CAG domina o RAG
(um dataset favorável ao RAG é uma necessidade da Fase 3); o `compressed_rag` foi inválido nesta
execução porque o tratamento de compressão nunca foi aplicado (já corrigido no código, reexecutar
na Fase 3); o `hybrid_warm` usou estatística não pareada. Análise completa:
`phase2_archive/PHASE2_ANALYSIS.md`. Custo ~US$ 3,1; toda a infraestrutura no GCP foi desligada
para US$ 0 e os dados estão arquivados localmente.

---

## 6. Pilha tecnológica

- **Inferência:** vLLM 0.11.0 (fixado), API HTTP compatível com OpenAI; `--enforce-eager` na L4.
- **Modelos:** Qwen3-4B (Fase 1 CPU), Qwen3-8B (Fase 2); candidatos da Fase 3: Qwen3-14B/32B e DeepSeek-V2-Lite (para MTP). Configs em `configs/model/`.
- **Recuperação:** FAISS `IndexFlatIP` + embeddings `intfloat/e5-large-v2` + `BAAI/bge-reranker-large`.
- **Compressão:** LLMLingua-2 (prompt, lado do cliente) e FP8 no cache KV (`--kv-cache-dtype fp8` no servidor).
- **Decodificação especulativa:** ngram + EAGLE-3 (`AngelSlim/Qwen3-8B_eagle3`) via `--speculative-config`.
- **Modelos de qualidade:** LettuceDetect (ModernBERT) para fundamentação; DeBERTa-mnli para fidelidade NLI.
- **Cache:** Redis (artefatos de recuperação) + cache de prefixo nativo do vLLM (blocos KV).
- **Telemetria:** cage-stats (`--vllm-telemetry`) + pynvml.
- **Infraestrutura:** GCP (g2-standard-8 + L4), Terraform, bucket GCS durável, além de uma suíte de preservação de logs e desligamento seguro. Há manifestos Docker Compose / Kubernetes para uso local e em cluster.
- **Runtime:** Python 3.12, torch, transformers, datasets, sentence-transformers, faiss-cpu, llmlingua.

---

## 7. Status e próximos passos

- **Fase 1 (CPU): CONCLUÍDA** - protocolo validado.
- **Fase 2 (GPU L4 única): CONCLUÍDA (27/06/2026)** - eixos de qualidade e serviço medidos com estatística; os quatro achados acima; infraestrutura em US$ 0, dados locais.
- **Fase 3 (HPC multi-nó): A SEGUIR** - transferência real de tensores KV entre nós via conector KV do vLLM (LMCache/NIXL) com política de contexto fragmentada (substituindo o modelo analítico/simulado atual), prefill desagregado, decodificação especulativa mais ampla (MTP via DeepSeek-V2-Lite), um dataset favorável ao RAG e intervalos de confiança com múltiplos trials. Plano: [`PHASE3_PLAN.md`](PHASE3_PLAN.md); modelos: `docs/PHASE3_MODELS.md`.

### Mudanças e correções recentes (neste ciclo)
- **Determinismo da geração:** temperatura 0,0 + `stop=["\n"]` para conter o vazamento de cadeia de raciocínio do Qwen3.
- **Estabilidade do vLLM:** encerramento do vazamento de GPU do EngineCore no restart; argumentos do servidor montados como array; gate de FP8 × cache de prefixo.
- **Correções de métricas:** `retrieval_hit` agora usa fallback por texto normalizado (era um zero falso); `completeness_bertscore` retorna None em referências vazias (era um sentinela negativo).
- **Validade da compressão:** `llmlingua` adicionado aos requirements + um modo estrito `CAGE_REQUIRE_COMPRESSION=1` para que o arm `compressed_rag` nunca mais falhe silenciosamente.
- **Operações:** uma suíte de preservação de logs (`collect_logs.sh`, `log_sync_daemon.sh`, `gcp_shutdown_hook.sh`) e um `teardown_vm.sh` que falha de forma segura, verificando que os logs chegaram ao GCS antes de excluir uma VM.
