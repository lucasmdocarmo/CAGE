# Framework CAGE - Descrição da Solução

**Última atualização:** 02/07/2026 · **Status:** ATUAL (Fase 1 concluída; re-execução limpa da Fase 2 em L4 única pronta em código; Fase 3 adiada)
**Autor:** Lucas Mariano do Carmo · **Instituição:** Pontifícia Universidade Católica de Minas Gerais (PUC Minas) · **Contato:** lucas.mariano.carmo@gmail.com

> Detalhamento técnico complementar: [`TECHNICAL_ARCHITECTURE.pt-BR.md`](TECHNICAL_ARCHITECTURE.pt-BR.md).
> Referências autoritativas: [`KNOWLEDGE_BASE.md`](KNOWLEDGE_BASE.md), [`RUNBOOK.md`](RUNBOOK.md).
> Resultados da Fase 2: `phase2_archive/PHASE2_ANALYSIS.md`. Versão em inglês: [`SOLUTION_DESCRIPTION.md`](SOLUTION_DESCRIPTION.md).

---

## 1. O que é o CAGE?

O CAGE (Cache-Augmented Generation Evaluation) é um framework de avaliação que, pela primeira
vez, mede de forma **conjunta** as duas dimensões que as demais ferramentas medem isoladamente:
a **eficiência de serviço** de LLMs (latência, vazão, reuso de cache chave-valor, KV) e a
**qualidade da resposta** (fundamentação, fidelidade). Ele existe para responder, com rigor, a
uma pergunta: *quando o reuso de estado de cache chave-valor (KV) pré-computado, ou seja, a
Geração Aumentada por Cache (CAG), supera a Geração Aumentada por Recuperação (RAG), e a que
custo de qualidade da resposta?*

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
- `src/data/loader.py` - carregamento de datasets em objetos `CAGExample`. Onze loaders: SQuAD v2, HotpotQA, Qasper, TriviaQA, Natural Questions, MuSiQue, CRAG, ShareGPT (QA/serviço), além de HumanEval, MBPP, hpc_code (código). O CRAG traz uma resposta gold ao lado de documentos candidatos recuperados (justo para RAG); o ShareGPT é um traço de carga de serviço sem resposta gold (apenas referência, `no_gold_answer=True`).
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
6. **distributed** - N réplicas vLLM atrás de um roteador por hash de prefixo (Fase 3 para transferência real de KV entre nós).
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

**Serviço (via cage-stats + pynvml):** QPS, tokens/s, TTFT, TPOT, latência fim a fim (média + p50/p95/p99), razão de acerto de prefixo, utilização do cache KV, memória/potência/temperatura de GPU e (quando o especulativo está ativo) taxa de aceitação.

**A telemetria é somente ao vivo (garantia rígida).** O caminho de telemetria sintética/mock foi removido do código. `capture_snapshot` resolve nesta ordem: `cage_stats.api` em processo, depois a CLI `cage-stats --once --json`, depois um coletor de `/metrics` em stdlib sem dependências. Se nenhum servidor vLLM estiver acessível (ou se o payload coletado não contiver nenhuma série `vllm:`), o amostrador retorna `None`; ele nunca fabrica zeros nem valores sintéticos. Este é um invariante imposto pelo código, não uma convenção.

**Camada estatística (`scripts/statistical_tests.py`):** testes de **Wilcoxon** de postos sinalizados por consulta contra um baseline de referência, correção de comparações múltiplas de **Holm**, tamanho de efeito **delta de Cliff** e intervalos de confiança por **bootstrap**.

---

## 5. Resultados

### Fase 1 (CPU, validação de protocolo, apenas relativa)
Configuração: Qwen3-4B, SQuAD v2, CPU Apple M4 Pro, 50 consultas × 3 trials × 7 baselines. As
latências absolutas em CPU não são generalizáveis; os rankings são. O prefix cache vence (−37,4%
de latência, −65,7% de TTFT, qualidade igual); o RAG é o mais lento e perde fidelidade; o
distribuído mostra espalhamento de cauda p95/p50 de 7,6×; o BERTScore é não discriminativo.

### Fase 2 (GPU NVIDIA L4 única, o baseline relevante para produção) - re-execução limpa PRONTA EM CÓDIGO
Configuração: **Qwen3-8B, vLLM 0.11.0, SQuAD v2, L4 única (24 GB), greedy (T=0)**, em 8 das 9
famílias (distribuído adiado para a Fase 3). Significância contra `no_cache`, com correção de
Holm. Métrica primária = fundamentação.

> **Status dos resultados:** a execução original da Fase 2 foi substituída (baselines inválidos) e
> a re-execução limpa está pronta em código, mas ainda não foi reexecutada, de modo que os números
> abaixo são direcionais, oriundos da execução anterior, e não devem ser tratados como
> finais/validados até que a re-execução seja concluída. **A contagem de consultas está em aberto:**
> `scripts/run_phase2.sh` usa `NUM_QUERIES=500` por padrão (× 3 trials): contagem fixada em 500 × 3.

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
- **Datasets:** onze loaders registrados em `src/data/loader.py` - SQuAD v2, HotpotQA, Qasper, TriviaQA, Natural Questions, MuSiQue, CRAG, ShareGPT (QA/serviço), além de HumanEval, MBPP, hpc_code (código). Todos os loaders de QA aplicam shuffle(seed) antes do select para independência entre trials. CRAG (resposta gold + documentos candidatos recuperados, justo para RAG) e ShareGPT (traço de carga de serviço, sem resposta gold) estão ligados de ponta a ponta (registro + `run_experiment.py --dataset` + `scripts/download_datasets.py`).
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

- **Fase 1 (CPU): CONCLUÍDA** - protocolo validado; este é o estado que a dissertação reporta atualmente (decodificação estocástica, temperatura 0,7).
- **Fase 2 (GPU L4 única): re-execução limpa PRONTA EM CÓDIGO** - eixos de qualidade e serviço em greedy (T=0) ligados com estatística; a execução original foi substituída e a re-execução ainda não foi reexecutada, de modo que seus números não são finais. Infraestrutura em US$ 0, dados locais. Contagem de consultas fixada em 500 × 3.
- **Fase 3 (HPC multi-nó): ADIADA** - transferência real de tensores KV entre nós via um conector KV do vLLM (LMCache/NIXL) com política de contexto fragmentada (substituindo o modelo analítico/simulado atual), prefill desagregado, decodificação especulativa mais ampla (MTP via DeepSeek-V2-Lite) e intervalos de confiança com múltiplos trials. A necessidade de um dataset favorável ao RAG agora está parcialmente atendida: **CRAG** (resposta gold + documentos candidatos recuperados, justo para RAG) e **ShareGPT** (traço de carga de serviço) estão ligados de ponta a ponta e disponíveis para a carga da Fase 3. A transferência entre nós permanece analítica/simulada até que o RDMA real seja implementado. Plano: [`PHASE3_PLAN.md`](PHASE3_PLAN.md); modelos: `docs/PHASE3_MODELS.md`.

### Mudanças e correções recentes (neste ciclo)
- **Determinismo da geração:** temperatura 0,0 + `stop=["\n"]` para conter o vazamento de cadeia de raciocínio do Qwen3.
- **Estabilidade do vLLM:** encerramento do vazamento de GPU do EngineCore no restart; argumentos do servidor montados como array; gate de FP8 × cache de prefixo.
- **Correções de métricas:** `retrieval_hit` agora usa fallback por texto normalizado (era um zero falso); `completeness_bertscore` retorna None em referências vazias (era um sentinela negativo).
- **Validade da compressão:** `llmlingua` adicionado aos requirements, e a compressão agora é **estrita por padrão**, de modo que o arm `compressed_rag` nunca mais possa falhar silenciosamente sem aplicar compressão. Se o LLMLingua-2 estiver indisponível, ele lança um erro em vez de deixar o prompt passar sem compressão. O opt-out ao vivo é `CAGE_ALLOW_NO_COMPRESSION`; `CAGE_DISABLE_COMPRESSION` desabilita a compressão por completo (pass-through).
- **Operações:** uma suíte de preservação de logs (`collect_logs.sh`, `log_sync_daemon.sh`, `gcp_shutdown_hook.sh`) e um `teardown_vm.sh` que falha de forma segura, verificando que os logs chegaram ao GCS antes de excluir uma VM.

---

## 8. Créditos / trabalhos anteriores

O CAGE é construído sobre, e se posiciona diante de, um corpo de trabalhos anteriores; as chaves bib abaixo (em `Main.bib`) creditam essa linhagem.

- **Motor de serviço instrumentado:** vLLM / PagedAttention (`kwon2023efficient`). Serviço com consciência de cache relacionado, diante do qual o CAGE se posiciona: SGLang/RadixAttention (`zheng2024sglang`), DistServe (`zhong2024distserve`), Mooncake (`qin2024mooncake`), LMCache (`lmcache2024`), CacheBlend (`cacheblend2025`), CacheGen (`cachegen2024`, arXiv 2310.07240).
- **Métrica de qualidade primária:** detecção de alucinação em nível de trecho LettuceDetect (`lettucedetect2025`).
- **Linhagem de avaliação de RAG** contra a qual o CAGE co-mede: RAGAS (`espejel2023ragas`), ARES (`ares2024`), além do benchmark de QA factual CRAG (`yang2024crag`).
- **A espinha "a compressão carrega um custo de qualidade mensurável":** The Pitfalls of KV Cache Compression (`chen2025pitfalls`) e SCBench (`li2025scbench`, ICLR 2025, Microsoft e University of Surrey, arXiv 2412.10319), o trabalho anterior de cache-mais-qualidade mais próximo (sem métrica de fundamentação, sem latência de serviço por método).
- **A decisão CAG vs RAG que o CAGE operacionaliza:** "Don't Do RAG" / geração aumentada por cache (`yu2024dontdorag`).
- **Método de compressão executado:** LLMLingua-2 para compressão de prompt no lado do texto (`llmlingua2`).

(As chaves bib são espelhadas verbatim do `Main.bib` da dissertação para compatibilidade com `\cite`; algumas chaves diferem deliberadamente do primeiro autor do artigo e são mantidas inalteradas.)
