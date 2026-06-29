# Arquitetura Técnica do CAGE

**Última atualização:** 28/06/2026 · **Status:** ATUAL (reflete o código e os resultados das Fases 1 e 2)

> **Objetivo:** referência técnica detalhada de como o código funciona, como os resultados são
> gerados, como o vLLM é integrado e como cada componente se conecta. Escrito para uma IA ou
> pessoa desenvolvedora assumindo o projeto do zero. Visão de alto nível:
> [`SOLUTION_DESCRIPTION.pt-BR.md`](SOLUTION_DESCRIPTION.pt-BR.md). Comandos: [`RUNBOOK.md`](RUNBOOK.md).
> Métricas/status atuais: [`KNOWLEDGE_BASE.md`](KNOWLEDGE_BASE.md). Versão em inglês:
> [`TECHNICAL_ARCHITECTURE.md`](TECHNICAL_ARCHITECTURE.md).

---

## Como um experimento roda de ponta a ponta

Um baseline é executado com (flags reais e atuais):

```bash
python scripts/run_experiment.py \
  --baseline prefix_cache --baseline-label cag_full \
  --model Qwen/Qwen3-8B --dataset squad_v2 \
  --num-queries 100 --num-trials 1 --seed 42 \
  --context-source auto --vllm-telemetry \
  --output-dir analysis/phase1/results/prefix_cache
```

> Observação: a CLI usa `--baseline`, `--num-trials`, `--num-queries`, `--context-source`
> (`auto|gold|retrieved`). As flags antigas `--phase`/`--all-baselines`/`--trials`/`--queries`
> não existem. O servidor vLLM é iniciado separadamente por `scripts/manage_vllm_server.sh`.

```
1. CLI lida -> config do baseline em src/orchestration/baselines.py
2. Dataset carregado -> src/data/loader.py -> objetos CAGExample
3. Índice IR construído se for baseline de recuperação OU --context-source retrieved -> src/orchestration/ir.py (FAISS)
4. Compressor criado se baseline_config.compress_method estiver definido -> src/orchestration/compression.py
5. Amostrador de telemetria do vLLM iniciado (se --vllm-telemetry) -> src/monitoring/vllm_telemetry.py
6. Para cada trial, para cada consulta:
   a. Contexto selecionado: gold (example.context) OU recuperado (ir_index.search) conforme --context-source
   b. Compressão opcional do prompt (compressed_rag) -> ContextCompressor.compress()
   c. Prompt montado -> src/utils/prompting.py -> format_qa_prompt()
   d. Requisição enviada ao vLLM (streaming) -> src/inference/vllm_adapter.py -> TTFT no primeiro chunk SSE
   e. Resposta + telemetria de uso (prompt_tokens, cached_prompt_tokens) coletadas
   f. Qualidade avaliada -> src/evaluation/quality.py -> fundamentação (primária), fidelidade, F1/EM, ROUGE-L
   g. Linha por consulta adicionada (campos de serviço + qualidade + recuperação + compressão)
7. Amostrador de telemetria encerrado -> aggregate() incorporado
8. Agregados calculados (mean_or_none ignora None) + CSV por consulta + metrics.json gravados
9. Resultados sincronizados ao GCS (cloud_run.sh / sync_results_to_gcs.sh)
```

### Estrutura de saída (por baseline)
```
analysis/<fase>/results/<baseline>/
├── results.csv                              # canônico: uma linha por consulta (serviço + qualidade + recuperação + compressão)
├── metrics.json                             # canônico: métricas agregadas + metadados + telemetria
├── vllm_telemetry.json                      # snapshot e agregados de telemetria de GPU/KV/serviço
├── commands.log                             # a(s) linha(s) exata(s) de invocação
└── <baseline>_<dataset>_<timestamp>_results.csv / _metrics.json   # cópias brutas por execução (proveniência)
```
O `results.csv` / `metrics.json` sem sufixo são os autoritativos (a última execução válida); as cópias
com timestamp são mantidas para proveniência. Os dados da Fase 2 ficam localmente em `phase2_archive/`
(o bucket no GCS foi excluído).

---

## Módulos do código-fonte

### `src/data/loader.py` - carregamento de datasets
Carrega datasets do HuggingFace em objetos `CAGExample`:
```python
@dataclass
class CAGExample:
    id: str
    question: str
    context: List[str]   # passagens gold
    answer: str          # resposta de referência
    metadata: Dict[str, Any]
```
Loaders: SQuAD v2, HotpotQA, TriviaQA, Natural Questions, MuSiQue (+ datasets de código para `code_evaluator`).
Baselines de contexto gold usam `example.context`; baselines de recuperação (e `--context-source retrieved`)
ignoram-no e buscam passagens no índice IR.

### `src/inference/engine.py` - interface abstrata de inferência
Define `InferenceEngine` e os dataclasses `InferenceRequest`/`InferenceResponse`. O `InferenceResponse`
carrega `generated_text`, `ttft_ms`, `total_time_ms`, `num_tokens`, `finish_reason`, `prompt_tokens`,
`cached_prompt_tokens` e `kv_transfer_params`. Os campos `prompt_tokens`/`cached_prompt_tokens` só são
preenchidos quando o vLLM retorna telemetria de uso (`--enable-prompt-tokens-details`).

### `src/inference/vllm_adapter.py` - cliente HTTP do vLLM (+ `gemini_adapter.py`, `ollama_adapter.py`)
Comunica-se com o vLLM pela API compatível com OpenAI `/v1/completions`.
- **TTFT** medido em modo streaming: `perf_counter()` no envio vs no primeiro chunk SSE `data:`.
- **Telemetria de uso** vem do chunk final com `stream_options={"include_usage": true}` (fornece `prompt_tokens` e `prompt_tokens_details.cached_tokens`, a razão de prompt em cache).
- **Parâmetros de geração (correção de validade):** `temperature=0.0` + `stop=["\n"]` + `max_tokens=100`. A decodificação gulosa torna as execuções comparáveis; o `stop=["\n"]` é necessário porque o Qwen3-8B, caso contrário, anexa cadeia de raciocínio após uma única quebra de linha.
- Para o baseline distribuído, o adaptador aponta para o roteador (`:9000`) em vez de um servidor único (`:8000`).

### `src/orchestration/baselines.py` - configuração de baselines
Enum `BaselineType` e `get_baseline_config(name)` retornando um `BaselineConfig` (API base, flag de cache
de prefixo, config de IR, config de Redis, `compress_method`, `compress_target_ratio`, `top_k_retrieval`,
modelos de embedding/reranker). O que muda entre baselines é a fonte de contexto, se o Redis é consultado,
se compressão/especulação está ativa e qual endpoint recebe a requisição, não o código da aplicação. A
config do `compressed_rag` define `compress_method="llmlingua2"`, `compress_target_ratio=0.5`.

### `src/orchestration/ir.py` - recuperação de informação
- `build_corpus_from_contexts()` deduplica todas as passagens do dataset por `stable_text_id()` (SHA1 do texto).
- `FaissIRIndex` faz embedding das passagens com `intfloat/e5-large-v2` em um `IndexFlatIP` (produto interno), persistido em `experiments/ir_index/` para reuso.
- Recuperação: embedding da consulta, top-k FAISS (k=3), rerank opcional com CrossEncoder `BAAI/bge-reranker-large`.
- **`retrieval_hit_rate()` (corrigido):** a verificação primária é casamento exato de `doc_id`; um fallback por texto normalizado agora também conta acerto quando o texto da passagem gold está entre os textos recuperados. Isso corrigiu um bug da Fase 2 em que a métrica lia 0,0 falso em todas as linhas (divergência de hash de doc-id) mesmo com similaridade top-1 ~0,99. Reporte `retrieval_top1_score`, não a flag bruta de acerto, para qualidade de recuperação.

### `src/orchestration/compression.py` - compressão de contexto/prompt
`ContextCompressor` (método `llmlingua2`, carrega `llmlingua.PromptCompressor` de forma preguiçosa).
`compress()` retorna `(compressed_docs, CompressionStats)` com `compression_ratio`, `compression_applied`
e contagens de tokens. **Modo estrito (correção de validade):** com `CAGE_REQUIRE_COMPRESSION=1`, um
compressor ausente/falho LEVANTA exceção em vez de passar adiante silenciosamente. Na Fase 2 o `llmlingua`
não estava instalado, então esse arm falhou silenciosamente (razão 1,0); o `llmlingua` agora está no
`requirements.txt` e os scripts de compressão fazem pré-checagem de `import llmlingua` e rodam estrito.

### `src/evaluation/quality.py` - métricas de qualidade
- **Fundamentação (LettuceDetect, PRIMÁRIA):** detector de alucinação token/trecho ModernBERT; `grounding_score = 1 − razão_de_trechos_alucinados`. Desative com `CAGE_DISABLE_LETTUCEDETECT=1` (cai para NLI).
- **Fidelidade (secundária):** resposta dividida em afirmações; cada uma checada por entailment NLI contra o contexto (DeBERTa-mnli / fallback BART-mnli); pontuação = fração de afirmações com entailment.
- **F1 / Exact Match, ROUGE-L, relevância de contexto** (relevância é diagnóstica, cosseno de embeddings).
- **`evaluate_completeness()` (corrigida):** retorna `None` (não um sentinela) quando a referência é vazia, excluindo os itens não respondíveis do SQuAD v2 do agregado. **O BERTScore está descontinuado** (não discriminativo).
- A agregação usa `mean_or_none()`, que ignora `None`, então métricas indefinidas por linha nunca poluem a média.

### `src/evaluation/performance.py` - métricas de desempenho
`PerformanceEvaluator` agrega o timing por requisição em QPS, tokens/s e média/p50/p95/p99 de TTFT,
TPOT (`(total_time_ms − ttft_ms)/(num_tokens − 1)`) e latência fim a fim. `CacheMetricsTracker` calcula a
razão de prompt em cache a partir de `prompt_tokens`/`cached_prompt_tokens`.

### `src/evaluation/compression.py` - pegada analítica de KV
`analytical_kv_footprint()` liga a arquitetura do modelo a uma estimativa de bytes do cache KV para que o
FP8 (compressed_cag) possa ser comparado analiticamente com um baseline em precisão plena (o modelo de KV
do eixo de compressão).

### `src/monitoring/vllm_telemetry.py` - amostrador de telemetria de serviço/GPU
`VllmTelemetrySampler` é um poller em thread do `capture_snapshot()` do cage-stats que roda durante todo o
baseline. `.aggregate()` retorna medidores de pico/média (uso de GPU, memória, potência, temperatura,
utilização do cache KV, acerto de prefixo) mais contadores finais, gravados em `vllm_telemetry.json`.
Ativado com `--vllm-telemetry`.

### `src/orchestration/router.py` - roteador distribuído (FastAPI)
Recebe `/v1/completions`, faz hash do prefixo do prompt (`sha256(prompt[:prefix_length])`), mapeia para
uma réplica (`hash % num_replicas`), encaminha a requisição (bloqueante ou em streaming) e reporta a
réplica que atendeu. A afinidade de prefixo mantém o cache de prefixo de uma réplica aquecido; um cold miss
na réplica errada foi o que produziu a cauda pesada do distribuído na Fase 1.

### `src/orchestration/redis_cache.py` - cache de recuperação em Redis
Faz cache de resultados de recuperação (consulta -> ids de documento), NÃO de tensores KV. `get/set` com
chave SHA1 da consulta. Limpo antes de baselines frios; baselines quentes o pré-populam.

### `src/orchestration/cache_manager.py` - políticas de distribuição do cache KV
Define a interface de políticas: `REPLICATED` (em uso agora), `SHARDED_TENSOR`, `SHARDED_CONTEXT`,
`OFFLOAD_CPU`, `OFFLOAD_NVME`. O `SimulatedKVCacheManager` deriva bytes/latência de transferência entre nós
analiticamente, a partir da pegada do cache e da banda do interconnect (nenhum tensor se move). **A Fase 3
substitui isso por um conector KV real do vLLM (LMCache/NIXL) sob uma política fragmentada.** Até lá, todos
os números de transferência são simulados.

### `src/utils/prompting.py` - templates de prompt
`format_qa_prompt(question, contexts, system_prefix=...)`. O prefixo de sistema é idêntico entre as
requisições (para que o cache de prefixo reutilize seu KV) e agora instrui uma resposta CURTA e direta para
suprimir o vazamento de raciocínio.

---

## Integração com o vLLM

### `scripts/manage_vllm_server.sh` - ciclo de vida do servidor
Inicia/para/reinicia um único servidor vLLM (caminhos ancorados na raiz do repositório; logs em
`logs/vllm/`). O argv é montado como um **array de bash**, de modo que valores com espaços (o JSON de
config especulativa) nunca sofram word-splitting. Alavancas (todas via variáveis de ambiente):
- `--enable-prefix-caching` (ou `--no-enable-prefix-caching`).
- `--kv-cache-dtype fp8` via `VLLM_KV_CACHE_DTYPE` (compressed_cag).
- `--speculative-config '<json>'` via `VLLM_SPECULATIVE_CONFIG` (API atual; a antiga `--speculative-model` está descontinuada).
- `--max-model-len ${VLLM_MAX_MODEL_LEN:-8192}` e `--gpu-memory-utilization ${VLLM_GPU_MEMORY_UTILIZATION:-0.92}`.
- `--enforce-eager` via `VLLM_ENFORCE_EAGER=1` (pula torch.compile/CUDA-graph; inicialização mais rápida e confiável na L4).
- `--enable-prompt-tokens-details` (necessário para a telemetria de tokens em cache).

`stop_server()` encerra `vllm serve`, o processo separado `VLLM::EngineCore` e qualquer processo ainda
segurando a GPU. Isso corrige um vazamento real: o vLLM v1 sobe o EngineCore como processo próprio; matar só
o `vllm serve` o deixava órfão, ele mantinha a VRAM e o próximo start falhava. `get_vllm_pid()` usa
`head -n1` para que um casamento de múltiplos PIDs não quebre a checagem de modo do cache de prefixo.

### Decodificação especulativa
Configurada via JSON em `--speculative-config`. Métodos verificados: ngram, draft_model, eagle, eagle3,
medusa, mlp_speculator, mtp (+ deepseek_mtp/ernie_mtp/mimo_mtp). Em uma única L4 só **ngram** e **EAGLE-3**
(`AngelSlim/Qwen3-8B_eagle3`, `num_speculative_tokens=5`) são viáveis; a matriz especulativa limita
`--max-model-len 4096` para a cabeça EAGLE caber ao lado do alvo de 8B. A especulação é sem perda na saída.

### Gate de FP8 x cache de prefixo
`scripts/check_fp8_prefix_cache.sh` verifica que o FP8 KV NÃO desativa o cache de prefixo antes de rodar o
compressed_cag (caso contrário esse arm ficaria confundido como "sem reuso + compressão").

---

## Eixos de compressão e especulação

A **compressão 2×2** é produzida por `scripts/run_compression.sh` (cag_full, rag_full, compressed_cag,
compressed_rag), com gate da checagem de FP8 e (para o compressed_rag) da pré-checagem do llmlingua. Dois
mecanismos: o FP8 KV é uma alavanca no lançamento do servidor (não muda os tokens de prompt); o LLMLingua-2
é compressão de prompt no lado do cliente (reduz os tokens de prompt). `scripts/rerun_compressed_rag.sh`
reexecuta o arm RAG com recuperação forçada e compressão estrita.

A **especulação 2×2** é produzida por `scripts/run_speculative_matrix.sh`: {ngram, eagle3} × {CAG gold,
RAG recuperado}. Isola o efeito de serviço (aceitação/TTFT/vazão) de cada método especulativo sob cada
estratégia de contexto, um cruzamento que outros frameworks não medem.

---

## Camada estatística

`scripts/statistical_tests.py` lê o `results.csv` por baseline, roda testes de **Wilcoxon** por consulta
contra um baseline de `--reference` (pareado por `example_id`; cai para Mann-Whitney não pareado quando os
ids compartilhados são insuficientes, p.ex. hybrid_warm), aplica correção de **Holm** e reporta **delta de
Cliff** e ICs por **bootstrap**. Emite um resumo JSON (`--output`) e uma tabela LaTeX pronta para o artigo
(`--latex-out`). Requer scipy. Saída da Fase 2: `phase2_archive/analysis/all_results/phase2_stats.{json,tex}`.

---

## Telemetria, logs e desligamento seguro

- **`scripts/sync_results_to_gcs.sh`** espelha um diretório local para o bucket GCS; um 3º argumento opcional
  define o subcaminho remoto (usado para namespacear logs por host); `-c` compara por checksum, protegendo
  contra uploads parciais.
- **`scripts/collect_logs.sh`** reúne TODOS os logs (servidor vLLM, stdout das execuções, timeline de status)
  mais forenses de sistema (nvidia-smi, dmesg OOM/Xid, journalctl, pip freeze, env, docker logs) em
  `vm_logs/<host>/` e grava um **sentinela de sucesso** por execução como último upload.
- **`scripts/teardown_vm.sh`** roda a coleta, **verifica que o sentinela da execução está no GCS e se recusa
  a excluir a VM se ele estiver ausente** (falha de forma segura; `--force` sobrescreve). Existe porque a
  primeira desativação da Fase 2 perdeu logs que só existiam na VM.
- **`scripts/log_sync_daemon.sh`** / **`scripts/_log_guard.sh`** espelham logs+resultados continuamente
  durante uma execução; **`scripts/gcp_shutdown_hook.sh`** coleta em preempção de spot / desligamento ACPI -
  mas SOMENTE se for explicitamente conectado na criação da VM
  (`gcloud ... --metadata-from-file shutdown-script=scripts/gcp_shutdown_hook.sh`); nenhum script de setup o
  instala automaticamente. Em uma L4 **on-demand** desativada via `teardown_vm.sh` ele nao e necessario (o
  collect do trap EXIT + o teardown fail-closed cobrem isso); conecte-o apenas para instancias **spot**.
- **`scripts/cloud_run.sh`** orquestra a suíte principal e sincroniza resultados + logs a cada intervalo e na
  saída (traps de EXIT + SIGTERM).

---

## Infraestrutura

- **Caminho principal (Fase 2):** uma única VM GCP `g2-standard-8` + L4; vLLM executado diretamente via
  `manage_vllm_server.sh`; orquestração + telemetria via `cloud_run.sh`; resultados para um bucket GCS durável.
- **Terraform (`terraform/gcp/`):** provisiona o cluster (roteador + N réplicas de GPU + Redis + GCS), com
  GVNIC + MTU 8896 para o interconnect de alta banda da Fase 3 (`num_replicas`, `nic_type`, `network_mtu`,
  `vllm_extra_args` são tfvars).
- **Docker Compose / Kubernetes (`docker/`, `k8s/`):** manifestos legados/locais e de cluster (roteador +
  réplicas + Redis). A Fase 2 usou o caminho direto de `vllm serve`, não o compose.

---

## Arquivos de configuração
- `configs/experiment/*.yaml` - baseline, num_queries, flags de avaliação, diretório de saída.
- `configs/model/*.yaml` - qwen3-4b/8b/14b/30b-a3b, qwen2.5-7b-instruct (nome, max_tokens, dtype, hardware).
- `configs/dataset/*.yaml` - squad_v2, hotpotqa e os loaders adicionais.

---

## Como baselines específicos funcionam (atual)
- **no_cache:** servidor sem cache de prefixo; contexto gold; prefill completo a cada requisição.
- **prefix_cache / cag_full:** servidor com cache de prefixo; contexto gold; KV de prefixo compartilhado reutilizado.
- **rag / rag_full:** recuperação (FAISS + rerank); gold não usado; recomputação completa.
- **redis (frio/quente):** RAG com resultados de recuperação em cache no Redis.
- **hybrid (frio/quente):** recuperação + cache de prefixo (+ Redis); quente pré-popula ambos os caches.
- **compressed_cag:** cache de prefixo + `--kv-cache-dtype fp8` (KV pela metade; tokens de prompt inalterados por construção).
- **compressed_rag:** RAG + compressão de prompt LLMLingua-2 (rodado estrito para não falhar silenciosamente).
- **speculative (ngram | eagle3) × (CAG | RAG):** a estratégia de contexto subjacente mais um método de rascunho; sem perda na saída, varia apenas a velocidade de serviço.
- **distributed:** N réplicas atrás do roteador por hash de prefixo; política replicada agora (transferência = 0); fragmentada + transferência real de KV na Fase 3.

---

## Geração de gráficos e verificação
- `scripts/generate_publication_plots.py` / `generate_additional_plots.py` / `generate_compact_figures.py` - figuras a partir de `metrics.json` (latência, vazão, Pareto, radar, heatmap, cauda de latência, ranking).
- `scripts/run_status.py` - status por baseline (iniciado / rodando / concluído / erros); `scripts/extract_qa_evidence.py` - evidência de Q/A início/meio/fim por baseline; `scripts/verify_results.py` - checagens de sanidade dos resultados.

## Suíte de testes (`tests/`)
`test_inference.py`, `test_ir.py`, `test_baselines.py`, `test_router_integration.py`,
`test_vllm_integration.py`, `test_data.py`. Execute com `pytest tests/ -v` (ou `scripts/run_tests.sh`).
