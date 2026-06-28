# CAGE Dissertation Defense — Bilingual Presentation Guide
# CAGE — Guia de Apresentação de Defesa (Bilíngue)

> Honest-scope guide. This is a **Phase-1, single-CPU-node framework-validation** dissertation. Cross-node KV transfer is **simulated** (analytical, zeroed in the policy actually run); FP8 and speculative decoding are **launch-time/record-only levers**; statistical testing is a **standalone post-hoc script**; generation is **stochastic (temperature 0.7, no sampling seed)**. Present what is delivered now versus what is future work, and you will be defensible.

> Guia de escopo honesto. Esta é uma dissertação de **validação de framework em Fase 1, em um único nó de CPU**. A transferência de KV entre nós é **simulada** (analítica, zerada na política efetivamente executada); FP8 e decodificação especulativa são **alavancas de tempo de lançamento / apenas-registro**; o teste estatístico é um **script pós-hoc independente**; a geração é **estocástica (temperatura 0.7, sem semente de amostragem)**. Apresente o que está entregue agora versus o que é trabalho futuro, e a defesa será sustentável.

---

## (a) Elevator Pitch / Discurso de Elevador

**EN.** CAGE is a measurement framework that asks a question the LLM-serving community keeps answering one dimension at a time: when you trade retrieval for cached context, what do you actually win in serving efficiency and what do you lose in answer faithfulness? CAGE co-measures both on the same queries across a controlled taxonomy of nine baselines (no-cache, prefix-cache, Redis, RAG, distributed, hybrid, speculative, and two compression arms) plus a 2x2 compression axis, instrumented from outside a real vLLM server. Phase 1 validates the framework end-to-end on a single CPU node and already shows a clean result: prefix caching cuts latency by 37.4% and time-to-first-token by 65.7% while keeping faithfulness at parity with no caching. The at-scale, memory-pressure, cross-node story is the framework's designed future, not a claim I make today.

**PT.** O CAGE é um framework de medição que faz uma pergunta que a comunidade de serving de LLMs costuma responder uma dimensão de cada vez: ao trocar recuperação por contexto em cache, o que de fato se ganha em eficiência de serving e o que se perde em fidelidade da resposta? O CAGE co-mede ambos nas mesmas consultas, sobre uma taxonomia controlada de nove baselines (sem-cache, prefix-cache, Redis, RAG, distribuído, híbrido, especulativo e dois braços de compressão) mais um eixo de compressão 2x2, instrumentado por fora de um servidor vLLM real. A Fase 1 valida o framework de ponta a ponta em um único nó de CPU e já mostra um resultado limpo: o prefix caching reduz a latência em 37,4% e o tempo até o primeiro token em 65,7%, mantendo a fidelidade em paridade com a ausência de cache. A história em escala, sob pressão de memória e entre nós, é o futuro projetado do framework, não uma alegação que eu faça hoje.

---

## (b) Slide-by-Slide Outline / Roteiro Slide a Slide

### Slide 1 — Title & Framing / Título e Enquadramento
- **EN.** CAGE: a framework to jointly measure serving efficiency and information-retrieval quality across caching and retrieval strategies. Master's dissertation, PUC Minas.
- **EN.** State the scope up front: Phase 1 is local CPU validation; distributed and HPC scaling are designed future phases.
- **EN.** Name the central tension in one line: efficiency wins and faithfulness wins are usually reported separately; CAGE reports them together.
- **PT.** CAGE: um framework para medir conjuntamente a eficiência de serving e a qualidade de recuperação de informação entre estratégias de cache e recuperação. Dissertação de mestrado, PUC Minas.
- **PT.** Declare o escopo logo no início: a Fase 1 é validação local em CPU; o escalonamento distribuído e para HPC são fases futuras projetadas.
- **PT.** Nomeie a tensão central em uma linha: ganhos de eficiência e ganhos de fidelidade são em geral relatados separadamente; o CAGE os relata juntos.

### Slide 2 — Problem & Motivation / Problema e Motivação
- **EN.** LLM serving is bottlenecked by the KV cache: the prefill builds it, decode reuses it, and memory pressure grows with context length.
- **EN.** Two families address long context differently: RAG fetches external passages each turn; CAG (cache-augmented generation) keeps a reusable cached context and avoids per-turn retrieval overhead.
- **EN.** Practitioners pick one axis (lower latency, or higher retrieval quality) without a shared way to see the trade-off on the same workload.
- **PT.** O serving de LLMs é limitado pelo KV cache: o prefill o constrói, o decode o reutiliza, e a pressão de memória cresce com o comprimento do contexto.
- **PT.** Duas famílias tratam o contexto longo de formas distintas: o RAG busca trechos externos a cada turno; o CAG (geração aumentada por cache) mantém um contexto em cache reutilizável e evita o custo de recuperação por turno.
- **PT.** Os profissionais escolhem um eixo (menor latência, ou maior qualidade de recuperação) sem uma forma compartilhada de enxergar o trade-off no mesmo workload.

### Slide 3 — The Efficiency-vs-Quality Gap / A Lacuna Eficiência-vs-Qualidade
- **EN.** Serving papers optimize latency and throughput; retrieval papers optimize faithfulness and relevance; few co-measure them on identical queries.
- **EN.** The hidden confound: comparing a cached arm against a retrieval arm often compares gold context against retrieved context, not caching against no-caching.
- **EN.** CAGE's contribution is methodological: separate the context source from the reuse policy so the two effects can be read apart.
- **PT.** Artigos de serving otimizam latência e throughput; artigos de recuperação otimizam fidelidade e relevância; poucos os co-medem nas mesmas consultas.
- **PT.** O confundimento oculto: comparar um braço com cache contra um braço de recuperação muitas vezes compara contexto gold contra contexto recuperado, e não cache contra ausência de cache.
- **PT.** A contribuição do CAGE é metodológica: separar a fonte do contexto da política de reuso para que os dois efeitos possam ser lidos isoladamente.

### Slide 4 — Research Questions & Hypotheses / Questões de Pesquisa e Hipóteses
- **EN.** RQ1: what metric suite jointly captures serving efficiency, cache reuse, and faithfulness? (Delivered.)
- **EN.** RQ2: measurable effect of cache reuse on performance and faithfulness versus retrieval/hybrid? (Performance delivered; faithfulness causal claim is confounded by context source, stated honestly.)
- **EN.** RQ3: how does the design hold under real KV-cache memory pressure at scale? (Future; the at-scale trade-off has no data yet.) RQ4: which telemetry signals attribute behavior to local reuse, retrieval, or cross-node transfer? (Local/retrieval attribution delivered; transfer attribution future.)
- **PT.** RQ1: que conjunto de métricas captura conjuntamente eficiência de serving, reuso de cache e fidelidade? (Entregue.)
- **PT.** RQ2: efeito mensurável do reuso de cache sobre desempenho e fidelidade versus recuperação/híbrido? (Desempenho entregue; a alegação causal de fidelidade é confundida pela fonte de contexto, declarado com honestidade.)
- **PT.** RQ3: como o projeto se sustenta sob pressão real de memória de KV cache em escala? (Futuro; o trade-off em escala ainda não tem dados.) RQ4: que sinais de telemetria atribuem o comportamento a reuso local, recuperação ou transferência entre nós? (Atribuição local/recuperação entregue; atribuição de transferência futura.)

### Slide 5 — CAGE Architecture / Arquitetura do CAGE
- **EN.** Instrument-from-outside: CAGE drives a real vLLM server and records telemetry; it does not modify the model or the serving engine internals.
- **EN.** A prefix-aware router dispatches requests; the cache manager is a simulated KV-cache manager running a replicated policy (every node holds the context).
- **EN.** An evaluation layer computes the quality metrics offline from the recorded generations and contexts.
- **PT.** Instrumentar por fora: o CAGE conduz um servidor vLLM real e registra telemetria; não modifica o modelo nem os internos do motor de serving.
- **PT.** Um roteador ciente de prefixo despacha as requisições; o gerenciador de cache é um gerenciador de KV cache simulado executando uma política replicada (cada nó mantém o contexto).
- **PT.** Uma camada de avaliação calcula as métricas de qualidade offline a partir das gerações e contextos registrados.

### Slide 6 — The Nine Baselines / Os Nove Baselines
- **EN.** The taxonomy: no-cache, prefix-cache, Redis, RAG, distributed, hybrid, speculative, compressed-RAG, compressed-CAG. Each varies exactly one axis (context source or reuse policy).
- **EN.** Honest note on two of them: speculative decoding and FP8 KV compression are vLLM launch-time settings, not per-request knobs; the runner records the intended flag but does not inject it into inference.
- **EN.** Phase 1 reports seven of the nine; the two compression arms belong to the compression axis, which is currently future work for results.
- **PT.** A taxonomia: sem-cache, prefix-cache, Redis, RAG, distribuído, híbrido, especulativo, RAG-comprimido, CAG-comprimido. Cada um varia exatamente um eixo (fonte de contexto ou política de reuso).
- **PT.** Observação honesta sobre dois deles: a decodificação especulativa e a compressão de KV em FP8 são configurações de tempo de lançamento do vLLM, não botões por requisição; o executor registra o flag pretendido mas não o injeta na inferência.
- **PT.** A Fase 1 reporta sete dos nove; os dois braços de compressão pertencem ao eixo de compressão, que é, no momento, trabalho futuro para resultados.

### Slide 7 — The 2x2 Compression Axis / O Eixo de Compressão 2x2
- **EN.** Two orthogonal compression levers: prompt-side text compression (LLMLingua-style) and server-side KV compression (FP8), at a matched roughly 2x operating point.
- **EN.** Status is honest: the compression baselines exist in code, but the analytical cache-footprint model has no runtime caller and FP8 is a launch-time/record-only flag. No compression results are reported yet.
- **EN.** A known threat is already documented: FP8 interacting with prefix caching needs a per-vLLM-version compatibility gate before any compression claim is trusted.
- **PT.** Duas alavancas ortogonais de compressão: compressão de texto no prompt (estilo LLMLingua) e compressão de KV no servidor (FP8), em um ponto de operação equiparado de aproximadamente 2x.
- **PT.** O status é honesto: os baselines de compressão existem no código, mas o modelo analítico de footprint de cache não tem chamador em tempo de execução e o FP8 é um flag de tempo de lançamento / apenas-registro. Nenhum resultado de compressão é reportado ainda.
- **PT.** Uma ameaça conhecida já está documentada: a interação do FP8 com o prefix caching exige um portão de compatibilidade por versão do vLLM antes de confiar em qualquer alegação de compressão.

### Slide 8 — Methodology & Metrics / Metodologia e Métricas
- **EN.** Quality suite: LettuceDetect span grounding is the declared PRIMARY metric; claim-level NLI faithfulness (RAGAS-style, max-over-context entailment, DeBERTa NLI); BERTScore baseline-rescaled as a completeness metric; ROUGE-L, token F1, and exact match.
- **EN.** Serving suite: latency, time-to-first-token, time-per-output-token, throughput; cache telemetry: prompt-cached ratio, retrieval hit, and a heuristic cache hit/miss inferred from cached prompt tokens.
- **EN.** Design choices stated plainly: a model that fails to load yields None (not zero); an empty answer is a genuine zero; the unit of analysis is the query.
- **PT.** Conjunto de qualidade: o grounding por spans do LettuceDetect é a métrica PRIMÁRIA declarada; fidelidade por NLI em nível de afirmação (estilo RAGAS, entailment máximo sobre o contexto, NLI DeBERTa); BERTScore com rescala de baseline como métrica de completude; ROUGE-L, F1 por token e correspondência exata.
- **PT.** Conjunto de serving: latência, tempo até o primeiro token, tempo por token de saída, throughput; telemetria de cache: razão de prompt em cache, acerto de recuperação e um acerto/erro de cache heurístico inferido dos tokens de prompt em cache.
- **PT.** Escolhas de projeto declaradas com clareza: um modelo que falha ao carregar resulta em None (não zero); uma resposta vazia é um zero genuíno; a unidade de análise é a consulta.

### Slide 9 — Statistical Rigor / Rigor Estatístico
- **EN.** Per-query paired Wilcoxon signed-rank tests, with a normal approximation fallback when SciPy is absent.
- **EN.** Holm-Bonferroni correction within each metric across baseline comparisons; 10,000-iteration bootstrap confidence intervals on the mean difference.
- **EN.** Honest integration note: this lives in a standalone post-hoc script that reads the trial result files; it is not yet called by the experiment runner, which reports mean and standard deviation across trials. Significance testing is a separate manual step.
- **PT.** Testes de Wilcoxon pareados por consulta (signed-rank), com fallback de aproximação normal quando o SciPy está ausente.
- **PT.** Correção de Holm-Bonferroni dentro de cada métrica entre as comparações de baseline; intervalos de confiança por bootstrap de 10.000 iterações sobre a diferença de médias.
- **PT.** Observação honesta de integração: isto reside em um script pós-hoc independente que lê os arquivos de resultado dos ensaios; ainda não é chamado pelo executor do experimento, que reporta média e desvio padrão entre ensaios. O teste de significância é uma etapa manual separada.

### Slide 10 — Phase-1 Results / Resultados da Fase 1
- **EN.** Prefix caching versus no-cache: 37.4% lower latency, 65.7% lower time-to-first-token, with faithfulness at parity (both around 0.570). This is clean because prefix caching is output-preserving by design.
- **EN.** The cache telemetry separates concerns cleanly: RAG and Redis show roughly 98% retrieval hit but 0% serving cache hit and no time-to-first-token benefit. Retrieval success is not serving efficiency.
- **EN.** Tail behavior: about a 7.6x spread between the 95th and 50th percentile time-to-first-token, driven by cold local prefix caches (not by transfer cost, which is zero by construction here).
- **PT.** Prefix caching versus sem-cache: 37,4% menos latência, 65,7% menos tempo até o primeiro token, com fidelidade em paridade (ambos em torno de 0,570). O resultado é limpo porque o prefix caching preserva a saída por construção.
- **PT.** A telemetria de cache separa as preocupações com clareza: RAG e Redis mostram cerca de 98% de acerto de recuperação mas 0% de acerto de cache de serving e nenhum benefício de tempo até o primeiro token. Sucesso de recuperação não é eficiência de serving.
- **PT.** Comportamento de cauda: cerca de 7,6x de dispersão entre os percentis 95 e 50 do tempo até o primeiro token, causada por caches de prefixo locais frios (não por custo de transferência, que aqui é zero por construção).

### Slide 11 — What Is Simulated / Preliminary / O Que É Simulado / Preliminar
- **EN.** Cross-node KV transfer is simulated by an analytical bandwidth model, and the replicated policy actually run zeroes the transfer cost. The "distributed" arm measures routing, not transfer.
- **EN.** Everything ran on a single CPU node; absolute latency is non-generalizable and the setup does not induce the memory pressure that motivates CAG.
- **EN.** Generation is stochastic at temperature 0.7 with no sampling seed; only dataset sampling is seeded. Small quality deltas below the standard deviation are within noise and should not be read as findings.
- **PT.** A transferência de KV entre nós é simulada por um modelo analítico de largura de banda, e a política replicada efetivamente executada zera o custo de transferência. O braço "distribuído" mede roteamento, não transferência.
- **PT.** Tudo rodou em um único nó de CPU; a latência absoluta não é generalizável e o setup não induz a pressão de memória que motiva o CAG.
- **PT.** A geração é estocástica a temperatura 0,7 sem semente de amostragem; apenas a amostragem do dataset tem semente. Pequenas diferenças de qualidade abaixo do desvio padrão estão dentro do ruído e não devem ser lidas como achados.

### Slide 12 — Contributions / Contribuições
- **EN.** A controlled baseline taxonomy that separates context source from reuse policy, with the equalization mode (gold versus retrieved) implemented in code.
- **EN.** A jointly-computed metric suite spanning serving, cache telemetry, and faithfulness, with documented model-unavailable and empty-answer semantics.
- **EN.** A reproducible, instrument-from-outside Phase-1 validation on a controlled state, plus an analytical cache-footprint model and a documented compression axis provided as the foundation for the future scaling phases.
- **PT.** Uma taxonomia controlada de baselines que separa fonte de contexto de política de reuso, com o modo de equiparação (gold versus recuperado) implementado no código.
- **PT.** Um conjunto de métricas computado conjuntamente abrangendo serving, telemetria de cache e fidelidade, com semântica documentada para modelo indisponível e resposta vazia.
- **PT.** Uma validação de Fase 1 reprodutível e instrumentada por fora sobre um estado controlado, mais um modelo analítico de footprint de cache e um eixo de compressão documentado, fornecidos como base para as fases futuras de escalonamento.

### Slide 13 — Threats to Validity / Ameaças à Validade
- **EN.** Gold-vs-retrieved confound: under the default, cached arms get gold context and retrieval arms get retrieved context, so the faithfulness gap is a context-source effect, not a caching effect. The equalized arms that isolate it were not run.
- **EN.** Stochastic decoding without a sampling seed makes small quality deltas unreliable; the highest faithfulness number in the distributed arm is most likely sampling noise and should not be presented as a win.
- **EN.** Simulated transfer, launch-time-only FP8/speculative levers, heuristic cache hit/miss dependent on a vLLM launch flag, non-streaming time-to-first-token equal to total latency, and a non-integrated significance script.
- **PT.** Confundimento gold-vs-recuperado: por padrão, os braços com cache recebem contexto gold e os braços de recuperação recebem contexto recuperado, então a diferença de fidelidade é um efeito de fonte de contexto, não de cache. Os braços equiparados que o isolariam não foram executados.
- **PT.** A decodificação estocástica sem semente de amostragem torna pequenas diferenças de qualidade não confiáveis; o maior número de fidelidade no braço distribuído é muito provavelmente ruído de amostragem e não deve ser apresentado como vitória.
- **PT.** Transferência simulada, alavancas de FP8/especulativo apenas em tempo de lançamento, acerto/erro de cache heurístico dependente de um flag de lançamento do vLLM, tempo até o primeiro token sem streaming igual à latência total e um script de significância não integrado.

### Slide 14 — Roadmap: Phase 2 / Phase 3 / Roteiro: Fase 2 / Fase 3
- **EN.** Phase 2 (GPU): the GPU metrics tracker is already wired and runs during the measured stage; this brings real memory-pressure evidence on a single accelerator.
- **EN.** Phase 3 (multi-node HPC): replace the simulated transfer with a real vLLM KV connector (the KV transfer config plus a connector such as LMCache or NIXL), then exercise the sharded-context policy that actually pays transfer cost.
- **EN.** Cross-cutting near-term fixes: add a sampling seed, run the equalized context-source arms, integrate the significance script, report the LettuceDetect grounding column, and gate FP8-times-prefix-cache per vLLM version.
- **PT.** Fase 2 (GPU): o rastreador de métricas de GPU já está integrado e roda durante o estágio medido; isso traz evidência real de pressão de memória em um único acelerador.
- **PT.** Fase 3 (HPC multi-nó): substituir a transferência simulada por um conector de KV real do vLLM (a configuração de transferência de KV mais um conector como LMCache ou NIXL), e então exercitar a política de contexto fragmentado que de fato paga o custo de transferência.
- **PT.** Correções transversais de curto prazo: adicionar semente de amostragem, executar os braços equiparados de fonte de contexto, integrar o script de significância, reportar a coluna de grounding do LettuceDetect e criar o portão FP8-vezes-prefix-cache por versão do vLLM.

### Slide 15 — Conclusion / Conclusão
- **EN.** CAGE delivers a credible Phase-1 framework: a controlled taxonomy, a jointly-computed metric suite, and a clean demonstration that prefix caching buys serving efficiency at faithfulness parity.
- **EN.** The titular at-scale, memory-pressure trade-off is the designed next phase, not a present claim, and the dissertation states this honestly.
- **EN.** The contribution is the measurement methodology itself: a shared, reproducible way to read serving efficiency and retrieval quality on the same workload.
- **PT.** O CAGE entrega um framework de Fase 1 crível: uma taxonomia controlada, um conjunto de métricas computado conjuntamente e uma demonstração limpa de que o prefix caching compra eficiência de serving com paridade de fidelidade.
- **PT.** O trade-off em escala e sob pressão de memória do título é a próxima fase projetada, não uma alegação presente, e a dissertação declara isso com honestidade.
- **PT.** A contribuição é a própria metodologia de medição: uma forma compartilhada e reprodutível de ler eficiência de serving e qualidade de recuperação no mesmo workload.

---

## (c) Narrative Arc / Arco Narrativo

**EN.** Open with the bottleneck everyone shares (the KV cache) and the choice everyone faces (retrieve versus cache). Show that the field answers efficiency and quality separately, and that the most common comparison hides a confound. Position CAGE as the missing shared instrument: a controlled taxonomy that varies one axis at a time, a metric suite that reads both sides on the same queries, and rigorous query-level statistics. Then deliver the honest Phase-1 result: prefix caching wins on serving at faithfulness parity, and the telemetry proves retrieval success is not serving efficiency. Pivot deliberately to candor: this is one CPU node, transfer is simulated, decoding is stochastic, compression has no results yet. Turn that candor into a roadmap (GPU memory pressure, real cross-node transfer, equalized arms, integrated statistics) and close on the durable contribution, which is the methodology, not any single number.

**PT.** Abra com o gargalo que todos compartilham (o KV cache) e a escolha que todos enfrentam (recuperar versus cachear). Mostre que a área responde eficiência e qualidade separadamente e que a comparação mais comum esconde um confundimento. Posicione o CAGE como o instrumento compartilhado que falta: uma taxonomia controlada que varia um eixo por vez, um conjunto de métricas que lê os dois lados nas mesmas consultas e estatística rigorosa em nível de consulta. Em seguida, entregue o resultado honesto da Fase 1: o prefix caching vence em serving com paridade de fidelidade, e a telemetria prova que sucesso de recuperação não é eficiência de serving. Faça uma virada deliberada para a franqueza: é um único nó de CPU, a transferência é simulada, a decodificação é estocástica, a compressão ainda não tem resultados. Converta essa franqueza em um roteiro (pressão de memória em GPU, transferência real entre nós, braços equiparados, estatística integrada) e encerre na contribuição durável, que é a metodologia, não qualquer número isolado.

---

## (d) Anticipated Examiner Questions + Strong Answers / Perguntas Antecipadas da Banca + Respostas Fortes

**Q1. Why is cross-node KV transfer simulated rather than measured?**
- **EN.** Because Phase 1 deliberately validates the framework on a controlled single-node state before introducing real network variability. The only cache manager is a simulated KV-cache manager, and the replicated policy I ran zeroes transfer cost by construction, so the distributed arm measures routing, not transfer. The analytical bandwidth model is there for offline estimation; real transfer arrives in Phase 3 by swapping in a real vLLM KV connector. I am explicit that no number in this dissertation reflects real cross-node transfer.
- **PT.** Porque a Fase 1 valida deliberadamente o framework em um estado controlado de nó único antes de introduzir a variabilidade real de rede. O único gerenciador de cache é simulado, e a política replicada que executei zera o custo de transferência por construção, então o braço distribuído mede roteamento, não transferência. O modelo analítico de largura de banda existe para estimativa offline; a transferência real chega na Fase 3 trocando por um conector de KV real do vLLM. Sou explícito de que nenhum número desta dissertação reflete transferência real entre nós.

**Q2. Is BERTScore your primary metric or a control?**
- **EN.** Neither as a primary, and I will be precise. The declared PRIMARY quality metric in the code is LettuceDetect span grounding. BERTScore is computed and labeled as a completeness metric, baseline-rescaled. In the Results discussion I interpret its near-constant behavior across baselines as functioning like a negative control, but that is my interpretation, not a label asserted by the code. I keep that distinction explicit so the committee knows what is measured versus what is inferred.
- **PT.** Nenhum como primária, e serei preciso. A métrica de qualidade PRIMÁRIA declarada no código é o grounding por spans do LettuceDetect. O BERTScore é computado e rotulado como métrica de completude, com rescala de baseline. Na discussão dos resultados, interpreto seu comportamento quase constante entre os baselines como atuando como um controle negativo, mas isso é interpretação minha, não um rótulo afirmado pelo código. Mantenho essa distinção explícita para a banca saber o que é medido versus o que é inferido.

**Q3. Your faithfulness drop from cache to retrieval: isn't that just a gold-vs-retrieved confound?**
- **EN.** Yes, and I state it as such. Under the default context source, cached arms receive gold context and retrieval arms receive retrieved context, so the roughly 11.6% faithfulness change measures gold-to-retrieved, not cache-to-no-cache. The code already supports equalized modes (force gold, or force retrieved) that would isolate caching; I did not run them in Phase 1. So I present this as a retrieval-quality effect consistent with the literature, and I list the equalized run as the immediate next step to deliver the isolation objective properly.
- **PT.** Sim, e declaro isso explicitamente. Na fonte de contexto padrão, os braços com cache recebem contexto gold e os braços de recuperação recebem contexto recuperado, então a mudança de fidelidade de cerca de 11,6% mede gold-para-recuperado, não cache-para-sem-cache. O código já suporta modos equiparados (forçar gold ou forçar recuperado) que isolariam o cache; não os executei na Fase 1. Então apresento isto como um efeito de qualidade de recuperação consistente com a literatura, e listo a execução equiparada como o próximo passo imediato para entregar adequadamente o objetivo de isolamento.

**Q4. Why temperature 0.7 instead of greedy decoding?**
- **EN.** Temperature 0.7 with top-p 0.95 reflects a realistic serving configuration rather than an idealized deterministic one, which keeps the faithfulness and latency measurements representative of deployment. The honest cost is that generation is stochastic and I did not pin a sampling seed, so small quality deltas below the standard deviation are within noise. My corrective commitment is to add a sampling seed and to re-state any quality delta smaller than its standard deviation as not a finding. For strict reproducibility of quality claims, greedy or seeded decoding is the right Phase-2 choice.
- **PT.** Temperatura 0,7 com top-p 0,95 reflete uma configuração realista de serving em vez de uma determinística idealizada, o que mantém as medições de fidelidade e latência representativas de produção. O custo honesto é que a geração é estocástica e não fixei uma semente de amostragem, então pequenas diferenças de qualidade abaixo do desvio padrão estão dentro do ruído. Meu compromisso corretivo é adicionar uma semente de amostragem e reapresentar qualquer diferença de qualidade menor que seu desvio padrão como não-achado. Para reprodutibilidade estrita das alegações de qualidade, decodificação greedy ou com semente é a escolha certa para a Fase 2.

**Q5. What does Phase 1 actually prove?**
- **EN.** It proves the framework works end-to-end and produces a coherent, defensible result on a controlled state: the orchestration, the nine-baseline taxonomy, and every metric module run together and agree. Concretely, it shows prefix caching delivers 37.4% lower latency and 65.7% lower time-to-first-token at faithfulness parity, and that retrieval hit and serving cache hit are distinct signals. It does not prove anything about scale or memory pressure; that is explicitly deferred.
- **PT.** Prova que o framework funciona de ponta a ponta e produz um resultado coerente e defensável em um estado controlado: a orquestração, a taxonomia de nove baselines e cada módulo de métrica rodam juntos e concordam. Concretamente, mostra que o prefix caching entrega 37,4% menos latência e 65,7% menos tempo até o primeiro token com paridade de fidelidade, e que acerto de recuperação e acerto de cache de serving são sinais distintos. Não prova nada sobre escala ou pressão de memória; isso é explicitamente adiado.

**Q6. The distributed router shows the highest faithfulness (0.636). Why does a router over replicated gold context beat the gold baseline?**
- **EN.** It should not, and I do not claim it as a finding. The replicated arm sees the same gold context as the no-cache baseline, so a higher score is not mechanistically explicable. Given temperature 0.7 without a sampling seed and a standard deviation around 0.078 over fifty queries by three trials, 0.636 versus 0.570 is within sampling noise. I will un-bold it and report that all quality deltas below the standard deviation are not findings.
- **PT.** Não deveria, e não a apresento como achado. O braço replicado vê o mesmo contexto gold que o baseline sem-cache, então uma pontuação maior não é mecanicamente explicável. Dada a temperatura 0,7 sem semente de amostragem e um desvio padrão em torno de 0,078 sobre cinquenta consultas por três ensaios, 0,636 versus 0,570 está dentro do ruído de amostragem. Vou remover o negrito e reportar que todas as diferenças de qualidade abaixo do desvio padrão não são achados.

**Q7. Your title talks about HPC workloads and at-scale trade-offs. Where is that evidence?**
- **EN.** It is not in this dissertation, and I will reframe the general objective to say so. Everything ran on a single CPU node, which by my own admission does not induce the KV-cache memory pressure that motivates CAG. The at-scale quantification is a proposed and designed objective, with the GPU tracker already wired for Phase 2 and a real KV connector planned for Phase 3. The delivered claim is propose, design, implement, and locally validate; the future claim is quantify the trade-off under memory pressure.
- **PT.** Não está nesta dissertação, e vou reformular o objetivo geral para dizer isso. Tudo rodou em um único nó de CPU, que por minha própria admissão não induz a pressão de memória de KV cache que motiva o CAG. A quantificação em escala é um objetivo proposto e projetado, com o rastreador de GPU já integrado para a Fase 2 e um conector de KV real planejado para a Fase 3. A alegação entregue é propor, projetar, implementar e validar localmente; a alegação futura é quantificar o trade-off sob pressão de memória.

**Q8. Why are there nine baselines in the code but only seven in the results?**
- **EN.** The two extra baselines are the compression arms, compressed-RAG and compressed-CAG, which belong to the 2x2 compression axis. That axis has no results yet because the cache-footprint model is analytical with no runtime caller and FP8 is a launch-time-only flag, plus there is an unresolved FP8-times-prefix-cache compatibility threat. So Phase 1 honestly reports the seven non-compression baselines, and the compression axis is presented as designed future work, not as delivered results.
- **PT.** Os dois baselines extras são os braços de compressão, RAG-comprimido e CAG-comprimido, que pertencem ao eixo de compressão 2x2. Esse eixo ainda não tem resultados porque o modelo de footprint de cache é analítico, sem chamador em tempo de execução, e o FP8 é um flag apenas de tempo de lançamento, além de uma ameaça não resolvida de compatibilidade FP8-vezes-prefix-cache. Então a Fase 1 reporta honestamente os sete baselines sem compressão, e o eixo de compressão é apresentado como trabalho futuro projetado, não como resultados entregues.

**Q9. Why isn't the statistical test integrated into the experiment runner?**
- **EN.** By design for Phase 1 the runner reports mean and standard deviation across trials, and significance testing lives in a standalone post-hoc script that reads the trial result files. That script does the rigorous part: per-query paired Wilcoxon, Holm-Bonferroni correction within each metric, and 10,000-iteration bootstrap confidence intervals, with the query as the unit of analysis. The integration hook is planned; until then significance is a documented manual step, which I disclose rather than hide.
- **PT.** Por projeto, na Fase 1 o executor reporta média e desvio padrão entre ensaios, e o teste de significância reside em um script pós-hoc independente que lê os arquivos de resultado dos ensaios. Esse script faz a parte rigorosa: Wilcoxon pareado por consulta, correção de Holm-Bonferroni dentro de cada métrica e intervalos de confiança por bootstrap de 10.000 iterações, com a consulta como unidade de análise. O gancho de integração está planejado; até lá, a significância é uma etapa manual documentada, que divulgo em vez de esconder.

**Q10. Your primary metric is LettuceDetect grounding, but the Results tables report NLI faithfulness. Where is grounding?**
- **EN.** That is a real gap and I own it. Grounding is the declared primary quality metric and it is computed by the framework, but the Phase-1 Results tables report the NLI claim-level faithfulness and do not yet show a grounding column. The immediate fix is to add the LettuceDetect grounding column so the primary metric is visible alongside faithfulness, completeness, and the serving and cache columns.
- **PT.** Essa é uma lacuna real e a assumo. O grounding é a métrica de qualidade primária declarada e é computado pelo framework, mas as tabelas de Resultados da Fase 1 reportam a fidelidade por NLI em nível de afirmação e ainda não mostram uma coluna de grounding. A correção imediata é adicionar a coluna de grounding do LettuceDetect para que a métrica primária fique visível ao lado de fidelidade, completude e das colunas de serving e cache.

**Q11. Is the cache hit/miss signal ground truth?**
- **EN.** No, it is heuristic. Cache hit and miss are inferred from the cached prompt tokens reported by vLLM plus the distributed transfer parameters, not from the engine's ground-truth cache state. Coverage also depends on launching vLLM with the prompt-tokens-details flag; without it the prompt-cache ratios are unavailable. So I treat the cache-hit signal as an attribution heuristic with stated coverage, not as an exact count, and I soften any necessary-and-sufficient claim accordingly.
- **PT.** Não, é heurístico. Acerto e erro de cache são inferidos dos tokens de prompt em cache reportados pelo vLLM mais os parâmetros de transferência distribuída, não do estado de cache real do motor. A cobertura também depende de lançar o vLLM com o flag de detalhes de tokens de prompt; sem ele, as razões de prompt em cache ficam indisponíveis. Então trato o sinal de acerto de cache como uma heurística de atribuição com cobertura declarada, não como uma contagem exata, e suavizo qualquer alegação de necessário-e-suficiente de acordo.

**Q12. The relevance metric is constant across baselines. Is it a dead metric?**
- **EN.** It is constant by construction, not dead. Relevance is a retriever diagnostic, not an answer-quality score, so within a single context source it is identical across reuse policies. I report it as a retrieval diagnostic with that caveat, and I exclude its None values from means. It belongs in the suite as a context-quality signal, not as a discriminator between caching strategies.
- **PT.** É constante por construção, não morto. A relevância é um diagnóstico do recuperador, não uma pontuação de qualidade da resposta, então dentro de uma única fonte de contexto ela é idêntica entre as políticas de reuso. Reporto-a como diagnóstico de recuperação com essa ressalva e excluo seus valores None das médias. Ela pertence ao conjunto como sinal de qualidade de contexto, não como discriminador entre estratégias de cache.

**Q13. Is your time-to-first-token a real TTFT?**
- **EN.** For streaming backends, yes; for non-streaming, no, and I document it. The vLLM and Ollama paths stream, so they report a real time-to-first-token. For non-streaming requests the framework sets time-to-first-token equal to total response time, honestly labeled as unobservable. The 65.7% reduction I report is on the streaming vLLM path, so it is a genuine first-token measurement.
- **PT.** Para backends com streaming, sim; para sem streaming, não, e documento isso. Os caminhos do vLLM e do Ollama fazem streaming, então reportam um tempo até o primeiro token real. Para requisições sem streaming, o framework define o tempo até o primeiro token igual ao tempo total de resposta, honestamente rotulado como não observável. A redução de 65,7% que reporto é no caminho com streaming do vLLM, então é uma medição genuína de primeiro token.

---

## (e) Timed Pitches / Discursos Cronometrados

### 60-Second Pitch / Discurso de 60 Segundos

**EN.** LLM serving forces a choice: retrieve fresh context every turn, or keep a reusable cached context. The community measures the efficiency side and the quality side separately, and the most common comparison quietly confounds caching with the context source. CAGE is a framework that fixes this. It drives a real vLLM server from the outside, varies one axis at a time across a taxonomy of nine baselines plus a 2x2 compression axis, and co-measures serving efficiency and answer faithfulness on the same queries with query-level Wilcoxon statistics. Phase 1 validates the whole pipeline on a single CPU node and already shows prefix caching cutting latency by 37.4% and time-to-first-token by 65.7% at faithfulness parity, while proving retrieval success is not serving efficiency. I am explicit about scope: cross-node transfer is simulated, decoding is stochastic, and compression has no results yet. Those are the designed next phases. The contribution is the measurement methodology itself.

**PT.** O serving de LLMs força uma escolha: recuperar contexto novo a cada turno, ou manter um contexto em cache reutilizável. A comunidade mede o lado da eficiência e o lado da qualidade separadamente, e a comparação mais comum confunde silenciosamente o cache com a fonte de contexto. O CAGE é um framework que corrige isso. Ele conduz um servidor vLLM real por fora, varia um eixo por vez sobre uma taxonomia de nove baselines mais um eixo de compressão 2x2, e co-mede eficiência de serving e fidelidade da resposta nas mesmas consultas com estatística de Wilcoxon em nível de consulta. A Fase 1 valida todo o pipeline em um único nó de CPU e já mostra o prefix caching reduzindo a latência em 37,4% e o tempo até o primeiro token em 65,7% com paridade de fidelidade, provando que sucesso de recuperação não é eficiência de serving. Sou explícito quanto ao escopo: a transferência entre nós é simulada, a decodificação é estocástica e a compressão ainda não tem resultados. Essas são as próximas fases projetadas. A contribuição é a própria metodologia de medição.

### 3-Minute Pitch / Discurso de 3 Minutos

**EN.** Large language model serving is dominated by the KV cache. The prefill builds it, the decode reuses it, and memory pressure grows with context length. To handle long context, practitioners pick between two families: retrieval-augmented generation, which fetches external passages every turn, and cache-augmented generation, which keeps a reusable cached context and avoids per-turn retrieval. The problem is that the field evaluates these one dimension at a time. Serving papers report latency and throughput; retrieval papers report faithfulness and relevance. Almost nobody reads both on the same workload, and worse, the obvious comparison hides a confound: a cached arm usually runs on gold context while a retrieval arm runs on retrieved context, so what looks like a caching effect is often a context-source effect.

CAGE is my answer. It is a measurement framework that instruments a real vLLM server from the outside, without modifying the model or the engine. It defines a controlled taxonomy of nine baselines, from no-cache and prefix-cache through Redis, RAG, distributed, hybrid, speculative, and two compression arms, where each baseline varies exactly one axis, the context source or the reuse policy. On top, it adds a 2x2 compression axis combining prompt-side text compression with server-side FP8 KV compression at a matched operating point. It co-computes a full metric suite on the same queries: LettuceDetect span grounding as the primary quality metric, claim-level NLI faithfulness, BERTScore completeness, and ROUGE, F1, and exact match, alongside latency, time-to-first-token, time-per-output-token, throughput, and cache telemetry. Significance uses per-query paired Wilcoxon tests with Holm-Bonferroni correction and 10,000-iteration bootstrap intervals, with the query as the unit of analysis.

Phase 1 validates this entire pipeline on a single CPU node and delivers a clean, defensible result. Prefix caching cuts latency by 37.4% and time-to-first-token by 65.7% relative to no caching, at faithfulness parity, which is exactly what we expect because prefix caching preserves the output. The cache telemetry separates concerns the field usually blurs: RAG and Redis hit retrieval around 98% but show zero serving cache hit and no first-token benefit, proving retrieval success and serving efficiency are different signals.

I am deliberately honest about what Phase 1 does not show. Everything ran on one CPU node, so absolute latency is non-generalizable and the setup does not induce real memory pressure. Cross-node KV transfer is simulated by an analytical model, and the replicated policy I ran zeroes transfer cost, so the distributed arm measures routing, not transfer. Decoding is stochastic at temperature 0.7 with no sampling seed, so small quality deltas are within noise. The compression axis has no results yet, and the significance script is standalone rather than integrated. These are not flaws hidden from the committee; they are the designed boundary between Phase 1 and the future phases. Phase 2 brings GPU memory pressure with the tracker already wired, and Phase 3 swaps in a real vLLM KV connector to measure true cross-node transfer. The durable contribution is the methodology: a shared, reproducible way to read serving efficiency and retrieval quality together on the same workload.

**PT.** O serving de modelos de linguagem de grande porte é dominado pelo KV cache. O prefill o constrói, o decode o reutiliza, e a pressão de memória cresce com o comprimento do contexto. Para lidar com contexto longo, os profissionais escolhem entre duas famílias: a geração aumentada por recuperação, que busca trechos externos a cada turno, e a geração aumentada por cache, que mantém um contexto em cache reutilizável e evita a recuperação por turno. O problema é que a área avalia isso uma dimensão de cada vez. Artigos de serving reportam latência e throughput; artigos de recuperação reportam fidelidade e relevância. Quase ninguém lê os dois no mesmo workload e, pior, a comparação óbvia esconde um confundimento: um braço com cache normalmente roda sobre contexto gold enquanto um braço de recuperação roda sobre contexto recuperado, então o que parece efeito de cache é muitas vezes efeito da fonte de contexto.

O CAGE é minha resposta. É um framework de medição que instrumenta um servidor vLLM real por fora, sem modificar o modelo ou o motor. Define uma taxonomia controlada de nove baselines, de sem-cache e prefix-cache passando por Redis, RAG, distribuído, híbrido, especulativo e dois braços de compressão, em que cada baseline varia exatamente um eixo, a fonte de contexto ou a política de reuso. Acima disso, adiciona um eixo de compressão 2x2 combinando compressão de texto no prompt com compressão de KV em FP8 no servidor, em um ponto de operação equiparado. Ele co-computa um conjunto completo de métricas nas mesmas consultas: grounding por spans do LettuceDetect como métrica primária de qualidade, fidelidade por NLI em nível de afirmação, completude por BERTScore e ROUGE, F1 e correspondência exata, ao lado de latência, tempo até o primeiro token, tempo por token de saída, throughput e telemetria de cache. A significância usa testes de Wilcoxon pareados por consulta com correção de Holm-Bonferroni e intervalos de bootstrap de 10.000 iterações, com a consulta como unidade de análise.

A Fase 1 valida todo esse pipeline em um único nó de CPU e entrega um resultado limpo e defensável. O prefix caching reduz a latência em 37,4% e o tempo até o primeiro token em 65,7% em relação à ausência de cache, com paridade de fidelidade, exatamente o que esperamos porque o prefix caching preserva a saída. A telemetria de cache separa preocupações que a área costuma misturar: RAG e Redis acertam recuperação em torno de 98% mas mostram zero acerto de cache de serving e nenhum benefício de primeiro token, provando que sucesso de recuperação e eficiência de serving são sinais diferentes.

Sou deliberadamente honesto sobre o que a Fase 1 não mostra. Tudo rodou em um nó de CPU, então a latência absoluta não é generalizável e o setup não induz pressão real de memória. A transferência de KV entre nós é simulada por um modelo analítico, e a política replicada que executei zera o custo de transferência, então o braço distribuído mede roteamento, não transferência. A decodificação é estocástica a temperatura 0,7 sem semente de amostragem, então pequenas diferenças de qualidade estão dentro do ruído. O eixo de compressão ainda não tem resultados, e o script de significância é independente em vez de integrado. Isso não são falhas escondidas da banca; são a fronteira projetada entre a Fase 1 e as fases futuras. A Fase 2 traz pressão de memória em GPU com o rastreador já integrado, e a Fase 3 troca por um conector de KV real do vLLM para medir a transferência verdadeira entre nós. A contribuição durável é a metodologia: uma forma compartilhada e reprodutível de ler eficiência de serving e qualidade de recuperação juntas no mesmo workload.

---

## Defense-Day Checklist / Checklist do Dia da Defesa

**EN.**
- Lead with scope (Phase 1, CPU, simulated transfer) before any number, so no claim is heard as more than it is.
- Never present the 0.636 distributed faithfulness as a win; call it sampling noise if asked.
- When you say "distributed," immediately add "replicated, zero-transfer simulation."
- Frame the faithfulness-versus-retrieval result as a gold-vs-retrieved effect, not a caching effect.
- Keep "grounding is primary; BERTScore is completeness; negative-control framing is my interpretation" ready as one sentence.

**PT.**
- Comece pelo escopo (Fase 1, CPU, transferência simulada) antes de qualquer número, para que nenhuma alegação seja ouvida como mais do que é.
- Nunca apresente a fidelidade distribuída de 0,636 como vitória; chame-a de ruído de amostragem se perguntarem.
- Ao dizer "distribuído", acrescente imediatamente "replicado, simulação de transferência zero".
- Enquadre o resultado fidelidade-versus-recuperação como efeito gold-vs-recuperado, não efeito de cache.
- Mantenha pronta a frase "grounding é primário; BERTScore é completude; o enquadramento de controle negativo é minha interpretação".
