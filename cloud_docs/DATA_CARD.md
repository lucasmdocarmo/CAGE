# CAGE Framework — Data Card

**Version:** 2.0
**Last Updated:** 2026-04-08
**Author:** Lucas Mariano do Carmo

---

## 1. Overview

This Data Card documents all datasets used or supported in the CAGE framework, following best practices for ML dataset documentation.

## 2. Primary Dataset (Phase 1)

### SQuAD v2 (Stanford Question Answering Dataset)

**Role in CAGE:** Primary benchmark for Phase 1 experiments. All 7 baselines were evaluated on this dataset.

- **Source:** HuggingFace Hub `rajpurkar/squad_v2`
- **Split Used:** Validation (11,873 examples total; 50 sampled per trial with seed=42)
- **License:** CC BY-SA 4.0
- **Domain:** Wikipedia articles (English)

**Schema in CAGE:**
```python
CAGExample(
    id="56be85543aeaaa14008c9063",
    question="When did Beyonce start becoming popular?",
    context=["Beyoncé Giselle Knowles-Carter..."],  # Gold passage
    answer="in the late 1990s",
    metadata={"title": "Beyoncé", "dataset": "squad_v2"}
)
```

**How it's used:**
- Gold-context baselines (No Cache, Prefix Cache, Distributed): use `example.context[0]` directly
- Retrieval baselines (RAG, Redis, Hybrid): ignore gold passage, retrieve via FAISS index instead

**Citation:**
```bibtex
@inproceedings{rajpurkar2018know,
  title={Know What You Don't Know: Unanswerable Questions for SQuAD},
  author={Rajpurkar, Pranav and Jia, Robin and Liang, Percy},
  booktitle={ACL},
  year={2018}
}
```

## 3. Supported Additional Datasets

### HotpotQA
- **Source:** HuggingFace Hub `hotpot_qa` (distractor subset)
- **Task:** Multi-hop reasoning across multiple documents
- **Config:** `configs/dataset/hotpotqa.yaml`
- **License:** CC BY-SA 4.0
- **Status:** Loader implemented (`src/data/loader.py`), FAISS index built (`experiments/ir_index/`)

### TriviaQA
- **Source:** HuggingFace Hub `trivia_qa` (rc subset)
- **Task:** Multi-evidence QA from web documents
- **License:** Apache 2.0
- **Status:** Loader implemented, FAISS index built

### QASPER
- **Source:** HuggingFace Hub `allenai/qasper`
- **Task:** Scientific paper QA (long-context)
- **License:** CC BY 4.0
- **Status:** Loader implemented

### HumanEval / MBPP
- **Source:** `openai_humaneval`, `mbpp`
- **Task:** Code generation (Python)
- **Status:** Loaders implemented, not used in Phase 1

## 4. FAISS Index Details

FAISS indexes are pre-built and persisted under `experiments/ir_index/`:

| Index | Embedding Model | Dataset |
|---|---|---|
| `ir_squad_v2_intfloat_e5-large-v2/` | intfloat/e5-large-v2 | SQuAD v2 |
| `ir_squad_v2_all-MiniLM-L6-v2/` | all-MiniLM-L6-v2 | SQuAD v2 |
| `ir_trivia_qa_intfloat_e5-large-v2/` | intfloat/e5-large-v2 | TriviaQA |
| `ir_trivia_qa_all-MiniLM-L6-v2/` | all-MiniLM-L6-v2 | TriviaQA |

Each index directory contains: `faiss.index`, `documents.jsonl`, `meta.json`.

## 5. Loading Datasets

```python
from src.data.loader import get_loader

loader = get_loader("squad_v2", split="validation", seed=42)
examples = loader.load(max_examples=50)
# Returns List[CAGExample]
```

## 6. Downloading Datasets

```bash
python scripts/download_datasets.py
# Downloads to ~/.cache/huggingface/datasets/
```

## 7. Ethical Considerations

- All datasets are publicly available under open licenses
- QA datasets are English-only (known bias)
- No personal information is present in the evaluation data
- CAGE does not train models — datasets are used exclusively for evaluation
