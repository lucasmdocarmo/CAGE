"""
Dataset loaders for CAGE evaluation.

Supports loading and formatting HuggingFace datasets:
- hotpotqa: multi-hop reasoning
- qasper: scientific paper QA (charter D5 item 4 — full papers, evidence qrels)
- squad_v2: reading comprehension
- trivia_qa: multi-evidence questions
- humaneval: code generation (HPC Layer 1)
- mbpp: code generation (HPC Layer 1)
- hpc_code: CUDA/OpenMP code generation prompts (HPC Layer 1)

Plus two charter D5 instruments living in sibling modules (registered in
``get_loader`` via lazy imports so this module stays importable without them):
- ruler: synthetic RULER-style length instrument (src/data/ruler.py; D5 item 5)
- scbench: SCBench two-subset external-validation slice (src/data/scbench.py;
  D5 item 6)
"""

from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple
import os
import random

# NOTE: `datasets` is imported lazily inside each loader's load() so this module
# (CAGExample, get_loader, registry) stays importable in environments without the
# HuggingFace `datasets` package (e.g. the local analysis venv running unit tests).


class DatasetUnavailableError(RuntimeError):
    """A required dataset failed to load or violated its expected schema.

    Raised INSTEAD of silently returning an empty/partial example list: a
    silently-degraded dataset would let a campaign cell run on the wrong (or no)
    workload under the same cell name, voiding cross-cell comparability. Mirrors
    the fail-closed contract of
    ``src.evaluation.quality.InstrumentUnavailableError``.
    """

    def __init__(self, dataset: str, source: str, cause: str) -> None:
        self.dataset = dataset
        self.source = source
        self.cause = cause
        super().__init__(
            f"Dataset '{dataset}' unavailable from '{source}': {cause}. "
            f"Fail-closed by design — stage it first (see "
            f"scripts/1_setup/download_datasets.py) or fix the named cause; "
            f"no silent fallback workload is substituted."
        )


@dataclass
class CAGExample:
    """Single example for CAG evaluation."""
    
    id: str        
    question: str
    context: List[str]  # Supporting documents/passages
    answer: str
    metadata: Dict[str, Any]
    
    def format_prompt(self, include_context: bool = True) -> str:
        """Format as prompt for LLM inference."""
        if not include_context or not self.context:
            return f"Question: {self.question}\nAnswer:"
        
        context_str = "\n\n".join([f"Context {i+1}: {c}" for i, c in enumerate(self.context)])
        return f"{context_str}\n\nQuestion: {self.question}\nAnswer:"


def gold_only(example: CAGExample) -> List[str]:
    """Filter ``example.context`` down to just its gold paragraph(s).

    Loaders that keep ALL paragraphs (gold + distractors) in ``.context`` --
    HotpotQA, MuSiQue -- record the gold titles in ``metadata["supporting_titles"]``
    and title-prefix every context string as ``"<title>: <text>"``; this returns only
    the entries whose title matches. Loaders without that metadata (SQuAD v2, and
    anything else whose ``.context`` already IS the gold paragraph(s)) are returned
    unchanged.

    Intended as the ``context_selector`` passed to
    ``src.data.manifest.build_manifest`` so shared true-CAG corpus blocks are built
    from gold paragraphs only, not unique-per-question distractor text.
    """
    titles = (example.metadata or {}).get("supporting_titles")
    if not titles:
        return example.context
    return [c for c in (example.context or []) if any(c.startswith(f"{t}: ") for t in titles)]


class DatasetLoader:
    """Base class for dataset loaders."""
    
    def __init__(self, dataset_name: str, split: str = "validation", seed: int = 42):
        self.dataset_name = dataset_name
        self.split = split
        self.seed = seed
        random.seed(seed)
    
    def load(self, max_examples: Optional[int] = None) -> List[CAGExample]:
        """Load and format dataset."""
        raise NotImplementedError
    
    def sample(self, examples: List[CAGExample], n: int) -> List[CAGExample]:
        """Sample n examples randomly."""
        if n >= len(examples):
            return examples
        return random.sample(examples, n)


class HotpotQALoader(DatasetLoader):
    """Loader for HotpotQA multi-hop QA (distractor setting).

    Emits ALL 10 paragraphs (2 gold + 8 distractors) as title-prefixed context
    strings so the retrieval arms have a real selection job; gold paragraphs are
    recoverable via metadata["supporting_titles"] (titles from supporting_facts).
    Every HotpotQA item is answerable (there is no unanswerable half), so — like
    MuSiQue/NQ and unlike SQuAD v2 — the gold answer is always non-empty and no
    is_impossible flag is emitted (SQuAD v2 signals unanswerable via empty answer
    + metadata["is_impossible"]).
    """

    def __init__(self, split: str = "validation", seed: int = 42):
        super().__init__("hotpotqa", split, seed)

    def load(self, max_examples: Optional[int] = None) -> List[CAGExample]:
        """Load HotpotQA (distractor) dataset."""
        from datasets import load_dataset  # lazy: see module-level note

        dataset = load_dataset("hotpot_qa", "distractor", split=self.split)

        if max_examples:
            # Seeded shuffle BEFORE select so different seeds (per trial) draw
            # different, reproducible samples — fixes the trial-independence bug
            # where every trial saw the identical first-N examples.
            dataset = dataset.shuffle(seed=self.seed).select(range(min(max_examples, len(dataset))))

        examples = []
        for item in dataset:
            # context = {"title": [...], "sentences": [[...], ...]} -> one
            # "<title>: <concatenated sentences>" paragraph per title. HotpotQA
            # sentences carry their own leading whitespace, so plain "".join is
            # the canonical concatenation. Keep ALL paragraphs (gold + distractors).
            context_docs = []
            ctx = item.get("context") or {}
            titles = ctx.get("title") if isinstance(ctx, dict) else None
            sentences_list = ctx.get("sentences") if isinstance(ctx, dict) else None
            if titles and sentences_list:
                for title, sentences in zip(titles, sentences_list):
                    doc_text = "".join(sentences) if isinstance(sentences, list) else str(sentences)
                    context_docs.append(f"{title}: {doc_text}")

            # supporting_facts = {"title": [...], "sent_id": [...]}; dedupe the
            # titles (one gold paragraph can contribute several sentences) so
            # corpus/gold selection can find the gold paragraphs by title prefix.
            sf = item.get("supporting_facts") or {}
            sf_titles = sf.get("title") if isinstance(sf, dict) else []
            supporting_titles = list(dict.fromkeys(sf_titles or []))

            examples.append(CAGExample(
                id=str(item.get("id", len(examples))),
                question=item["question"],
                context=context_docs,
                answer=item["answer"],  # always non-empty: all items answerable
                metadata={
                    "dataset": "hotpotqa",
                    "type": item.get("type", "unknown"),
                    "level": item.get("level", "unknown"),
                    "supporting_titles": supporting_titles,
                },
            ))

        return examples


class QasperLoader(DatasetLoader):
    """Loader for QASPER (Dasigi et al., NAACL 2021) — charter D5 item 4.

    Context (FULL paper, never LongBench-style truncation): one title-prefixed
    doc per unit, in paper order — optional ``"Title: <paper title>"`` (only
    when ``include_title=True``; default off preserves the pinned context
    layout), then ``"Abstract: <abstract>"``, then one ``"<section name>:
    <paragraphs joined by newline>"`` doc per full_text section. The real HF
    schema is COLUMNAR and abstract is TOP-LEVEL (crash class fixed 2026-08-04,
    pinned by tests/test_dataset_loaders.py): ``full_text = {"section_name":
    [...], "paragraphs": [[...], ...]}`` (index-aligned lists), ``qas`` a dict
    of parallel lists, and each per-question ``answers`` entry is ``{"answer":
    [one dict per annotator], "annotation_id": [...], "worker_id": [...]}``.

    Answer resolution — deterministic rule (charter D5#4: yes/no + unanswerable
    special-cased; abstention axis preserved):
      1. Each annotator dict is resolved independently with the fixed
         precedence ``unanswerable -> yes_no -> free_form_answer ->
         extractive_spans`` (unanswerable FIRST because a stale/default yes_no
         can co-occur with ``unanswerable=True``; yes/no resolves to the
         literal ``"Yes"``/``"No"``; extractive spans join with ``"; "``;
         an annotator matching none of the four resolves to nothing).
      2. Question-level gold: if EVERY resolvable annotator resolved
         unanswerable, the question is unanswerable — empty ``answer`` +
         ``metadata["is_impossible"]=True`` (same convention as SQuAD v2).
         Otherwise the primary gold is the FIRST annotator (dataset order)
         resolving to a non-empty answer; every annotator's non-empty
         resolution is kept order-preserving-deduplicated (primary first) in
         ``metadata["all_answers"]`` for max-over-golds scoring (same key the
         SQuAD v2 loader emits, consumed by evaluation/quality.py). Questions
         where no annotator resolves to anything are skipped.

    Evidence (qrels-ready, feeds D8 §8.2 Layer-0): ``metadata["evidence"]`` /
    ``metadata["highlighted_evidence"]`` hold the order-preserving-deduplicated
    union of the human gold evidence texts across ALL annotators (exact
    paragraph texts; figure/table refs appear as ``"FLOAT SELECTED: ..."`` and
    are kept verbatim). ``metadata["evidence_doc_ids"]`` maps each evidence
    text to the index of the emitted ``context`` doc containing it verbatim
    (unmatched texts — e.g. floats not in full_text — contribute no id);
    ``metadata["supporting_titles"]`` holds the section names of the matched
    docs so ``gold_only()`` and the corpus/qrels machinery work exactly as for
    HotpotQA/MuSiQue (note: duplicate section names within one paper make the
    title-prefix match over-select; ``evidence_doc_ids`` is the precise form).
    """

    def __init__(self, split: str = "validation", seed: int = 42,
                 include_title: bool = False):
        super().__init__("allenai/qasper", split, seed)
        self.include_title = include_title

    # -- answer resolution -------------------------------------------------

    @staticmethod
    def _resolve_annotator(annotator: Dict[str, Any]) -> Tuple[str, Optional[str]]:
        """Resolve ONE annotator dict -> (answer_text, answer_type).

        Fixed precedence documented in the class docstring; answer_type is one
        of "unanswerable" | "yes_no" | "abstractive" | "extractive" | None
        (None = annotation resolves to nothing and is ignored).
        """
        annotator = annotator or {}
        if annotator.get("unanswerable"):
            return "", "unanswerable"
        if annotator.get("yes_no") is not None:
            return ("Yes" if annotator["yes_no"] else "No"), "yes_no"
        if annotator.get("free_form_answer"):
            return annotator["free_form_answer"], "abstractive"
        if annotator.get("extractive_spans"):
            return "; ".join(annotator["extractive_spans"]), "extractive"
        return "", None

    def load(self, max_examples: Optional[int] = None) -> List[CAGExample]:
        """Load QASPER dataset (one CAGExample per question; ``max_examples``
        bounds PAPERS, each contributing all its questions)."""
        from datasets import load_dataset  # lazy: see module-level note

        dataset = load_dataset("allenai/qasper", split=self.split)

        if max_examples:
            # Seeded shuffle BEFORE select so different seeds (per trial) draw
            # different, reproducible samples — fixes the trial-independence bug
            # where every trial saw the identical first-N examples.
            dataset = dataset.shuffle(seed=self.seed).select(range(min(max_examples, len(dataset))))

        examples = []
        for item in dataset:
            # QASPER full text (charter D5#4: "full papers, never LongBench
            # truncation"). Real schema: item["abstract"] is TOP-LEVEL (not nested
            # under full_text); full_text = {"section_name": [...], "paragraphs":
            # [[...], ...]} -- a list of paragraph strings PER section, aligned by
            # index with section_name.
            context_docs: List[str] = []
            doc_titles: List[str] = []  # parallel: title-prefix of each doc ("" if none)
            title = item.get("title") or ""
            if self.include_title and title:
                context_docs.append(f"Title: {title}")
                doc_titles.append("Title")
            abstract = item.get("abstract") or ""
            if abstract:
                context_docs.append(f"Abstract: {abstract}")
                doc_titles.append("Abstract")

            full_text = item.get("full_text") or {}
            section_names = full_text.get("section_name") or []
            section_paragraphs = full_text.get("paragraphs") or []
            for section_name, paragraphs in zip(section_names, section_paragraphs):
                body = "\n".join(p for p in (paragraphs or []) if p)
                if not body:
                    continue
                context_docs.append(f"{section_name}: {body}" if section_name else body)
                doc_titles.append(section_name or "")

            # qas is COLUMNAR (dict of parallel lists), not a list of per-question
            # records: {"question": [...], "question_id": [...], "answers": [...]}.
            qas = item.get("qas") or {}
            questions = qas.get("question") or []
            question_ids = qas.get("question_id") or []
            answers_list = qas.get("answers") or []
            for i, question in enumerate(questions):
                question_id = question_ids[i] if i < len(question_ids) else str(len(examples))
                answers_struct = answers_list[i] if i < len(answers_list) else {}
                annotator_answers = (answers_struct or {}).get("answer") or []

                # Resolve EVERY annotator (deterministic rule in class docstring).
                resolved: List[Tuple[str, str]] = []  # (text, type), type != None
                for annotator in annotator_answers:
                    text, answer_type = self._resolve_annotator(annotator)
                    if answer_type is not None:
                        resolved.append((text, answer_type))
                if not question or not resolved:
                    continue  # nothing resolvable: skip (pinned pre-existing filter)

                non_empty = [(t, a) for t, a in resolved if t]
                if not non_empty:
                    # every resolvable annotator said unanswerable -> abstention axis
                    answer_text, answer_type, is_impossible = "", "unanswerable", True
                    all_answers: List[str] = []
                else:
                    answer_text, answer_type = non_empty[0]
                    is_impossible = False
                    all_answers = list(dict.fromkeys(t for t, _ in non_empty))

                # Human gold evidence, unioned across ALL annotators (qrels basis).
                evidence = list(dict.fromkeys(
                    e for annotator in annotator_answers
                    for e in ((annotator or {}).get("evidence") or []) if e
                ))
                highlighted = list(dict.fromkeys(
                    h for annotator in annotator_answers
                    for h in ((annotator or {}).get("highlighted_evidence") or []) if h
                ))
                evidence_doc_ids: List[int] = []
                for ev in evidence:
                    for doc_idx, doc in enumerate(context_docs):
                        if ev in doc:
                            if doc_idx not in evidence_doc_ids:
                                evidence_doc_ids.append(doc_idx)
                            break
                supporting_titles = list(dict.fromkeys(
                    doc_titles[doc_idx] for doc_idx in evidence_doc_ids if doc_titles[doc_idx]
                ))

                examples.append(CAGExample(
                    id=f"{item.get('id', '')}_{question_id}",
                    question=question,
                    context=context_docs,
                    answer=answer_text,
                    metadata={
                        "dataset": "qasper",
                        "paper_id": item.get("id", ""),
                        "title": title,
                        # Yes/no + unanswerable scoring special-cased (charter D5#4).
                        "is_impossible": is_impossible,
                        "answer_type": answer_type,
                        "all_answers": all_answers,
                        "num_annotators": len(resolved),
                        "num_unanswerable_annotators": sum(
                            1 for _, a in resolved if a == "unanswerable"),
                        # Qrels-ready gold evidence (D8 §8.2 Layer-0).
                        "evidence": evidence,
                        "highlighted_evidence": highlighted,
                        "evidence_doc_ids": evidence_doc_ids,
                        "supporting_titles": supporting_titles,
                    }
                ))

        return examples


class SquadV2Loader(DatasetLoader):
    """Loader for SQuAD v2 dataset."""
    
    def __init__(self, split: str = "validation", seed: int = 42):
        super().__init__("squad_v2", split, seed)
    
    def load(self, max_examples: Optional[int] = None) -> List[CAGExample]:
        """Load SQuAD v2 dataset."""
        from datasets import load_dataset  # lazy: see module-level note

        dataset = load_dataset("squad_v2", split=self.split)
        
        if max_examples:
            # Seeded shuffle BEFORE select so different seeds (per trial) draw
            # different, reproducible samples — fixes the trial-independence bug
            # where every trial saw the identical first-N examples.
            dataset = dataset.shuffle(seed=self.seed).select(range(min(max_examples, len(dataset))))
        
        examples = []
        for item in dataset:
            # SQuAD v2 has context paragraph + question + answers
            answers = item.get("answers", {})
            answer_text = answers.get("text", [""])[0] if answers.get("text") else ""
            # ALL gold answers, deduplicated order-preserving (audit 2026-07-16 M5):
            # official SQuAD v2 F1/EM take the MAX over every gold answer; keeping only
            # text[0] understated answerable F1 ~5pp / EM ~10pp. Empty list = unanswerable
            # (official SQuAD semantics). Consumed by evaluation/quality.py.
            all_answers = list(dict.fromkeys(t for t in (answers.get("text") or []) if t))
            # DERIVED, not read from the schema: the real squad_v2 HF payload has no
            # "is_impossible" field (only id/title/context/question/answers), so
            # item.get("is_impossible", False) was silently always False. Official
            # SQuAD v2 semantics: an empty gold-answer list means unanswerable.
            is_impossible = len(all_answers) == 0

            examples.append(CAGExample(
                id=item.get("id", str(len(examples))),
                question=item["question"],
                context=[item["context"]],
                answer=answer_text,
                metadata={
                    "title": item.get("title", ""),
                    "is_impossible": is_impossible,
                    "all_answers": all_answers,
                }
            ))

        return examples


class TriviaQALoader(DatasetLoader):
    """Loader for TriviaQA dataset."""
    
    def __init__(self, split: str = "validation", seed: int = 42):
        super().__init__("trivia_qa", split, seed)
    
    def load(self, max_examples: Optional[int] = None) -> List[CAGExample]:
        """Load TriviaQA dataset."""
        from datasets import load_dataset  # lazy: see module-level note

        dataset = load_dataset("trivia_qa", "rc", split=self.split)
        
        if max_examples:
            # Seeded shuffle BEFORE select so different seeds (per trial) draw
            # different, reproducible samples — fixes the trial-independence bug
            # where every trial saw the identical first-N examples.
            dataset = dataset.shuffle(seed=self.seed).select(range(min(max_examples, len(dataset))))
        
        examples = []
        for item in dataset:
            # TriviaQA provides question + answer + supporting facts
            answer = item.get("answer", {})
            answer_text = answer.get("value", "") if isinstance(answer, dict) else str(answer)
            
            # Get entity pages as context
            entity_pages = item.get("entity_pages", {})
            context_docs = []
            if entity_pages:
                for title, content in zip(
                    entity_pages.get("title", []),
                    entity_pages.get("wiki_context", [])
                ):
                    context_docs.append(f"{title}: {content}")
            
            examples.append(CAGExample(
                id=item.get("question_id", str(len(examples))),
                question=item["question"],
                context=context_docs,
                answer=answer_text,
                metadata={
                    "question_source": item.get("question_source", ""),
                }
            ))
        
        return examples


class HumanEvalLoader(DatasetLoader):
    """Loader for HumanEval code generation benchmark.
    
    HumanEval tests functional correctness of code generation.
    Each problem has a function signature, docstring, and test cases.
    """
    
    def __init__(self, split: str = "test", seed: int = 42):
        super().__init__("openai_humaneval", split, seed)
    
    def load(self, max_examples: Optional[int] = None) -> List[CAGExample]:
        """Load HumanEval dataset."""
        from datasets import load_dataset  # lazy: see module-level note

        dataset = load_dataset("openai_humaneval", split=self.split)
        
        if max_examples:
            # Seeded shuffle BEFORE select so different seeds (per trial) draw
            # different, reproducible samples — fixes the trial-independence bug
            # where every trial saw the identical first-N examples.
            dataset = dataset.shuffle(seed=self.seed).select(range(min(max_examples, len(dataset))))
        
        examples = []
        for item in dataset:
            # HumanEval provides function signature + docstring as prompt
            prompt = item.get("prompt", "")
            canonical_solution = item.get("canonical_solution", "")
            test_code = item.get("test", "")
            entry_point = item.get("entry_point", "")
            
            # Context is the function signature and docstring
            # Answer is the canonical solution
            examples.append(CAGExample(
                id=item.get("task_id", str(len(examples))),
                question=f"Complete the following Python function:\n\n{prompt}",
                context=[prompt],  # The prompt itself serves as context
                answer=canonical_solution,
                metadata={
                    "task_id": item.get("task_id", ""),
                    "entry_point": entry_point,
                    "test_code": test_code,
                    "dataset_type": "code_generation",
                }
            ))
        
        return examples


class MBPPLoader(DatasetLoader):
    """Loader for MBPP (Mostly Basic Python Problems) benchmark.
    
    MBPP contains 974 programming problems designed to be solvable by
    entry-level programmers.
    """
    
    def __init__(self, split: str = "test", seed: int = 42):
        super().__init__("mbpp", split, seed)
    
    def load(self, max_examples: Optional[int] = None) -> List[CAGExample]:
        """Load MBPP dataset."""
        from datasets import load_dataset  # lazy: see module-level note

        dataset = load_dataset("mbpp", split=self.split)
        
        if max_examples:
            # Seeded shuffle BEFORE select so different seeds (per trial) draw
            # different, reproducible samples — fixes the trial-independence bug
            # where every trial saw the identical first-N examples.
            dataset = dataset.shuffle(seed=self.seed).select(range(min(max_examples, len(dataset))))
        
        examples = []
        for item in dataset:
            task_description = item.get("text", "")
            code_solution = item.get("code", "")
            test_list = item.get("test_list", [])
            
            # Format test cases as context
            test_context = "\n".join(test_list) if test_list else ""
            
            examples.append(CAGExample(
                id=str(item.get("task_id", len(examples))),
                question=f"Write a Python function to solve:\n{task_description}",
                context=[f"Test cases:\n{test_context}"] if test_context else [],
                answer=code_solution,
                metadata={
                    "task_id": item.get("task_id", ""),
                    "test_list": test_list,
                    "dataset_type": "code_generation",
                }
            ))
        
        return examples


class HPCCodeLoader(DatasetLoader):
    """Loader for HPC-specific code generation tasks.
    
    Provides prompts for:
    - CUDA kernel generation
    - OpenMP parallelization
    - MPI communication patterns
    - Scientific computing code porting
    
    This is a synthetic dataset for HPC workload characterization (Layer 1).
    """
    
    # HPC code generation prompts
    HPC_PROMPTS = [
        # CUDA kernels
        {
            "id": "cuda_vector_add",
            "question": "Write a CUDA kernel to perform element-wise vector addition of two arrays.",
            "context": [
                "CUDA kernels use __global__ qualifier.",
                "Use threadIdx.x and blockIdx.x for indexing.",
                "Ensure bounds checking for array access."
            ],
            "answer": '''__global__ void vectorAdd(float *a, float *b, float *c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}''',
            "category": "cuda",
        },
        {
            "id": "cuda_matrix_mul",
            "question": "Write a CUDA kernel for matrix multiplication C = A * B using shared memory tiling.",
            "context": [
                "Use __shared__ memory for tile-based computation.",
                "Typical tile size is 16x16 or 32x32.",
                "Synchronize threads with __syncthreads()."
            ],
            "answer": '''#define TILE_SIZE 16

__global__ void matMul(float *A, float *B, float *C, int N) {
    __shared__ float tileA[TILE_SIZE][TILE_SIZE];
    __shared__ float tileB[TILE_SIZE][TILE_SIZE];
    
    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;
    float sum = 0.0f;
    
    for (int t = 0; t < (N + TILE_SIZE - 1) / TILE_SIZE; t++) {
        if (row < N && t * TILE_SIZE + threadIdx.x < N)
            tileA[threadIdx.y][threadIdx.x] = A[row * N + t * TILE_SIZE + threadIdx.x];
        else
            tileA[threadIdx.y][threadIdx.x] = 0.0f;
            
        if (col < N && t * TILE_SIZE + threadIdx.y < N)
            tileB[threadIdx.y][threadIdx.x] = B[(t * TILE_SIZE + threadIdx.y) * N + col];
        else
            tileB[threadIdx.y][threadIdx.x] = 0.0f;
            
        __syncthreads();
        
        for (int k = 0; k < TILE_SIZE; k++)
            sum += tileA[threadIdx.y][k] * tileB[k][threadIdx.x];
            
        __syncthreads();
    }
    
    if (row < N && col < N)
        C[row * N + col] = sum;
}''',
            "category": "cuda",
        },
        {
            "id": "cuda_reduction",
            "question": "Write a CUDA kernel for parallel sum reduction of an array.",
            "context": [
                "Use shared memory for block-level reduction.",
                "Apply sequential addressing to avoid bank conflicts.",
                "Handle arrays of arbitrary size."
            ],
            "answer": '''__global__ void reduce(float *input, float *output, int n) {
    extern __shared__ float sdata[];
    
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    sdata[tid] = (i < n) ? input[i] : 0.0f;
    __syncthreads();
    
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    
    if (tid == 0) output[blockIdx.x] = sdata[0];
}''',
            "category": "cuda",
        },
        # OpenMP parallelization
        {
            "id": "openmp_parallel_for",
            "question": "Convert this serial loop to use OpenMP parallel for with reduction:\n\nfloat sum = 0.0f;\nfor (int i = 0; i < n; i++) {\n    sum += arr[i];\n}",
            "context": [
                "Use #pragma omp parallel for.",
                "Use reduction clause for sum operations.",
                "Consider scheduling options for load balancing."
            ],
            "answer": '''float sum = 0.0f;
#pragma omp parallel for reduction(+:sum)
for (int i = 0; i < n; i++) {
    sum += arr[i];
}''',
            "category": "openmp",
        },
        {
            "id": "openmp_matrix_mul",
            "question": "Parallelize this matrix multiplication using OpenMP with proper loop ordering for cache efficiency.",
            "context": [
                "Use collapse clause for nested loops.",
                "Consider loop interchange for better cache performance.",
                "Use schedule(static) or schedule(dynamic) based on workload."
            ],
            "answer": '''#pragma omp parallel for collapse(2) schedule(static)
for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
        float sum = 0.0f;
        for (int k = 0; k < N; k++) {
            sum += A[i * N + k] * B[k * N + j];
        }
        C[i * N + j] = sum;
    }
}''',
            "category": "openmp",
        },
        {
            "id": "openmp_sections",
            "question": "Use OpenMP sections to parallelize independent tasks A, B, and C.",
            "context": [
                "Use #pragma omp parallel sections.",
                "Each section runs in parallel.",
                "Sections are useful for task parallelism."
            ],
            "answer": '''#pragma omp parallel sections
{
    #pragma omp section
    {
        taskA();
    }
    #pragma omp section
    {
        taskB();
    }
    #pragma omp section
    {
        taskC();
    }
}''',
            "category": "openmp",
        },
        # Scientific computing
        {
            "id": "stencil_jacobi",
            "question": "Implement a 2D Jacobi stencil iteration for solving Laplace equation using OpenMP.",
            "context": [
                "Jacobi iteration: u_new[i][j] = 0.25 * (u[i-1][j] + u[i+1][j] + u[i][j-1] + u[i][j+1]).",
                "Use double buffering to avoid race conditions.",
                "Parallelize the outer loop."
            ],
            "answer": '''void jacobi_iteration(float **u, float **u_new, int N) {
    #pragma omp parallel for collapse(2)
    for (int i = 1; i < N - 1; i++) {
        for (int j = 1; j < N - 1; j++) {
            u_new[i][j] = 0.25f * (u[i-1][j] + u[i+1][j] + 
                                   u[i][j-1] + u[i][j+1]);
        }
    }
    
    // Swap pointers
    float **temp = u;
    u = u_new;
    u_new = temp;
}''',
            "category": "scientific",
        },
        {
            "id": "fft_cuda",
            "question": "Write CUDA code to perform a simple radix-2 FFT butterfly operation.",
            "context": [
                "FFT butterfly: X[k] = E[k] + W * O[k], X[k+N/2] = E[k] - W * O[k].",
                "W is the twiddle factor: exp(-2*pi*i*k/N).",
                "Use cuComplex for complex arithmetic."
            ],
            "answer": '''__device__ cuFloatComplex butterfly(cuFloatComplex a, cuFloatComplex b, 
                                         cuFloatComplex w) {
    cuFloatComplex wb = cuCmulf(w, b);
    return cuCaddf(a, wb);
}

__global__ void fft_butterfly(cuFloatComplex *data, int N, int step) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int k = idx % (N / 2);
    int block = idx / (N / 2);
    
    float angle = -2.0f * M_PI * k / N;
    cuFloatComplex w = make_cuFloatComplex(cosf(angle), sinf(angle));
    
    int i1 = block * N + k;
    int i2 = i1 + N / 2;
    
    cuFloatComplex t1 = data[i1];
    cuFloatComplex t2 = data[i2];
    
    data[i1] = cuCaddf(t1, cuCmulf(w, t2));
    data[i2] = cuCsubf(t1, cuCmulf(w, t2));
}''',
            "category": "scientific",
        },
        # Code porting
        {
            "id": "port_serial_to_openmp",
            "question": "Port this serial N-body simulation loop to OpenMP:\n\nfor (int i = 0; i < n; i++) {\n    for (int j = 0; j < n; j++) {\n        if (i != j) {\n            float dx = pos[j].x - pos[i].x;\n            float dy = pos[j].y - pos[i].y;\n            float dist = sqrt(dx*dx + dy*dy + eps);\n            float f = mass[j] / (dist * dist * dist);\n            acc[i].x += f * dx;\n            acc[i].y += f * dy;\n        }\n    }\n}",
            "context": [
                "Each particle's acceleration can be computed independently.",
                "Inner loop has no loop-carried dependencies for acc[i].",
                "Use schedule(dynamic) for load balancing."
            ],
            "answer": '''#pragma omp parallel for schedule(dynamic)
for (int i = 0; i < n; i++) {
    float ax = 0.0f, ay = 0.0f;
    for (int j = 0; j < n; j++) {
        if (i != j) {
            float dx = pos[j].x - pos[i].x;
            float dy = pos[j].y - pos[i].y;
            float dist = sqrt(dx*dx + dy*dy + eps);
            float f = mass[j] / (dist * dist * dist);
            ax += f * dx;
            ay += f * dy;
        }
    }
    acc[i].x = ax;
    acc[i].y = ay;
}''',
            "category": "porting",
        },
        {
            "id": "port_numpy_to_cuda",
            "question": "Convert this NumPy operation to a CUDA kernel: result = np.exp(a) + np.sin(b)",
            "context": [
                "Use CUDA math functions: expf(), sinf().",
                "One thread per element.",
                "Arrays a and b have the same length n."
            ],
            "answer": '''__global__ void numpy_to_cuda(float *a, float *b, float *result, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        result[idx] = expf(a[idx]) + sinf(b[idx]);
    }
}

// Launch configuration
int blockSize = 256;
int numBlocks = (n + blockSize - 1) / blockSize;
numpy_to_cuda<<<numBlocks, blockSize>>>(d_a, d_b, d_result, n);''',
            "category": "porting",
        },
    ]
    
    def __init__(self, split: str = "test", seed: int = 42):
        super().__init__("hpc_code", split, seed)
    
    def load(self, max_examples: Optional[int] = None) -> List[CAGExample]:
        """Load HPC code generation prompts."""
        prompts = self.HPC_PROMPTS.copy()
        random.shuffle(prompts)
        
        if max_examples:
            prompts = prompts[:max_examples]
        
        examples = []
        for prompt in prompts:
            examples.append(CAGExample(
                id=prompt["id"],
                question=prompt["question"],
                context=prompt["context"],
                answer=prompt["answer"],
                metadata={
                    "category": prompt["category"],
                    "dataset_type": "hpc_code_generation",
                }
            ))
        
        return examples
    
    @classmethod
    def get_prompts_by_category(cls, category: str) -> List[Dict[str, Any]]:
        """Get prompts filtered by category (cuda, openmp, scientific, porting)."""
        return [p for p in cls.HPC_PROMPTS if p["category"] == category]


class NaturalQuestionsLoader(DatasetLoader):
    """Loader for Natural Questions (open) — used by LongLLMLingua/RECOMP for RAG comparability."""

    def __init__(self, split: str = "validation", seed: int = 42):
        super().__init__("nq_open", split, seed)

    def load(self, max_examples: Optional[int] = None) -> List[CAGExample]:
        from datasets import load_dataset  # lazy: see module-level note

        # nq_open has question + short answers; no gold passage shipped, so context is
        # left empty and the retrieval path supplies documents (fair RAG setup).
        dataset = load_dataset("nq_open", split=self.split)
        if max_examples:
            dataset = dataset.shuffle(seed=self.seed).select(range(min(max_examples, len(dataset))))
        examples = []
        for item in dataset:
            answers = item.get("answer") or []
            examples.append(CAGExample(
                id=str(len(examples)),
                question=item["question"],
                context=[],  # open-domain: retrieval supplies context
                answer=answers[0] if answers else "",
                metadata={"all_answers": answers, "dataset": "nq_open"},
            ))
        return examples


class MuSiQueLoader(DatasetLoader):
    """Loader for MuSiQue multi-hop QA (used by CompAct/long-context compression work)."""

    def __init__(self, split: str = "validation", seed: int = 42):
        super().__init__("musique", split, seed)

    def load(self, max_examples: Optional[int] = None) -> List[CAGExample]:
        from datasets import load_dataset  # lazy: see module-level note

        # dgslibisey/MuSiQue mirrors the answerable split with paragraphs + question + answer,
        # so — like HotpotQA/NQ and unlike SQuAD v2 — the gold answer is always non-empty and
        # no is_impossible flag is emitted.
        dataset = load_dataset("dgslibisey/MuSiQue", split=self.split)
        if max_examples:
            # Seeded shuffle BEFORE select so different seeds (per trial) draw
            # different, reproducible samples — fixes the trial-independence bug
            # where every trial saw the identical first-N examples.
            dataset = dataset.shuffle(seed=self.seed).select(range(min(max_examples, len(dataset))))
        examples = []
        for item in dataset:
            # paragraphs = [{"idx", "title", "paragraph_text", "is_supporting"}, ...] ->
            # title-prefixed paragraph strings (same convention as HotpotQA/TriviaQA).
            # Keep ALL paragraphs (gold + distractors); gold paragraphs are recoverable
            # via metadata["supporting_titles"] (is_supporting=True).
            paragraphs = item.get("paragraphs") or []
            contexts: List[str] = []
            supporting_titles: List[str] = []
            for p in paragraphs:
                if isinstance(p, dict):
                    text = p.get("paragraph_text", "") or ""
                    title = p.get("title", "") or ""
                    doc = f"{title}: {text}" if title else text
                    if doc:
                        contexts.append(doc)
                    if p.get("is_supporting") and title:
                        supporting_titles.append(title)
                elif p:
                    contexts.append(str(p))
            decomposition = item.get("question_decomposition") or []
            examples.append(CAGExample(
                id=str(item.get("id", len(examples))),
                question=item.get("question", ""),
                context=contexts,
                answer=item.get("answer", ""),  # answerable split: gold always populated
                metadata={
                    "dataset": "musique",
                    # Hop COUNT (was: the raw question_decomposition list stored
                    # under a count-named key).
                    "num_hops": len(decomposition) if isinstance(decomposition, list) else None,
                    "supporting_titles": list(dict.fromkeys(supporting_titles)),
                },
            ))
        return examples


class CRAGLoader(DatasetLoader):
    """Loader for CRAG (Comprehensive RAG Benchmark, Meta / KDD Cup 2024).

    CRAG pairs a natural-language ``query`` with a gold ``answer`` and a set of retrieved
    web ``search_results`` (the candidate context a RAG system must ground on), spanning
    multiple domains and question types (simple, conditional, comparison, aggregation,
    multi-hop, false-premise). This makes it a strong RAG-fairness + retrieval-quality
    dataset for CAGE's rag / compressed_rag arms.

    The HF distribution path is NOT fixed across mirrors, so it is configurable via the
    ``hf_path`` argument or the ``CAGE_CRAG_HF_PATH`` env var. Field mapping is defensive
    (query/question, answer, search_results/contexts). Run a 5-query smoke test to validate
    the exact schema of your chosen mirror before a full run.
    """

    def __init__(self, split: str = "validation", seed: int = 42, hf_path: Optional[str] = None):
        self.hf_path = hf_path or os.getenv("CAGE_CRAG_HF_PATH", "crag")
        super().__init__(self.hf_path, split, seed)

    def load(self, max_examples: Optional[int] = None) -> List[CAGExample]:
        from datasets import load_dataset  # lazy: see module-level note

        dataset = load_dataset(self.hf_path, split=self.split)
        if max_examples:
            dataset = dataset.shuffle(seed=self.seed).select(range(min(max_examples, len(dataset))))

        examples = []
        for item in dataset:
            question = item.get("query") or item.get("question") or ""
            answer = item.get("answer") or item.get("gold_answer") or ""
            # search_results may be a list of dicts (page_snippet/page_result/text) or strings.
            raw = item.get("search_results") or item.get("contexts") or item.get("context") or []
            context_docs = []
            if isinstance(raw, list):
                for r in raw:
                    if isinstance(r, dict):
                        txt = (r.get("page_snippet") or r.get("page_result")
                               or r.get("text") or r.get("snippet") or "")
                    else:
                        txt = str(r)
                    if txt:
                        context_docs.append(txt)
            elif isinstance(raw, str) and raw:
                context_docs = [raw]

            if not question:
                continue
            examples.append(CAGExample(
                id=str(item.get("interaction_id", item.get("id", len(examples)))),
                question=question,
                context=context_docs,
                answer=answer,
                metadata={
                    "dataset": "crag",
                    "question_type": item.get("question_type", ""),
                    "static_or_dynamic": item.get("static_or_dynamic", ""),
                    "domain": item.get("domain", ""),
                },
            ))
        return examples


class ShareGPTLoader(DatasetLoader):
    """Loader for ShareGPT conversations as a realistic SERVING-WORKLOAD trace.

    ShareGPT is a corpus of real user<->assistant conversations with highly variable prompt
    lengths and turn counts. It has NO extractive gold answer, so CAGE uses it as a
    serving-pressure / workload-shape trace (TTFT / TPOT / throughput / KV behaviour under
    realistic, heterogeneous prompts), NOT as a QA quality benchmark. The first assistant
    turn is kept as a REFERENCE response (similarity signal only), never as extractive gold;
    quality metrics on this dataset are therefore diagnostic, not primary.

    HF path is configurable (``hf_path`` / ``CAGE_SHAREGPT_HF_PATH``); the default is the
    52K-conversation mirror. Validate with a 5-query smoke test before a full run.
    """

    def __init__(self, split: str = "train", seed: int = 42, hf_path: Optional[str] = None):
        self.hf_path = hf_path or os.getenv("CAGE_SHAREGPT_HF_PATH", "RyokoAI/ShareGPT52K")
        super().__init__(self.hf_path, split, seed)

    @staticmethod
    def _role(turn) -> str:
        return (turn.get("from") or turn.get("role") or "") if isinstance(turn, dict) else ""

    @staticmethod
    def _text(turn) -> str:
        return (turn.get("value") or turn.get("content") or "") if isinstance(turn, dict) else str(turn)

    def load(self, max_examples: Optional[int] = None) -> List[CAGExample]:
        from datasets import load_dataset  # lazy: see module-level note

        dataset = load_dataset(self.hf_path, split=self.split)
        if max_examples:
            dataset = dataset.shuffle(seed=self.seed).select(range(min(max_examples, len(dataset))))

        examples = []
        for item in dataset:
            convo = item.get("conversations") or item.get("conversation") or item.get("items") or []
            if not isinstance(convo, list) or not convo:
                continue
            # First human/user turn is the question; first assistant/gpt turn is the reference.
            question, reference = "", ""
            for turn in convo:
                role = self._role(turn).lower()
                if not question and role in {"human", "user"}:
                    question = self._text(turn)
                elif question and not reference and role in {"gpt", "assistant"}:
                    reference = self._text(turn)
                    break
            if not question:
                # Some dumps open with a system/gpt turn; fall back to the first turn's text.
                question = self._text(convo[0])
            if not question:
                continue
            examples.append(CAGExample(
                id=str(item.get("id", len(examples))),
                question=question,
                context=[],  # open conversation: no supplied gold context
                answer=reference,  # reference response (similarity signal only, NOT gold)
                metadata={
                    "dataset": "sharegpt",
                    "dataset_type": "conversation_trace",
                    "no_gold_answer": True,
                    "num_turns": len(convo),
                },
            ))
        return examples


def get_loader(dataset_name: str, split: str = "validation", seed: int = 42) -> DatasetLoader:
    """Factory function to get appropriate dataset loader.

    "ruler" (synthetic length instrument) and "scbench" (external-validation
    slice) live in sibling modules imported lazily so this module keeps zero
    non-stdlib imports; both are configured via env vars documented in their
    classes (e.g. CAGE_RULER_CONTEXT_TOKENS, CAGE_SCBENCH_SUBSET). NOTE:
    microsoft/SCBench publishes a "test" split — pass split="test" (or set
    CAGE_SCBENCH_SPLIT) for scbench; the loader fails closed otherwise.
    """
    loaders = {
        "hotpotqa": HotpotQALoader,
        "qasper": QasperLoader,
        "squad_v2": SquadV2Loader,
        "trivia_qa": TriviaQALoader,
        "natural_questions": NaturalQuestionsLoader,
        "musique": MuSiQueLoader,
        "crag": CRAGLoader,
        "sharegpt": ShareGPTLoader,
        "humaneval": HumanEvalLoader,
        "mbpp": MBPPLoader,
        "hpc_code": HPCCodeLoader,
    }

    if dataset_name == "ruler":
        from src.data.ruler import RulerLoader  # lazy: see docstring
        return RulerLoader(split=split, seed=seed)
    if dataset_name == "scbench":
        from src.data.scbench import SCBenchLoader  # lazy: see docstring
        return SCBenchLoader(split=split, seed=seed)

    if dataset_name not in loaders:
        supported = list(loaders.keys()) + ["ruler", "scbench"]
        raise ValueError(f"Unknown dataset: {dataset_name}. Supported: {supported}")

    return loaders[dataset_name](split=split, seed=seed)
