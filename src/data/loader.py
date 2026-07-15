"""
Dataset loaders for CAGE evaluation.

Supports loading and formatting HuggingFace datasets:
- hotpotqa: multi-hop reasoning
- qasper: scientific paper QA
- squad_v2: reading comprehension
- trivia_qa: multi-evidence questions
- humaneval: code generation (HPC Layer 1)
- mbpp: code generation (HPC Layer 1)
- hpc_code: CUDA/OpenMP code generation prompts (HPC Layer 1)
"""

from dataclasses import dataclass
from typing import List, Dict, Any, Optional
import os
import random

# NOTE: `datasets` is imported lazily inside each loader's load() so this module
# (CAGExample, get_loader, registry) stays importable in environments without the
# HuggingFace `datasets` package (e.g. the local analysis venv running unit tests).


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
    """Loader for QASPER dataset (scientific papers)."""
    
    def __init__(self, split: str = "validation", seed: int = 42):
        super().__init__("allenai/qasper", split, seed)
    
    def load(self, max_examples: Optional[int] = None) -> List[CAGExample]:
        """Load QASPER dataset."""
        from datasets import load_dataset  # lazy: see module-level note

        dataset = load_dataset("allenai/qasper", split=self.split)
        
        if max_examples:
            # Seeded shuffle BEFORE select so different seeds (per trial) draw
            # different, reproducible samples — fixes the trial-independence bug
            # where every trial saw the identical first-N examples.
            dataset = dataset.shuffle(seed=self.seed).select(range(min(max_examples, len(dataset))))
        
        examples = []
        for item in dataset:
            # QASPER has paper full text + questions
            paper_text = item.get("full_text", {})
            
            # Extract abstract and intro as context
            context_docs = []
            if "abstract" in paper_text:
                context_docs.append(f"Abstract: {paper_text['abstract']}")
            
            # Process questions
            for qa in item.get("qas", []):
                question = qa.get("question", "")
                # Use first answer if available
                answers = qa.get("answers", [])
                answer_text = answers[0].get("answer", "") if answers else ""
                
                if question and answer_text:
                    examples.append(CAGExample(
                        id=f"{item.get('id', '')}_{qa.get('question_id', len(examples))}",
                        question=question,
                        context=context_docs,
                        answer=answer_text,
                        metadata={
                            "paper_id": item.get("id", ""),
                            "title": item.get("title", ""),
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
            
            examples.append(CAGExample(
                id=item.get("id", str(len(examples))),
                question=item["question"],
                context=[item["context"]],
                answer=answer_text,
                metadata={
                    "title": item.get("title", ""),
                    "is_impossible": item.get("is_impossible", False),
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
    """Factory function to get appropriate dataset loader."""
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
    
    if dataset_name not in loaders:
        raise ValueError(f"Unknown dataset: {dataset_name}. Supported: {list(loaders.keys())}")
    
    return loaders[dataset_name](split=split, seed=seed)
