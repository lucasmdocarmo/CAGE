"""
Quality metrics for CAGE evaluation.

Metrics:
- Faithfulness: NLI-based entailment score
- Relevance: Embedding similarity  
- Completeness: BERTScore and ROUGE
- F1-score: Token-level precision/recall (QA standard metric)
- Cache Relevance: Proportion of cache blocks that contributed to the answer
"""

from dataclasses import dataclass
from typing import List, Dict, Any, Optional
import os
import numpy as np
import warnings

# Suppress BERTScore warning about empty candidates
warnings.filterwarnings("ignore", message=".*Empty candidate sentence detected.*")


@dataclass
class CacheRelevanceMetrics:
    """Cache relevance evaluation results."""
    
    cache_relevance: float  # 0-1, proportion of cache blocks that contributed
    relevant_block_count: int  # Number of blocks that contributed
    total_block_count: int  # Total number of cache blocks accessed
    per_block_scores: List[float]  # Relevance score for each block
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "cache_relevance": self.cache_relevance,
            "relevant_block_count": self.relevant_block_count,
            "total_block_count": self.total_block_count,
            "per_block_scores": self.per_block_scores,
        }


@dataclass
class QualityMetrics:
    """Quality evaluation results."""
    
    faithfulness: float  # 0-1, NLI entailment score
    relevance: float  # 0-1, embedding similarity
    completeness_bertscore: Optional[float]  # 0-1, BERTScore F1
    completeness_rouge_l: Optional[float]  # 0-1, ROUGE-L F1
    f1_score: float = 0.0  # 0-1, token-level F1 (QA standard metric)
    precision: float = 0.0  # 0-1, token-level precision
    recall: float = 0.0  # 0-1, token-level recall
    exact_match: float = 0.0  # 0 or 1, exact string match
    cache_relevance: Optional[float] = None  # 0-1, proportion of useful cache blocks
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        result = {
            "faithfulness": self.faithfulness,
            "relevance": self.relevance,
            "completeness_bertscore": self.completeness_bertscore,
            "completeness_rouge_l": self.completeness_rouge_l,
            "f1_score": self.f1_score,
            "precision": self.precision,
            "recall": self.recall,
            "exact_match": self.exact_match,
        }
        if self.cache_relevance is not None:
            result["cache_relevance"] = self.cache_relevance
        return result


class QualityEvaluator:
    """Evaluates quality of generated responses."""

    def __init__(
        self,
        use_nli: bool = True,
        use_embeddings: bool = True,
        use_bertscore: bool = True,
        use_rouge: bool = True,
        device: str | int = "cpu",
        nli_model_name: Optional[str] = None,
        embedding_model_name: Optional[str] = None,
        bertscore_model_name: Optional[str] = None,
    ):
        self.use_nli = use_nli
        self.use_embeddings = use_embeddings
        self.use_bertscore = use_bertscore
        self.use_rouge = use_rouge
        self.device = device

        # Allow override via env vars or constructor args.
        self.nli_model_name = (
            nli_model_name
            or os.getenv("CAGE_NLI_MODEL", "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli")
        )
        self.nli_model_fallbacks = [
            name.strip()
            for name in os.getenv("CAGE_NLI_FALLBACKS", "facebook/bart-large-mnli").split(",")
            if name.strip()
        ]
        self.embedding_model_name = (
            embedding_model_name
            or os.getenv("CAGE_EMBEDDING_MODEL", "all-MiniLM-L6-v2")
        )
        self.bertscore_model_name = (
            bertscore_model_name
            or os.getenv("CAGE_BERTSCORE_MODEL", "roberta-base")
        )
        self.bertscore_model_fallbacks = [
            name.strip()
            for name in os.getenv(
                "CAGE_BERTSCORE_FALLBACKS",
                "distilbert-base-uncased,distilroberta-base,microsoft/deberta-base-mnli",
            ).split(",")
            if name.strip()
        ]
        
        # Lazy loading of models
        self._nli_model = None
        self._embedding_model = None
        self._bertscore_model = None
        self._bertscore_model_active_name = None
        self._rouge_scorer = None
        self._bertscore_disabled_reason = None
    
    def _hf_pipeline_device(self) -> int:
        """Convert device setting to a value compatible with transformers.pipeline."""
        if isinstance(self.device, int):
            return self.device

        d = str(self.device).lower()
        if d in {"cpu", "mps"}:
            return -1

        if d.startswith("cuda"):
            # cuda or cuda:0
            parts = d.split(":", 1)
            if len(parts) == 2 and parts[1].isdigit():
                return int(parts[1])
            return 0

        return -1

    @property
    def nli_model(self):
        """Lazy load NLI model for faithfulness."""
        if self._nli_model is None and self.use_nli:
            try:
                from transformers import pipeline
                tried = []
                for candidate in [self.nli_model_name, *self.nli_model_fallbacks]:
                    if not candidate or candidate in tried:
                        continue
                    tried.append(candidate)
                    try:
                        self._nli_model = pipeline(
                            "text-classification",
                            model=candidate,
                            device=self._hf_pipeline_device(),
                        )
                        if self._nli_model:
                            break
                    except Exception as e:
                        print(f"Warning: Failed to load NLI model '{candidate}': {e}")
            except Exception as e:
                print(f"Warning: Failed to initialize NLI pipeline: {e}")
                self._nli_model = None
        return self._nli_model
    
    @property
    def embedding_model(self):
        """Lazy load embedding model for relevance."""
        if self._embedding_model is None and self.use_embeddings:
            try:
                from sentence_transformers import SentenceTransformer
                self._embedding_model = SentenceTransformer(
                    self.embedding_model_name,
                    device=self.device,
                )
            except Exception as e:
                print(f"Warning: Failed to load embedding model: {e}")
                self._embedding_model = None
        return self._embedding_model
    
    @property
    def bertscore_model(self):
        """Lazy load BERTScore."""
        if self._bertscore_model is None and self.use_bertscore:
            self._load_bertscore_model()
        return self._bertscore_model

    def _iter_bertscore_candidates(self, exclude: Optional[set[str]] = None) -> List[str]:
        exclude = exclude or set()
        ordered: List[str] = []
        for candidate in [self.bertscore_model_name, *self.bertscore_model_fallbacks]:
            if not candidate or candidate in exclude or candidate in ordered:
                continue
            ordered.append(candidate)
        return ordered

    def _probe_bertscore_model(self, scorer: Any) -> None:
        """Run a tiny score call to catch models that load but fail at inference time."""
        _, _, f1 = scorer.score(["cage sanity check"], ["cage sanity check"])
        _ = float(f1[0].cpu().numpy())

    def _disable_bertscore(self, reason: str) -> None:
        """Disable BERTScore for the remainder of the run after an unrecoverable compatibility failure."""
        if self.use_bertscore:
            print(f"Warning: Disabling BERTScore for this run: {reason}")
        self.use_bertscore = False
        self._bertscore_model = None
        self._bertscore_model_active_name = None
        self._bertscore_disabled_reason = reason

    def _load_bertscore_model(self, exclude: Optional[set[str]] = None) -> Any:
        self._bertscore_model = None
        self._bertscore_model_active_name = None
        failures: List[str] = []
        try:
            from bert_score import BERTScorer

            for candidate in self._iter_bertscore_candidates(exclude=exclude):
                try:
                    scorer = BERTScorer(
                        model_type=candidate,
                        device=self.device,
                    )
                    self._probe_bertscore_model(scorer)
                    self._bertscore_model = scorer
                    self._bertscore_model_active_name = candidate
                    break
                except Exception as e:
                    failures.append(f"{candidate}: {e}")
                    print(f"Warning: Failed to initialize BERTScore model '{candidate}': {e}")
            if self._bertscore_model is None and failures:
                self._disable_bertscore(
                    "all configured BERTScore models failed to initialize under the current bert-score/transformers stack"
                )
        except Exception as e:
            self._disable_bertscore(f"failed to import bert-score: {e}")
        return self._bertscore_model
    
    @property
    def rouge_scorer(self):
        """Lazy load ROUGE scorer."""
        if self._rouge_scorer is None and self.use_rouge:
            try:
                from rouge_score import rouge_scorer
                self._rouge_scorer = rouge_scorer.RougeScorer(
                    ["rouge1", "rouge2", "rougeL"],
                    use_stemmer=True,
                )
            except Exception as e:
                print(f"Warning: Failed to load ROUGE: {e}")
                self._rouge_scorer = None
        return self._rouge_scorer
    
    def evaluate_faithfulness(
        self, generated_text: str, context: List[str]
    ) -> float:
        """
        Evaluate faithfulness using NLI.
        
        Checks if generated text is entailed by context.
        Returns average entailment score across context documents.
        """
        if not self.nli_model or not context:
            return 0.5  # Neutral score
        
        try:
            # Check entailment for each context document
            scores = []
            for ctx in context:
                # NLI: premise=context, hypothesis=generated_text
                result = self.nli_model(
                    f"{ctx} [SEP] {generated_text}",
                    top_k=1,
                )
                
                # Extract entailment probability
                label = str(result[0].get("label", "")).lower()
                is_entailment = (
                    "entail" in label or label in {"label_2"}  # common MNLI mapping
                )
                if result and is_entailment:
                    scores.append(result[0]["score"])
                else:
                    scores.append(0.0)
            
            return float(np.mean(scores)) if scores else 0.5
            
        except Exception as e:
            print(f"Error in faithfulness evaluation: {e}")
            return 0.5
    
    def evaluate_relevance(
        self, question: str, context: List[str]
    ) -> float:
        """
        Evaluate relevance using embedding similarity.
        
        Computes cosine similarity between question and context.
        Returns maximum similarity score across context documents.
        """
        if not self.embedding_model or not context:
            return 0.5
        
        try:
            # Encode question and context
            question_emb = self.embedding_model.encode(
                question, convert_to_tensor=True
            )
            context_embs = self.embedding_model.encode(
                context, convert_to_tensor=True
            )
            
            # Compute cosine similarities
            from sentence_transformers.util import cos_sim
            similarities = cos_sim(question_emb, context_embs)[0]
            
            # Return max similarity
            return float(similarities.max().cpu().numpy())
            
        except Exception as e:
            print(f"Error in relevance evaluation: {e}")
            return 0.5
    
    def evaluate_completeness(
        self, generated_text: str, reference_answer: str
    ) -> Dict[str, Optional[float]]:
        """
        Evaluate completeness using BERTScore and ROUGE.
        
        Compares generated text to reference answer.
        Returns dict with bertscore_f1 and rouge_l_f1.
        """
        results: Dict[str, Optional[float]] = {"bertscore_f1": None, "rouge_l_f1": None}
        
        # Handle empty generation gracefully to avoid BERTScore warnings
        if not generated_text or not generated_text.strip():
            if self.use_bertscore:
                results["bertscore_f1"] = 0.0
            if self.use_rouge:
                results["rouge_l_f1"] = 0.0
            return results
        
        # BERTScore
        if self.bertscore_model:
            try:
                P, R, F1 = self.bertscore_model.score(
                    [generated_text], [reference_answer]
                )
                results["bertscore_f1"] = float(F1[0].cpu().numpy())
            except Exception as e:
                active_model = self._bertscore_model_active_name or self.bertscore_model_name
                print(f"Error in BERTScore with model '{active_model}': {e}")
                fallback = self._load_bertscore_model(exclude={active_model})
                if fallback is not None:
                    try:
                        P, R, F1 = fallback.score(
                            [generated_text], [reference_answer]
                        )
                        results["bertscore_f1"] = float(F1[0].cpu().numpy())
                    except Exception as retry_error:
                        print(f"Error in BERTScore fallback: {retry_error}")
                        self._disable_bertscore(
                            f"runtime scoring failed after fallback attempt: {retry_error}"
                        )
        
        # ROUGE
        if self.rouge_scorer:
            try:
                scores = self.rouge_scorer.score(reference_answer, generated_text)
                results["rouge_l_f1"] = scores["rougeL"].fmeasure
            except Exception as e:
                print(f"Error in ROUGE: {e}")
        
        return results
    
    def evaluate_f1_score(
        self, generated_text: str, reference_answer: str
    ) -> Dict[str, float]:
        """
        Compute token-level F1 score (standard QA metric).
        
        F1 score is the harmonic mean of precision and recall at the token level.
        This is the standard metric used in SQuAD, HotpotQA, and other QA benchmarks.
        
        Args:
            generated_text: Model's generated answer
            reference_answer: Ground truth answer
            
        Returns:
            Dict with f1, precision, recall, and exact_match scores
        """
        import re
        import string
        
        def normalize_text(text: str) -> str:
            """Normalize text for comparison (lowercase, remove punctuation/articles)."""
            text = text.lower()
            # Remove punctuation
            text = text.translate(str.maketrans("", "", string.punctuation))
            # Remove articles
            text = re.sub(r"\b(a|an|the)\b", " ", text)
            # Normalize whitespace
            text = " ".join(text.split())
            return text
        
        def get_tokens(text: str) -> List[str]:
            """Tokenize normalized text."""
            return normalize_text(text).split()
        
        # Handle empty inputs
        if not generated_text or not generated_text.strip():
            return {"f1": 0.0, "precision": 0.0, "recall": 0.0, "exact_match": 0.0}
        if not reference_answer or not reference_answer.strip():
            return {"f1": 0.0, "precision": 0.0, "recall": 0.0, "exact_match": 0.0}
        
        # Exact match check
        exact_match = 1.0 if normalize_text(generated_text) == normalize_text(reference_answer) else 0.0
        
        # Get token sets
        pred_tokens = get_tokens(generated_text)
        ref_tokens = get_tokens(reference_answer)
        
        if not pred_tokens or not ref_tokens:
            return {"f1": 0.0, "precision": 0.0, "recall": 0.0, "exact_match": exact_match}
        
        # Count common tokens
        common_tokens = set(pred_tokens) & set(ref_tokens)
        num_common = sum(min(pred_tokens.count(t), ref_tokens.count(t)) for t in common_tokens)
        
        # Compute precision and recall
        precision = num_common / len(pred_tokens) if pred_tokens else 0.0
        recall = num_common / len(ref_tokens) if ref_tokens else 0.0
        
        # Compute F1
        if precision + recall > 0:
            f1 = 2 * precision * recall / (precision + recall)
        else:
            f1 = 0.0
        
        return {
            "f1": f1,
            "precision": precision,
            "recall": recall,
            "exact_match": exact_match,
        }
    
    def evaluate(
        self,
        question: str,
        context: List[str],
        generated_text: str,
        reference_answer: str,
    ) -> QualityMetrics:
        """
        Perform full quality evaluation.
        
        Args:
            question: The input question
            context: List of context documents
            generated_text: Model's generated answer
            reference_answer: Ground truth answer
        
        Returns:
            QualityMetrics with all scores
        """
        faithfulness = self.evaluate_faithfulness(generated_text, context)
        relevance = self.evaluate_relevance(question, context)
        completeness = self.evaluate_completeness(generated_text, reference_answer)
        f1_metrics = self.evaluate_f1_score(generated_text, reference_answer)
        
        return QualityMetrics(
            faithfulness=faithfulness,
            relevance=relevance,
            completeness_bertscore=completeness["bertscore_f1"],
            completeness_rouge_l=completeness["rouge_l_f1"],
            f1_score=f1_metrics["f1"],
            precision=f1_metrics["precision"],
            recall=f1_metrics["recall"],
            exact_match=f1_metrics["exact_match"],
        )
    
    def batch_evaluate(
        self,
        questions: List[str],
        contexts: List[List[str]],
        generated_texts: List[str],
        reference_answers: List[str],
    ) -> List[QualityMetrics]:
        """Batch evaluation (sequential for now)."""
        results = []
        for q, ctx, gen, ref in zip(questions, contexts, generated_texts, reference_answers):
            metrics = self.evaluate(q, ctx, gen, ref)
            results.append(metrics)
        return results
    
    def evaluate_cache_relevance(
        self,
        generated_text: str,
        reference_answer: str,
        cache_blocks: List[str],
        relevance_threshold: float = 0.3,
    ) -> CacheRelevanceMetrics:
        """
        Evaluate cache relevance - what proportion of accessed cache blocks
        actually contributed to generating the correct answer.
        
        This is a key metric for distributed CAG systems where we want to
        minimize unnecessary KV cache transfers between nodes.
        
        Args:
            generated_text: The model's generated answer
            reference_answer: Ground truth answer
            cache_blocks: List of cache block contents (context chunks)
            relevance_threshold: Minimum similarity score to consider a block "relevant"
        
        Returns:
            CacheRelevanceMetrics with per-block and aggregate scores
        """
        if not cache_blocks:
            return CacheRelevanceMetrics(
                cache_relevance=0.0,
                relevant_block_count=0,
                total_block_count=0,
                per_block_scores=[],
            )
        
        per_block_scores = []
        
        # Method 1: Embedding similarity between each block and the reference answer
        # This measures whether each block contains information relevant to the answer
        if self.embedding_model:
            try:
                from sentence_transformers.util import cos_sim
                
                # Encode reference answer (what we're trying to generate)
                ref_emb = self.embedding_model.encode(
                    reference_answer, convert_to_tensor=True
                )
                
                # Encode each cache block
                block_embs = self.embedding_model.encode(
                    cache_blocks, convert_to_tensor=True
                )
                
                # Compute similarity of each block to the reference answer
                similarities = cos_sim(ref_emb, block_embs)[0]
                per_block_scores = [float(s.cpu().numpy()) for s in similarities]
                
            except Exception as e:
                print(f"Error computing cache relevance embeddings: {e}")
                # Fall back to lexical overlap
                per_block_scores = self._lexical_cache_relevance(
                    reference_answer, cache_blocks
                )
        else:
            # Fallback: lexical overlap (token-based)
            per_block_scores = self._lexical_cache_relevance(
                reference_answer, cache_blocks
            )
        
        # Count blocks above relevance threshold
        relevant_count = sum(1 for s in per_block_scores if s >= relevance_threshold)
        total_count = len(cache_blocks)
        
        # Cache relevance = proportion of blocks that were actually useful
        cache_relevance = relevant_count / total_count if total_count > 0 else 0.0
        
        return CacheRelevanceMetrics(
            cache_relevance=cache_relevance,
            relevant_block_count=relevant_count,
            total_block_count=total_count,
            per_block_scores=per_block_scores,
        )
    
    def _lexical_cache_relevance(
        self,
        reference_answer: str,
        cache_blocks: List[str],
    ) -> List[float]:
        """
        Compute lexical overlap between reference answer and each cache block.
        Fallback method when embedding model is not available.
        
        Uses token-level Jaccard similarity.
        """
        import re
        
        def tokenize(text: str) -> set:
            # Simple whitespace + punctuation tokenization
            tokens = re.findall(r'\b\w+\b', text.lower())
            return set(tokens)
        
        ref_tokens = tokenize(reference_answer)
        if not ref_tokens:
            return [0.0] * len(cache_blocks)
        
        scores = []
        for block in cache_blocks:
            block_tokens = tokenize(block)
            if not block_tokens:
                scores.append(0.0)
                continue
            
            # Jaccard similarity
            intersection = len(ref_tokens & block_tokens)
            union = len(ref_tokens | block_tokens)
            jaccard = intersection / union if union > 0 else 0.0
            scores.append(jaccard)
        
        return scores
    
    def evaluate_with_cache_relevance(
        self,
        question: str,
        context: List[str],
        generated_text: str,
        reference_answer: str,
        cache_blocks: Optional[List[str]] = None,
        relevance_threshold: float = 0.3,
    ) -> QualityMetrics:
        """
        Full quality evaluation including cache relevance.
        
        Args:
            question: The input question
            context: List of context documents (for faithfulness/relevance)
            generated_text: Model's generated answer
            reference_answer: Ground truth answer
            cache_blocks: Optional list of cache block contents to evaluate.
                          If None, uses context as cache blocks.
            relevance_threshold: Threshold for considering a block "relevant"
        
        Returns:
            QualityMetrics with all scores including cache_relevance
        """
        # Base metrics
        faithfulness = self.evaluate_faithfulness(generated_text, context)
        relevance = self.evaluate_relevance(question, context)
        completeness = self.evaluate_completeness(generated_text, reference_answer)
        
        # Cache relevance (use context as cache blocks if not provided)
        blocks_to_evaluate = cache_blocks if cache_blocks is not None else context
        cache_rel = self.evaluate_cache_relevance(
            generated_text, reference_answer, blocks_to_evaluate, relevance_threshold
        )
        
        return QualityMetrics(
            faithfulness=faithfulness,
            relevance=relevance,
            completeness_bertscore=completeness["bertscore_f1"],
            completeness_rouge_l=completeness["rouge_l_f1"],
            cache_relevance=cache_rel.cache_relevance,
        )
