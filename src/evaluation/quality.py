"""
Quality metrics for CAGE evaluation.

Metrics:
- Hallucination (PRIMARY): LettuceDetect token/span-level grounding detector.
  Encoder (ModernBERT) trained on RAGTruth; flags answer spans not supported by
  the context. Reports a span ratio and a derived faithfulness score.
- Faithfulness (NLI): claim-level entailment. The answer is split into claims;
  each claim's entailment probability is taken as the MAX over context documents
  (a claim is faithful if supported by ANY provided context), then averaged over
  claims (RAGAS-style). Premise/hypothesis are passed as a proper sentence PAIR
  and the entailment class index is resolved from the model config (never hard
  coded), so the score is comparable across NLI checkpoints.
- Relevance (retriever diagnostic): question<->context embedding similarity.
  NOTE: this characterises the retriever, NOT answer quality. Reported under the
  ``context_relevance`` name; ``relevance`` is kept as an alias for back-compat.
- Completeness: BERTScore (with baseline rescaling) and ROUGE-L.
- F1-score: Token-level precision/recall (QA standard metric).
- Cache Relevance: Proportion of cache blocks that contributed to the answer.

Design intent for cloud/HPC + publication: every metric returns ``None`` (not a
silent 0.5/0.0 sentinel) when its model is unavailable, so undisclosed model-load
failures cannot contaminate reported means.
"""

from dataclasses import dataclass
from typing import List, Dict, Any, Optional
import os
import re
import numpy as np
import warnings

# Suppress BERTScore warning about empty candidates
warnings.filterwarnings("ignore", message=".*Empty candidate sentence detected.*")


# --------------------------------------------------------------------------- #
# SQuAD v2 no-answer (abstention) detection
# --------------------------------------------------------------------------- #
# SQuAD v2 is ~52% UNANSWERABLE: the gold reference is empty and the correct model
# behaviour is to ABSTAIN ("the context does not answer this"). A vLLM model never emits
# a null-answer token -- it emits prose -- so scoring abstention requires detecting a
# no-answer PREDICTION from the generated text. This detector powers proper SQuAD v2
# EM/F1 (a correct abstention on a no-answer item scores 1, not 0) in evaluate_f1_score.
#
# It is deliberately tuned to AVOID FALSE POSITIVES (misreading a real answer as an
# abstention), because that is the dangerous direction -- it would wrongly credit a
# hallucination. A missed verbose abstention (false negative) merely scores 0, which is
# exactly the pre-fix behaviour, so the detector degrades gracefully and never regresses
# a previously-correct score. Bare "no"/"yes" are VALID answers and are NOT matched.
_NO_ANSWER_RE = re.compile(
    r"\b("
    r"no\s+answer|"
    r"cannot\s+(be\s+)?answer(ed)?|can'?t\s+answer|unable\s+to\s+(answer|determine|find)|"
    r"unanswerable|not\s+answerable|"
    r"no\s+(information|mention|indication|answer|idea)|"
    r"insufficient\s+(information|context|detail)|"
    r"not\s+enough\s+(information|context|details?)|"
    r"not\s+(in|found\s+in|provided|mentioned|stated|specified|available|present|given|sure)|"
    r"does\s+not\s+(say|mention|provide|contain|specify|state|give|include)|"
    r"doesn'?t\s+(say|mention|provide|contain|specify|state|give|include)|"
    r"do(es)?\s+not\s+have\s+(the\s+)?answer|"
    # Leading "I" is OPTIONAL: the system prompt instructs "say you don't know", and models
    # emit the bare form ("Don't know.") as often as the first-person one. Requiring the "i"
    # scored those correct abstentions as attempted answers (2026-07-15 audit: 12/12 missed).
    r"(i\s+)?don'?t\s+know|(i\s+)?do\s+not\s+know|"
    r"cannot\s+(be\s+)?(determined?|found)|can'?t\s+(be\s+)?(determined?|found)|"
    r"the\s+(context|passage|text|document|article)\s+does\s+not"
    r")",
    re.IGNORECASE,
)
# Whole-answer abstention tokens. These words appear inside legitimate answers ("Unknown
# Pleasures"), so they only count as abstention when they ARE the entire answer. Bare
# "none"/"NA" are deliberately excluded: both occur as real SQuAD gold spans ("none",
# sodium's symbol), and a false positive here wrongly zeroes a correct answer.
_NO_ANSWER_EXACT_RE = re.compile(r"^\W*(unknown|n/a|no\s+idea|not\s+sure)[\s.!?]*$", re.IGNORECASE)
# Answers longer than this are assumed to be real content, not an abstention, even if they
# happen to contain a matched phrase ("There is no doubt the answer is Paris"). SQuAD v2
# abstention outputs are short; the cap trades a few verbose-abstention misses (safe: score
# 0) for near-zero false positives (unsafe: falsely credited).
_NO_ANSWER_MAX_WORDS = 20


def is_no_answer_prediction(text: Optional[str]) -> bool:
    """True if the generated text is a no-answer / abstention prediction (SQuAD v2).

    Empty output counts as abstention. Otherwise, a SHORT response containing an explicit
    abstention phrase counts. Conservative by design -- see _NO_ANSWER_RE note above.
    """
    t = (text or "").strip()
    if not t:
        return True
    if _NO_ANSWER_EXACT_RE.match(t):
        return True
    if len(t.split()) <= _NO_ANSWER_MAX_WORDS and _NO_ANSWER_RE.search(t):
        return True
    return False


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
    """Quality evaluation results.

    Faithfulness/quality fields are ``Optional``: ``None`` means "metric model
    unavailable for this sample" and must be excluded from means, never treated
    as a real score.
    """

    faithfulness: Optional[float]  # 0-1, claim-level NLI entailment (None if NLI unavailable)
    relevance: Optional[float]  # 0-1, question<->context similarity (retriever diagnostic)
    completeness_bertscore: Optional[float]  # 0-1, BERTScore F1 (baseline-rescaled)
    completeness_rouge_l: Optional[float]  # 0-1, ROUGE-L F1
    f1_score: float = 0.0  # 0-1, token-level F1 (SQuAD v2 official: abstention-aware)
    precision: float = 0.0  # 0-1, token-level precision
    recall: float = 0.0  # 0-1, token-level recall
    exact_match: float = 0.0  # 0 or 1, exact match (SQuAD v2 official: abstention-aware)
    # SQuAD v2 no-answer decomposition (fix #4). answerable-only variants are None on
    # no-answer items so downstream None-exclusion reports F1/EM over the answerable subset;
    # no_answer_correct is None on answerable items so its mean is abstention accuracy.
    is_answerable: Optional[float] = None  # 1.0 answerable / 0.0 no-answer / None if not scored
    predicted_no_answer: Optional[float] = None  # 1.0 if the model abstained
    f1_answerable: Optional[float] = None  # token-F1 over answerable items only
    exact_match_answerable: Optional[float] = None  # EM over answerable items only
    no_answer_correct: Optional[float] = None  # 1.0/0.0 on no-answer items; abstention accuracy
    abstention_precision: Optional[float] = None  # 1.0/0.0 on abstained rows only; mean = precision
    cache_relevance: Optional[float] = None  # 0-1, proportion of useful cache blocks
    # Hallucination (LettuceDetect, PRIMARY grounding signal)
    grounding_score: Optional[float] = None  # 0-1, 1 - hallucinated_span_ratio (None if detector unavailable)
    hallucination_detected: Optional[bool] = None  # True if any answer span is unsupported
    hallucinated_span_ratio: Optional[float] = None  # 0-1, fraction of answer characters flagged unsupported
    supported_claim_ratio: Optional[float] = None  # 0-1, fraction of claims with entailment >= 0.5
    faithfulness_method: str = "nli_claim_max"  # provenance of the faithfulness number
    # Evidence-only, NOT a metric: raw LettuceDetect answer spans flagged unsupported, for the
    # per-query qa_evidence.jsonl. Deliberately EXCLUDED from to_dict() so it never becomes a
    # CSV column or enters metric aggregation. None when the detector is unavailable.
    hallucinated_spans: Optional[List[Dict[str, Any]]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary. Numeric fields are auto-aggregated downstream."""
        result: Dict[str, Any] = {
            "faithfulness": self.faithfulness,
            # Honest name for the retriever-diagnostic, plus a back-compat alias.
            "context_relevance": self.relevance,
            "relevance": self.relevance,
            "completeness_bertscore": self.completeness_bertscore,
            "completeness_rouge_l": self.completeness_rouge_l,
            "f1_score": self.f1_score,
            "precision": self.precision,
            "recall": self.recall,
            "exact_match": self.exact_match,
            # SQuAD v2 no-answer decomposition (fix #4). Emitted unconditionally so the CSV
            # columns are stable; None values are excluded from means / stats automatically.
            "is_answerable": self.is_answerable,
            "predicted_no_answer": self.predicted_no_answer,
            "f1_answerable": self.f1_answerable,
            "exact_match_answerable": self.exact_match_answerable,
            "no_answer_correct": self.no_answer_correct,
            "abstention_precision": self.abstention_precision,
            "grounding_score": self.grounding_score,
            "hallucinated_span_ratio": self.hallucinated_span_ratio,
            "supported_claim_ratio": self.supported_claim_ratio,
            "faithfulness_method": self.faithfulness_method,
        }
        if self.hallucination_detected is not None:
            # Stored as 0/1 so it aggregates to a hallucination RATE across a run.
            result["hallucination_detected"] = 1.0 if self.hallucination_detected else 0.0
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
        use_lettucedetect: bool = True,
        device: str | int = "cpu",
        nli_model_name: Optional[str] = None,
        embedding_model_name: Optional[str] = None,
        bertscore_model_name: Optional[str] = None,
        lettucedetect_model_name: Optional[str] = None,
        bertscore_rescale_with_baseline: bool = True,
        bertscore_lang: str = "en",
        nli_max_length: int = 512,
    ):
        self.use_nli = use_nli
        self.use_embeddings = use_embeddings
        self.use_bertscore = use_bertscore
        self.use_rouge = use_rouge
        # LettuceDetect can be force-disabled via env (e.g. CPU-only smoke tests).
        self.use_lettucedetect = use_lettucedetect and os.getenv(
            "CAGE_DISABLE_LETTUCEDETECT", ""
        ).strip().lower() not in {"1", "true", "yes"}
        self.device = device
        self.bertscore_rescale_with_baseline = bertscore_rescale_with_baseline
        self.bertscore_lang = bertscore_lang
        self.nli_max_length = nli_max_length

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
        self.lettucedetect_model_name = (
            lettucedetect_model_name
            or os.getenv(
                "CAGE_LETTUCEDETECT_MODEL",
                "KRLabsOrg/lettucedect-base-modernbert-en-v1",
            )
        )

        # Lazy loading of models
        self._nli_model = None
        self._nli_entail_index = None  # resolved entailment class index for the loaded NLI model
        self._embedding_model = None
        self._bertscore_model = None
        self._bertscore_model_active_name = None
        self._rouge_scorer = None
        self._bertscore_disabled_reason = None
        self._lettucedetect_model = None
        self._lettucedetect_disabled_reason = None
    
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
                    # rescale_with_baseline is REQUIRED for discriminative scores:
                    # raw RoBERTa F1 sits in a compressed ~0.3 band and is flat across
                    # systems. Baseline rescaling restores dynamic range. lang selects
                    # the correct baseline file.
                    try:
                        scorer = BERTScorer(
                            model_type=candidate,
                            device=self.device,
                            lang=self.bertscore_lang,
                            rescale_with_baseline=self.bertscore_rescale_with_baseline,
                        )
                    except Exception as baseline_err:
                        # Some custom model_types have no published baseline file;
                        # fall back to unrescaled rather than dropping the model.
                        print(
                            f"Warning: BERTScore baseline rescaling unavailable for "
                            f"'{candidate}' ({baseline_err}); using unrescaled scores."
                        )
                        scorer = BERTScorer(model_type=candidate, device=self.device)
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

    @property
    def lettucedetect_model(self):
        """Lazy load the LettuceDetect hallucination detector (PRIMARY grounding signal)."""
        if (
            self._lettucedetect_model is None
            and self.use_lettucedetect
            and self._lettucedetect_disabled_reason is None
        ):
            try:
                from lettucedetect.models.inference import HallucinationDetector

                # device: HallucinationDetector accepts a torch-style device string.
                device_str = "cpu"
                d = str(self.device).lower()
                if isinstance(self.device, int):
                    device_str = f"cuda:{self.device}" if self.device >= 0 else "cpu"
                elif d.startswith("cuda"):
                    device_str = d
                self._lettucedetect_model = HallucinationDetector(
                    method="transformer",
                    model_path=self.lettucedetect_model_name,
                    device=device_str,
                )
            except Exception as e:
                self._lettucedetect_disabled_reason = str(e)
                print(
                    f"Warning: LettuceDetect unavailable ({e}); "
                    f"falling back to NLI faithfulness only."
                )
                self._lettucedetect_model = None
        return self._lettucedetect_model

    @staticmethod
    def _split_claims(text: str) -> List[str]:
        """Split an answer into atomic claims (sentence-level).

        Dependency-free splitter: breaks on sentence terminators and newlines.
        Short answers (no terminator) are returned as a single claim.
        """
        import re

        if not text or not text.strip():
            return []
        # Split on ., !, ? followed by whitespace, and on newlines/semicolons.
        parts = re.split(r"(?<=[.!?])\s+|\n+|;\s+", text.strip())
        claims = [p.strip() for p in parts if p and p.strip()]
        return claims or [text.strip()]

    def _resolve_nli_entail_index(self) -> Optional[int]:
        """Resolve the entailment class index from the loaded NLI model config.

        Never hard-code LABEL_2: DeBERTa-mnli-fever-anli uses
        {0: entailment, 1: neutral, 2: contradiction} whereas bart-large-mnli uses
        the reverse. We read id2label and find the 'entailment' class.
        """
        if self._nli_entail_index is not None:
            return self._nli_entail_index
        try:
            id2label = self.nli_model.model.config.id2label
            for idx, label in id2label.items():
                if "entail" in str(label).lower():
                    self._nli_entail_index = int(idx)
                    return self._nli_entail_index
        except Exception:
            pass
        return None

    def _nli_entailment_prob(self, premise: str, hypothesis: str) -> Optional[float]:
        """P(entailment) for hypothesis given premise, as a proper sentence pair."""
        try:
            # Pass a PAIR (text/text_pair) so the model sees premise vs hypothesis
            # with correct segment encoding. top_k=None returns all class scores.
            result = self.nli_model(
                {"text": premise, "text_pair": hypothesis},
                top_k=None,
                truncation=True,
                max_length=self.nli_max_length,
            )
            # transformers may nest the result as [[...]] for a single pair.
            if result and isinstance(result[0], list):
                result = result[0]
            if not result:
                return None
            by_label = {str(d.get("label", "")).lower(): float(d.get("score", 0.0)) for d in result}
            # Prefer a named 'entailment' class.
            for label, score in by_label.items():
                if "entail" in label:
                    return score
            # Otherwise resolve LABEL_x via the model config.
            idx = self._resolve_nli_entail_index()
            if idx is not None:
                return by_label.get(f"label_{idx}")
            return None
        except Exception as e:
            print(f"Error in NLI entailment: {e}")
            return None

    def evaluate_faithfulness(
        self, generated_text: str, context: List[str]
    ) -> Dict[str, Optional[float]]:
        """Claim-level NLI faithfulness.

        The answer is split into claims; each claim's entailment probability is the
        MAX over context documents (faithful if supported by ANY context), then
        averaged over claims. Returns ``{"faithfulness": <0-1 or None>,
        "supported_claim_ratio": <0-1 or None>}``. ``None`` means NLI unavailable.
        """
        empty = {"faithfulness": None, "supported_claim_ratio": None}
        if not self.use_nli or not self.nli_model:
            return empty
        nonempty_ctx = [c for c in (context or []) if c and str(c).strip()]
        claims = self._split_claims(generated_text or "")
        if not nonempty_ctx or not claims:
            return empty

        try:
            claim_scores: List[float] = []
            for claim in claims:
                best = 0.0
                have_score = False
                for ctx in nonempty_ctx:
                    p = self._nli_entailment_prob(str(ctx), claim)
                    if p is not None:
                        best = max(best, p)
                        have_score = True
                if have_score:
                    claim_scores.append(best)
            if not claim_scores:
                return empty
            faithfulness = float(np.mean(claim_scores))
            supported = float(np.mean([1.0 if s >= 0.5 else 0.0 for s in claim_scores]))
            return {"faithfulness": faithfulness, "supported_claim_ratio": supported}
        except Exception as e:
            print(f"Error in faithfulness evaluation: {e}")
            return empty

    def evaluate_hallucination(
        self, question: str, context: List[str], generated_text: str
    ) -> Dict[str, Any]:
        """Token/span-level hallucination detection via LettuceDetect (PRIMARY).

        Returns ``{"grounding_score", "hallucination_detected",
        "hallucinated_span_ratio"}``. All ``None`` if the detector is unavailable.
        """
        empty: Dict[str, Any] = {
            "grounding_score": None,
            "hallucination_detected": None,
            "hallucinated_span_ratio": None,
            "hallucinated_spans": None,
        }
        detector = self.lettucedetect_model
        answer = generated_text or ""
        nonempty_ctx = [str(c) for c in (context or []) if c and str(c).strip()]
        if detector is None or not nonempty_ctx or not answer.strip():
            return empty
        try:
            spans = detector.predict(
                context=nonempty_ctx,
                question=question or "",
                answer=answer,
                output_format="spans",
            )
            # spans: list of dicts with 'start','end' (char offsets into the answer).
            # Mark flagged characters in a boolean array so OVERLAPPING spans are counted
            # ONCE: summing raw span lengths would double-count shared characters, inflate the
            # ratio, and deflate the PRIMARY grounding_score. Identical to the old sum when
            # spans are disjoint (the normal case).
            total = len(answer)
            flagged_chars = [False] * total
            for s in spans or []:
                start = max(0, min(int(s.get("start", 0)), total))
                end = max(0, min(int(s.get("end", 0)), total))
                for i in range(start, end):
                    flagged_chars[i] = True
            flagged = sum(flagged_chars)
            ratio = (flagged / total) if total > 0 else 0.0
            ratio = max(0.0, min(1.0, ratio))
            norm_spans = [
                {"start": int(s.get("start", 0)), "end": int(s.get("end", 0)),
                 "text": s.get("text")}
                for s in (spans or [])
            ]
            return {
                "grounding_score": 1.0 - ratio,
                # Flag a hallucination only when characters were actually flagged, not merely
                # because the detector returned a (possibly zero-length) span.
                "hallucination_detected": flagged > 0,
                "hallucinated_span_ratio": ratio,
                "hallucinated_spans": norm_spans,
            }
        except Exception as e:
            print(f"Error in LettuceDetect hallucination detection: {e}")
            return empty
    
    def evaluate_relevance(
        self, question: str, context: List[str]
    ) -> Optional[float]:
        """
        Retriever diagnostic: question<->context embedding similarity.

        NOTE: this is a property of the retriever + dataset and is INDEPENDENT of
        the generated answer. It is NOT an answer-quality metric. Returns the max
        cosine similarity across context documents, or ``None`` if the embedding
        model is unavailable.
        """
        nonempty_ctx = [c for c in (context or []) if c and str(c).strip()]
        if not self.embedding_model or not nonempty_ctx:
            return None

        try:
            # Encode question and context
            question_emb = self.embedding_model.encode(
                question, convert_to_tensor=True
            )
            context_embs = self.embedding_model.encode(
                nonempty_ctx, convert_to_tensor=True
            )

            # Compute cosine similarities
            from sentence_transformers.util import cos_sim
            similarities = cos_sim(question_emb, context_embs)[0]

            # Return max similarity
            return float(similarities.max().cpu().numpy())

        except Exception as e:
            print(f"Error in relevance evaluation: {e}")
            return None
    
    def evaluate_completeness(
        self, generated_text: str, reference_answer: str
    ) -> Dict[str, Optional[float]]:
        """
        Evaluate completeness using BERTScore and ROUGE.
        
        Compares generated text to reference answer.
        Returns dict with bertscore_f1 and rouge_l_f1.
        """
        results: Dict[str, Optional[float]] = {"bertscore_f1": None, "rouge_l_f1": None}
        
        # Empty/blank reference (e.g. SQuAD v2 unanswerable items, ~52% of the set):
        # completeness is UNDEFINED against a non-existent reference. Return None so these
        # rows are EXCLUDED from the aggregate (mean_or_none skips None) rather than
        # averaging in BERTScore's large-negative baseline-rescaled value (a misleading
        # sentinel near -4.4) that dragged the Phase-2 aggregate to an implausible ~-2.0.
        if not reference_answer or not reference_answer.strip():
            return results

        # Empty generation: a missing answer scores 0 on overlap metrics (this is a
        # genuine 0, not a model-unavailable sentinel).
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
        self, generated_text: str, reference_answer: str,
        all_answers: Optional[List[str]] = None,
    ) -> Dict[str, float]:
        """
        Compute token-level F1 / EM with SQuAD v2 no-answer credit.

        F1 is the harmonic mean of token-level precision and recall (SQuAD / HotpotQA
        standard). This implementation also handles SQuAD v2 UNANSWERABLE items: a correct
        abstention on a no-answer question scores 1 (not 0), and answerable-only variants are
        emitted so extraction quality can be reported separately from abstention accuracy.

        Args:
            generated_text: Model's generated answer
            reference_answer: Ground truth answer ("" / blank == SQuAD v2 no-answer item)
            all_answers: Optional list of ALL gold answers (audit 2026-07-16 M5): the
                official SQuAD v2 metric is the MAX over every gold answer
                (metric_max_over_ground_truths); scoring only text[0] understated
                answerable F1 ~5pp / EM ~10pp. Sourced from
                CAGExample.metadata["all_answers"]. Empty list = unanswerable item
                (official semantics); None falls back to the single reference_answer
                (older evidence files / datasets without the field).

        Returns:
            Dict with:
              f1, precision, recall, exact_match       -- SQuAD v2 official (abstention-aware)
              is_answerable                            -- 1.0 answerable / 0.0 no-answer item
              predicted_no_answer                      -- 1.0 if the model abstained
              f1_answerable, exact_match_answerable    -- None on no-answer items (answerable-only)
              no_answer_correct                        -- None on answerable items; 1.0/0.0 on
                                                          no-answer items (abstention accuracy)
        """
        # Max over ALL gold answers (audit 2026-07-16 M5). Explicit class calls keep the
        # method free of instance state (tests invoke it unbound with self=None).
        if all_answers is not None:
            golds = [a for a in all_answers if (a or "").strip()]
            if not golds:
                # Official SQuAD v2 semantics: no gold answers == unanswerable item.
                return QualityEvaluator.evaluate_f1_score(self, generated_text, "")
            per_gold = [
                QualityEvaluator.evaluate_f1_score(self, generated_text, g) for g in golds
            ]
            # EM and F1 are maximized INDEPENDENTLY (official semantics); precision/recall
            # accompany the F1-maximizing gold so the P/R/F1 triplet stays coherent. The
            # abstention fields are identical across golds (they depend only on the
            # prediction and answerability), so any per-gold copy is correct.
            merged = dict(max(per_gold, key=lambda r: r["f1"]))
            merged["exact_match"] = max(r["exact_match"] for r in per_gold)
            if merged.get("f1_answerable") is not None:
                merged["f1_answerable"] = merged["f1"]
            if merged.get("exact_match_answerable") is not None:
                merged["exact_match_answerable"] = merged["exact_match"]
            return merged

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
        
        # ------------------------------------------------------------------ #
        # SQuAD v2 scoring with no-answer credit  (fix #4, options A + B)
        # ------------------------------------------------------------------ #
        # gold_no_answer: this is an UNANSWERABLE item (empty reference). ~52% of SQuAD v2.
        # pred_no_answer: the model produced an abstention (empty or an explicit phrase).
        gold_no_answer = not (reference_answer or "").strip()
        pred_no_answer = is_no_answer_prediction(generated_text)

        # (A) Official SQuAD v2 semantics on the UNANSWERABLE half. Before this fix the
        # function returned 0 unconditionally here, so a CORRECT abstention scored 0 and was
        # indistinguishable from a hallucination -- deflating F1/EM and blinding them to the
        # abstention behaviour that cache/serving configs can regress. Now: abstain -> 1, else 0.
        # (B) f1_answerable / exact_match_answerable are None on no-answer items so the downstream
        # None-exclusion computes the answerable-only F1/EM automatically. no_answer_correct is the
        # abstention-accuracy signal (mean over no-answer rows).
        if gold_no_answer:
            correct = 1.0 if pred_no_answer else 0.0
            return {
                "f1": correct, "precision": correct, "recall": correct, "exact_match": correct,
                "is_answerable": 0.0,
                "predicted_no_answer": 1.0 if pred_no_answer else 0.0,
                "f1_answerable": None, "exact_match_answerable": None,
                "no_answer_correct": correct,
                # Per-row indicator whose None-excluded mean IS abstention precision: defined
                # only on rows where the model abstained; 1.0 = the abstention was right
                # (item truly unanswerable). Recall over unanswerable rows is mean(no_answer_correct).
                "abstention_precision": 1.0 if pred_no_answer else None,
            }

        # ANSWERABLE item but the model abstained -> wrong (standard SQuAD v2: predicting
        # no-answer when an answer exists scores 0). Kept explicit so the abstention diagnostics
        # are populated and the answerable-only columns record the miss.
        if pred_no_answer:
            return {
                "f1": 0.0, "precision": 0.0, "recall": 0.0, "exact_match": 0.0,
                "is_answerable": 1.0, "predicted_no_answer": 1.0,
                "f1_answerable": 0.0, "exact_match_answerable": 0.0,
                "no_answer_correct": None,
                "abstention_precision": 0.0,  # abstained on an answerable item: wrong abstention
            }

        # ANSWERABLE item, model attempted an answer: standard token-level F1 / EM.
        exact_match = 1.0 if normalize_text(generated_text) == normalize_text(reference_answer) else 0.0
        pred_tokens = get_tokens(generated_text)
        ref_tokens = get_tokens(reference_answer)

        if not pred_tokens or not ref_tokens:
            return {
                "f1": 0.0, "precision": 0.0, "recall": 0.0, "exact_match": exact_match,
                "is_answerable": 1.0, "predicted_no_answer": 0.0,
                "f1_answerable": 0.0, "exact_match_answerable": exact_match,
                "no_answer_correct": None,
                "abstention_precision": None,
            }

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
            # (B) answerable subset: these equal the headline metric here and are None on
            # no-answer items, so mean(f1_answerable) is F1 over answerable questions only.
            "is_answerable": 1.0,
            "predicted_no_answer": 0.0,
            "f1_answerable": f1,
            "exact_match_answerable": exact_match,
            "no_answer_correct": None,
            "abstention_precision": None,
        }
    
    def evaluate(
        self,
        question: str,
        context: List[str],
        generated_text: str,
        reference_answer: str,
        all_answers: Optional[List[str]] = None,
    ) -> QualityMetrics:
        """
        Perform full quality evaluation.

        Args:
            question: The input question
            context: List of context documents
            generated_text: Model's generated answer
            reference_answer: Ground truth answer
            all_answers: Optional list of ALL gold answers for max-over-golds F1/EM
                (audit 2026-07-16 M5); see evaluate_f1_score.

        Returns:
            QualityMetrics with all scores
        """
        f1_metrics = self.evaluate_f1_score(generated_text, reference_answer, all_answers)
        relevance = self.evaluate_relevance(question, context)

        # Abstention-aware grounding/faithfulness (2026-07-15 audit): an abstention like
        # "Don't know." is by construction unsupported by the context, so LettuceDetect and
        # NLI mathematically MUST flag it -- scoring a CORRECT abstention as a hallucination
        # and penalizing whichever arm abstains more (~52% of SQuAD v2 is unanswerable).
        # An abstention is neither grounded nor hallucinated: those metrics are N/A (None,
        # excluded from means), and abstention correctness is scored by evaluate_f1_score
        # (no_answer_correct / abstention_precision). Completeness (reference similarity)
        # is equally meaningless for an abstention phrase. Relevance is question<->context
        # only, so it stays.
        if is_no_answer_prediction(generated_text):
            faith = {"faithfulness": None, "supported_claim_ratio": None}
            halluc = {
                "grounding_score": None,
                "hallucination_detected": None,
                "hallucinated_span_ratio": None,
                "hallucinated_spans": None,
            }
            completeness = {"bertscore_f1": None, "rouge_l_f1": None}
        else:
            faith = self.evaluate_faithfulness(generated_text, context)
            halluc = self.evaluate_hallucination(question, context, generated_text)
            completeness = self.evaluate_completeness(generated_text, reference_answer)

        return QualityMetrics(
            faithfulness=faith["faithfulness"],
            relevance=relevance,
            completeness_bertscore=completeness["bertscore_f1"],
            completeness_rouge_l=completeness["rouge_l_f1"],
            f1_score=f1_metrics["f1"],
            precision=f1_metrics["precision"],
            recall=f1_metrics["recall"],
            exact_match=f1_metrics["exact_match"],
            is_answerable=f1_metrics.get("is_answerable"),
            predicted_no_answer=f1_metrics.get("predicted_no_answer"),
            f1_answerable=f1_metrics.get("f1_answerable"),
            exact_match_answerable=f1_metrics.get("exact_match_answerable"),
            no_answer_correct=f1_metrics.get("no_answer_correct"),
            abstention_precision=f1_metrics.get("abstention_precision"),
            grounding_score=halluc["grounding_score"],
            hallucination_detected=halluc["hallucination_detected"],
            hallucinated_span_ratio=halluc["hallucinated_span_ratio"],
            supported_claim_ratio=faith["supported_claim_ratio"],
            hallucinated_spans=halluc.get("hallucinated_spans"),
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
        all_answers: Optional[List[str]] = None,
    ) -> QualityMetrics:
        """
        Full quality evaluation including cache relevance.

        Refactored (audit 2026-07-16 DEAD-EVAL-PATH): this used to duplicate the
        evaluate() pipeline WITHOUT the abstention short-circuit, so any future caller
        would reintroduce the "correct abstention scored as hallucination" bug. It now
        delegates to evaluate() (abstention guard, abstention_precision, max-over-golds
        F1/EM included) and attaches cache_relevance to the result.

        Args:
            question: The input question
            context: List of context documents (for faithfulness/relevance)
            generated_text: Model's generated answer
            reference_answer: Ground truth answer
            cache_blocks: Optional list of cache block contents to evaluate.
                          If None, uses context as cache blocks.
            relevance_threshold: Threshold for considering a block "relevant"
            all_answers: Optional list of ALL gold answers (see evaluate_f1_score)

        Returns:
            QualityMetrics with all scores including cache_relevance
        """
        metrics = self.evaluate(
            question, context, generated_text, reference_answer, all_answers=all_answers
        )

        # Cache relevance (use context as cache blocks if not provided)
        blocks_to_evaluate = cache_blocks if cache_blocks is not None else context
        cache_rel = self.evaluate_cache_relevance(
            generated_text, reference_answer, blocks_to_evaluate, relevance_threshold
        )
        metrics.cache_relevance = cache_rel.cache_relevance
        return metrics
