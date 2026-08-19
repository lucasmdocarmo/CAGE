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

D8 additions (build pass 2026-08-04):
- Windowed grounding (D8 §8.5, LAUNCH-BLOCKER): contexts beyond LettuceDetect's
  runtime-derived L_max (max_position_embeddings from the loaded model config --
  never hardcoded) are scored via sentence-aligned overlapping windows with
  MAX-SUPPORT aggregation per answer character; affected rows carry
  ``scored_windowed=True`` (windowing is an alert, never silent).
- Per-row instrument provenance (D8 §8.1): ``grounding_instrument``,
  ``faithfulness_instrument`` (id+version), ``calibration_id`` (None until the
  D9 calibration exists), ``scored_windowed`` -- all emitted unconditionally.
- BERTScore IDF weighting (D8 §8.4): env-gated via CAGE_BERTSCORE_IDF; default
  OFF (current behavior); requires a reference corpus (fail-closed).
- Real batching: ``batch_evaluate`` accumulates NLI pairs and BERTScore texts
  across rows into single batched model calls with identical per-row outputs.
- Instrument B seams (D8 §8.5): MiniCheck/AlignScore claim checkers, lazily
  loaded fail-closed, selected via CAGE_CLAIM_CHECKER (default 'nli' since the
  2026-08-19 owner decision #120/F8 -- in-process-safe; AlignScore remains the
  selected Instrument B via its dedicated out-of-process runner
  scripts/4_analysis/score_instrument_b.py; instrument selection history in
  MyDocs/registration/
  instrument_selection_2026-08-05/DECISION.md; in the project venv the
  alignscore package can never install, so the default fails closed and
  points to the out-of-process runner scripts/4_analysis/score_instrument_b.py).
- 3-class NLI reporting (D8 §8.5 "contradiction and neutral reported
  separately"): env-gated via CAGE_NLI_THREE_CLASS; default OFF (current
  behavior byte-identical). When ON, per-row faithfulness_contradiction /
  faithfulness_neutral columns (per-claim max-over-premises, mean over claims).
- Claim-decomposition seam: ``ClaimDecomposer`` protocol with the sentence
  splitter as default; the entity-resolved decomposer plugs in later.

Design intent for cloud/HPC + publication (P0-5 fail-closed, audit 2026-08-02):
instruments are PINNED. An instrument that cannot load or score is NEVER silently
replaced by a fallback model scoring under the same column name -- a mid-run
instrument swap voids the D8/D9 predicate symmetry behind Y (serving yield). In
strict mode (the D8 default) any load/score failure raises
``InstrumentUnavailableError``; in non-strict mode (long-run harness survival)
the row records score=``None`` plus an ``instrument_status`` token such as
``nli:unavailable:<model>``. The only substitution retained is the explicitly
labeled cache-relevance DIAGNOSTIC (lexical Jaccard), which writes its method
into the row (``cache_relevance_method``).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, NamedTuple, Optional, Protocol, Sequence, Tuple
import os
import re
import numpy as np
import warnings

# Suppress BERTScore warning about empty candidates
warnings.filterwarnings("ignore", message=".*Empty candidate sentence detected.*")


class InstrumentUnavailableError(RuntimeError):
    """A pinned quality instrument failed to load or score.

    Raised in strict mode (the D8 default) INSTEAD of substituting a fallback
    model: a substitute would score later rows with a DIFFERENT model under the
    same column name, voiding predicate symmetry across cells (audit P0-5).
    """

    def __init__(self, instrument: str, model: str, cause: str) -> None:
        self.instrument = instrument
        self.model = model
        self.cause = cause
        super().__init__(
            f"quality instrument '{instrument}' ({model}) unavailable: {cause}"
        )


def _package_version(package: str) -> str:
    """Installed distribution version for provenance strings (D8 §8.1).

    Provenance METADATA, not a score: an unresolvable version (e.g. a test stub
    injected via sys.modules with no distribution) records ``"unknown"`` rather
    than failing the row -- the instrument id (model name) still pins WHAT scored.
    """
    try:
        from importlib.metadata import version

        return version(package)
    except Exception:
        return "unknown"


# --------------------------------------------------------------------------- #
# F9/#147 instrument revision provenance (quality-module review 2026-08-19)
# --------------------------------------------------------------------------- #
# The four in-process HF checkpoints are pinned by repo NAME only -- a silent
# upstream repo update changes the instrument under the same provenance id.
# Each lazy load best-effort captures the RESOLVED HF commit hash; the optional
# CAGE_*_REVISION env pins turn a mismatch (or an unresolvable hash) into a
# standard fail-closed LOAD failure. Unpinned runs are record-only.
_REVISION_PIN_ENVS: Dict[str, str] = {
    "nli": "CAGE_NLI_REVISION",
    "embedding": "CAGE_EMBEDDING_REVISION",
    "bertscore": "CAGE_BERTSCORE_REVISION",
    "lettucedetect": "CAGE_LETTUCEDETECT_REVISION",
}


def _hf_commit_hash(*candidates: Any) -> Optional[str]:
    """Best-effort resolved HF commit hash from loaded-model internals.

    transformers stamps the resolved repo revision on every loaded config as
    ``config._commit_hash``; each candidate is a model-like object whose
    ``.config`` is walked (a config object itself also works). Provenance
    METADATA, not a score: never raises -- unresolvable internals (test stubs,
    local checkpoints, older library versions) record ``None``. The fail-closed
    path is the CAGE_*_REVISION pin check, which treats pinned+None as a load
    failure; unpinned runs record-only.
    """
    for obj in candidates:
        if obj is None:
            continue
        try:
            cfg = getattr(obj, "config", obj)
            rev = getattr(cfg, "_commit_hash", None)
        except Exception:
            continue
        if isinstance(rev, str) and rev.strip():
            return rev
    return None


def _split_sentences(text: str) -> List[str]:
    """Sentence-level split shared by claim decomposition and window building.

    Dependency-free splitter: breaks on sentence terminators and newlines.
    Short answers (no terminator) are returned as a single sentence.
    """
    if not text or not text.strip():
        return []
    # Split on ., !, ? followed by whitespace, and on newlines/semicolons.
    parts = re.split(r"(?<=[.!?])\s+|\n+|;\s+", text.strip())
    claims = [p.strip() for p in parts if p and p.strip()]
    return claims or [text.strip()]


# --------------------------------------------------------------------------- #
# Claim-decomposition seam (D8 §8.5: "entity-resolved atomic decomposition")
# --------------------------------------------------------------------------- #
class ClaimDecomposer(Protocol):
    """Protocol for splitting a generated answer into atomic claims.

    D8 §8.5 pre-registers ENTITY-RESOLVED atomic decomposition as the binding
    claim-pipeline protocol. This seam lets that decomposer plug into
    QualityEvaluator (constructor arg ``claim_decomposer``) without touching any
    scoring code; the default remains the sentence splitter the pilot ran with.
    """

    def decompose(self, text: str) -> List[str]:
        """Return the ordered list of atomic claims in ``text``."""
        ...


class SentenceClaimDecomposer:
    """Default ClaimDecomposer: sentence-level split (the pre-seam behavior).

    Identical to the historical ``QualityEvaluator._split_claims`` output, so the
    default scored behavior is unchanged. The entity-resolved decomposer required
    by D8 §8.5 replaces this class at calibration time via the constructor seam.
    """

    def decompose(self, text: str) -> List[str]:
        return _split_sentences(text)


# --------------------------------------------------------------------------- #
# Instrument B seams (D8 §8.5): trained claim checkers, fail-closed lazy load
# --------------------------------------------------------------------------- #
# The charter pre-registers "MiniCheck vs AlignScore, selection ... decided at
# calibration" with generic DeBERTa-MNLI demoted to legacy fallback. These
# classes are the SEAMS: lazily loaded, fail-closed (InstrumentUnavailableError
# when the package is absent -- never a silent skip), selected via the
# CAGE_CLAIM_CHECKER env var. DEFAULT = "nli" since 2026-08-19 (owner decision
# #120/F8: the default must be in-process-loadable so as-scripted preflight and
# rescore paths work; 'alignscore' by design cannot load in this venv and is
# requested EXPLICITLY by its dedicated out-of-process runner,
# scripts/4_analysis/score_instrument_b.py). Instrument selection itself is
# unchanged (2026-08-05 owner decision,
# MyDocs/registration/instrument_selection_2026-08-05/DECISION.md;
# charter stamp PUBLICATION.md §8.6(c)): AlignScore-large won the selection
# calibration. Its 2023 stack can NEVER install into the project venv, so the
# in-process default fails closed and every unavailability message points to
# the sanctioned out-of-process runner scripts/4_analysis/score_instrument_b.py.
# Dependencies are NOT added here; the isolated env is managed by
# src/evaluation/instrument_b_runner.py.
class MiniCheckClaimChecker:
    """MiniCheck claim checker (Instrument B candidate).

    Citation: Tang, Laban & Durrett 2024, "MiniCheck: Efficient Fact-Checking of
    LLMs on Grounding Documents" (EMNLP 2024, arXiv:2404.10774). Scores
    P(claim supported by doc) per (doc, claim) pair.

    Constructor kwargs are VERIFY-LIVE against the pinned minicheck release at
    calibration time; the model id is env-pinned via CAGE_MINICHECK_MODEL.
    """

    name: str = "minicheck"

    def __init__(
        self, model_name: Optional[str] = None, device: str | int = "cpu"
    ) -> None:
        self.model_name = model_name or os.getenv(
            "CAGE_MINICHECK_MODEL", "flan-t5-large"
        )
        self.device = device
        self._scorer: Optional[Any] = None

    @property
    def instrument_id(self) -> str:
        """Instrument id+version per D8 §8.1 row provenance."""
        return f"minicheck:{self.model_name}@minicheck-{_package_version('minicheck')}"

    def _load(self) -> Any:
        """Lazy fail-closed load: absent package raises, never degrades."""
        if self._scorer is None:
            try:
                from minicheck.minicheck import MiniCheck
            except Exception as e:  # ImportError and transitive load failures
                raise InstrumentUnavailableError(
                    "claim_checker", self.instrument_id,
                    f"minicheck package unavailable: {e}",
                ) from e
            self._scorer = MiniCheck(model_name=self.model_name)
        return self._scorer

    def score_pairs(self, pairs: List[Tuple[str, str]]) -> List[Optional[float]]:
        """Support probability per (premise, claim) pair, order-aligned."""
        scorer = self._load()
        docs = [p for p, _ in pairs]
        claims = [c for _, c in pairs]
        # MiniCheck.score returns (pred_labels, raw_probs, span_logits, spans);
        # raw_probs is the calibrated support probability (Tang et al. 2024 §3).
        _, raw_probs, _, _ = scorer.score(docs=docs, claims=claims)
        return [float(p) for p in raw_probs]


class AlignScoreClaimChecker:
    """AlignScore claim checker (Instrument B, SELECTED 2026-08-05).

    Citation: Zha, Yang, Li & Hu 2023, "AlignScore: Evaluating Factual
    Consistency with a Unified Alignment Function" (ACL 2023, arXiv:2305.16739).
    Scores alignment(context, claim) in [0, 1].

    Requires a trained checkpoint: CAGE_ALIGNSCORE_CKPT (fail-closed when
    unset -- an unpinned checkpoint is a missing instrument, not a default).
    Model architecture is env-pinned via CAGE_ALIGNSCORE_MODEL.
    """

    name: str = "alignscore"

    def __init__(
        self,
        model_name: Optional[str] = None,
        ckpt_path: Optional[str] = None,
        device: str | int = "cpu",
    ) -> None:
        self.model_name = model_name or os.getenv(
            "CAGE_ALIGNSCORE_MODEL", "roberta-large"
        )
        self.ckpt_path = ckpt_path or os.getenv("CAGE_ALIGNSCORE_CKPT") or None
        self.device = device
        self._scorer: Optional[Any] = None

    @property
    def instrument_id(self) -> str:
        """Instrument id+version per D8 §8.1 row provenance."""
        return (
            f"alignscore:{self.model_name}@alignscore-{_package_version('alignscore')}"
        )

    def _device_str(self) -> str:
        if isinstance(self.device, int):
            return f"cuda:{self.device}" if self.device >= 0 else "cpu"
        return str(self.device)

    def _load(self) -> Any:
        """Lazy fail-closed load: absent package OR unpinned checkpoint raises."""
        if self._scorer is None:
            if not self.ckpt_path:
                raise InstrumentUnavailableError(
                    "claim_checker", self.instrument_id,
                    "AlignScore checkpoint not configured (set CAGE_ALIGNSCORE_CKPT); "
                    "the sanctioned execution path is the out-of-process runner: "
                    "scripts/4_analysis/score_instrument_b.py",
                )
            try:
                from alignscore import AlignScore
            except Exception as e:
                raise InstrumentUnavailableError(
                    "claim_checker", self.instrument_id,
                    f"alignscore package unavailable: {e}; it can never install "
                    "into the project venv (2023 dependency stack) -- score via "
                    "the out-of-process runner: "
                    "scripts/4_analysis/score_instrument_b.py",
                ) from e
            # evaluation_mode='nli_sp' is the paper's headline configuration
            # (Zha et al. 2023 §4: sentence-splitting + NLI-style aggregation).
            self._scorer = AlignScore(
                model=self.model_name,
                batch_size=32,
                device=self._device_str(),
                ckpt_path=self.ckpt_path,
                evaluation_mode="nli_sp",
            )
        return self._scorer

    def score_pairs(self, pairs: List[Tuple[str, str]]) -> List[Optional[float]]:
        """Alignment score per (premise, claim) pair, order-aligned."""
        scorer = self._load()
        contexts = [p for p, _ in pairs]
        claims = [c for _, c in pairs]
        scores = scorer.score(contexts=contexts, claims=claims)
        return [float(s) for s in scores]


# Valid CAGE_CLAIM_CHECKER values. DEFAULT = 'nli' (flipped 2026-08-19, owner
# decision #120/F8 — in-process-safe; 'alignscore' is requested explicitly by
# scripts/4_analysis/score_instrument_b.py, the Instrument-B runner); 'nli'
# (DeBERTa-MNLI path) and
# 'minicheck' (documented runner-up) remain selectable.
_CLAIM_CHECKER_NAMES: Tuple[str, ...] = ("nli", "minicheck", "alignscore")


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


# --------------------------------------------------------------------------- #
# Answer sanitizer (B4 scoring side, 2026-07-16 pre-run package)
# --------------------------------------------------------------------------- #
# Few-shot / QA-formatted prompts make models emit a scaffold prefix ("A: Paris",
# "Answer: Paris") and, on runaway generations, a fabricated continuation of the prompt
# template ("...\nQuestion 2: ..." / "Context: ..."). Both are prompt-format artifacts,
# not answer content: the prefix deflates EM/token-F1 against short gold spans, and the
# fabricated continuation is un-grounded text that LettuceDetect/NLI correctly flag --
# penalizing the SERVING arm for a PROMPT artifact. Scoring therefore runs on the
# sanitized text; the raw generation is never overwritten (stored separately as
# generated_answer, with sanitized_answer alongside).
_ANSWER_SCAFFOLD_PREFIX_RE = re.compile(r"^\s*(A|Answer)\s*[:.]\s*")
_FABRICATED_CONTINUATION_RE = re.compile(r"\n?\s*(Context|Question)\s*\d*\s*:")


def sanitize_answer(text: Optional[str]) -> str:
    """Strip a leading answer-scaffold token and truncate fabricated continuations.

    1. Removes ONE leading scaffold token ('A:', 'Answer:', 'A.', 'Answer.').
    2. Truncates at the first fabricated prompt-template continuation
       ('Context:', 'Question 2:', ...), which is model runaway, not answer text.

    Never applied destructively: callers keep the raw generation and store this result
    as ``sanitized_answer`` alongside it. ALL quality scoring (grounding, NLI,
    completeness, F1/EM, abstention detection) runs on the sanitized text.
    """
    t = text or ""
    t = _ANSWER_SCAFFOLD_PREFIX_RE.sub("", t, count=1)
    m = _FABRICATED_CONTINUATION_RE.search(t)
    if m:
        t = t[: m.start()]
    return t.strip()


@dataclass
class CacheRelevanceMetrics:
    """Cache relevance evaluation results."""

    cache_relevance: float  # 0-1, proportion of cache blocks that contributed
    relevant_block_count: int  # Number of blocks that contributed
    total_block_count: int  # Total number of cache blocks accessed
    per_block_scores: List[float]  # Relevance score for each block
    # P0-5: which scorer produced the numbers -- 'embedding:<model>' or the
    # explicitly-labeled 'lexical_jaccard' substitute ('none' when no blocks).
    # This diagnostic keeps its fallback ONLY because the method is written
    # into the row; the score never masquerades as the embedding scorer's.
    method: str = "embedding"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "cache_relevance": self.cache_relevance,
            "relevant_block_count": self.relevant_block_count,
            "total_block_count": self.total_block_count,
            "per_block_scores": self.per_block_scores,
            "cache_relevance_method": self.method,
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
    # Provenance of the faithfulness number: None when faithfulness was not scored
    # (abstention/unavailable), 'nli_claim_max' when it was.
    faithfulness_method: Optional[str] = None
    # B2 (2026-07-16 audit): 'direct' = every context doc fit the NLI premise budget as-is;
    # 'windowed' = at least one doc was split into sentence-aligned <=400-token windows and
    # scored max-over-windows. None when faithfulness was not scored (abstention/unavailable).
    faithfulness_premise_mode: Optional[str] = None
    # Review-fix: number of NLI premises (context docs x windows) the per-claim max was
    # taken over. mean-of-per-claim-max is premise-count-sensitive (more premises -> higher
    # expected max by chance alone, independent of true faithfulness), so this lets downstream
    # analysis condition on / covariate-adjust for premise count when comparing faithfulness
    # across arms with structurally different context sizes (gold-* vs retr-* vs corpus-*).
    # None when faithfulness was not scored (abstention/unavailable).
    faithfulness_premise_count: Optional[int] = None
    # B4: the scaffold-stripped / continuation-truncated text ALL quality scoring ran on.
    # generated_answer (the raw text) is never overwritten; this column sits alongside it.
    sanitized_answer: Optional[str] = None
    # P0-5 fail-closed: 'ok', or ';'-joined tokens ('nli:unavailable:<model>',
    # 'bertscore:error:<ExcType>') for every pinned instrument consulted on THIS row
    # that failed to produce a score. A missing score is never a substitute's score.
    instrument_status: str = "ok"
    # ---- D8 §8.1 per-row instrument provenance (emitted unconditionally) ---- #
    # True when Instrument A (LettuceDetect) scored this row via the D8 §8.5
    # windowed max-support pass because the context exceeded the detector's
    # runtime-derived L_max. Windowing is an ALERT, never silent: the flag rides
    # every affected row. (NLI premise windowing has its own flag:
    # faithfulness_premise_mode.) False when scored natively or not scored.
    scored_windowed: bool = False
    # Instrument id+version of the grounding detector consulted for this row
    # ('<model>@lettucedetect-<pkg-version>'); None when not consulted
    # (config-off / abstention / empty inputs).
    grounding_instrument: Optional[str] = None
    # Instrument id+version of the claim checker consulted for faithfulness
    # ('<model>@transformers-<v>' for the default NLI path, or the
    # MiniCheck/AlignScore instrument_id); None when not consulted.
    faithfulness_instrument: Optional[str] = None
    # D9 calibration-manifest id these instrument thresholds were frozen under.
    # None until the D9 §9.7 calibration exists (pre-registration gate).
    calibration_id: Optional[str] = None
    # Whether BERTScore IDF weighting (Zhang et al. 2020 §3; D8 §8.4) was active
    # for this row's completeness_bertscore. None when BERTScore did not score.
    bertscore_idf: Optional[bool] = None
    # Labeled diagnostic provenance for cache_relevance (see CacheRelevanceMetrics.method).
    cache_relevance_method: Optional[str] = None
    # Evidence-only, NOT a metric: raw LettuceDetect answer spans flagged unsupported, for the
    # per-query qa_evidence.jsonl. Deliberately EXCLUDED from to_dict() so it never becomes a
    # CSV column or enters metric aggregation. None when the detector is unavailable.
    hallucinated_spans: Optional[List[Dict[str, Any]]] = None
    # ---- D8 §8.5 3-class NLI reporting (CAGE_NLI_THREE_CLASS, default OFF) ---- #
    # Charter: "contradiction and neutral reported separately (misread evidence vs
    # invented claim — different bugs)". Per-claim aggregation mirrors faithfulness:
    # MAX over premises per class (2026-08-04 review, quality L3 row), then mean over
    # claims. MNLI 3-class scheme: entailment/neutral/contradiction (Williams,
    # Nangia & Bowman, NAACL 2018). None when not scored (flag off / abstention /
    # NLI unavailable / non-NLI claim checker).
    faithfulness_contradiction: Optional[float] = None
    faithfulness_neutral: Optional[float] = None
    # Column-emission switch, set by the evaluator from CAGE_NLI_THREE_CLASS.
    # False (the default) keeps to_dict() byte-identical to the pre-flag output;
    # True emits both columns unconditionally (stable CSV columns within a
    # flagged run, None on unscored rows). Not itself a metric column.
    nli_three_class: bool = False

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
            "faithfulness_premise_mode": self.faithfulness_premise_mode,
            "faithfulness_premise_count": self.faithfulness_premise_count,
            "sanitized_answer": self.sanitized_answer,
            # Emitted unconditionally so the CSV column is stable across rows/runs.
            "instrument_status": self.instrument_status,
            # D8 §8.1 provenance -- ALL emitted unconditionally (stable columns).
            # scored_windowed as 0/1 so it aggregates to a windowed-row RATE
            # (the D8 §8.5 alert surface), like hallucination_detected.
            "scored_windowed": 1.0 if self.scored_windowed else 0.0,
            "grounding_instrument": self.grounding_instrument,
            "faithfulness_instrument": self.faithfulness_instrument,
            "calibration_id": self.calibration_id,
            "bertscore_idf": self.bertscore_idf,
        }
        if self.nli_three_class:
            # D8 §8.5 flag-gated columns: emitted for EVERY row of a flagged run
            # (stable CSV columns; None on unscored rows, excluded from means).
            # Flag off: keys absent — output byte-identical to pre-flag behavior.
            result["faithfulness_contradiction"] = self.faithfulness_contradiction
            result["faithfulness_neutral"] = self.faithfulness_neutral
        if self.hallucination_detected is not None:
            # Stored as 0/1 so it aggregates to a hallucination RATE across a run.
            result["hallucination_detected"] = 1.0 if self.hallucination_detected else 0.0
        if self.cache_relevance is not None:
            result["cache_relevance"] = self.cache_relevance
            if self.cache_relevance_method is not None:
                result["cache_relevance_method"] = self.cache_relevance_method
        return result


class NLIProbs(NamedTuple):
    """One (premise, hypothesis) pair's parsed NLI class probabilities.

    MNLI 3-class scheme (Williams, Nangia & Bowman, NAACL 2018):
    entailment / neutral / contradiction. ``entailment`` is always resolved
    (same fail-closed contract as the historical scalar parse); ``neutral``
    and ``contradiction`` are resolved only when 3-class reporting is active
    (CAGE_NLI_THREE_CLASS — D8 §8.5 "contradiction is not neutral") and are
    ``None`` otherwise or when the class label is unresolvable.
    """

    entailment: float
    neutral: Optional[float] = None
    contradiction: Optional[float] = None


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
        strict: Optional[bool] = None,
        bertscore_idf: Optional[bool] = None,
        bertscore_idf_sents: Optional[List[str]] = None,
        claim_checker: Optional[str] = None,
        claim_decomposer: Optional[ClaimDecomposer] = None,
        calibration_id: Optional[str] = None,
        nli_three_class: Optional[bool] = None,
    ) -> None:
        # P0-5 fail-closed switch (D8 default: strict). strict=None reads
        # CAGE_QUALITY_STRICT (unset/"1" -> True). Strict raises
        # InstrumentUnavailableError on any instrument load/score failure;
        # non-strict records score=None + an instrument_status token instead.
        if strict is None:
            strict = os.getenv("CAGE_QUALITY_STRICT", "1").strip().lower() not in {
                "0", "false", "no",
            }
        self.strict = strict
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
        # D8 §8.4 BERTScore IDF weighting (Zhang et al. 2020 §3: rare-word
        # importance weighting "where references share boilerplate"). DEFAULT
        # OFF, matching the pre-existing scored behavior; env-gated on via
        # CAGE_BERTSCORE_IDF=1 (constructor arg wins when not None). When ON,
        # bertscore_idf_sents (the dataset's reference-answer corpus, computed
        # once per dataset/cell by the caller) is REQUIRED -- enabling IDF with
        # no corpus is a missing instrument, not a silent no-op (fail-closed).
        if bertscore_idf is None:
            bertscore_idf = os.getenv("CAGE_BERTSCORE_IDF", "").strip().lower() in {
                "1", "true", "yes",
            }
        self.bertscore_idf = bool(bertscore_idf)
        self.bertscore_idf_sents = (
            list(bertscore_idf_sents) if bertscore_idf_sents is not None else None
        )
        # D8 §8.5 3-class NLI reporting ("contradiction and neutral reported
        # separately — misread evidence vs invented claim, different bugs").
        # DEFAULT OFF: scored behavior and to_dict() output stay byte-identical
        # to the pre-flag pipeline. Env-gated on via CAGE_NLI_THREE_CLASS=1
        # (constructor arg wins when not None), same pattern as bertscore_idf.
        # When ON, per-row faithfulness_contradiction / faithfulness_neutral
        # columns are emitted (NLI path only; the Instrument B checkers return
        # a single support probability, so their rows carry None).
        if nli_three_class is None:
            nli_three_class = os.getenv(
                "CAGE_NLI_THREE_CLASS", ""
            ).strip().lower() in {"1", "true", "yes"}
        self.nli_three_class = bool(nli_three_class)
        # D8 §8.5 Instrument B seam: which claim checker scores faithfulness.
        # DEFAULT 'nli' -- in-process-safe default since the 2026-08-19 owner
        # decision #120/F8 (as-scripted preflight and rescore paths must be
        # loadable in-process). Instrument SELECTION is unchanged (2026-08-05
        # owner decision, MyDocs/registration/instrument_selection_2026-08-05/
        # DECISION.md; charter stamp PUBLICATION.md §8.6(c)): AlignScore-large
        # remains the selected Instrument B (pooled AUC 0.8275 vs MiniCheck
        # 0.7734) and is requested EXPLICITLY by its out-of-process runner,
        # scripts/4_analysis/score_instrument_b.py -- its 2023 dependency stack
        # can never install into this venv, so 'alignscore' here fails closed
        # with a message pointing at that runner. 'minicheck' (documented
        # runner-up) remains selectable; the DeBERTa continuous-faithfulness
        # columns are a separate charter role and are untouched. Invalid names
        # raise (fail-closed).
        checker_name = (
            claim_checker
            or os.getenv("CAGE_CLAIM_CHECKER", "nli")
            or "nli"
        ).strip().lower()
        if checker_name not in _CLAIM_CHECKER_NAMES:
            raise ValueError(
                f"CAGE_CLAIM_CHECKER={checker_name!r} is not one of "
                f"{_CLAIM_CHECKER_NAMES} (fail-closed: refusing to guess)"
            )
        self.claim_checker_name = checker_name
        self._claim_checker: Optional[Any] = None
        if checker_name == "minicheck":
            self._claim_checker = MiniCheckClaimChecker(device=device)
        elif checker_name == "alignscore":
            self._claim_checker = AlignScoreClaimChecker(device=device)
        # D8 §8.5 claim-decomposition seam: default = sentence splitting (the
        # pre-seam behavior); the entity-resolved decomposer plugs in here.
        self.claim_decomposer: ClaimDecomposer = (
            claim_decomposer if claim_decomposer is not None else SentenceClaimDecomposer()
        )
        # D8 §8.1 calibration id: None until the D9 §9.7 calibration manifest
        # exists; threaded onto every row for provenance.
        self.calibration_id: Optional[str] = (
            calibration_id or os.getenv("CAGE_CALIBRATION_ID") or None
        )

        # Allow override via env vars or constructor args.
        self.nli_model_name = (
            nli_model_name
            or os.getenv("CAGE_NLI_MODEL", "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli")
        )
        # P0-5: fallback SUBSTITUTION is removed -- the attributes survive for API
        # compat but are never consulted. A configured fallback env var gets a loud
        # warning so operators learn the chain is gone before a multi-day run.
        for _env_key in ("CAGE_NLI_FALLBACKS", "CAGE_BERTSCORE_FALLBACKS"):
            if os.getenv(_env_key, "").strip():
                print(
                    f"Warning: {_env_key} is set but fallback substitution was removed "
                    f"(P0-5 fail-closed); pin the instrument via its primary env var."
                )
        self.nli_model_fallbacks: List[str] = []
        self.embedding_model_name = (
            embedding_model_name
            or os.getenv("CAGE_EMBEDDING_MODEL", "all-MiniLM-L6-v2")
        )
        self.bertscore_model_name = (
            bertscore_model_name
            or os.getenv("CAGE_BERTSCORE_MODEL", "roberta-base")
        )
        self.bertscore_model_fallbacks: List[str] = []
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
        # Resolved neutral/contradiction class indices (D8 §8.5 3-class seam);
        # populated lazily per class name, mirroring _nli_entail_index.
        self._nli_class_index: Dict[str, Optional[int]] = {}
        self._embedding_model = None
        self._bertscore_model = None
        self._rouge_scorer = None
        self._lettucedetect_model = None
        # P0-5 sticky unavailability: instrument -> pinned model that failed to load.
        # Short-circuits reload attempts (the old code retried the full model load on
        # EVERY row) and drives the 'unavailable:' row-status tokens.
        self._instrument_unavailable: Dict[str, str] = {}
        # F9/#147 revision provenance: instrument -> resolved HF commit hash,
        # captured best-effort at each lazy load (None when unresolvable or not
        # yet loaded). Read via instrument_provenance(); never a row column.
        self._instrument_revisions: Dict[str, Optional[str]] = {}
        # Optional fail-closed pins (CAGE_NLI_REVISION & co). Unset -> None ->
        # record-only, zero behavior change; set -> a resolved hash that differs
        # (or cannot be resolved) is a LOAD failure via _mark_unavailable.
        self._revision_pins: Dict[str, Optional[str]] = {
            inst: (os.getenv(env, "").strip() or None)
            for inst, env in _REVISION_PIN_ENVS.items()
        }
        # Per-row status tokens; reset at the top of evaluate().
        self._row_status_tokens: List[str] = []
        # One-shot console alert for the D8 §8.5 windowed grounding pass (the
        # per-row scored_windowed flag is the durable alert; this is a courtesy).
        self._windowed_alert_emitted = False

    # ------------------------------------------------------------------ #
    # P0-5 fail-closed plumbing
    # ------------------------------------------------------------------ #
    def _mark_unavailable(self, instrument: str, model: str, cause: Exception) -> None:
        """Register a permanent instrument LOAD failure (strict mode: raise)."""
        if self.strict:
            raise InstrumentUnavailableError(instrument, model, str(cause)) from cause
        if instrument not in self._instrument_unavailable:
            print(
                f"Warning: quality instrument '{instrument}' ({model}) unavailable: {cause}. "
                f"Rows will carry instrument_status '{instrument}:unavailable:{model}' with "
                f"score=None -- never a substitute model's score."
            )
            self._instrument_unavailable[instrument] = model

    def _enforce_revision_pin(
        self, instrument: str, model: str, model_attr: str
    ) -> None:
        """F9/#147 optional fail-closed revision pin (CAGE_*_REVISION).

        Called immediately after a successful lazy load, once the resolved HF
        commit hash has been recorded in ``_instrument_revisions``. With the
        pin env unset this is a no-op (record-only). With it set, a resolved
        hash that differs from the pin -- or one that could not be resolved --
        is treated as a LOAD failure: the just-loaded model is discarded
        (never score with an unpinned instrument) and the standard
        _mark_unavailable machinery fires (strict raises
        InstrumentUnavailableError; non-strict rows carry the usual
        'X:unavailable:<model>' token).
        """
        pinned = self._revision_pins.get(instrument)
        if not pinned:
            return
        resolved = self._instrument_revisions.get(instrument)
        if resolved == pinned:
            return
        setattr(self, model_attr, None)  # discard BEFORE the strict raise
        detail = (
            f"loaded checkpoint resolved to revision {resolved!r}"
            if resolved
            else "loaded checkpoint's revision could not be resolved"
        )
        self._mark_unavailable(
            instrument,
            model,
            ValueError(
                f"revision pin {_REVISION_PIN_ENVS[instrument]}={pinned} "
                f"failed: {detail} (fail-closed: a silent upstream repo "
                f"update would change the instrument under the same "
                f"provenance id)"
            ),
        )

    def _note_row_status(self, instrument: str, kind: str, detail: str) -> None:
        self._row_status_tokens.append(f"{instrument}:{kind}:{detail}")

    def _note_unavailable_row(self, instrument: str) -> None:
        self._note_row_status(
            instrument, "unavailable", self._instrument_unavailable[instrument]
        )

    def _scoring_failure(self, instrument: str, model: str, exc: Exception) -> None:
        """Per-row SCORING failure: strict raises, non-strict labels the row."""
        if self.strict:
            raise InstrumentUnavailableError(
                instrument, model, f"scoring failed: {exc}"
            ) from exc
        print(f"Error in {instrument} scoring ({model}): {exc}")
        self._note_row_status(instrument, "error", type(exc).__name__)

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
    def nli_model(self) -> Optional[Any]:
        """Lazy load the PINNED NLI model (no fallback substitution -- P0-5)."""
        if (
            self._nli_model is None
            and self.use_nli
            and "nli" not in self._instrument_unavailable
        ):
            try:
                from transformers import pipeline

                self._nli_model = pipeline(
                    "text-classification",
                    model=self.nli_model_name,
                    device=self._hf_pipeline_device(),
                )
            except Exception as e:
                self._mark_unavailable("nli", self.nli_model_name, e)
            else:
                # F9/#147: capture the resolved HF commit hash (record-only
                # unless CAGE_NLI_REVISION pins it -- then fail-closed).
                self._instrument_revisions["nli"] = _hf_commit_hash(
                    getattr(self._nli_model, "model", None)
                )
                self._enforce_revision_pin("nli", self.nli_model_name, "_nli_model")
        return self._nli_model

    @property
    def embedding_model(self) -> Optional[Any]:
        """Lazy load the PINNED embedding model for relevance."""
        if (
            self._embedding_model is None
            and self.use_embeddings
            and "embedding" not in self._instrument_unavailable
        ):
            try:
                from sentence_transformers import SentenceTransformer

                self._embedding_model = SentenceTransformer(
                    self.embedding_model_name,
                    device=self.device,
                )
            except Exception as e:
                self._mark_unavailable("embedding", self.embedding_model_name, e)
            else:
                # F9/#147: sentence-transformers wraps the HF checkpoint in its
                # first module's auto_model; walk there for the resolved hash.
                try:
                    first = self._embedding_model._first_module()
                except Exception:
                    first = None
                self._instrument_revisions["embedding"] = _hf_commit_hash(
                    getattr(first, "auto_model", None)
                )
                self._enforce_revision_pin(
                    "embedding", self.embedding_model_name, "_embedding_model"
                )
        return self._embedding_model

    @property
    def bertscore_model(self) -> Optional[Any]:
        """Lazy load BERTScore for the PINNED model only (no fallback chain -- P0-5)."""
        if (
            self._bertscore_model is None
            and self.use_bertscore
            and "bertscore" not in self._instrument_unavailable
        ):
            self._load_bertscore_model()
        return self._bertscore_model

    def _probe_bertscore_model(self, scorer: Any) -> None:
        """Run a tiny score call to catch models that load but fail at inference time."""
        _, _, f1 = scorer.score(["cage sanity check"], ["cage sanity check"])
        _ = float(f1[0].cpu().numpy())

    def _load_bertscore_model(self) -> Optional[Any]:
        """Load the pinned BERTScore model, probing one real score call.

        P0-5: no fallback model and no silent unrescaled retry. Baseline rescaling
        is part of the pinned instrument definition (raw RoBERTa F1 sits in a
        compressed ~0.3 band; §8.4 mandates rescaling), so losing it mid-run would
        change the metric's SCALE under the same column name. Any failure marks
        the instrument unavailable (strict: raises).

        D8 §8.4 IDF weighting (Zhang et al. 2020 §3): when ``bertscore_idf`` is
        on, the scorer precomputes IDF over ``bertscore_idf_sents`` (the
        dataset's reference-answer corpus). IDF-on with no corpus fails CLOSED
        here -- bert-score would otherwise error only at score time, or worse,
        a silent no-op would score under the same column as the IDF-weighted
        metric. IDF-off constructs the scorer EXACTLY as before (no new kwargs).
        """
        try:
            from bert_score import BERTScorer

            if self.bertscore_idf and not self.bertscore_idf_sents:
                raise ValueError(
                    "BERTScore IDF weighting enabled (CAGE_BERTSCORE_IDF/"
                    "bertscore_idf) but no bertscore_idf_sents reference corpus "
                    "was provided (required to precompute IDF; D8 §8.4)"
                )
            idf_kwargs: Dict[str, Any] = {}
            if self.bertscore_idf:
                idf_kwargs = {"idf": True, "idf_sents": list(self.bertscore_idf_sents or [])}
            scorer = BERTScorer(
                model_type=self.bertscore_model_name,
                device=self.device,
                lang=self.bertscore_lang,
                rescale_with_baseline=self.bertscore_rescale_with_baseline,
                **idf_kwargs,
            )
            self._probe_bertscore_model(scorer)
        except Exception as e:
            self._mark_unavailable("bertscore", self.bertscore_model_name, e)
            return None
        self._bertscore_model = scorer
        # F9/#147: BERTScorer keeps the underlying transformers model on
        # ``_model``; capture its resolved hash, then enforce any pin.
        self._instrument_revisions["bertscore"] = _hf_commit_hash(
            getattr(scorer, "_model", None)
        )
        self._enforce_revision_pin(
            "bertscore", self.bertscore_model_name, "_bertscore_model"
        )
        return self._bertscore_model

    @property
    def rouge_scorer(self) -> Optional[Any]:
        """Lazy load ROUGE scorer ('rouge_score' is the pinned implementation)."""
        if (
            self._rouge_scorer is None
            and self.use_rouge
            and "rouge" not in self._instrument_unavailable
        ):
            try:
                from rouge_score import rouge_scorer

                self._rouge_scorer = rouge_scorer.RougeScorer(
                    ["rouge1", "rouge2", "rougeL"],
                    use_stemmer=True,
                )
            except Exception as e:
                self._mark_unavailable("rouge", "rouge_score", e)
        return self._rouge_scorer

    @property
    def lettucedetect_model(self) -> Optional[Any]:
        """Lazy load the PINNED LettuceDetect detector (PRIMARY grounding signal)."""
        if (
            self._lettucedetect_model is None
            and self.use_lettucedetect
            and "lettucedetect" not in self._instrument_unavailable
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
                self._mark_unavailable(
                    "lettucedetect", self.lettucedetect_model_name, e
                )
            else:
                # F9/#147: getattr-walk the detector internals (the transformer
                # method nests the HF model at detector.model; tolerate either
                # nesting depth -- unresolvable records None).
                det = self._lettucedetect_model
                self._instrument_revisions["lettucedetect"] = _hf_commit_hash(
                    getattr(getattr(det, "detector", None), "model", None),
                    getattr(det, "model", None),
                )
                self._enforce_revision_pin(
                    "lettucedetect",
                    self.lettucedetect_model_name,
                    "_lettucedetect_model",
                )
        return self._lettucedetect_model

    @staticmethod
    def _split_claims(text: str) -> List[str]:
        """Split an answer into atomic claims (sentence-level).

        Back-compat wrapper over the module-level sentence splitter. Claim
        DECOMPOSITION now routes through ``self.claim_decomposer`` (D8 §8.5
        seam); window construction keeps using sentence splitting regardless of
        the configured decomposer (windows are sentence-ALIGNED by definition).
        """
        return _split_sentences(text)

    # ------------------------------------------------------------------ #
    # D8 §8.1 per-row instrument provenance ids
    # ------------------------------------------------------------------ #
    def instrument_provenance(self) -> Dict[str, Dict[str, Optional[str]]]:
        """F9/#147: instrument -> {model, revision} for the four HF checkpoints.

        ``model`` is the resolved constructor-time repo id (env overrides
        applied); ``revision`` is the HF commit hash captured best-effort at
        lazy-load time -- ``None`` until that instrument has actually loaded,
        or when the hash is unresolvable (test stubs, local paths). Manifest
        METADATA only: no row column is derived from this mapping, and the
        _grounding_instrument_id / _faithfulness_instrument_id row strings are
        deliberately untouched (their formats are pinned by tests).
        """
        return {
            inst: {"model": model, "revision": self._instrument_revisions.get(inst)}
            for inst, model in (
                ("nli", self.nli_model_name),
                ("embedding", self.embedding_model_name),
                ("bertscore", self.bertscore_model_name),
                ("lettucedetect", self.lettucedetect_model_name),
            )
        }

    def _grounding_instrument_id(self) -> str:
        """Instrument A id+version: '<model>@lettucedetect-<pkg-version>'."""
        return (
            f"{self.lettucedetect_model_name}"
            f"@lettucedetect-{_package_version('lettucedetect')}"
        )

    def _faithfulness_instrument_id(self) -> str:
        """Active claim-checker id+version (NLI default or Instrument B seam)."""
        if self._claim_checker is not None:
            return str(self._claim_checker.instrument_id)
        return f"{self.nli_model_name}@transformers-{_package_version('transformers')}"

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

    def _resolve_nli_class_index(self, class_key: str) -> Optional[int]:
        """Resolve a non-entailment NLI class index (``'neutral'`` /
        ``'contradi'`` substring) from the loaded model's id2label config.

        3-class seam companion to :meth:`_resolve_nli_entail_index` (which
        stays untouched: entailment resolution is gate-critical and its
        behavior must remain byte-identical). Cached per class key; ``None``
        when unresolvable — the optional D8 §8.5 reporting columns then stay
        ``None`` instead of failing the faithfulness gate.
        """
        if class_key in self._nli_class_index:
            return self._nli_class_index[class_key]
        idx: Optional[int] = None
        try:
            id2label = self.nli_model.model.config.id2label
            for i, label in id2label.items():
                if class_key in str(label).lower():
                    idx = int(i)
                    break
        except Exception:
            idx = None
        self._nli_class_index[class_key] = idx
        return idx

    def _nli_class_prob(
        self, by_label: Dict[str, float], class_key: str
    ) -> Optional[float]:
        """P(class) for a non-entailment class from one parsed label map:
        named label first, then LABEL_x via the model config; None when
        unresolvable (never raises — see _resolve_nli_class_index)."""
        for label, score in by_label.items():
            if class_key in label:
                return score
        idx = self._resolve_nli_class_index(class_key)
        if idx is not None:
            return by_label.get(f"label_{idx}")
        return None

    def _parse_nli_result(self, result: Any) -> Optional[NLIProbs]:
        """Parse one pipeline output into class probabilities (the SINGLE seam
        shared by the sequential and batched paths so both score identically).

        Entailment resolution is unchanged from the historical scalar parse:
        named 'entailment' label first, then LABEL_x via the model config;
        raises InstrumentUnavailableError when the entailment class is
        unresolvable (misconfiguration, not a per-input hiccup -- P0-5).
        D8 §8.5 extension: when ``nli_three_class`` is active, the neutral and
        contradiction probabilities (MNLI scheme -- Williams, Nangia & Bowman,
        NAACL 2018) are RETAINED on the returned :class:`NLIProbs` instead of
        being discarded; flag off leaves them ``None`` (zero extra work,
        scored behavior identical).
        """
        # transformers may nest the result as [[...]] for a single pair.
        if result and isinstance(result[0], list):
            result = result[0]
        if not result:
            return None
        by_label = {str(d.get("label", "")).lower(): float(d.get("score", 0.0)) for d in result}
        # Prefer a named 'entailment' class.
        entail: Optional[float] = None
        for label, score in by_label.items():
            if "entail" in label:
                entail = score
                break
        if entail is None:
            # Otherwise resolve LABEL_x via the model config.
            idx = self._resolve_nli_entail_index()
            if idx is None:
                # No entailment class resolvable: the pinned model cannot be
                # interpreted -- misconfiguration, not a per-input hiccup. Silently
                # skipping every claim here is exactly the fail-open path P0-5 bans.
                raise InstrumentUnavailableError(
                    "nli", self.nli_model_name,
                    "entailment class unresolvable from model labels/id2label",
                )
            entail = by_label.get(f"label_{idx}")
            if entail is None:
                # Historical contract: a resolvable index whose LABEL_x key is
                # absent from THIS output yields no score (None), not an error.
                return None
        if not self.nli_three_class:
            return NLIProbs(entailment=entail)
        return NLIProbs(
            entailment=entail,
            neutral=self._nli_class_prob(by_label, "neutral"),
            contradiction=self._nli_class_prob(by_label, "contradi"),
        )

    def _nli_pair_probs(self, premise: str, hypothesis: str) -> Optional[NLIProbs]:
        """Class probabilities for hypothesis given premise, as a proper sentence
        pair (sequential path; error handling identical to the historical
        _nli_entailment_prob)."""
        try:
            # Pass a PAIR (text/text_pair) so the model sees premise vs hypothesis
            # with correct segment encoding. top_k=None returns all class scores.
            result = self.nli_model(
                {"text": premise, "text_pair": hypothesis},
                top_k=None,
                truncation=True,
                max_length=self.nli_max_length,
            )
            return self._parse_nli_result(result)
        except InstrumentUnavailableError:
            if self.strict:
                raise
            self._note_row_status("nli", "error", "entailment-class-unresolved")
            return None
        except Exception as e:
            self._scoring_failure("nli", self.nli_model_name, e)
            return None

    def _nli_entailment_prob(self, premise: str, hypothesis: str) -> Optional[float]:
        """P(entailment) for hypothesis given premise (thin wrapper preserving
        the historical scalar contract over _nli_pair_probs)."""
        probs = self._nli_pair_probs(premise, hypothesis)
        return None if probs is None else probs.entailment

    def _three_class_result_fields(
        self, claim_contra: List[float], claim_neutral: List[float]
    ) -> Dict[str, Any]:
        """Row-level D8 §8.5 3-class values from per-claim maxima (shared by the
        sequential and batched faithfulness paths).

        Same claim aggregation as faithfulness: mean over claims of the
        per-claim MAX over premises. Returns {} when the flag is off so the
        default result dicts stay byte-identical; when on, values are None if
        no claim resolved that class (excluded from means downstream).
        """
        if not self.nli_three_class:
            return {}
        return {
            "contradiction": float(np.mean(claim_contra)) if claim_contra else None,
            "neutral": float(np.mean(claim_neutral)) if claim_neutral else None,
        }

    # B2 (2026-07-16 audit): premise budget for a single NLI window, in NLI-tokenizer
    # (DeBERTa) tokens. Any context doc longer than this is split into sentence-aligned
    # windows of <= this many tokens with ~50% overlap; entailment is the MAX over
    # windows (and, as before, over docs). Kept below nli_max_length=512 so the
    # premise+hypothesis pair never hits pipeline truncation.
    NLI_PREMISE_WINDOW_TOKENS = 400

    def _nli_tokenizer(self):
        """The loaded NLI pipeline's tokenizer, or None if unavailable."""
        try:
            return getattr(self.nli_model, "tokenizer", None)
        except Exception:
            return None

    def _split_premise_windows(self, doc: str) -> List[str]:
        """Split one context doc into sentence-aligned NLI premise windows.

        B2 fix (2026-07-16 audit): cag_true concatenates its corpus into a ~2.8k-token
        block; passed whole as the NLI premise it is TRUNCATED to the first
        ``nli_max_length`` tokens, so evidence past the truncation horizon can never
        entail a claim -- faithfulness collapsed to 0.107 even for in-window evidence.
        The premise must be shortened (paragraph-sized windows, max over windows), not
        the model window enlarged.

        Windows are <= NLI_PREMISE_WINDOW_TOKENS DeBERTa tokens with ~50% overlap
        (consecutive windows share roughly half their token mass so no evidence
        straddles a boundary unseen). Each sentence is tokenized ONCE; counts are
        reused when building every window. Docs that already fit are returned whole
        (premise_mode 'direct').
        """
        text = (doc or "").strip()
        if not text:
            return []
        sentences = _split_sentences(text)
        tokenizer = self._nli_tokenizer()
        counts = self._sentence_token_counts(sentences, tokenizer)
        return self._sentence_window_split(
            text, sentences, counts, self.NLI_PREMISE_WINDOW_TOKENS
        )

    @staticmethod
    def _sentence_token_counts(texts: List[str], tokenizer: Any) -> List[int]:
        """Token count per text via ONE batched tokenizer call (tokenize once).

        Whitespace fallback when no tokenizer is available or it errors: subword
        count >= word count for the model tokenizers in play, so windows only
        get smaller when the real tokenizer is present (conservative proxy).
        """
        if tokenizer is not None:
            try:
                enc = tokenizer(list(texts), add_special_tokens=False)
                return [len(ids) for ids in enc["input_ids"]]
            except Exception:
                pass
        return [len(t.split()) for t in texts]

    @staticmethod
    def _count_tokens_strict(texts: List[str], tokenizer: Any) -> List[int]:
        """Token count per text with NO silent fallback (LettuceDetect path).

        The D8 §8.5 windowing decision for Instrument A must run on the
        detector's OWN tokenizer counts -- a whitespace proxy could under-count
        and silently skip a REQUIRED windowed pass. Tokenizer errors propagate
        to the caller's typed-failure handling (fail-closed).
        """
        enc = tokenizer(list(texts), add_special_tokens=False)
        return [len(ids) for ids in enc["input_ids"]]

    @staticmethod
    def _sentence_window_split(
        text: str, sentences: List[str], counts: List[int], max_tokens: int
    ) -> List[str]:
        """Sentence-aligned ~50%-overlap windowing (shared core algorithm).

        Used by the NLI premise windows (B2) AND the LettuceDetect context
        windows (D8 §8.5). Behavior is identical to the original
        _split_premise_windows loop: windows are <= ``max_tokens`` by the given
        per-sentence ``counts``, consecutive windows share ~half their token
        mass, and a text that already fits (or is a single sentence) is
        returned whole.

        NOTE: a SINGLE sentence longer than ``max_tokens`` (terminator-free
        text) is emitted whole/over-budget by both the single-sentence
        early-out and the always-include-one rule below. Callers that must
        never rely on downstream truncation (the LettuceDetect D8 §8.5
        branch in ``evaluate_hallucination``) re-check every emitted window
        with the strict counter and hard-split via ``_hard_token_split``.
        """
        if sum(counts) <= max_tokens or len(sentences) == 1:
            return [text]

        windows: List[str] = []
        start = 0
        n = len(sentences)
        while start < n:
            end = start
            tok_sum = 0
            # Always include at least one sentence so an oversized single
            # sentence still forms a window (progress guarantee). Such a
            # window can EXCEED max_tokens: see the docstring NOTE -- the
            # LettuceDetect caller re-checks and hard-splits, never trusting
            # downstream truncation.
            while end < n and (end == start or tok_sum + counts[end] <= max_tokens):
                tok_sum += counts[end]
                end += 1
            windows.append(" ".join(sentences[start:end]))
            if end >= n:
                break
            # ~50% overlap: advance the start until at least half of THIS window's
            # token mass has been dropped, always by >=1 sentence (progress guarantee).
            dropped = 0
            new_start = start
            while new_start < end - 1 and dropped < tok_sum / 2:
                dropped += counts[new_start]
                new_start += 1
            start = max(new_start, start + 1)
        return windows

    @classmethod
    def _hard_token_split(
        cls, text: str, tokenizer: Any, max_tokens: int
    ) -> List[str]:
        """Token-budget hard split (~50% overlap) for a text sentence
        alignment cannot shrink (D8 §8.5, LettuceDetect branch only).

        A context doc that is one giant "sentence" (no terminators — e.g. a
        key-value dump or minified/code-like blob) defeats the
        sentence-aligned windower: it comes back whole and over budget, and
        the detector's internal max_length truncation would then silently
        drop the evidence past the horizon INSIDE a row reported as
        ``scored_windowed=True`` — the exact silent-evidence-loss class §8.5
        windowing exists to stop. This fallback splits at the character
        level, sizing each chunk by bisection against the detector's OWN
        strict token counts so every emitted chunk verifiably fits
        ``max_tokens``; consecutive chunks overlap by ~half their characters
        so any evidence span shorter than half a chunk appears whole in some
        chunk. Raises (fail-closed, into the caller's typed
        ``_scoring_failure`` path) when not even one character fits.
        """
        chunks: List[str] = []
        start = 0
        n = len(text)
        while start < n:
            lo, hi = 1, n - start
            best = 0
            while lo <= hi:
                mid = (lo + hi) // 2
                count = cls._count_tokens_strict(
                    [text[start : start + mid]], tokenizer
                )[0]
                if count <= max_tokens:
                    best = mid
                    lo = mid + 1
                else:
                    hi = mid - 1
            if best == 0:
                raise RuntimeError(
                    f"hard token split cannot fit even one character into the "
                    f"detector budget ({max_tokens} tokens) — refusing to "
                    f"score truncated"
                )
            chunks.append(text[start : start + best])
            if start + best >= n:
                break
            start += max(1, best // 2)  # ~50% overlap
        return chunks

    def _build_premises(self, contexts: List[str]) -> tuple[List[str], bool]:
        """Expand context docs into NLI premises. Returns (premises, any_windowed)."""
        premises: List[str] = []
        windowed = False
        for ctx in contexts:
            wins = self._split_premise_windows(str(ctx))
            if len(wins) > 1:
                windowed = True
            premises.extend(wins)
        return premises, windowed

    def evaluate_faithfulness(
        self, generated_text: str, context: List[str]
    ) -> Dict[str, Any]:
        """Claim-level NLI faithfulness with paragraph-sized premises (B2).

        The answer is split into claims. Each context doc longer than
        NLI_PREMISE_WINDOW_TOKENS DeBERTa tokens is split into sentence-aligned
        ~50%-overlap windows (see _split_premise_windows); each claim's entailment
        probability is the MAX over windows AND over docs (faithful if supported by
        ANY provided premise -- consistent with the existing max-over-docs rule),
        then averaged over claims. Returns ``{"faithfulness": <0-1 or None>,
        "supported_claim_ratio": <0-1 or None>, "premise_mode":
        'direct'|'windowed'|None, "premise_count": <int or None>}``. ``None`` means
        NLI unavailable. ``premise_count`` is the number of NLI premises (context docs
        x windows) the per-claim max was taken over -- provenance for the
        premise-count confound in mean-of-per-claim-max (see BACKLOG.md).

        The returned dict also carries D8 §8.1 provenance: ``instrument`` (the
        claim-checker id+version consulted; None on early-outs) and ``method``
        (e.g. 'nli_claim_max'; None when no score was produced). Claims come
        from the pluggable ``claim_decomposer`` (default: sentence split); the
        checker is the CAGE_CLAIM_CHECKER selection (default: 'nli' since
        2026-08-19, owner decision #120/F8 -- in-process-safe; 'alignscore'
        runs via scripts/4_analysis/score_instrument_b.py, which requests it
        explicitly).

        D8 §8.5 3-class reporting: when ``nli_three_class`` is on (CAGE_NLI_
        THREE_CLASS) and the NLI path scored, the dict ALSO carries
        ``contradiction`` and ``neutral`` (per-claim max over premises, mean
        over claims); with the flag off those keys are absent (dict unchanged).
        """
        empty: Dict[str, Any] = {
            "faithfulness": None, "supported_claim_ratio": None, "premise_mode": None,
            "premise_count": None, "instrument": None, "method": None,
        }
        if not self.use_nli:
            return empty
        nonempty_ctx = [c for c in (context or []) if c and str(c).strip()]
        claims = self.claim_decomposer.decompose(generated_text or "")
        if not nonempty_ctx or not claims:
            return empty
        # Instrument B seam: a configured MiniCheck/AlignScore checker replaces
        # the per-pair NLI scoring; premise construction and the per-claim
        # max-over-premises aggregation (D8 §8.5) are IDENTICAL in both branches.
        if self._claim_checker is not None:
            return self._faithfulness_via_checker(claims, nonempty_ctx)
        empty["instrument"] = self._faithfulness_instrument_id()
        # Instrument gate AFTER the input early-outs: rows the model was never
        # needed for neither raise nor carry a status. Strict load failure raises.
        if self.nli_model is None:
            self._note_unavailable_row("nli")
            return empty

        try:
            premises, windowed = self._build_premises(nonempty_ctx)
            if not premises:
                return empty
            claim_scores: List[float] = []
            # D8 §8.5 3-class (flag-gated): per-claim MAX over premises per
            # class, mirroring the entailment rule (2026-08-04 review, L3 row).
            claim_contra: List[float] = []
            claim_neutral: List[float] = []
            for claim in claims:
                best = 0.0
                have_score = False
                best_contra: Optional[float] = None
                best_neutral: Optional[float] = None
                for premise in premises:
                    probs = self._nli_pair_probs(premise, claim)
                    if probs is not None:
                        best = max(best, probs.entailment)
                        have_score = True
                        if probs.contradiction is not None:
                            best_contra = (
                                probs.contradiction if best_contra is None
                                else max(best_contra, probs.contradiction)
                            )
                        if probs.neutral is not None:
                            best_neutral = (
                                probs.neutral if best_neutral is None
                                else max(best_neutral, probs.neutral)
                            )
                if have_score:
                    claim_scores.append(best)
                    if best_contra is not None:
                        claim_contra.append(best_contra)
                    if best_neutral is not None:
                        claim_neutral.append(best_neutral)
            if not claim_scores:
                return empty
            faithfulness = float(np.mean(claim_scores))
            supported = float(np.mean([1.0 if s >= 0.5 else 0.0 for s in claim_scores]))
            return {
                "faithfulness": faithfulness,
                "supported_claim_ratio": supported,
                "premise_mode": "windowed" if windowed else "direct",
                "premise_count": len(premises),
                "instrument": empty["instrument"],
                "method": "nli_claim_max",
                # Keys present ONLY when the flag is on (default-off dicts are
                # byte-identical); consumers read them via .get().
                **self._three_class_result_fields(claim_contra, claim_neutral),
            }
        except InstrumentUnavailableError:
            raise  # strict-mode failure from _nli_entailment_prob: never swallow
        except Exception as e:
            self._scoring_failure("nli", self.nli_model_name, e)
            return empty

    def _faithfulness_via_checker(
        self, claims: List[str], contexts: List[str]
    ) -> Dict[str, Any]:
        """Faithfulness via the configured Instrument B claim checker.

        Same premise construction (_build_premises) and the same D8 §8.5
        per-claim max-over-premises aggregation as the NLI path; only the pair
        scorer differs. Fail-closed: a checker whose package/checkpoint is
        absent raises InstrumentUnavailableError in strict mode; non-strict
        marks the instrument sticky-unavailable and labels every affected row.
        """
        checker = self._claim_checker
        assert checker is not None  # caller-gated
        instrument_id = str(checker.instrument_id)
        empty: Dict[str, Any] = {
            "faithfulness": None, "supported_claim_ratio": None, "premise_mode": None,
            "premise_count": None, "instrument": instrument_id, "method": None,
        }
        if "claim_checker" in self._instrument_unavailable:
            self._note_unavailable_row("claim_checker")
            return empty
        premises, windowed = self._build_premises(contexts)
        if not premises:
            return empty
        # Claim-major pair order mirrors the NLI loop (claims outer, premises
        # inner) so the aggregation below can slice per claim.
        pairs: List[Tuple[str, str]] = [(p, c) for c in claims for p in premises]
        try:
            scores = checker.score_pairs(pairs)
        except InstrumentUnavailableError as e:
            # Load-tier failure (package absent / checkpoint unpinned): sticky.
            self._mark_unavailable("claim_checker", instrument_id, e)  # strict raises
            self._note_unavailable_row("claim_checker")
            return empty
        except Exception as e:
            self._scoring_failure("claim_checker", instrument_id, e)
            return empty
        n_prem = len(premises)
        claim_scores: List[float] = []
        for ci in range(len(claims)):
            claim_pair_scores = [
                s for s in scores[ci * n_prem: (ci + 1) * n_prem] if s is not None
            ]
            if claim_pair_scores:
                claim_scores.append(max(float(s) for s in claim_pair_scores))
        if not claim_scores:
            return empty
        return {
            "faithfulness": float(np.mean(claim_scores)),
            "supported_claim_ratio": float(
                np.mean([1.0 if s >= 0.5 else 0.0 for s in claim_scores])
            ),
            "premise_mode": "windowed" if windowed else "direct",
            "premise_count": len(premises),
            "instrument": instrument_id,
            "method": f"{checker.name}_claim_max",
        }

    # D8 §8.5 windowed grounding: token allowance reserved for the detector's
    # special tokens + prompt scaffolding around (context, question, answer).
    # This is a scaffold MARGIN, not the model limit -- L_max itself is derived
    # at runtime from the loaded model config (never hardcoded).
    LETTUCE_SPECIAL_TOKENS_MARGIN = 64

    def _lettucedetect_limits(self, detector: Any) -> Tuple[Optional[Any], Optional[int]]:
        """Derive (tokenizer, effective L_max) from the LOADED detector.

        L_max comes from the model config's ``max_position_embeddings``
        (ModernBERT-class ~8k at the pinned version -- D8 §8.5 VERIFY-LIVE),
        capped by the detector's own ``max_length`` truncation setting when
        present. NEVER hardcoded: a model swap re-derives the limit.
        """
        inner = getattr(detector, "detector", detector)  # HallucinationDetector wraps
        tokenizer = getattr(inner, "tokenizer", None)
        model = getattr(inner, "model", None)
        config = getattr(model, "config", None)
        l_max = getattr(config, "max_position_embeddings", None)
        cap = getattr(inner, "max_length", None)
        limits = [v for v in (l_max, cap) if isinstance(v, int) and v > 0]
        return tokenizer, (min(limits) if limits else None)

    @staticmethod
    def _spans_to_flagged_chars(spans: Any, total: int) -> List[bool]:
        """Char-level flag array from detector spans (overlap counted ONCE)."""
        flagged_chars = [False] * total
        for s in spans or []:
            start = max(0, min(int(s.get("start", 0)), total))
            end = max(0, min(int(s.get("end", 0)), total))
            for i in range(start, end):
                flagged_chars[i] = True
        return flagged_chars

    def evaluate_hallucination(
        self, question: str, context: List[str], generated_text: str
    ) -> Dict[str, Any]:
        """Token/span-level hallucination detection via LettuceDetect (PRIMARY).

        Citation: Kovacs et al. 2025, "LettuceDetect: A Hallucination Detection
        Framework for RAG Applications" (arXiv:2502.17125), trained on RAGTruth
        (Niu et al. 2024).

        Native scoring while (context + question + answer) fits the detector's
        L_max, derived AT RUNTIME from the loaded model config
        (max_position_embeddings, capped by the detector's own max_length) --
        never hardcoded. Beyond L_max, the D8 §8.5 windowed pass runs: over-long
        context docs are split into sentence-aligned ~50%-overlap windows
        (mirroring the NLI premise windows), the answer is scored against EACH
        window, and verdicts aggregate by MAX-SUPPORT per answer character -- a
        character is unsupported only if NO window supports it (the windowed
        union of the native multi-doc call). Affected rows carry
        ``scored_windowed=True``: windowing is an ALERT, never silent.

        Returns ``{"grounding_score", "hallucination_detected",
        "hallucinated_span_ratio", "hallucinated_spans", "scored_windowed",
        "instrument"}``. Scores are ``None`` if the detector is unavailable.
        Span char offsets refer to the (sanitized) answer; windowed-pass spans
        are rebuilt from the aggregated flag array.
        """
        empty: Dict[str, Any] = {
            "grounding_score": None,
            "hallucination_detected": None,
            "hallucinated_span_ratio": None,
            "hallucinated_spans": None,
            "scored_windowed": False,
            "instrument": None,
        }
        answer = generated_text or ""
        nonempty_ctx = [str(c) for c in (context or []) if c and str(c).strip()]
        if not self.use_lettucedetect or not nonempty_ctx or not answer.strip():
            return empty
        # Consulted from here on: provenance names the pinned instrument even
        # when it fails (instrument_status records the failure).
        empty["instrument"] = self._grounding_instrument_id()
        detector = self.lettucedetect_model  # strict: raises on load failure
        if detector is None:
            self._note_unavailable_row("lettucedetect")
            return empty
        # D8 §8.5 windowing gate: refusing to score blind is fail-closed -- a
        # detector whose tokenizer/limit cannot be derived could silently
        # truncate evidence (the exact failure class windowing exists to stop).
        tokenizer, l_max = self._lettucedetect_limits(detector)
        if tokenizer is None or l_max is None:
            self._scoring_failure(
                "lettucedetect", self.lettucedetect_model_name,
                RuntimeError(
                    "cannot derive tokenizer/L_max (max_position_embeddings) from "
                    "the loaded detector -- refusing to score without the D8 §8.5 "
                    "windowing gate"
                ),
            )
            return empty
        try:
            counts = self._count_tokens_strict(
                [question or "", answer] + nonempty_ctx, tokenizer
            )
            q_tokens, a_tokens = counts[0], counts[1]
            doc_counts = counts[2:]
            budget = l_max - q_tokens - a_tokens - self.LETTUCE_SPECIAL_TOKENS_MARGIN
            if budget <= 0:
                # Question + answer alone overflow the detector: windowing the
                # CONTEXT cannot fix this. Typed failure, never a blind score.
                self._scoring_failure(
                    "lettucedetect", self.lettucedetect_model_name,
                    RuntimeError(
                        f"question+answer ({q_tokens}+{a_tokens} tokens) exceed the "
                        f"detector context (L_max={l_max}); context windowing cannot help"
                    ),
                )
                return empty

            total = len(answer)
            if sum(doc_counts) <= budget:
                # Native path (unchanged pre-existing behavior): everything fits.
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
                flagged_chars = self._spans_to_flagged_chars(spans, total)
                norm_spans = [
                    {"start": int(s.get("start", 0)), "end": int(s.get("end", 0)),
                     "text": s.get("text")}
                    for s in (spans or [])
                ]
                scored_windowed = False
            else:
                # D8 §8.5 windowed max-support pass (LAUNCH-BLOCKER item). Docs
                # that fit stay whole; over-long docs split sentence-aligned
                # with ~50% overlap (same core algorithm as the NLI premises).
                windows: List[str] = []
                for doc, doc_count in zip(nonempty_ctx, doc_counts):
                    if doc_count <= budget:
                        windows.append(doc)
                        continue
                    sents = _split_sentences(doc)
                    sent_counts = self._count_tokens_strict(sents, tokenizer)
                    windows.extend(
                        self._sentence_window_split(doc, sents, sent_counts, budget)
                    )
                # Re-check EVERY window against the budget with the strict
                # counter: sentence alignment guarantees nothing for a single
                # "sentence" longer than the budget (terminator-free text is
                # returned whole / the always-include-one rule emits it over
                # budget). Relying on detector-internal truncation there would
                # silently drop evidence inside a scored_windowed=True row, so
                # over-budget windows are hard-split at token level instead.
                window_counts = self._count_tokens_strict(windows, tokenizer)
                checked: List[str] = []
                for window, w_count in zip(windows, window_counts):
                    if w_count <= budget:
                        checked.append(window)
                    else:
                        checked.extend(
                            self._hard_token_split(window, tokenizer, budget)
                        )
                windows = checked
                if not self._windowed_alert_emitted:
                    print(
                        f"ALERT (D8 §8.5): context exceeds LettuceDetect "
                        f"L_max={l_max}; scoring via {len(windows)} sentence-aligned "
                        f"windows with max-support aggregation. Affected rows carry "
                        f"scored_windowed=True."
                    )
                    self._windowed_alert_emitted = True
                # Max-support per answer character: unsupported only if flagged
                # in EVERY window (no window's evidence supports it) -- the
                # windowed equivalent of the native call's union of evidence.
                flagged_chars = [True] * total
                for window in windows:
                    spans = detector.predict(
                        context=[window],
                        question=question or "",
                        answer=answer,
                        output_format="spans",
                    )
                    window_flags = self._spans_to_flagged_chars(spans, total)
                    flagged_chars = [a and b for a, b in zip(flagged_chars, window_flags)]
                # Rebuild spans from the aggregated flag array (contiguous runs).
                norm_spans = []
                run_start: Optional[int] = None
                for i in range(total + 1):
                    on = i < total and flagged_chars[i]
                    if on and run_start is None:
                        run_start = i
                    elif not on and run_start is not None:
                        norm_spans.append(
                            {"start": run_start, "end": i, "text": answer[run_start:i]}
                        )
                        run_start = None
                scored_windowed = True

            flagged = sum(flagged_chars)
            ratio = (flagged / total) if total > 0 else 0.0
            ratio = max(0.0, min(1.0, ratio))
            return {
                "grounding_score": 1.0 - ratio,
                # Flag a hallucination only when characters were actually flagged, not merely
                # because the detector returned a (possibly zero-length) span.
                "hallucination_detected": flagged > 0,
                "hallucinated_span_ratio": ratio,
                "hallucinated_spans": norm_spans,
                "scored_windowed": scored_windowed,
                "instrument": empty["instrument"],
            }
        except InstrumentUnavailableError:
            raise  # strict-mode typed failure from the gates above: never swallow
        except Exception as e:
            self._scoring_failure("lettucedetect", self.lettucedetect_model_name, e)
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
        if not self.use_embeddings or not nonempty_ctx:
            return None
        model = self.embedding_model  # strict: raises on load failure
        if model is None:
            self._note_unavailable_row("embedding")
            return None

        try:
            # Encode question and context
            question_emb = model.encode(question, convert_to_tensor=True)
            context_embs = model.encode(nonempty_ctx, convert_to_tensor=True)

            # Compute cosine similarities
            from sentence_transformers.util import cos_sim
            similarities = cos_sim(question_emb, context_embs)[0]

            # Return max similarity
            return float(similarities.max().cpu().numpy())

        except Exception as e:
            self._scoring_failure("embedding", self.embedding_model_name, e)
            return None
    
    def evaluate_completeness(
        self, generated_text: str, reference_answer: str
    ) -> Dict[str, Optional[float]]:
        """
        Evaluate completeness using BERTScore and ROUGE.
        
        Compares generated text to reference answer.
        Returns dict with bertscore_f1, rouge_l_f1, and bertscore_idf (D8 §8.4
        provenance: whether IDF weighting was active for the produced BERTScore;
        None when BERTScore produced no score for this row).
        """
        results: Dict[str, Any] = {
            "bertscore_f1": None, "rouge_l_f1": None, "bertscore_idf": None,
        }

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
                results["bertscore_idf"] = self.bertscore_idf
            if self.use_rouge:
                results["rouge_l_f1"] = 0.0
            return results
        
        # BERTScore. P0-5: the old path reloaded a FALLBACK model on a runtime
        # scoring error and kept scoring under the same column -- a silent mid-run
        # instrument swap. Now: strict raises, non-strict labels the row.
        if self.use_bertscore:
            scorer = self.bertscore_model  # strict: raises on load failure
            if scorer is None:
                self._note_unavailable_row("bertscore")
            else:
                try:
                    P, R, F1 = scorer.score([generated_text], [reference_answer])
                    results["bertscore_f1"] = float(F1[0].cpu().numpy())
                    results["bertscore_idf"] = self.bertscore_idf
                except Exception as e:
                    self._scoring_failure("bertscore", self.bertscore_model_name, e)

        # ROUGE
        if self.use_rouge:
            rs = self.rouge_scorer  # strict: raises on load failure
            if rs is None:
                self._note_unavailable_row("rouge")
            else:
                try:
                    scores = rs.score(reference_answer, generated_text)
                    results["rouge_l_f1"] = scores["rougeL"].fmeasure
                except Exception as e:
                    self._scoring_failure("rouge", "rouge_score", e)

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
        # B4 sanitizer (2026-07-16 pre-run package): ALL quality scoring -- F1/EM,
        # abstention detection, grounding, NLI, completeness -- runs on the sanitized
        # text (scaffold prefix stripped, fabricated prompt-continuation truncated).
        # The raw generation is NEVER overwritten: callers keep generated_answer and
        # this result carries sanitized_answer alongside. NOTE: hallucinated_spans
        # char offsets refer to the SANITIZED text.
        # P0-5: per-row instrument status. Tokens accumulate as instruments are
        # consulted; 'ok' means every pinned instrument asked to score this row
        # produced a score (config-disabled instruments are not consulted).
        self._row_status_tokens = []
        sanitized_text = sanitize_answer(generated_text)
        f1_metrics = self.evaluate_f1_score(sanitized_text, reference_answer, all_answers)
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
        if is_no_answer_prediction(sanitized_text):
            faith: Dict[str, Any] = {
                "faithfulness": None, "supported_claim_ratio": None, "premise_mode": None,
                "premise_count": None, "instrument": None, "method": None,
            }
            halluc = {
                "grounding_score": None,
                "hallucination_detected": None,
                "hallucinated_span_ratio": None,
                "hallucinated_spans": None,
                "scored_windowed": False,
                "instrument": None,
            }
            completeness = {
                "bertscore_f1": None, "rouge_l_f1": None, "bertscore_idf": None,
            }
        else:
            faith = self.evaluate_faithfulness(sanitized_text, context)
            halluc = self.evaluate_hallucination(question, context, sanitized_text)
            completeness = self.evaluate_completeness(sanitized_text, reference_answer)

        # Dedupe (per-claim NLI errors repeat) while preserving consult order.
        instrument_status = (
            ";".join(dict.fromkeys(self._row_status_tokens))
            if self._row_status_tokens
            else "ok"
        )

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
            # 'method' is set by the scoring branch that produced a score
            # ('nli_claim_max' default; '<checker>_claim_max' for Instrument B),
            # and None whenever faithfulness was not scored -- same contract as
            # the old hardcoded "nli_claim_max"-if-scored expression.
            faithfulness_method=faith.get("method"),
            faithfulness_premise_mode=faith.get("premise_mode"),
            faithfulness_premise_count=faith.get("premise_count"),
            sanitized_answer=sanitized_text,
            instrument_status=instrument_status,
            hallucinated_spans=halluc.get("hallucinated_spans"),
            # D8 §8.1 per-row provenance (emitted unconditionally in to_dict).
            scored_windowed=bool(halluc.get("scored_windowed")),
            grounding_instrument=halluc.get("instrument"),
            faithfulness_instrument=faith.get("instrument"),
            calibration_id=self.calibration_id,
            bertscore_idf=completeness.get("bertscore_idf"),
            # D8 §8.5 flag-gated 3-class columns: keys exist in `faith` only
            # when the flag is on and the NLI path scored; .get() keeps every
            # other path (abstention/checker/unavailable) at None.
            faithfulness_contradiction=faith.get("contradiction"),
            faithfulness_neutral=faith.get("neutral"),
            nli_three_class=self.nli_three_class,
        )

    def batch_evaluate(
        self,
        questions: List[str],
        contexts: List[List[str]],
        generated_texts: List[str],
        reference_answers: List[str],
        all_answers: Optional[Sequence[Optional[List[str]]]] = None,
        batched: bool = True,
        nli_batch_size: int = 32,
    ) -> List[QualityMetrics]:
        """Batch evaluation with REAL cross-row batching (D8 §8.1).

        The GPU-batched post-serving scoring pass is the module's execution
        model ("inline scoring was ~90% of wall-clock" -- D8 §8.1); the old
        implementation looped ``evaluate`` row by row. This version accumulates
        (premise, claim) NLI pairs and BERTScore (candidate, reference) texts
        ACROSS rows and issues single batched model calls, preserving
        per-row outputs identical to the sequential path (same parsing, same
        aggregation, same instrument_status token order; proven by test).

        Args:
            all_answers: optional per-row gold-answer lists (max-over-golds
                F1/EM); index-aligned, None entries fall back to the single
                reference.
            batched: False reproduces the historical row-by-row loop exactly
                (also the comparison baseline for the equivalence test).
            nli_batch_size: forwarded to the HF pipeline's ``batch_size`` for
                the single batched NLI call.
        """

        def _aa(i: int) -> Optional[List[str]]:
            if all_answers is None or i >= len(all_answers):
                return None
            return all_answers[i]

        rows = list(zip(questions, contexts, generated_texts, reference_answers))
        if not batched:
            return [
                self.evaluate(q, ctx, gen, ref, all_answers=_aa(i))
                for i, (q, ctx, gen, ref) in enumerate(rows)
            ]

        n = len(rows)
        # Per-row status-token lists; helpers append into the active row's list
        # via self._row_status_tokens (pointer swap), so token ORDER per row
        # matches the sequential pipeline: embedding -> nli -> lettucedetect ->
        # bertscore -> rouge.
        row_tokens: List[List[str]] = [[] for _ in range(n)]
        sanitized = [sanitize_answer(gen) for (_, _, gen, _) in rows]
        f1s = [
            self.evaluate_f1_score(sanitized[i], rows[i][3], _aa(i)) for i in range(n)
        ]
        abstained = [is_no_answer_prediction(s) for s in sanitized]

        # Relevance (question<->context; computed for ALL rows, like evaluate()).
        relevances: List[Optional[float]] = []
        for i in range(n):
            self._row_status_tokens = row_tokens[i]
            relevances.append(self.evaluate_relevance(rows[i][0], rows[i][1]))

        # Faithfulness: ONE batched call across every row's (premise, claim) pairs.
        faiths = self._batch_faithfulness(
            sanitized, [ctx for (_, ctx, _, _) in rows], abstained, row_tokens,
            nli_batch_size,
        )

        # Grounding: per-row detector calls (windowing is per-row by nature).
        halluc_empty: Dict[str, Any] = {
            "grounding_score": None, "hallucination_detected": None,
            "hallucinated_span_ratio": None, "hallucinated_spans": None,
            "scored_windowed": False, "instrument": None,
        }
        hallucs: List[Dict[str, Any]] = []
        for i in range(n):
            if abstained[i]:
                hallucs.append(dict(halluc_empty))
            else:
                self._row_status_tokens = row_tokens[i]
                hallucs.append(
                    self.evaluate_hallucination(rows[i][0], rows[i][1], sanitized[i])
                )

        # Completeness: ONE batched BERTScore call; ROUGE per row (CPU-cheap).
        completes = self._batch_completeness(
            sanitized, [ref for (_, _, _, ref) in rows], abstained, row_tokens
        )

        self._row_status_tokens = []
        out: List[QualityMetrics] = []
        for i in range(n):
            status = (
                ";".join(dict.fromkeys(row_tokens[i])) if row_tokens[i] else "ok"
            )
            faith = faiths[i]
            halluc = hallucs[i]
            completeness = completes[i]
            f1_metrics = f1s[i]
            out.append(
                QualityMetrics(
                    faithfulness=faith["faithfulness"],
                    relevance=relevances[i],
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
                    faithfulness_method=faith.get("method"),
                    faithfulness_premise_mode=faith.get("premise_mode"),
                    faithfulness_premise_count=faith.get("premise_count"),
                    sanitized_answer=sanitized[i],
                    instrument_status=status,
                    hallucinated_spans=halluc.get("hallucinated_spans"),
                    scored_windowed=bool(halluc.get("scored_windowed")),
                    grounding_instrument=halluc.get("instrument"),
                    faithfulness_instrument=faith.get("instrument"),
                    calibration_id=self.calibration_id,
                    bertscore_idf=completeness.get("bertscore_idf"),
                    faithfulness_contradiction=faith.get("contradiction"),
                    faithfulness_neutral=faith.get("neutral"),
                    nli_three_class=self.nli_three_class,
                )
            )
        return out

    def _batch_faithfulness(
        self,
        sanitized: List[str],
        contexts_list: List[List[str]],
        abstained: List[bool],
        row_tokens: List[List[str]],
        nli_batch_size: int,
    ) -> List[Dict[str, Any]]:
        """Cross-row faithfulness: one batched model call over all pairs.

        Mirrors evaluate_faithfulness row semantics exactly: same early-outs,
        same premise construction, same per-claim max-over-premises aggregation
        (D8 §8.5), same strict/non-strict failure labeling. A whole-batch call
        failure is equivalent to every pair failing in the sequential loop.
        """
        empty: Dict[str, Any] = {
            "faithfulness": None, "supported_claim_ratio": None, "premise_mode": None,
            "premise_count": None, "instrument": None, "method": None,
        }
        results = [dict(empty) for _ in sanitized]
        if not self.use_nli:
            return results
        prepared: List[Tuple[int, List[str], List[str]]] = []  # (row, claims, ctx)
        for i, (text, ctx) in enumerate(zip(sanitized, contexts_list)):
            if abstained[i]:
                continue
            nonempty_ctx = [c for c in (ctx or []) if c and str(c).strip()]
            claims = self.claim_decomposer.decompose(text or "")
            if not nonempty_ctx or not claims:
                continue
            prepared.append((i, claims, nonempty_ctx))
        if not prepared:
            return results

        if self._claim_checker is not None:
            return self._batch_faithfulness_checker(prepared, results, row_tokens)

        instrument = self._faithfulness_instrument_id()
        for (i, _, _) in prepared:
            results[i]["instrument"] = instrument
        # Lazy load once; a strict load failure raises exactly as it would on
        # the first sequential row that needed the model.
        self._row_status_tokens = row_tokens[prepared[0][0]]
        if self.nli_model is None:
            for (i, _, _) in prepared:
                self._row_status_tokens = row_tokens[i]
                self._note_unavailable_row("nli")
            return results

        plans: List[Tuple[int, List[str], List[str], bool]] = []
        for (i, claims, ctx) in prepared:
            premises, windowed = self._build_premises(ctx)
            if not premises:
                continue
            plans.append((i, claims, premises, windowed))
        if not plans:
            return results
        inputs: List[Dict[str, str]] = []
        for (_, claims, premises, _) in plans:
            # Claim-major order mirrors the sequential loop (claims outer).
            for c in claims:
                for p in premises:
                    inputs.append({"text": p, "text_pair": c})
        try:
            raw = self.nli_model(
                inputs,
                top_k=None,
                truncation=True,
                max_length=self.nli_max_length,
                batch_size=nli_batch_size,
            )
        except Exception as e:
            # Whole-call failure == every pair failing sequentially: strict
            # raises on the first affected row; non-strict labels each row.
            for (i, _, _, _) in plans:
                self._row_status_tokens = row_tokens[i]
                self._scoring_failure("nli", self.nli_model_name, e)
            return results
        k = 0
        for (i, claims, premises, windowed) in plans:
            self._row_status_tokens = row_tokens[i]
            claim_scores: List[float] = []
            # D8 §8.5 3-class (flag-gated): same per-claim max-over-premises
            # aggregation as the sequential path (single seam: _parse_nli_result).
            claim_contra: List[float] = []
            claim_neutral: List[float] = []
            for _claim in claims:
                best = 0.0
                have_score = False
                best_contra: Optional[float] = None
                best_neutral: Optional[float] = None
                for _premise in premises:
                    pair_result = raw[k]
                    k += 1
                    probs: Optional[NLIProbs]
                    try:
                        probs = self._parse_nli_result(pair_result)
                    except InstrumentUnavailableError:
                        if self.strict:
                            raise
                        self._note_row_status(
                            "nli", "error", "entailment-class-unresolved"
                        )
                        probs = None
                    except Exception as e:
                        self._scoring_failure("nli", self.nli_model_name, e)
                        probs = None
                    if probs is not None:
                        best = max(best, probs.entailment)
                        have_score = True
                        if probs.contradiction is not None:
                            best_contra = (
                                probs.contradiction if best_contra is None
                                else max(best_contra, probs.contradiction)
                            )
                        if probs.neutral is not None:
                            best_neutral = (
                                probs.neutral if best_neutral is None
                                else max(best_neutral, probs.neutral)
                            )
                if have_score:
                    claim_scores.append(best)
                    if best_contra is not None:
                        claim_contra.append(best_contra)
                    if best_neutral is not None:
                        claim_neutral.append(best_neutral)
            if claim_scores:
                results[i] = {
                    "faithfulness": float(np.mean(claim_scores)),
                    "supported_claim_ratio": float(
                        np.mean([1.0 if s >= 0.5 else 0.0 for s in claim_scores])
                    ),
                    "premise_mode": "windowed" if windowed else "direct",
                    "premise_count": len(premises),
                    "instrument": instrument,
                    "method": "nli_claim_max",
                    **self._three_class_result_fields(claim_contra, claim_neutral),
                }
        return results

    def _batch_faithfulness_checker(
        self,
        prepared: List[Tuple[int, List[str], List[str]]],
        results: List[Dict[str, Any]],
        row_tokens: List[List[str]],
    ) -> List[Dict[str, Any]]:
        """Cross-row Instrument B scoring: one score_pairs call over all pairs."""
        checker = self._claim_checker
        assert checker is not None  # caller-gated
        instrument = str(checker.instrument_id)
        plans: List[Tuple[int, List[str], List[str], bool]] = []
        for (i, claims, ctx) in prepared:
            results[i]["instrument"] = instrument
            premises, windowed = self._build_premises(ctx)
            if not premises:
                continue
            plans.append((i, claims, premises, windowed))
        if not plans:
            return results
        if "claim_checker" in self._instrument_unavailable:
            for (i, _, _, _) in plans:
                self._row_status_tokens = row_tokens[i]
                self._note_unavailable_row("claim_checker")
            return results
        all_pairs: List[Tuple[str, str]] = []
        for (_, claims, premises, _) in plans:
            all_pairs.extend((p, c) for c in claims for p in premises)
        try:
            scores = checker.score_pairs(all_pairs)
        except InstrumentUnavailableError as e:
            self._row_status_tokens = row_tokens[plans[0][0]]
            self._mark_unavailable("claim_checker", instrument, e)  # strict raises
            for (i, _, _, _) in plans:
                self._row_status_tokens = row_tokens[i]
                self._note_unavailable_row("claim_checker")
            return results
        except Exception as e:
            for (i, _, _, _) in plans:
                self._row_status_tokens = row_tokens[i]
                self._scoring_failure("claim_checker", instrument, e)
            return results
        k = 0
        for (i, claims, premises, windowed) in plans:
            n_prem = len(premises)
            claim_scores: List[float] = []
            for _ci in range(len(claims)):
                vals = [s for s in scores[k: k + n_prem] if s is not None]
                k += n_prem
                if vals:
                    claim_scores.append(max(float(s) for s in vals))
            if claim_scores:
                results[i] = {
                    "faithfulness": float(np.mean(claim_scores)),
                    "supported_claim_ratio": float(
                        np.mean([1.0 if s >= 0.5 else 0.0 for s in claim_scores])
                    ),
                    "premise_mode": "windowed" if windowed else "direct",
                    "premise_count": len(premises),
                    "instrument": instrument,
                    "method": f"{checker.name}_claim_max",
                }
        return results

    def _batch_completeness(
        self,
        sanitized: List[str],
        references: List[str],
        abstained: List[bool],
        row_tokens: List[List[str]],
    ) -> List[Dict[str, Any]]:
        """Cross-row completeness: one batched BERTScore call, ROUGE per row.

        Mirrors evaluate_completeness row semantics: abstention/empty-reference
        rows stay None, empty generations score a genuine 0.0, and the
        BERTScore-before-ROUGE token order is preserved per row.
        """
        results: List[Dict[str, Any]] = [
            {"bertscore_f1": None, "rouge_l_f1": None, "bertscore_idf": None}
            for _ in sanitized
        ]
        pending: List[int] = []
        rouge_rows: List[int] = []
        for i, (gen, ref) in enumerate(zip(sanitized, references)):
            if abstained[i]:
                continue
            if not ref or not ref.strip():
                continue
            if not gen or not gen.strip():
                if self.use_bertscore:
                    results[i]["bertscore_f1"] = 0.0
                    results[i]["bertscore_idf"] = self.bertscore_idf
                if self.use_rouge:
                    results[i]["rouge_l_f1"] = 0.0
                continue
            if self.use_bertscore:
                pending.append(i)
            if self.use_rouge:
                rouge_rows.append(i)
        if pending:
            # Strict load failure raises here, as on the first sequential row.
            self._row_status_tokens = row_tokens[pending[0]]
            scorer = self.bertscore_model
            if scorer is None:
                for i in pending:
                    self._row_status_tokens = row_tokens[i]
                    self._note_unavailable_row("bertscore")
            else:
                try:
                    cands = [sanitized[i] for i in pending]
                    refs = [references[i] for i in pending]
                    P, R, F1 = scorer.score(cands, refs)
                    for j, i in enumerate(pending):
                        results[i]["bertscore_f1"] = float(F1[j].cpu().numpy())
                        results[i]["bertscore_idf"] = self.bertscore_idf
                except Exception as e:
                    for i in pending:
                        self._row_status_tokens = row_tokens[i]
                        self._scoring_failure(
                            "bertscore", self.bertscore_model_name, e
                        )
        for i in rouge_rows:
            self._row_status_tokens = row_tokens[i]
            rs = self.rouge_scorer
            if rs is None:
                self._note_unavailable_row("rouge")
                continue
            try:
                scores = rs.score(references[i], sanitized[i])
                results[i]["rouge_l_f1"] = scores["rougeL"].fmeasure
            except Exception as e:
                self._scoring_failure("rouge", "rouge_score", e)
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
                method="none",
            )

        per_block_scores = []

        # Method 1: Embedding similarity between each block and the reference answer.
        # P0-5: the lexical fallback survives ONLY because this is an explicitly
        # labeled DIAGNOSTIC (never in Y) and the method used is written into the
        # row -- the substitute score never masquerades as the embedding scorer's.
        # In strict mode a broken embedding model raises via the property.
        method = f"embedding:{self.embedding_model_name}"
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
                # Fall back to lexical overlap (labeled below).
                method = "lexical_jaccard"
                per_block_scores = self._lexical_cache_relevance(
                    reference_answer, cache_blocks
                )
        else:
            # Fallback: lexical overlap (token-based), labeled.
            method = "lexical_jaccard"
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
            method=method,
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
        metrics.cache_relevance_method = cache_rel.method  # P0-5 labeled diagnostic
        return metrics
