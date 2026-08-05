"""SCBench two-subset external-validation slice — charter D5 item 6.

SCBench (Li et al., ICLR 2025, "SCBench: A KV-Cache-Centric Analysis of
Long-Context Methods") provides SESSION-shaped items: one shared long context
plus an ordered list of follow-up requests over that same context — exactly
the multi-request reuse shape CAGE's locality findings must replicate on
community data. The charter scopes CAGE to a two-subset slice
(``scbench_kv`` + ``scbench_qa_eng``), run under the CAGE client on the >=70B
rungs (Groups B and D), scored with SCBench-native metrics only (labeled
serving+native-quality; NEVER feeds the D8 Y predicate).

Loader semantics:
- One ``CAGExample`` per session TURN, emitted in session order; the session
  structure is preserved via ``metadata["session_id"] / turn_index /
  num_turns`` so the harness can replay turns of a session against the same
  shared context (prefix-reuse shape intact).
- ``max_examples`` bounds SESSIONS (each contributing all its turns), matching
  the Qasper precedent where ``max_examples`` bounds papers, not questions;
  session selection uses the pinned seeded shuffle-before-select pattern.
- FAIL-CLOSED: if the dataset is not downloaded/reachable, or an item does not
  match the expected session schema, a typed ``DatasetUnavailableError`` is
  raised (mirroring ``InstrumentUnavailableError`` in
  ``src/evaluation/quality.py``) — never an empty or partial example list.

NOTE: ``microsoft/SCBench`` publishes a ``"test"`` split; construct with
``split="test"`` (or leave default / set ``CAGE_SCBENCH_SPLIT``).
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

from src.data.loader import CAGExample, DatasetLoader, DatasetUnavailableError

#: Charter D5#6: the scoped two-subset slice (never the full benchmark).
SCBENCH_SUBSETS = ("scbench_kv", "scbench_qa_eng")

DEFAULT_HF_PATH = "microsoft/SCBench"


class SCBenchLoader(DatasetLoader):
    """Session-shaped loader over the charter's two-subset SCBench slice.

    Args:
        split: HF split (default: env ``CAGE_SCBENCH_SPLIT``, else "test" —
            the split microsoft/SCBench publishes).
        seed: seed for the session-level shuffle-before-select draw.
        subset: "scbench_kv" | "scbench_qa_eng"
            (default: env ``CAGE_SCBENCH_SUBSET``, else "scbench_kv").
        hf_path: HF dataset path (default: env ``CAGE_SCBENCH_HF_PATH``, else
            "microsoft/SCBench").
    """

    def __init__(
        self,
        split: Optional[str] = None,
        seed: int = 42,
        subset: Optional[str] = None,
        hf_path: Optional[str] = None,
    ):
        subset = subset or os.getenv("CAGE_SCBENCH_SUBSET", "scbench_kv")
        if subset not in SCBENCH_SUBSETS:
            raise ValueError(
                f"Unknown SCBench subset '{subset}'. The charter (D5#6) scopes "
                f"CAGE to the two-subset slice {list(SCBENCH_SUBSETS)}."
            )
        self.subset = subset
        self.hf_path = hf_path or os.getenv("CAGE_SCBENCH_HF_PATH", DEFAULT_HF_PATH)
        split = split or os.getenv("CAGE_SCBENCH_SPLIT", "test")
        super().__init__(self.hf_path, split, seed)

    # -- schema helpers -----------------------------------------------------

    @staticmethod
    def _turn_question(turn: Dict[str, Any]) -> str:
        return (turn.get("input") or turn.get("question") or turn.get("prompt") or "") \
            if isinstance(turn, dict) else ""

    @staticmethod
    def _turn_answers(turn: Dict[str, Any]) -> List[str]:
        """All gold answers for a turn (SCBench answers may be scalar or list)."""
        if not isinstance(turn, dict):
            return []
        raw = turn.get("answer")
        if raw is None:
            raw = turn.get("ground_truth")
        if raw is None:
            return []
        if isinstance(raw, list):
            return [str(a) for a in raw if a is not None and str(a)]
        return [str(raw)] if str(raw) else []

    def load(self, max_examples: Optional[int] = None) -> List[CAGExample]:
        """Load the slice; ``max_examples`` bounds SESSIONS (all turns kept)."""
        from datasets import load_dataset  # lazy: see loader.py module note

        source = f"{self.hf_path}/{self.subset}[{self.split}]"
        try:
            dataset = load_dataset(self.hf_path, self.subset, split=self.split)
        except DatasetUnavailableError:
            raise
        except Exception as exc:  # fail-closed: typed, never silent
            raise DatasetUnavailableError(
                dataset="scbench",
                source=source,
                cause=f"load_dataset failed ({type(exc).__name__}: {exc})",
            ) from exc

        if max_examples:
            # Seeded shuffle BEFORE select at the SESSION level (pinned pattern:
            # different seeds draw different, reproducible session samples).
            dataset = dataset.shuffle(seed=self.seed).select(
                range(min(max_examples, len(dataset))))

        examples: List[CAGExample] = []
        for s_idx, item in enumerate(dataset):
            context = item.get("context") or item.get("prompt") or ""
            turns = item.get("multi_turns") or item.get("turns") or []
            if not context or not isinstance(turns, list) or not turns:
                raise DatasetUnavailableError(
                    dataset="scbench",
                    source=source,
                    cause=(
                        f"item {s_idx} does not match the expected SCBench "
                        f"session schema (shared 'context' + non-empty "
                        f"'multi_turns' list); refusing to emit a degraded "
                        f"session"
                    ),
                )
            session_id = str(item.get("id", s_idx))
            num_turns = len(turns)
            for t_idx, turn in enumerate(turns):
                question = self._turn_question(turn)
                answers = self._turn_answers(turn)
                if not question:
                    raise DatasetUnavailableError(
                        dataset="scbench",
                        source=source,
                        cause=(
                            f"session {session_id} turn {t_idx} has no "
                            f"input/question field"
                        ),
                    )
                examples.append(CAGExample(
                    id=f"{self.subset}_{session_id}_turn{t_idx}",
                    question=question,
                    context=[context],  # the session's SHARED context, verbatim
                    answer=answers[0] if answers else "",
                    metadata={
                        "dataset": "scbench",
                        "subset": self.subset,
                        # Multi-request session structure (charter D5#6): the
                        # harness replays turns in order against one context.
                        "session_id": session_id,
                        "turn_index": t_idx,
                        "num_turns": num_turns,
                        "all_answers": answers,
                        # SCBench-native metrics only; never feeds the D8 Y
                        # predicate (labeled serving+native-quality).
                        "native_metrics_only": True,
                    },
                ))
        return examples
