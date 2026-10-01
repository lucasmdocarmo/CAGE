"""The NLI checkpoint loads with TorchScript disabled, then the flag is restored (S0F-4, ADR-0129).

The S0 finding, corrected by the fact-finding of 2026-10-01: the pod could not
import the NLI instrument (MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli) because
CPython 3.13.8's ``inspect.getsourcelines`` stops at a comment that sits between
a ``@torch.jit.script`` decorator and its ``def`` (fixed in 3.13.9), and
transformers' ``modeling_deberta_v2.py:105-107`` has exactly that layout; under
torch 2.10 the import raised ``IndentationError`` from ``torch/_sources.py``.
torch reads ``PYTORCH_JIT`` once at import, so an environment variable set from
inside Python after torch is loaded does nothing, and a variable set in the shell
leaked into the fp8 gate's vLLM restart at S0. ``torch.jit._state.disable()``
works at any time, and ``torch.jit.script`` then returns the function unchanged.

Pinned here with a fake ``transformers.pipeline`` (the real library is not the
thing under test; the 3.13.8 bug itself cannot be reproduced on this Mac):
1. while the pipeline loads, TorchScript is disabled;
2. after the load the PRIOR flag is restored, both when it was on and when the
   process was started with it off (``enable()`` would have forced it on);
3. a failing load restores the flag too and still fails closed;
4. the loader never writes ``PYTORCH_JIT`` (no environment leak into engines).
"""
from __future__ import annotations

import re
import sys
import types
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

torch = pytest.importorskip("torch")
from torch.jit import _state as jit_state  # noqa: E402

QUALITY_PY = REPO_ROOT / "src" / "evaluation" / "quality.py"


class _FakeModel:
    config = types.SimpleNamespace(id2label={0: "entailment", 1: "neutral", 2: "contradiction"})


def _install_fake_transformers(monkeypatch, seen, fail=False):
    def pipeline(task, model, device):
        seen.append(bool(jit_state._enabled))
        if fail:
            raise RuntimeError("simulated load failure")
        return types.SimpleNamespace(model=_FakeModel(), task=task, model_name=model)

    fake = types.ModuleType("transformers")
    fake.pipeline = pipeline
    monkeypatch.setitem(sys.modules, "transformers", fake)


def _evaluator():
    from src.evaluation.quality import QualityEvaluator

    return QualityEvaluator(
        use_nli=True, use_embeddings=False, use_bertscore=False, use_rouge=False,
        use_lettucedetect=False, strict=True,
    )


@pytest.fixture()
def jit_flag_restored(monkeypatch):
    monkeypatch.delenv("CAGE_NLI_REVISION", raising=False)
    prior = jit_state._enabled.enabled
    yield
    jit_state._enabled.enabled = prior


def test_nli_loads_with_torchscript_disabled_and_restores_an_enabled_flag(
        monkeypatch, jit_flag_restored) -> None:
    jit_state._enabled.enabled = True
    seen = []
    _install_fake_transformers(monkeypatch, seen)
    ev = _evaluator()
    assert ev.nli_model is not None
    assert seen == [False], "TorchScript must be OFF while the checkpoint loads"
    assert bool(jit_state._enabled) is True, "the prior (enabled) flag is restored"


def test_nli_load_restores_a_disabled_flag_instead_of_forcing_it_on(
        monkeypatch, jit_flag_restored) -> None:
    # A process started with PYTORCH_JIT=0 (the S0 rescore) must stay off
    # afterwards: torch.jit._state.enable() sets True unconditionally, so the
    # loader restores the saved value, never calls enable().
    jit_state._enabled.enabled = False
    seen = []
    _install_fake_transformers(monkeypatch, seen)
    assert _evaluator().nli_model is not None
    assert seen == [False]
    assert bool(jit_state._enabled) is False


def test_failed_nli_load_restores_the_flag_and_fails_closed(
        monkeypatch, jit_flag_restored) -> None:
    from src.evaluation.quality import InstrumentUnavailableError

    jit_state._enabled.enabled = True
    seen = []
    _install_fake_transformers(monkeypatch, seen, fail=True)
    ev = _evaluator()
    with pytest.raises(InstrumentUnavailableError, match="simulated load failure"):
        _ = ev.nli_model
    assert seen == [False]
    assert bool(jit_state._enabled) is True


def test_loader_never_touches_the_environment_and_names_the_cause() -> None:
    text = QUALITY_PY.read_text(encoding="utf-8")
    code = "\n".join(l for l in text.splitlines() if not l.lstrip().startswith("#"))
    assert not re.search(r"environ\[[\"']PYTORCH_JIT|setdefault\([\"']PYTORCH_JIT", code), (
        "the fix must not set PYTORCH_JIT: an env var leaks into child engines and is read "
        "by torch only at import time")
    assert "jit_state.disable()" in code or "_state.disable()" in code
    assert "3.13.8" in text and "ADR-0129" in text
