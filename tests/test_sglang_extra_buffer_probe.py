"""The SGLang extra-buffer mechanism probe (owner 2026-11-08, decision 2 option 3,
MyDocs/RunPod/S2_SGLANG_EXTRA_BUFFER_PROBE_SPEC_2026-11-08.md, option 1).

One cell of cd-act1 serves arm B3 on SGLang with ``--mamba-scheduler-strategy
extra_buffer --disable-overlap-schedule`` as its own serving family. The
variant is a CellSpec coordinate (``engine_variant``), legal on SGLang only,
carried in the row key as ``variant=<name>`` so every reader labels the cell
apart from the default SGLang cells; absent, every key, flat dict and record
stays byte-identical to before.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
for _p in (str(REPO_ROOT / "scripts" / "4_analysis"), str(REPO_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import figure_pipeline as fp  # noqa: E402
import organize_results as org  # noqa: E402
from src.analysis import conditioned_curves as cc  # noqa: E402
from src.analysis import l0_retrieval as l0  # noqa: E402
from src.analysis.cellspec import ENGINE_VARIANTS, CellSpec, CellSpecError, InvalidCellSpecError  # noqa: E402
from src.orchestration.campaign_session import CampaignSessionError, derive_cell_spec  # noqa: E402


def _spec(**overrides: object) -> CellSpec:
    base: dict[str, object] = dict(
        arm="corpus-reuse", retriever="none", policy="none", topology="single",
        engine="sglang", model="qwen3-next-80b", family="F1",
    )
    base.update(overrides)
    return CellSpec(**base)  # type: ignore[arg-type]


XB_KEY = "corpus-reuse|none|none|single|sglang|qwen3-next-80b|F1|variant=extra_buffer"


# --- the axis -------------------------------------------------------------------


def test_the_variant_is_registered_for_sglang_alone() -> None:
    assert ENGINE_VARIANTS == {"sglang": frozenset({"extra_buffer"})}


def test_the_variant_is_legal_on_sglang_and_enters_the_row_key_last() -> None:
    xb = _spec(engine_variant="extra_buffer")
    assert xb.engine_variant == "extra_buffer" and xb.to_row_key() == XB_KEY
    assert xb != _spec() and xb.to_row_key() != _spec().to_row_key()
    pressured = _spec(family="F3", budget_r=1.0, rate_frac=0.95, engine_variant="extra_buffer")
    assert pressured.to_row_key() == "corpus-reuse|none|none|single|sglang|qwen3-next-80b|F3|r1|lam0.95|variant=extra_buffer"
    assert CellSpec.from_baseline("B3", engine="sglang", model="qwen3-next-80b", engine_variant="extra_buffer") == xb


@pytest.mark.parametrize(
    "overrides",
    [
        dict(engine="vllm", engine_variant="extra_buffer"),   # the variant is SGLang's
        dict(engine="hf", engine_variant="extra_buffer"),
        dict(engine_variant="no_buffer"),                      # the default is absence, never a value
        dict(engine_variant=""),
        dict(engine_variant=1),
    ],
)
def test_illegal_variants_refuse(overrides: dict[str, object]) -> None:
    with pytest.raises(InvalidCellSpecError):
        _spec(**overrides)


def test_the_flat_dict_carries_the_variant_only_when_set() -> None:
    flat = _spec().to_flat_dict()
    assert "engine_variant" not in flat
    xb = _spec(engine_variant="extra_buffer")
    assert xb.to_flat_dict()["engine_variant"] == "extra_buffer"
    assert CellSpec.from_flat_dict(xb.to_flat_dict()) == xb
    for absent in (None, "", float("nan")):
        assert CellSpec.from_flat_dict(dict(flat, engine_variant=absent)) == _spec()
    with pytest.raises(CellSpecError):
        CellSpec.from_flat_dict(dict(flat, engine_variant="no_buffer"))


# --- every row-key reader -----------------------------------------------------------


def test_the_strict_row_key_parsers_round_trip_the_variant() -> None:
    xb = _spec(engine_variant="extra_buffer")
    assert fp.parse_row_key(XB_KEY) == xb.to_flat_dict()
    assert fp.parse_row_key(_spec().to_row_key()) == _spec().to_flat_dict()
    assert org.parse_row_key_dir(XB_KEY) == xb
    retr = CellSpec.from_baseline("B7", engine="sglang", model="qwen3-next-80b", engine_variant="extra_buffer")
    assert l0._validate_row_key(retr.to_row_key()) == retr
    for bad in (XB_KEY.replace("variant=extra_buffer", "variant=no_buffer"), XB_KEY.replace("variant=", "var=")):
        with pytest.raises(fp.FigureDataError):
            fp.parse_row_key(bad)
        with pytest.raises(org.OrganizeError):
            org.parse_row_key_dir(bad)


def test_the_curve_identity_keeps_the_variant_as_a_mechanism_coordinate() -> None:
    pressured = _spec(family="F3", budget_r=1.0, rate_frac=0.95, engine_variant="extra_buffer").to_row_key()
    assert cc.mechanism_engine_key(pressured) == "corpus-reuse|none|none|single|sglang|qwen3-next-80b|F3|variant=extra_buffer"
    assert cc.mechanism_engine_key(_spec(family="F3", budget_r=1.0, rate_frac=0.95).to_row_key()) == (
        "corpus-reuse|none|none|single|sglang|qwen3-next-80b|F3"
    )


# --- the runner seam --------------------------------------------------------------------


def test_the_runner_derives_the_variant_from_its_own_env_key() -> None:
    env = {"CAGE_CELL_ARM": "corpus-reuse", "CAGE_CELL_RETRIEVER": "none", "CAGE_CELL_ENGINE_VARIANT": "extra_buffer"}
    spec = derive_cell_spec(baseline="cag_true_on", baseline_label=None, backend="sglang",
                            model="Qwen/Qwen3-Next-80B-A3B-Instruct", env=env)
    assert spec == _spec(engine_variant="extra_buffer")
    plain = derive_cell_spec(baseline="cag_true_on", baseline_label=None, backend="sglang",
                             model="Qwen/Qwen3-Next-80B-A3B-Instruct", env={k: v for k, v in env.items() if "VARIANT" not in k})
    assert plain == _spec()
    with pytest.raises(CampaignSessionError, match="charter-illegal"):
        derive_cell_spec(baseline="cag_true_on", baseline_label=None, backend="vllm",
                         model="Qwen/Qwen3-Next-80B-A3B-Instruct", env=env)
