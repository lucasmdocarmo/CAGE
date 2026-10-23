"""The pod environment lock (S1 audit 2026-10-23): four constraints files minted
from the S0 attempt 5 bootstrap log, one per pip step, read by
scripts/runpod/setup_runpod.sh (PIP_CONSTRAINT for the cage-env steps, -c for
the engine venvs) and, for the final cage-env state, by preflight gate (i).
Static pins: the files parse, agree with requirements.txt's exact pins, carry the
two cage-env states S0's log recorded, and the bootstrap wires each one where it
belongs.
"""
from __future__ import annotations

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SETUP = REPO_ROOT / "scripts" / "runpod" / "setup_runpod.sh"
LOCKS = {
    "vllm-step": REPO_ROOT / "requirements.lock.gpu.vllm-step.txt",
    "final": REPO_ROOT / "requirements.lock.gpu.txt",
    "sglang": REPO_ROOT / "requirements.lock.sglang-env.txt",
    "lmdeploy": REPO_ROOT / "requirements.lock.lmdeploy-env.txt",
}
_PIN = re.compile(r"^([A-Za-z0-9][A-Za-z0-9._-]*)==([^\s#]+)$")


def _lock(name: str) -> dict:
    out = {}
    for raw in LOCKS[name].read_text(encoding="utf-8").splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        m = _PIN.match(line)
        assert m, f"{LOCKS[name].name}: not a name==version line: {raw!r}"
        assert m.group(1) not in out, f"{LOCKS[name].name}: {m.group(1)} listed twice"
        out[m.group(1)] = m.group(2)
    return out


def test_every_lock_parses_and_is_sized_like_the_s0_log() -> None:
    sizes = {k: len(_lock(k)) for k in LOCKS}
    assert sizes == {"vllm-step": 183, "final": 344, "sglang": 180, "lmdeploy": 112}, sizes
    for name, lock in LOCKS.items():
        head = lock.read_text(encoding="utf-8").splitlines()[1]
        assert "S0 attempt 5" in head and "2026-10-14" in lock.read_text(encoding="utf-8")


def test_the_final_lock_agrees_with_every_exact_pin_of_requirements_txt() -> None:
    final = _lock("final")
    pins = {}
    for raw in (REPO_ROOT / "requirements.txt").read_text(encoding="utf-8").splitlines():
        line = raw.split("#", 1)[0].strip()
        m = _PIN.match(line)
        if m:
            pins[m.group(1)] = m.group(2)
    assert pins, "requirements.txt carries no exact pin?"
    missing = {k: v for k, v in pins.items() if final.get(k) != v}
    assert not missing, f"Tier-1 pins the final lock does not carry at the same version: {missing}"
    assert "cage-stats" not in final  # pinned by its git commit, never a version constraint
    assert final["pydantic-settings"] == "2.15.0" and final["python-dotenv"] == "1.0.1"


def test_the_two_cage_env_states_are_the_ones_s0_recorded_adr_0126() -> None:
    step, final = _lock("vllm-step"), _lock("final")
    assert step["vllm"] == final["vllm"] == "0.19.1"
    # vLLM's own numba pin at the vLLM step; the Tier-1 numba and numpy after the requirements step
    assert step["numba"] == "0.61.2" and final["numba"] == "0.67.0"
    assert step["numpy"] == "2.2.6" and final["numpy"] == "2.5.1"
    assert step["transformers"].startswith("5.") and final["transformers"] == "4.57.6"
    assert set(step) <= set(final) | {"numba"}  # the step set survives into the final state
    for k in ("pip", "setuptools", "wheel"):
        assert step[k] == final[k]


def test_the_engine_venv_locks_pin_the_served_builds() -> None:
    sgl, lmd = _lock("sglang"), _lock("lmdeploy")
    assert sgl["sglang"] == "0.5.10.post1" and sgl["torch"] == "2.9.1" and sgl["flash-attn-4"] == "4.0.0b34"
    assert lmd["lmdeploy"] == "0.17.0" and lmd["torch"] == "2.10.0"


def test_the_bootstrap_wires_each_lock_to_its_pip_step() -> None:
    text = SETUP.read_text(encoding="utf-8")
    lines = text.splitlines()

    def first(needle: str) -> int:
        hits = [i for i, l in enumerate(lines) if needle in l]
        assert hits, f"setup_runpod.sh lacks {needle!r}"
        return hits[0]

    vllm_lock = first('export PIP_CONSTRAINT="$_lock_vllm_step"')
    tooling = first("pip install --upgrade pip setuptools wheel")
    vllm = first('pip install "vllm==${VLLM_VERSION}"')
    final_lock = first('export PIP_CONSTRAINT="$_lock_final"')
    reqs = first("pip install -r requirements.txt")
    openai = first('pip install -U "openai>=2.0"')
    unset = [i for i, l in enumerate(lines) if l.strip() == "unset PIP_CONSTRAINT"][-1]
    sglang = first("install_engine_venv sglang-env")
    lmdeploy = first("install_engine_venv lmdeploy-env")
    assert vllm_lock < tooling < vllm < final_lock < reqs < openai < unset < sglang < lmdeploy
    assert any("requirements.lock.sglang-env.txt" in l for l in lines[sglang - 3:sglang])
    assert any("requirements.lock.lmdeploy-env.txt" in l for l in lines[lmdeploy - 2:lmdeploy])
    assert '${_lock_sglang[@]+"${_lock_sglang[@]}"} "sglang==${SGLANG_VERSION}"' in text
    assert '${_lock_lmdeploy[@]+"${_lock_lmdeploy[@]}"} "torch==${LMDEPLOY_TORCH_VERSION}"' in text
    # a missing lock never stops the bootstrap, it warns
    assert text.count("resolves against today's PyPI (unlocked)") == 3
