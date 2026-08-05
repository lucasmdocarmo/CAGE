"""Offline analysis package — pure domain logic (stdlib + pandas/numpy/scipy only).

Import submodules by full path (e.g. ``from src.analysis.cellspec import CellSpec``);
this package initializer stays minimal by design.
"""


def __getattr__(name: str):  # PEP 562 lazy export — appended, keeps imports light
    """Expose ``src.analysis.l0_retrieval`` (charter D8 §8.2 Layer-0 scorer)
    as an attribute without importing pandas at package-import time."""
    if name == "l0_retrieval":
        import importlib

        return importlib.import_module("src.analysis.l0_retrieval")
    raise AttributeError(f"module 'src.analysis' has no attribute {name!r}")
