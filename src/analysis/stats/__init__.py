"""D9 statistics machinery (PUBLICATION.md §9.1-§9.13).

Test engine (§9.3-§9.5): ``families`` (the §7.8 contrast registry + family
map), ``tests_by_unit`` (paired Wilcoxon / McNemar / batch-means by §9.4
unit), ``corrections`` (Holm, BH-FDR), ``equivalence`` (conditional two-layer
TOST), ``gatekeeping`` (the §9.3 chain), ``wlt`` (mandatory win/loss/tie).
Proof-of-honesty modules (§9.6-§9.13): ``blinding``, ``calibration``,
``ledger``, ``power_sim``, ``prereg``.

Import submodules by full path (e.g. ``from src.analysis.stats.ledger import
hash_artifacts``); this package initializer stays minimal by design. No file
I/O in the test-engine modules — plain numpy/pandas in, typed results out.
"""
