#!/usr/bin/env python3
"""
Verify experiment outputs by checking that each metrics JSON has a matching
CSV with the expected number of rows and consistent baseline metadata.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import pandas as pd


def verify_dir(results_dir: Path) -> dict:
    report = {
        "results_dir": str(results_dir),
        "checks": [],
        "ok": True,
    }

    # Descend into trial_*/ -- multi-trial runs write per-trial
    # <label>_<dataset>_<ts>_metrics.json under trial_N/, not at the cell root. Exclude the
    # cell-root aggregated_metrics.json (it has no sibling *_results.csv). Hard-fail on zero
    # matches so a misdirected --results-dir cannot silently pass (audit false-pass fix).
    metrics_files = [
        p for p in sorted(results_dir.rglob("*_metrics.json"))
        if p.name != "aggregated_metrics.json"
    ]
    if not metrics_files:
        report["ok"] = False
        report["errors"] = ["no_per_trial_metrics_found"]

    for metrics_path in metrics_files:
        with open(metrics_path, "r") as f:
            metrics = json.load(f)

        baseline = metrics.get("experiment", {}).get("baseline")
        expected_requests = metrics.get("performance", {}).get("total_requests")
        dataset = metrics.get("experiment", {}).get("dataset")
        model = metrics.get("experiment", {}).get("model")

        csv_path = metrics_path.with_name(metrics_path.name.replace("_metrics.json", "_results.csv"))
        check = {
            "baseline": baseline,
            "dataset": dataset,
            "model": model,
            "metrics_file": str(metrics_path),
            "csv_file": str(csv_path),
            "expected_requests": expected_requests,
            "actual_rows": None,
            "ok": True,
            "errors": [],
        }

        if not csv_path.exists():
            check["ok"] = False
            check["errors"].append("missing_results_csv")
        else:
            df = pd.read_csv(csv_path)
            check["actual_rows"] = int(len(df))
            if expected_requests is not None and check["actual_rows"] != expected_requests:
                check["ok"] = False
                check["errors"].append("row_count_mismatch")

        report["checks"].append(check)
        if not check["ok"]:
            report["ok"] = False

    # Metric-coverage section (2026-07-15 audit): per cell x key metric, how many valid
    # rows actually carry a value, split by trial. Makes coverage pathologies visible
    # (e.g. the fixture's bertscore 1/3/1 rows per trial, silently averaged before) so a
    # sparse metric can't masquerade as a well-estimated one. Advisory: never flips ok.
    try:
        import sys as _sys
        _sys.path.insert(0, str(Path(__file__).resolve().parent))
        from _results_loader import load_results_long, metric_values, valid_rows

        cov_metrics = ["grounding_score", "faithfulness", "completeness_bertscore",
                       "completeness_rouge_l", "ttft_ms", "abstention_precision"]
        long_df = load_results_long(results_dir)
        v = valid_rows(long_df)
        coverage = []
        for cell, df_cell in v.groupby("cell", sort=True):
            for metric in cov_metrics:
                scored = metric_values(df_cell, metric).notna()
                by_trial = {int(t): int(scored[df_cell["trial"] == t].sum())
                            for t in sorted(df_cell["trial"].unique())}
                coverage.append({
                    "cell": cell, "metric": metric,
                    "n_valid_rows": int(len(df_cell)),
                    "n_scored": int(scored.sum()),
                    "per_trial_scored": by_trial,
                })
        report["metric_coverage"] = coverage
    except SystemExit:
        pass  # no results.csv trees under this dir (e.g. bare metrics check) -- skip
    except Exception as exc:  # advisory section must never break verification
        report["metric_coverage_error"] = f"{type(exc).__name__}: {exc}"

    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify experiment results in a directory.")
    parser.add_argument("--results-dir", required=True, help="Directory containing metrics/CSV files.")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    report = verify_dir(results_dir)

    report_path = results_dir / "verification_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    txt_path = results_dir / "verification_report.txt"
    with open(txt_path, "w") as f:
        f.write(f"Results dir: {results_dir}\n")
        f.write(f"Overall OK: {report['ok']}\n\n")
        for check in report["checks"]:
            f.write(
                f"{check['baseline']} | rows={check['actual_rows']} "
                f"expected={check['expected_requests']} | ok={check['ok']}\n"
            )
            if check["errors"]:
                f.write(f"  errors: {', '.join(check['errors'])}\n")
        for cov in report.get("metric_coverage", []):
            if cov["n_scored"] < cov["n_valid_rows"]:
                f.write(
                    f"COVERAGE {cov['cell']} {cov['metric']}: "
                    f"{cov['n_scored']}/{cov['n_valid_rows']} rows scored "
                    f"(per-trial {cov['per_trial_scored']})\n"
                )

    print(f"Wrote {report_path}")
    print(f"Wrote {txt_path}")


if __name__ == "__main__":
    main()
