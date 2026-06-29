"""Compare two eval reports and print the per-metric delta.

  Usage:  python eval/compare_reports.py <baseline.json> <current.json>

  Each report is what run_eval.py writes: {"scores": {...}, "thresholds": {...}, ...}.
  The deltas are the numbers you cite ("+14% context precision", etc.).
"""
from __future__ import annotations
import json
import sys
from pathlib import Path


def _load_scores(path: str) -> dict[str, float]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return data.get("scores", data)  # tolerate a bare scores dict too


def _fmt(v) -> str:
    return "n/a" if v is None else f"{v:.3f}"


def main() -> None:
    if len(sys.argv) != 3:
        sys.exit("Usage: python eval/compare_reports.py <baseline.json> <current.json>")

    baseline = _load_scores(sys.argv[1])
    current = _load_scores(sys.argv[2])
    metrics = sorted(set(baseline) | set(current))

    print(f"\n{'metric':<20}{'baseline':>10}{'current':>10}{'delta':>10}")
    print("-" * 52)
    improved = regressed = 0
    for m in metrics:
        b, c = baseline.get(m), current.get(m)
        if b is None or c is None:
            print(f"{m:<20}{_fmt(b):>10}{_fmt(c):>10}{'n/a':>10}")
            continue
        d = c - b
        arrow = "UP" if d > 0 else ("DOWN" if d < 0 else "=")
        improved += d > 0
        regressed += d < 0
        print(f"{m:<20}{b:>10.3f}{c:>10.3f}{d:>+10.3f}  {arrow}")
    print("-" * 52)
    print(f"{improved} improved, {regressed} regressed\n")


if __name__ == "__main__":
    main()