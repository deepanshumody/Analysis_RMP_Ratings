"""Run the full analysis: save every figure to reports/figures/ and the scalar
results to reports/results.json. Reproducible entry point for the README/site.

Usage:  python scripts/run_analysis.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: render to files, never open a window

from ape import (
    questions,  # noqa: E402
    viz,  # noqa: E402
)

REPORTS = Path(__file__).resolve().parents[1] / "reports"
FIGURES = REPORTS / "figures"


def main() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    all_results = {}
    for i in range(1, 12):
        results, figures = getattr(questions, f"q{i}")()
        all_results[f"q{i}"] = results
        for name, fig in figures.items():
            viz.save_figure(fig, name, out_dir=FIGURES)
            fig.clf()
    (REPORTS / "results.json").write_text(json.dumps(all_results, indent=2, default=str))
    print(f"Wrote {REPORTS / 'results.json'} and {len(list(FIGURES.glob('*.png')))} figures.")


if __name__ == "__main__":
    main()
