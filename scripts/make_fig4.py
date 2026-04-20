"""Generate a figure-ready plot for integrated spin-current scans versus chain length."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from ciss_ladder.plotting import plot_length_scan


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=REPO_ROOT / "data" / "processed" / "fig4_Iz_scan.csv")
    parser.add_argument("--output", type=Path, default=REPO_ROOT / "figures" / "fig4_Iz_scan.png")
    parser.add_argument("--columns", nargs="+", default=["Iz"])
    args = parser.parse_args()
    plot_length_scan(args.input, "N", args.columns, args.output, ylabel=r"$I_z$")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
