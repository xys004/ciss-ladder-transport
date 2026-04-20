"""Generate a figure-ready plot for the coherent charge-transmission kernel T^0(E)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from ciss_ladder.plotting import plot_energy_kernel


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=REPO_ROOT / "data" / "raw" / "fig3_T0_coherent_N10_eta0.csv")
    parser.add_argument("--output", type=Path, default=REPO_ROOT / "figures" / "fig3_T0_coherent.png")
    args = parser.parse_args()
    plot_energy_kernel(args.input, "energy", "T0", r"$T^0(E)$", args.output, title="Coherent charge-transmission kernel")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
