#!/usr/bin/env python3
"""Verify exact 4x4 Landauer polarization for the top near-90 legacy cases."""

from __future__ import annotations

import csv
import math
from pathlib import Path

import numpy as np
from scipy.integrate import quad

KB = 8.617e-5
TEMP = 300.0


def ferm(e: float) -> float:
    x = e / (KB * TEMP)
    if x > 700:
        return 0.0
    if x < -700:
        return 1.0
    return 1.0 / (1.0 + math.exp(x))


def exact_p(Lambda: float, t: float, Gamma: float, alpha_factor: float, a0: float, DP: float) -> tuple[float, float, float, float]:
    alpha = alpha_factor * math.pi / 5.0
    Du = math.sqrt(a0) * Lambda * np.exp(0.5j * alpha)
    Dd = math.sqrt(a0) * Lambda * np.exp(-0.5j * alpha)
    H = np.array(
        [[0, 0, t, Du], [0, 0, Dd, t], [t, np.conj(Dd), 0, 0], [np.conj(Du), t, 0, 0]],
        dtype=complex,
    )
    sigma = np.diag([-0.5j * Gamma, -0.5j * Gamma, -0.5j * Gamma, -0.5j * Gamma])
    I4 = np.eye(4)
    eig = list(np.linalg.eigvalsh(H).real) + [0.0, DP]
    pts = sorted({round(float(x), 15) for x in eig if -1 < x < 1})

    def trans(e: float) -> tuple[float, float]:
        G = np.linalg.inv((e + 1e-16j) * I4 - H - sigma)
        tu = Gamma**2 * (abs(G[2, 0]) ** 2 + abs(G[2, 1]) ** 2)
        td = Gamma**2 * (abs(G[3, 0]) ** 2 + abs(G[3, 1]) ** 2)
        return float(tu), float(td)

    def integ(spin: int) -> float:
        def f(e: float) -> float:
            tu, td = trans(e)
            return (tu if spin == 1 else td) * (ferm(e - DP) - ferm(e))

        val, _ = quad(f, -1, 1, points=pts, epsabs=1e-12, epsrel=1e-10, limit=2000)
        return float(val)

    iu, idn = integ(1), integ(2)
    P = (iu - idn) / (iu + idn) if abs(iu + idn) > 1e-300 else float("nan")
    # Peak asymmetry on a local grid.
    lim = min(1.0, max(0.01, 3 * max(t, math.sqrt(a0) * Lambda, Gamma)))
    E = np.linspace(-lim, lim, 2001)
    T = np.array([trans(float(e)) for e in E])
    peak = abs(T[:, 0].max() - T[:, 1].max()) / (T[:, 0].max() + T[:, 1].max())
    return iu, idn, P, float(peak)


def main() -> None:
    root = Path(__file__).resolve().parent
    inp = root / "find_90_results" / "top_90_candidates.csv"
    out = root / "find_90_results" / "top_legacy_exact4x4_check.csv"
    rows = list(csv.DictReader(inp.open()))
    seen = set()
    checks = []
    for r in rows:
        if r["family"] != "legacy":
            continue
        key = (r["Lambda"], r["t"], r["Gamma"], r["alpha_factor"], r["a0"], r["context"])
        if key in seen:
            continue
        seen.add(key)
        DP = float(r["context"].split("DP=", 1)[1])
        iu, idn, P, peak = exact_p(float(r["Lambda"]), float(r["t"]), float(r["Gamma"]), float(r["alpha_factor"]), float(r["a0"]), DP)
        checks.append(
            {
                "Lambda": r["Lambda"],
                "t": r["t"],
                "Gamma": r["Gamma"],
                "alpha_factor": r["alpha_factor"],
                "a0": r["a0"],
                "DP": DP,
                "legacy_observable": r["observable"],
                "legacy_abs_refined": r.get("abs_value_refined") or r["abs_value"],
                "Iup_exact": iu,
                "Idown_exact": idn,
                "P_exact_4x4": P,
                "A_peak_exact_4x4": peak,
            }
        )
        if len(checks) >= 12:
            break
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(checks[0].keys()))
        w.writeheader()
        w.writerows(checks)
    print(out)


if __name__ == "__main__":
    main()
