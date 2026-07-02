#!/usr/bin/env python3
"""Audit the legacy observable apparently used in 2sitiosCompletoCRRT.nb."""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.integrate import quad

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

KB = 8.617e-5
TEMP = 300.0


@dataclass(frozen=True)
class LegacyParams:
    en1: float = 0.0
    en2: float = 0.0
    Lambda: float = 2.0e-3
    t: float = 5.0e-3
    Gamma: float = 1.0e-3
    alpha: float = 0.1 * math.pi / 5.0
    a0: float = 1.0

    @property
    def deco1(self) -> complex:
        return self.a0 * self.Lambda**2 * complex(math.cos(self.alpha), math.sin(self.alpha))

    @property
    def deco2(self) -> complex:
        return self.a0 * self.Lambda**2 * complex(math.cos(self.alpha), -math.sin(self.alpha))


def ferm(e: float, T: float = TEMP) -> float:
    x = e / (KB * T)
    if x > 700:
        return 0.0
    if x < -700:
        return 1.0
    return 1.0 / (1.0 + math.exp(x))


def denom2(E: float, p: LegacyParams) -> complex:
    return E - p.en2 + 0.5j * p.Gamma


def gr(E: float, p: LegacyParams, spin: int) -> complex:
    deco = p.deco1 if spin == 1 else p.deco2
    d2 = denom2(E, p)
    return 1.0 / (E - p.en1 + 0.5j * p.Gamma - p.t**2 / d2 - deco / d2)


def gars(E: float, p: LegacyParams, spin: int) -> complex:
    deco = p.deco1 if spin == 1 else p.deco2
    return p.Gamma * (p.t**2 + deco) / abs(denom2(E, p)) ** 2


def legacy_integrand(E: float, DP: float, p: LegacyParams, spin: int, version: str) -> float:
    G = gr(E, p, spin)
    pref_spin = 1 if (spin == 1 or version == "A") else 2
    Ga = gars(E, p, pref_spin)
    occ = (p.Gamma * ferm(E - DP, TEMP) + Ga * ferm(E, TEMP)) / (p.Gamma + Ga)
    return float(np.real(occ * 1j * G))


def points(DP: float, p: LegacyParams) -> list[float]:
    vals = [
        0.0,
        DP,
        math.sqrt(p.t**2 + abs(p.deco1)),
        -math.sqrt(p.t**2 + abs(p.deco1)),
        math.sqrt(p.t**2 + abs(p.deco2)),
        -math.sqrt(p.t**2 + abs(p.deco2)),
        math.sqrt(p.t**2 + p.Lambda**2),
        -math.sqrt(p.t**2 + p.Lambda**2),
    ]
    return sorted({round(float(x), 15) for x in vals if -2.0 < x < 2.0})


def integrate_legacy(DP: float, p: LegacyParams, spin: int, version: str, bound: float) -> tuple[float, float]:
    pts = [x for x in points(DP, p) if -bound < x < bound]
    val, err = quad(
        lambda E: legacy_integrand(E, DP, p, spin, version),
        -bound,
        bound,
        points=pts,
        epsabs=1e-11,
        epsrel=1e-9,
        limit=2000,
    )
    return float(val), float(err)


def safe_ratio(a: float, b: float) -> float:
    if abs(b) < 1e-28:
        return float("nan")
    return float(a / b)


def derived(i1: float, i2: float) -> dict[str, float]:
    s = i1 + i2
    d = i1 - i2
    return {
        "Delta": d,
        "P": safe_ratio(d, s),
        "A": abs(d) / max(abs(i1), abs(i2), 1e-28),
        "R": safe_ratio(i1, i2),
        "rel_supp_up": 1.0 - safe_ratio(i1, i2),
        "rel_supp_down": 1.0 - safe_ratio(i2, i1),
        "sum": s,
    }


def physical_t_dec(E: float, DP: float, p: LegacyParams, channel: int, mode: str) -> complex:
    deco = p.deco1 if channel == 1 else p.deco2
    if mode == "abs":
        K = p.t**2 + abs(deco)
    elif mode == "real":
        K = p.t**2 + deco.real
    elif mode == "complex":
        K = p.t**2 + deco
    else:
        raise ValueError(mode)
    den = (E + 0.5j * p.Gamma) ** 2 - K
    return p.Gamma**2 * K / abs(den) ** 2


def integrate_physical_dec(DP: float, p: LegacyParams, channel: int, mode: str, bound: float = 1.0) -> float:
    pts = [x for x in points(DP, p) if -bound < x < bound]

    def integrand(E: float) -> float:
        # Notebook bias convention: f(E-DP)-f(E), i.e. mu_L=DP, mu_R=0.
        return float(np.real(physical_t_dec(E, DP, p, channel, mode))) * (ferm(E - DP, TEMP) - ferm(E, TEMP))

    val, _ = quad(integrand, -bound, bound, points=pts, epsabs=1e-12, epsrel=1e-10, limit=2000)
    return float(val)


def h4(p: LegacyParams) -> np.ndarray:
    Du = p.Lambda * np.exp(0.5j * p.alpha)
    Dd = p.Lambda * np.exp(-0.5j * p.alpha)
    t = p.t
    return np.array(
        [
            [0.0, 0.0, t, Du],
            [0.0, 0.0, Dd, t],
            [t, np.conj(Dd), 0.0, 0.0],
            [np.conj(Du), t, 0.0, 0.0],
        ],
        dtype=complex,
    )


def exact_trans(E: float, p: LegacyParams) -> tuple[float, float]:
    Gm = p.Gamma
    sigma = np.diag([-0.5j * Gm, -0.5j * Gm, -0.5j * Gm, -0.5j * Gm])
    G = np.linalg.inv((E + 1e-16j) * np.eye(4) - h4(p) - sigma)
    tu = Gm * Gm * (abs(G[2, 0]) ** 2 + abs(G[2, 1]) ** 2)
    td = Gm * Gm * (abs(G[3, 0]) ** 2 + abs(G[3, 1]) ** 2)
    return float(tu), float(td)


def integrate_exact_4x4(DP: float, p: LegacyParams, spin: int, bound: float = 1.0) -> float:
    eigs = list(np.linalg.eigvalsh(h4(p)).real)
    pts = sorted({round(float(x), 15) for x in eigs + [0.0, DP] if -bound < x < bound})

    def integrand(E: float) -> float:
        tu, td = exact_trans(E, p)
        return (tu if spin == 1 else td) * (ferm(E - DP, TEMP) - ferm(E, TEMP))

    val, _ = quad(integrand, -bound, bound, points=pts, epsabs=1e-12, epsrel=1e-10, limit=2000)
    return float(val)


def pol(i1: float, i2: float) -> float:
    return safe_ratio(i1 - i2, i1 + i2)


def scan(p: LegacyParams, dps: np.ndarray, out_csv: Path) -> list[dict[str, float]]:
    rows = []
    for DP in dps:
        i1_a, _ = integrate_legacy(float(DP), p, 1, "A", 1.0)
        i2_a, _ = integrate_legacy(float(DP), p, 2, "A", 1.0)
        i1_b = i1_a
        i2_b, _ = integrate_legacy(float(DP), p, 2, "B", 1.0)
        da = derived(i1_a, i2_a)
        db = derived(i1_b, i2_b)

        i1_exact = integrate_exact_4x4(float(DP), p, 1, 1.0)
        i2_exact = integrate_exact_4x4(float(DP), p, 2, 1.0)
        i1_dec = integrate_physical_dec(float(DP), p, 1, "abs", 1.0)
        i2_dec = integrate_physical_dec(float(DP), p, 2, "abs", 1.0)
        rows.append(
            {
                "DP": float(DP),
                "i1_A": i1_a,
                "i2_A": i2_a,
                "Delta_A": da["Delta"],
                "P_A": da["P"],
                "A_A": da["A"],
                "R_A": da["R"],
                "rel_supp_up_A": da["rel_supp_up"],
                "rel_supp_down_A": da["rel_supp_down"],
                "sum_A": da["sum"],
                "i1_B": i1_b,
                "i2_B": i2_b,
                "Delta_B": db["Delta"],
                "P_B": db["P"],
                "A_B": db["A"],
                "R_B": db["R"],
                "rel_supp_up_B": db["rel_supp_up"],
                "rel_supp_down_B": db["rel_supp_down"],
                "sum_B": db["sum"],
                "P_exact_4x4": pol(i1_exact, i2_exact),
                "P_decoupled_physical": pol(i1_dec, i2_dec),
            }
        )
    with out_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return rows


def plot(rows: list[dict[str, float]], outdir: Path, suffix: str) -> None:
    x = np.array([r["DP"] for r in rows])
    specs = [
        ("legacy_i_A", ["i1_A", "i2_A"], "legacy i1/i2 version A"),
        ("legacy_i_B", ["i1_B", "i2_B"], "legacy i1/i2 version B"),
        ("legacy_P", ["P_A", "P_B"], "P_legacy A/B"),
        ("legacy_A", ["A_A", "A_B"], "A_legacy A/B"),
        ("legacy_R", ["R_A", "R_B"], "R_legacy A/B"),
        ("P_compare", ["P_A", "P_B", "P_exact_4x4", "P_decoupled_physical"], "legacy vs physical P"),
    ]
    for name, cols, title in specs:
        plt.figure(figsize=(7.0, 4.4))
        for col in cols:
            plt.plot(x, [r[col] for r in rows], label=col)
        plt.xlabel("DP (eV)")
        plt.ylabel(title)
        plt.grid(alpha=0.25)
        plt.legend()
        plt.tight_layout()
        plt.savefig(outdir / f"{name}_{suffix}.png", dpi=180)
        plt.close()


def complex_transmission_diagnostics(p: LegacyParams) -> str:
    xs = np.linspace(-0.05, 0.05, 1001)
    vals1 = np.array([physical_t_dec(float(x), 0.0, p, 1, "complex") for x in xs])
    vals2 = np.array([physical_t_dec(float(x), 0.0, p, 2, "complex") for x in xs])
    return "\n".join(
        [
            f"complex T1 max |Im| = {np.max(np.abs(vals1.imag)):.6g}, min Re = {np.min(vals1.real):.6g}",
            f"complex T2 max |Im| = {np.max(np.abs(vals2.imag)):.6g}, min Re = {np.min(vals2.real):.6g}",
            "complex T1/T2 are not real transmissions" if max(np.max(np.abs(vals1.imag)), np.max(np.abs(vals2.imag))) > 1e-12 else "complex T1/T2 real on sampled grid",
        ]
    )


def summarize(rows: list[dict[str, float]], p: LegacyParams, label: str) -> list[str]:
    def maxabs(col: str) -> float:
        return max(abs(r[col]) for r in rows if np.isfinite(r[col]))

    def minmax(col: str) -> tuple[float, float]:
        vals = [r[col] for r in rows if np.isfinite(r[col])]
        return min(vals), max(vals)

    eq = min(rows, key=lambda r: abs(r["DP"]))
    zero_cross = [r["DP"] for r in rows if abs(r["sum_A"]) < 1e-3 or abs(r["sum_B"]) < 1e-3]
    best = {
        "P_A": maxabs("P_A"),
        "P_B": maxabs("P_B"),
        "A_A": maxabs("A_A"),
        "A_B": maxabs("A_B"),
        "R_A": maxabs("R_A"),
        "R_B": maxabs("R_B"),
        "rel_supp_up_A": maxabs("rel_supp_up_A"),
        "rel_supp_down_A": maxabs("rel_supp_down_A"),
        "rel_supp_up_B": maxabs("rel_supp_up_B"),
        "rel_supp_down_B": maxabs("rel_supp_down_B"),
        "P_exact_4x4": maxabs("P_exact_4x4"),
        "P_decoupled_physical": maxabs("P_decoupled_physical"),
    }
    near90 = [k for k, v in best.items() if 0.85 <= v <= 0.95]
    lines = [
        f"## Scan {label}",
        f"DP range: {rows[0]['DP']:.3g} to {rows[-1]['DP']:.3g} eV, N={len(rows)}",
        f"DP=0: i1_A={eq['i1_A']:.9g}, i2_A={eq['i2_A']:.9g}, i2_B={eq['i2_B']:.9g}, P_A={eq['P_A']:.9g}, P_B={eq['P_B']:.9g}",
        f"i1_A min/max: {minmax('i1_A')[0]:.9g}, {minmax('i1_A')[1]:.9g}",
        f"i2_A min/max: {minmax('i2_A')[0]:.9g}, {minmax('i2_A')[1]:.9g}",
        f"max |P_A|={best['P_A']:.9g}, max |P_B|={best['P_B']:.9g}",
        f"max |A_A|={best['A_A']:.9g}, max |A_B|={best['A_B']:.9g}",
        f"max |R_A|={best['R_A']:.9g}, max |R_B|={best['R_B']:.9g}",
        f"max relative suppressions: up_A={best['rel_supp_up_A']:.9g}, down_A={best['rel_supp_down_A']:.9g}, up_B={best['rel_supp_up_B']:.9g}, down_B={best['rel_supp_down_B']:.9g}",
        f"max |P_exact_4x4|={best['P_exact_4x4']:.9g}, max |P_decoupled_physical|={best['P_decoupled_physical']:.9g}",
        f"near-90 quantities: {near90 if near90 else 'none'}",
        f"sum near zero count, threshold 1e-3: {len(zero_cross)}",
    ]
    return lines


def convergence_check(p: LegacyParams) -> str:
    vals = []
    for DP in [0.0, 0.2, 0.5]:
        i1_1, _ = integrate_legacy(DP, p, 1, "A", 1.0)
        i1_2, _ = integrate_legacy(DP, p, 1, "A", 2.0)
        i2_1, _ = integrate_legacy(DP, p, 2, "A", 1.0)
        i2_2, _ = integrate_legacy(DP, p, 2, "A", 2.0)
        vals.append(f"DP={DP:g}: i1[-1,1]={i1_1:.9g}, i1[-2,2]={i1_2:.9g}, diff={i1_2-i1_1:.3g}; i2 diff={i2_2-i2_1:.3g}")
    return "\n".join(vals)


def main() -> None:
    root = Path(__file__).resolve().parent
    out = root / "legacy_results"
    out.mkdir(exist_ok=True)
    p = LegacyParams()

    rows_full = scan(p, np.linspace(-0.5, 0.5, 101), out / "legacy_observables.csv")
    rows_paper = scan(p, np.linspace(-0.2, 0.2, 81), out / "legacy_observables_pm0p2.csv")
    plot(rows_full, out, "pm0p5")
    plot(rows_paper, out, "pm0p2")

    report = [
        "# Legacy Mathematica observable audit",
        "",
        f"Parameters: Lambda={p.Lambda}, t={p.t}, Gamma={p.Gamma}, alpha={p.alpha}, Deco1={p.deco1}, Deco2={p.deco2}",
        "",
        "## Range convergence",
        "```",
        convergence_check(p),
        "```",
        "",
        "## Complex transmission diagnostic",
        "```",
        complex_transmission_diagnostics(p),
        "```",
        "",
        *summarize(rows_full, p, "[-0.5,0.5]"),
        "",
        *summarize(rows_paper, p, "[-0.2,0.2]"),
        "",
        "## Diagnostic answers",
        "1. The notebook formula does not directly compute Landauer P=(Iup-Idown)/(Iup+Idown); it computes two legacy proxies i1/i2 and one can form ratios from them.",
        "2. i1/i2 are not physical net currents: at DP=0 they are nonzero.",
        "3. Version A is the literal notebook form with GaRs1 also in i2; version B is the symmetric corrected variant.",
        "4. The complex K transmission form is not a real positive transmission.",
        "5. Recommended wording: effective spin-channel imbalance / spin-dependent spectral asymmetry, separated from transport polarization.",
        "6. Recommendation: D) Separar claramente legacy spin preference de transport polarization; for physical transport use C) redo with Landauer/NEGF.",
    ]
    (out / "LEGACY_REPORT.md").write_text("\n".join(report), encoding="utf-8")
    print(out / "LEGACY_REPORT.md")


if __name__ == "__main__":
    main()
