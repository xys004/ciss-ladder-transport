#!/usr/bin/env python3
"""Independent CISS two-site validation: SymPy, Z3, exact 4x4 NEGF, and benchmarks."""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import sympy as sp
from scipy.integrate import quad
from z3 import And, Not, Or, Real, Solver, sat, unsat

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

KB_EV = 8.617333262e-5


@dataclass(frozen=True)
class Params:
    name: str
    eps1u: float = 0.0
    eps1d: float = 0.0
    eps2u: float = 0.0
    eps2d: float = 0.0
    V: float = 0.010
    GammaL: float = 0.001
    GammaR: float = 0.001
    Du: complex = 0.008 + 0j
    Dd: complex = -0.008 + 0j
    T: float = 300.0


def fermi(E: float, mu: float, T: float) -> float:
    x = (E - mu) / (KB_EV * T)
    if x > 700:
        return 0.0
    if x < -700:
        return 1.0
    return 1.0 / (math.exp(x) + 1.0)


def hamiltonian(p: Params) -> np.ndarray:
    return np.array(
        [
            [p.eps1u, 0.0, p.V, p.Du],
            [0.0, p.eps1d, p.Dd, p.V],
            [np.conj(p.V), np.conj(p.Dd), p.eps2u, 0.0],
            [np.conj(p.Du), np.conj(p.V), 0.0, p.eps2d],
        ],
        dtype=complex,
    )


def green(E: float, p: Params) -> np.ndarray:
    sigma = np.diag(
        [
            -0.5j * p.GammaL,
            -0.5j * p.GammaL,
            -0.5j * p.GammaR,
            -0.5j * p.GammaR,
        ]
    )
    return np.linalg.inv((E + 1e-16j) * np.eye(4) - hamiltonian(p) - sigma)


def transmissions_exact(E: float, p: Params) -> tuple[float, float]:
    G = green(E, p)
    tu = p.GammaL * p.GammaR * (abs(G[2, 0]) ** 2 + abs(G[2, 1]) ** 2)
    td = p.GammaL * p.GammaR * (abs(G[3, 0]) ** 2 + abs(G[3, 1]) ** 2)
    return float(max(tu.real, 0.0)), float(max(td.real, 0.0))


def transmission_dec(E: float, p: Params, spin: str) -> float:
    D = p.Du if spin == "up" else p.Dd
    eps1 = p.eps1u if spin == "up" else p.eps1d
    eps2 = p.eps2u if spin == "up" else p.eps2d
    K = p.V**2 + abs(D) ** 2
    denom = (E - eps1 + 0.5j * p.GammaL) * (E - eps2 + 0.5j * p.GammaR) - K
    return float((p.GammaL * p.GammaR * K / abs(denom) ** 2).real)


def integration_points(p: Params, muL: float, muR: float) -> list[float]:
    vals = list(np.linalg.eigvalsh(hamiltonian(p)).real)
    vals += [muL, muR]
    vals += [
        math.sqrt(p.V**2 + abs(p.Du) ** 2),
        -math.sqrt(p.V**2 + abs(p.Du) ** 2),
        math.sqrt(p.V**2 + abs(p.Dd) ** 2),
        -math.sqrt(p.V**2 + abs(p.Dd) ** 2),
    ]
    return sorted({round(float(x), 15) for x in vals})


def integration_bounds(points: list[float], p: Params) -> tuple[float, float]:
    margin = max(0.35, 25 * KB_EV * p.T, 80 * max(p.GammaL, p.GammaR))
    return min(points) - margin, max(points) + margin


def integrate_current(
    p: Params,
    muL: float,
    muR: float,
    which: str,
    mode: str,
    epsrel: float = 1e-10,
) -> tuple[float, float]:
    pts = integration_points(p, muL, muR)
    lo, hi = integration_bounds(pts, p)

    def integrand(E: float) -> float:
        if mode == "exact":
            tu, td = transmissions_exact(E, p)
            t = tu if which == "up" else td
        else:
            t = transmission_dec(E, p, which)
        return t * (fermi(E, muL, p.T) - fermi(E, muR, p.T))

    val, err = quad(integrand, lo, hi, points=pts, epsabs=1e-12, epsrel=epsrel, limit=1200)
    return float(val), float(err)


def integrate_current_grid(p: Params, muL: float, muR: float, which: str, mode: str, n: int) -> float:
    pts = integration_points(p, muL, muR)
    lo, hi = integration_bounds(pts, p)
    xs = np.linspace(lo, hi, n)
    vals = []
    for E in xs:
        if mode == "exact":
            tu, td = transmissions_exact(float(E), p)
            t = tu if which == "up" else td
        else:
            t = transmission_dec(float(E), p, which)
        vals.append(t * (fermi(float(E), muL, p.T) - fermi(float(E), muR, p.T)))
    return float(np.trapz(vals, xs))


def mus(bias: float, convention: str) -> tuple[float, float]:
    if convention == "left0":
        return 0.0, bias
    if convention == "symmetric":
        return 0.5 * bias, -0.5 * bias
    raise ValueError(convention)


def pol(iu: float, idn: float) -> float:
    den = iu + idn
    if abs(den) < 1e-28:
        return float("nan")
    return float((iu - idn) / den)


def symbolic_output() -> str:
    V = sp.symbols("V", real=True)
    Du, Dd, Duc, Ddc = sp.symbols("Du Dd conjugate(Du) conjugate(Dd)")
    C = sp.Matrix([[V, Du], [Dd, V]])
    Cdag = sp.Matrix([[V, Ddc], [Duc, V]])
    ccdag = sp.simplify(C * Cdag)
    off = sp.simplify(ccdag[0, 1])
    trs = sp.simplify(off.subs({Ddc: -Du}))
    out = [
        "C*Cdag =",
        str(ccdag),
        f"offdiag(0,1) = {off}",
        f"offdiag under Dd=-Du* = {trs}",
        "decoupled-channel exact only in TRS limit",
    ]
    return "\n".join(out)


def z3_output() -> str:
    xu, yu, xd, yd, V = Real("xu"), Real("yu"), Real("xd"), Real("yd"), Real("V")
    trs = And(xd == -xu, yd == yu)
    off_re = V * (xd + xu)
    off_im = V * (-yd + yu)
    mag_u = xu * xu + yu * yu
    mag_d = xd * xd + yd * yd
    v_nonzero = V != 0

    def check(name: str, constraints: list) -> str:
        s = Solver()
        s.add(*constraints)
        r = s.check()
        if r == sat:
            return f"{name}: sat; model={s.model()}"
        return f"{name}: {r}"

    return "\n".join(
        [
            check("1 TRS & V!=0 & offdiag!=0", [trs, v_nonzero, Or(off_re != 0, off_im != 0)]),
            check("2 offdiag=0 & V!=0 & |Du|^2!=|Dd|^2", [v_nonzero, off_re == 0, off_im == 0, mag_u != mag_d]),
            check("3 |Du|^2!=|Dd|^2 & V!=0 & offdiag=0", [v_nonzero, mag_u != mag_d, off_re == 0, off_im == 0]),
            check("4a |Du|^2!=|Dd|^2 & TRS", [mag_u != mag_d, trs]),
            check("4b |Du|^2!=|Dd|^2 & broken TRS", [mag_u != mag_d, Not(trs)]),
        ]
    )


def scan_case(p: Params, biases: np.ndarray, convention: str, out_csv: Path) -> dict:
    rows = []
    max_pe = 0.0
    max_pd = 0.0
    hi_pe = float("nan")
    for b in biases:
        muL, muR = mus(float(b), convention)
        iue, _ = integrate_current(p, muL, muR, "up", "exact")
        ide, _ = integrate_current(p, muL, muR, "down", "exact")
        iud, _ = integrate_current(p, muL, muR, "up", "dec")
        idd, _ = integrate_current(p, muL, muR, "down", "dec")
        pe, pd = pol(iue, ide), pol(iud, idd)
        if not math.isnan(pe):
            max_pe = max(max_pe, abs(pe))
            if abs(float(b)) >= 0.19:
                hi_pe = pe
        if not math.isnan(pd):
            max_pd = max(max_pd, abs(pd))
        rows.append([b, iue, ide, pe, iud, idd, pd])
    with out_csv.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["bias_eV", "Iup_exact", "Idown_exact", "P_exact", "Iup_dec", "Idown_dec", "P_dec"])
        writer.writerows(rows)
    return {"max_P_exact": max_pe, "max_P_dec": max_pd, "high_bias_P_exact": hi_pe, "rows": rows}


def plot_bias(csv_path: Path, png_path: Path, title: str) -> None:
    arr = np.genfromtxt(csv_path, delimiter=",", names=True)
    plt.figure(figsize=(7.0, 4.4))
    plt.plot(arr["bias_eV"], arr["P_exact"], label="Exact 4x4", lw=2)
    plt.plot(arr["bias_eV"], arr["P_dec"], "--", label="Decoupled", lw=2)
    plt.xlabel("bias (eV)")
    plt.ylabel("P")
    plt.title(title)
    plt.ylim(-1.05, 1.05)
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(png_path, dpi=180)
    plt.close()


def plot_transmissions(p: Params, png_exact: Path, png_dec: Path) -> None:
    pts = integration_points(p, -0.2, 0.2)
    lo, hi = min(pts) - 0.06, max(pts) + 0.06
    xs = np.linspace(lo, hi, 1600)
    te = np.array([transmissions_exact(float(x), p) for x in xs])
    td_u = np.array([transmission_dec(float(x), p, "up") for x in xs])
    td_d = np.array([transmission_dec(float(x), p, "down") for x in xs])
    plt.figure(figsize=(7.0, 4.4))
    plt.plot(xs, te[:, 0], label="T_up exact")
    plt.plot(xs, te[:, 1], label="T_down exact")
    plt.xlabel("E (eV)")
    plt.ylabel("T(E)")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(png_exact, dpi=180)
    plt.close()
    plt.figure(figsize=(7.0, 4.4))
    plt.plot(xs, td_u, label="T_up dec")
    plt.plot(xs, td_d, label="T_down dec")
    plt.xlabel("E (eV)")
    plt.ylabel("T_dec(E)")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(png_dec, dpi=180)
    plt.close()


def grid_alias_test(p: Params, bias: float, convention: str) -> dict:
    muL, muR = mus(bias, convention)
    iuq, _ = integrate_current(p, muL, muR, "up", "exact")
    idq, _ = integrate_current(p, muL, muR, "down", "exact")
    out = {"quad_P": pol(iuq, idq)}
    for n in [101, 201, 501, 1001]:
        iug = integrate_current_grid(p, muL, muR, "up", "exact", n)
        idg = integrate_current_grid(p, muL, muR, "down", "exact", n)
        out[f"grid_{n}_P"] = pol(iug, idg)
    return out


def make_cases() -> dict[str, Params]:
    alpha_m = 0.2 * math.pi / 5
    alpha_f = 0.3 * math.pi / 5
    return {
        "trs": Params(name="trs", V=0.010, GammaL=0.001, GammaR=0.001, Du=0.008, Dd=-0.008),
        "noso": Params(name="noso", V=0.010, GammaL=0.001, GammaR=0.001, Du=0.0, Dd=0.0),
        "moderate": Params(name="moderate", V=0.010, GammaL=0.001, GammaR=0.001, Du=0.008, Dd=-0.008 * math.exp(-2 * alpha_m)),
        "moderate_rev_alpha": Params(name="moderate_rev_alpha", V=0.010, GammaL=0.001, GammaR=0.001, Du=0.008, Dd=-0.008 * math.exp(2 * alpha_m)),
        "fig4_g1mev": Params(name="fig4_g1mev", V=0.100, GammaL=0.001, GammaR=0.001, Du=0.0055, Dd=-0.0055 * math.exp(-2 * alpha_f)),
        "fig4_g0p1mev": Params(name="fig4_g0p1mev", V=0.100, GammaL=0.0001, GammaR=0.0001, Du=0.0055, Dd=-0.0055 * math.exp(-2 * alpha_f)),
        "fig4_g0p06mev": Params(name="fig4_g0p06mev", V=0.100, GammaL=0.00006, GammaR=0.00006, Du=0.0055, Dd=-0.0055 * math.exp(-2 * alpha_f)),
    }


def main() -> None:
    root = Path(__file__).resolve().parent
    out = root / "full_results"
    out.mkdir(exist_ok=True)
    cases = make_cases()
    biases = np.linspace(-0.2, 0.2, 41)
    positive_biases = np.linspace(0.0, 0.2, 31)

    sym = symbolic_output()
    z3 = z3_output()
    (out / "sympy_output.txt").write_text(sym, encoding="utf-8")
    (out / "z3_output.txt").write_text(z3, encoding="utf-8")

    summary = {}
    for convention in ["left0", "symmetric"]:
        for name in ["trs", "noso", "moderate", "moderate_rev_alpha", "fig4_g1mev", "fig4_g0p1mev", "fig4_g0p06mev"]:
            bs = biases if name.startswith("fig4") else positive_biases
            csv_path = out / f"{name}_{convention}.csv"
            summary[(name, convention)] = scan_case(cases[name], bs, convention, csv_path)
            if name in ["moderate", "fig4_g0p06mev"]:
                plot_bias(csv_path, out / f"{name}_{convention}_polarization.png", f"{name} {convention}")

    plot_transmissions(
        cases["fig4_g0p06mev"],
        out / "fig4_g0p06mev_transmission_exact.png",
        out / "fig4_g0p06mev_transmission_decoupled.png",
    )

    grid = {
        "fig4_g1mev_left0": grid_alias_test(cases["fig4_g1mev"], 0.2, "left0"),
        "fig4_g0p1mev_left0": grid_alias_test(cases["fig4_g0p1mev"], 0.2, "left0"),
        "fig4_g0p06mev_left0": grid_alias_test(cases["fig4_g0p06mev"], 0.2, "left0"),
    }

    report_lines = [
        "# CISS validation report",
        "",
        "## SymPy",
        "```",
        sym,
        "```",
        "",
        "## Z3",
        "```",
        z3,
        "```",
        "",
        "## Summary",
        "| case | bias convention | max |P_exact| | max |P_dec| | high-bias P_exact |",
        "|---|---:|---:|---:|---:|",
    ]
    for key in [
        ("trs", "left0"),
        ("trs", "symmetric"),
        ("noso", "left0"),
        ("moderate", "left0"),
        ("moderate", "symmetric"),
        ("moderate_rev_alpha", "left0"),
        ("fig4_g1mev", "left0"),
        ("fig4_g0p1mev", "left0"),
        ("fig4_g0p06mev", "left0"),
        ("fig4_g1mev", "symmetric"),
        ("fig4_g0p1mev", "symmetric"),
        ("fig4_g0p06mev", "symmetric"),
    ]:
        s = summary[key]
        report_lines.append(
            f"| {key[0]} | {key[1]} | {s['max_P_exact']:.6g} | {s['max_P_dec']:.6g} | {s['high_bias_P_exact']:.6g} |"
        )
    report_lines += [
        "",
        "## Grid alias test",
        "```",
        "\n".join(f"{k}: {v}" for k, v in grid.items()),
        "```",
        "",
        "## Conclusions",
        f"P~90% exact 4x4 survives? {'yes' if max(summary[(n, 'left0')]['max_P_exact'] for n in ['fig4_g1mev','fig4_g0p1mev','fig4_g0p06mev']) > 0.85 else 'no'}",
        f"P~90% decoupled appears? {'yes' if max(summary[(n, 'left0')]['max_P_dec'] for n in ['fig4_g1mev','fig4_g0p1mev','fig4_g0p06mev']) > 0.85 else 'no'}",
        "Chirality reversal alpha->-alpha flips sign? "
        + (
            "yes"
            if np.sign(summary[("moderate", "left0")]["rows"][-1][3])
            == -np.sign(summary[("moderate_rev_alpha", "left0")]["rows"][-1][3])
            else "no"
        ),
        "Recommendation: C) Eliminar 90% y reformular resultados."
        if max(summary[(n, "left0")]["max_P_exact"] for n in ["fig4_g1mev", "fig4_g0p1mev", "fig4_g0p06mev"]) < 0.85
        else "Recommendation: A) Mantener claim fuerte.",
    ]
    (out / "REPORT.md").write_text("\n".join(report_lines), encoding="utf-8")
    print(out / "REPORT.md")


if __name__ == "__main__":
    main()
