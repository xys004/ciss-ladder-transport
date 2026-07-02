#!/usr/bin/env python3
"""Search which CISS observable can land near 90% over broad parameter sweeps."""

from __future__ import annotations

import csv
import itertools
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.integrate import quad

KB = 8.617e-5
TEMP = 300.0

LAMBDA_VALUES = [0.001, 0.002, 0.005, 0.008, 0.01, 0.02, 0.05, 0.1]
T_VALUES = [0.001, 0.005, 0.01, 0.05, 0.1]
GAMMA_VALUES = [0.00001, 0.00006, 0.0001, 0.001, 0.01]
ALPHA_FACTORS = [0.01, 0.05, 0.1, 0.2, 0.3, 0.5, 1.0, 2.0]
A0_VALUES = [0.1, 0.5, 1, 2, 5, 10]
UNIT_MODES = [("eV", 1.0), ("literal", 1.0), ("meV_to_eV", 1e-3)]
DP_VALUES = [-0.5, -0.2, 0.0, 0.2, 0.5]


@dataclass(frozen=True)
class P:
    Lambda: float
    t: float
    Gamma: float
    alpha_factor: float
    a0: float
    unit_mode: str

    @property
    def alpha(self) -> float:
        return self.alpha_factor * math.pi / 5.0

    @property
    def deco1(self) -> complex:
        return self.a0 * self.Lambda**2 * np.exp(1j * self.alpha)

    @property
    def deco2(self) -> complex:
        return self.a0 * self.Lambda**2 * np.exp(-1j * self.alpha)


def ferm_array(e: np.ndarray) -> np.ndarray:
    x = np.clip(e / (KB * TEMP), -700, 700)
    return 1.0 / (1.0 + np.exp(x))


def ferm(e: float) -> float:
    x = e / (KB * TEMP)
    if x > 700:
        return 0.0
    if x < -700:
        return 1.0
    return 1.0 / (1.0 + math.exp(x))


def metric_values(a: float, b: float) -> dict[str, float]:
    denom = a + b
    absden = abs(a) + abs(b)
    maxabs = max(abs(a), abs(b), 1e-300)
    out = {
        "P": (a - b) / denom if abs(denom) > 1e-300 else np.nan,
        "A_max": abs(a - b) / maxabs,
        "A_sum": abs(a - b) / absden if absden > 1e-300 else np.nan,
        "R": a / b if abs(b) > 1e-300 else np.nan,
        "S_up": 1.0 - b / a if abs(a) > 1e-300 else np.nan,
        "S_down": 1.0 - a / b if abs(b) > 1e-300 else np.nan,
    }
    return out


def add_candidate(cands: list[dict], p: P, family: str, observable: str, value: float, context: str, den: float = np.nan) -> None:
    if not np.isfinite(value):
        return
    score = abs(abs(value) - 0.90)
    if score <= 0.20:
        cands.append(
            {
                "score_to_0p90": score,
                "abs_value": abs(value),
                "value": value,
                "family": family,
                "observable": observable,
                "context": context,
                "Lambda": p.Lambda,
                "t": p.t,
                "Gamma": p.Gamma,
                "alpha_factor": p.alpha_factor,
                "alpha": p.alpha,
                "a0": p.a0,
                "unit_mode": p.unit_mode,
                "denominator": den,
            }
        )


def legacy_grid(p: P, cands: list[dict]) -> None:
    # A bounded fast search. Resonance resolution scales with Gamma but is capped for runtime.
    n = 1601 if p.Gamma >= 1e-4 else 3201
    E = np.linspace(-1.0, 1.0, n)
    d2 = E + 0.5j * p.Gamma
    absd2 = np.abs(d2) ** 2
    gr1 = 1.0 / (E + 0.5j * p.Gamma - p.t**2 / d2 - p.deco1 / d2)
    gr2 = 1.0 / (E + 0.5j * p.Gamma - p.t**2 / d2 - p.deco2 / d2)
    ga1 = p.Gamma * (p.t**2 + p.deco1) / absd2
    ga2 = p.Gamma * (p.t**2 + p.deco2) / absd2
    f0 = ferm_array(E)
    for DP in DP_VALUES:
        fdp = ferm_array(E - DP)
        occ1 = (p.Gamma * fdp + ga1 * f0) / (p.Gamma + ga1)
        occ2a = occ1
        occ2b = (p.Gamma * fdp + ga2 * f0) / (p.Gamma + ga2)
        i1 = float(np.trapz(np.real(occ1 * 1j * gr1), E))
        i2a = float(np.trapz(np.real(occ2a * 1j * gr2), E))
        i2b = float(np.trapz(np.real(occ2b * 1j * gr2), E))
        for version, i2 in [("A", i2a), ("B", i2b)]:
            metrics = metric_values(i1, i2)
            for key, val in metrics.items():
                name = {
                    "P": "P_legacy",
                    "A_max": "A_legacy",
                    "R": "R_legacy",
                    "S_up": "S_legacy_up",
                    "S_down": "S_legacy_down",
                    "A_sum": "A_sum_legacy",
                }[key]
                den = i1 + i2 if key == "P" else max(abs(i1), abs(i2))
                add_candidate(cands, p, "legacy", f"{name}_{version}", val, f"DP={DP}", den)


def exact_trans_grid(p: P, E: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    # Hermitian 4x4 with Du^2-like=Deco1 and Dd^2-like=Deco2.
    Du = math.sqrt(p.a0) * p.Lambda * np.exp(0.5j * p.alpha)
    Dd = math.sqrt(p.a0) * p.Lambda * np.exp(-0.5j * p.alpha)
    H = np.array(
        [[0, 0, p.t, Du], [0, 0, Dd, p.t], [p.t, np.conj(Dd), 0, 0], [np.conj(Du), p.t, 0, 0]],
        dtype=complex,
    )
    sigma = np.diag([-0.5j * p.Gamma, -0.5j * p.Gamma, -0.5j * p.Gamma, -0.5j * p.Gamma])
    tu = np.empty_like(E)
    td = np.empty_like(E)
    I = np.eye(4)
    for k, e in enumerate(E):
        G = np.linalg.inv((float(e) + 1e-16j) * I - H - sigma)
        tu[k] = p.Gamma**2 * (abs(G[2, 0]) ** 2 + abs(G[2, 1]) ** 2)
        td[k] = p.Gamma**2 * (abs(G[3, 0]) ** 2 + abs(G[3, 1]) ** 2)
    return tu, td


def spectral_and_current_grid(p: P, cands: list[dict]) -> None:
    scale = max(p.t, math.sqrt(p.a0) * p.Lambda, p.Gamma, 1e-12)
    lim = min(1.0, max(0.01, 3.0 * scale))
    n = 1201
    E = np.linspace(-lim, lim, n)

    # Decoupled physical abs/real modes. For this conjugate pair these should be identical between channels.
    for mode in ["abs", "real"]:
        K1 = p.t**2 + (abs(p.deco1) if mode == "abs" else p.deco1.real)
        K2 = p.t**2 + (abs(p.deco2) if mode == "abs" else p.deco2.real)
        T1 = p.Gamma**2 * K1 / np.abs((E + 0.5j * p.Gamma) ** 2 - K1) ** 2
        T2 = p.Gamma**2 * K2 / np.abs((E + 0.5j * p.Gamma) ** 2 - K2) ** 2
        analyze_spectral_arrays(p, cands, E, T1.real, T2.real, f"decoupled_{mode}", exact=False)

    # Complex "transmission" diagnostic: not physical, but scan it because it may create artifacts.
    K1c = p.t**2 + p.deco1
    K2c = p.t**2 + p.deco2
    T1c = p.Gamma**2 * K1c / np.abs((E + 0.5j * p.Gamma) ** 2 - K1c) ** 2
    T2c = p.Gamma**2 * K2c / np.abs((E + 0.5j * p.Gamma) ** 2 - K2c) ** 2
    analyze_spectral_arrays(p, cands, E, T1c.real, T2c.real, "complex_K_realpart_nonphysical", exact=False)

    # Exact 4x4 is intentionally not swept for every point here: 28.8k
    # parameter sets times a dense spectral grid is expensive. The earlier
    # validation showed the Hermitian equal-magnitude construction is
    # polarization-neutral; this broad search focuses on locating any 90%
    # observable, then legacy candidates are adaptively refined below.


def analyze_spectral_arrays(p: P, cands: list[dict], E: np.ndarray, Tu: np.ndarray, Td: np.ndarray, family: str, exact: bool) -> None:
    mu_weight = {}
    peak_metrics = metric_values(float(np.max(Tu)), float(np.max(Td)))
    add_candidate(cands, p, "spectral" if exact else "spectral/decoupled", "A_peak", peak_metrics["A_sum"], family, np.max(Tu) + np.max(Td))

    for DP in [-0.5, -0.2, 0.2, 0.5]:
        w = ferm_array(E - DP) - ferm_array(E)
        iu = float(np.trapz(Tu * w, E))
        idn = float(np.trapz(Td * w, E))
        metrics = metric_values(iu, idn)
        for key in ["P", "A_max", "A_sum", "S_up", "S_down", "R"]:
            add_candidate(cands, p, "current_grid" if exact else "current_decoupled_grid", key, metrics[key], f"{family}, DP={DP}", iu + idn)

    peaks = []
    for T in [Tu, Td]:
        idx = int(np.argmax(T))
        peaks.append(float(E[idx]))
    for E0 in peaks:
        for mult in [1, 5, 10, 50]:
            delta = max(mult * p.Gamma, (E[1] - E[0]) * 2)
            mask = np.abs(E - E0) <= delta
            if mask.sum() < 3:
                continue
            iu = float(np.trapz(Tu[mask], E[mask]))
            idn = float(np.trapz(Td[mask], E[mask]))
            val = abs(iu - idn) / (abs(iu) + abs(idn)) if abs(iu) + abs(idn) > 0 else np.nan
            add_candidate(cands, p, "windowed_spectral" if exact else "windowed_decoupled", "A_window", val, f"{family}, E0={E0:.6g}, delta={mult}Gamma", iu + idn)


def refine_legacy_quad(p: P, version: str, DP: float) -> tuple[float, float, dict[str, float]]:
    def one(spin: int) -> float:
        def integrand(e: float) -> float:
            d2 = e + 0.5j * p.Gamma
            deco = p.deco1 if spin == 1 else p.deco2
            G = 1.0 / (e + 0.5j * p.Gamma - p.t**2 / d2 - deco / d2)
            pref_deco = p.deco1 if (spin == 1 or version == "A") else p.deco2
            Ga = p.Gamma * (p.t**2 + pref_deco) / abs(d2) ** 2
            occ = (p.Gamma * ferm(e - DP) + Ga * ferm(e)) / (p.Gamma + Ga)
            return float(np.real(occ * 1j * G))

        pts = [0.0, DP]
        val, _ = quad(integrand, -1, 1, points=pts, epsabs=1e-11, epsrel=1e-9, limit=2000)
        return float(val)

    i1, i2 = one(1), one(2)
    return i1, i2, metric_values(i1, i2)


def parse_context_dp(context: str) -> float | None:
    if "DP=" not in context:
        return None
    tail = context.split("DP=", 1)[1].split(",", 1)[0]
    try:
        return float(tail)
    except ValueError:
        return None


def main() -> None:
    root = Path(__file__).resolve().parent
    out = root / "find_90_results"
    out.mkdir(exist_ok=True)
    candidates: list[dict] = []

    combos = list(itertools.product(LAMBDA_VALUES, T_VALUES, GAMMA_VALUES, ALPHA_FACTORS, A0_VALUES, UNIT_MODES))
    for idx, (La0, t0, G0, af, a0, (mode, scale)) in enumerate(combos, 1):
        p = P(La0 * scale, t0 * scale, G0 * scale, af, a0, mode)
        legacy_grid(p, candidates)
        spectral_and_current_grid(p, candidates)
        if idx % 500 == 0:
            print(f"scanned {idx}/{len(combos)}, candidates={len(candidates)}")

    candidates.sort(key=lambda r: r["score_to_0p90"])
    prelim = candidates[:500]

    # Refine top legacy candidates with adaptive quad and update their value.
    refined = []
    for r in prelim:
        rr = dict(r)
        if r["family"] == "legacy":
            DP = parse_context_dp(r["context"])
            version = "_A" if r["observable"].endswith("_A") else "_B"
            version = version[-1]
            p = P(r["Lambda"], r["t"], r["Gamma"], r["alpha_factor"], r["a0"], r["unit_mode"])
            if DP is not None:
                i1, i2, m = refine_legacy_quad(p, version, DP)
                key = r["observable"].replace("_A", "").replace("_B", "")
                metric_key = {
                    "P_legacy": "P",
                    "A_legacy": "A_max",
                    "A_sum_legacy": "A_sum",
                    "R_legacy": "R",
                    "S_legacy_up": "S_up",
                    "S_legacy_down": "S_down",
                }.get(key)
                if metric_key:
                    rr["value_refined"] = m[metric_key]
                    rr["abs_value_refined"] = abs(m[metric_key])
                    rr["score_refined"] = abs(abs(m[metric_key]) - 0.90)
                    rr["i1_refined"] = i1
                    rr["i2_refined"] = i2
                    rr["denominator_refined"] = i1 + i2 if metric_key == "P" else max(abs(i1), abs(i2))
        refined.append(rr)

    refined.sort(key=lambda r: r.get("score_refined", r["score_to_0p90"]))

    fields = sorted({k for r in refined for k in r.keys()})
    with (out / "top_90_candidates.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(refined[:200])

    top20 = refined[:20]
    lines = [
        "# Search for near-90 observables",
        "",
        f"Parameter sets scanned including duplicate literal/eV modes: {len(combos)}",
        f"Candidates within |abs(value)-0.90|<=0.20 before refinement: {len(candidates)}",
        "",
        "## Top 20 closest to 0.90",
        "| rank | abs value | observable | family | context | Lambda | t | Gamma | alpha_factor | a0 | unit | flags |",
        "|---:|---:|---|---|---|---:|---:|---:|---:|---:|---|---|",
    ]
    for i, r in enumerate(top20, 1):
        val = r.get("abs_value_refined", r["abs_value"])
        den = r.get("denominator_refined", r["denominator"])
        flags = []
        if r["family"] == "legacy":
            flags.append("not net current; DP=0 nonzero generally")
        if "window" in r["family"]:
            flags.append("window-dependent")
        if "complex_K" in r["context"]:
            flags.append("nonphysical complex K realpart")
        if np.isfinite(den) and abs(den) < 1e-8:
            flags.append("small denominator")
        lines.append(
            f"| {i} | {val:.6g} | {r['observable']} | {r['family']} | {r['context']} | "
            f"{r['Lambda']:.6g} | {r['t']:.6g} | {r['Gamma']:.6g} | {r['alpha_factor']:.3g} | {r['a0']:.3g} | {r['unit_mode']} | {'; '.join(flags)} |"
        )

    physical = [r for r in refined if r["family"] in ("current_grid", "current_decoupled_grid") and "complex_K" not in r["context"]]
    legacy = [r for r in refined if r["family"] == "legacy"]
    window = [r for r in refined if "window" in r["family"] and "complex_K" not in r["context"]]
    lines += [
        "",
        "## Diagnostics",
        f"Best legacy abs value near 0.90: {legacy[0].get('abs_value_refined', legacy[0]['abs_value']):.6g} ({legacy[0]['observable']}, {legacy[0]['context']})" if legacy else "No legacy candidates.",
        f"Best physical-current abs value near 0.90: {physical[0].get('abs_value_refined', physical[0]['abs_value']):.6g} ({physical[0]['observable']}, {physical[0]['context']})" if physical else "No physical-current candidates.",
        f"Best physical/windowed abs value near 0.90: {window[0].get('abs_value_refined', window[0]['abs_value']):.6g} ({window[0]['observable']}, {window[0]['context']})" if window else "No windowed candidates.",
        "",
        "Interpretation: legacy quantities are proxies, not net currents; exact physical current candidates are only grid-search estimates unless listed as legacy-refined.",
    ]
    (out / "FIND_90_REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    print(out / "FIND_90_REPORT.md")


if __name__ == "__main__":
    main()
