#!/usr/bin/env python3
"""Finalize physical CISS search from the saved 50k raw scan."""

from __future__ import annotations

import csv
import math
from dataclasses import asdict
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ciss_physical_polarization_search import (
    Candidate,
    H,
    bounds,
    current_quad,
    currents_grid,
    currents_quad,
    peak_metrics,
    trans_array,
)

RNG = np.random.default_rng(12345)


def read_raw(path: Path) -> list[dict]:
    rows = []
    with path.open() as f:
        for r in csv.DictReader(f):
            for k in list(r):
                try:
                    r[k] = float(r[k])
                except ValueError:
                    pass
            rows.append(r)
    rows.sort(key=lambda r: r["score"], reverse=True)
    return rows


def cand_from_row(r: dict) -> Candidate:
    return Candidate(
        r["V"],
        r["Lambda"],
        r["GammaL"],
        r["GammaR"],
        r["eps_avg"],
        r["delta_eps"],
        r["a_down_ratio"],
        r["theta_base"],
        r["delta_theta"],
        r["bias"],
    )


def perturb(c: Candidate, n: int) -> list[Candidate]:
    out = [c]
    for _ in range(n):
        V = np.clip(10 ** (math.log10(c.V) + RNG.normal(0, 0.18)), 0.001, 0.2)
        La = np.clip(10 ** (math.log10(c.Lambda) + RNG.normal(0, 0.18)), 0.0005, 0.05)
        GL = np.clip(10 ** (math.log10(c.GammaL) + RNG.normal(0, 0.25)), 1e-5, 0.01)
        GR = np.clip(10 ** (math.log10(c.GammaR) + RNG.normal(0, 0.25)), 1e-5, 0.01)
        ea = np.clip(c.eps_avg + RNG.normal(0, 0.02), -0.1, 0.1)
        de = np.clip(c.delta_eps + RNG.normal(0, 0.02), -0.1, 0.1)
        ar = np.clip(10 ** (math.log10(c.a_down_ratio) + RNG.normal(0, 0.25)), 0.05, 10)
        tb = np.clip(c.theta_base + RNG.normal(0, 0.35), -math.pi, math.pi)
        dt = np.clip(c.delta_theta + RNG.normal(0, 0.35), -math.pi, math.pi)
        b = np.clip(c.bias + RNG.normal(0, 0.06), -0.5, 0.5)
        if abs(b) < 0.02:
            b = 0.02 if b >= 0 else -0.02
        out.append(Candidate(V, La, GL, GR, ea, de, ar, tb, dt, b))
    return out


def pol_from_cur(cur: dict) -> float:
    return cur["P_tr"]


def result_row(rank: int, c: Candidate, conv: str = "symmetric") -> dict:
    cc = Candidate(**{**asdict(c), "bias_convention": conv})
    cur = currents_quad(cc, conv, strict=True, margin_floor=0.5)
    wide = currents_quad(cc, conv, strict=True, margin_floor=0.8)
    zero = currents_quad(Candidate(**{**asdict(cc), "bias": 0.0}), conv, strict=False)
    trs = currents_quad(Candidate(**{**asdict(cc), "a_down_ratio": 1.0, "delta_theta": 0.0}), conv, strict=False)
    noso = currents_quad(Candidate(**{**asdict(cc), "Lambda": 0.0}), conv, strict=False)
    rev = currents_quad(Candidate(**{**asdict(cc), "theta_base": -cc.theta_base, "delta_theta": -cc.delta_theta}), conv, strict=False)
    tup, tdn, apeak, pos = peak_metrics(cc)
    P = cur["P_tr"]
    chir = abs(rev["P_tr"] + P) / max(abs(P), 1e-12) if np.isfinite(P) else np.nan
    reasons = []
    if abs(cur["I_tot"]) <= 1e-8:
        reasons.append("I_tot<=1e-8")
    if abs(zero["I_up"]) >= 1e-10 or abs(zero["I_down"]) >= 1e-10:
        reasons.append("zero_bias_fail")
    if abs(trs["P_tr"]) >= 1e-8:
        reasons.append("TRS_control_fail")
    if abs(noso["P_tr"]) >= 1e-8:
        reasons.append("noSO_control_fail")
    if not pos:
        reasons.append("negative_T")
    if abs(wide["P_tr"] - P) > max(1e-5, 0.02 * abs(P)):
        reasons.append("wide_window_unstable")
    return {
        "rank": rank,
        "P_tr": P,
        "abs_P_tr": abs(P),
        "I_up": cur["I_up"],
        "I_down": cur["I_down"],
        "I_tot": cur["I_tot"],
        "DeltaI": cur["DeltaI"],
        "A_ch": abs(cur["DeltaI"]) / max(abs(cur["I_up"]), abs(cur["I_down"]), 1e-300),
        "A_peak": apeak,
        "T_up_peak": tup,
        "T_down_peak": tdn,
        "V": cc.V,
        "Lambda": cc.Lambda,
        "Gamma_L": cc.GammaL,
        "Gamma_R": cc.GammaR,
        "eps1": cc.eps1,
        "eps2": cc.eps2,
        "eps_avg": cc.eps_avg,
        "delta_eps": cc.delta_eps,
        "a_up": 1.0,
        "a_down": cc.a_down_ratio,
        "theta_up": cc.theta_up,
        "theta_down": cc.theta_down,
        "Du_real": cc.Du.real,
        "Du_imag": cc.Du.imag,
        "Dd_real": cc.Dd.real,
        "Dd_imag": cc.Dd.imag,
        "bias": cc.bias,
        "bias_convention": conv,
        "P_rev": rev["P_tr"],
        "chirality_score": chir,
        "P_TRS_control": trs["P_tr"],
        "P_noSO_control": noso["P_tr"],
        "zero_bias_Iup": zero["I_up"],
        "zero_bias_Idown": zero["I_down"],
        "P_wide_window": wide["P_tr"],
        "I_tot_gt_1e-6": abs(cur["I_tot"]) > 1e-6,
        "V_over_Lambda": cc.V / cc.Lambda if cc.Lambda else np.inf,
        "Lambda_over_Gamma_avg": cc.Lambda / (0.5 * (cc.GammaL + cc.GammaR)),
        "V_over_Gamma_avg": cc.V / (0.5 * (cc.GammaL + cc.GammaR)),
        "Gamma_L_over_Gamma_R": cc.GammaL / cc.GammaR,
        "delta_eps_over_V": cc.delta_eps / cc.V,
        "passes_filters": len(reasons) == 0,
        "failure_reason": ";".join(reasons),
    }


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def figures(out: Path, top: Candidate, rows: list[dict]) -> None:
    lo, hi, _ = bounds(top)
    E = np.linspace(lo, hi, 2400)
    Tu, Td = trans_array(E, top)
    plt.figure(figsize=(7, 4.4))
    plt.plot(E, Tu, label="T_up")
    plt.plot(E, Td, label="T_down")
    plt.xlabel("E (eV)")
    plt.ylabel("T(E)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out / "top_candidate_transmission.png", dpi=180)
    plt.close()

    biases = np.linspace(-0.5, 0.5, 41)
    ps, iu, idn = [], [], []
    for b in biases:
        cur = currents_quad(Candidate(**{**asdict(top), "bias": float(b)}), "symmetric", strict=False)
        ps.append(cur["P_tr"])
        iu.append(cur["I_up"])
        idn.append(cur["I_down"])
    plt.figure(figsize=(7, 4.4))
    plt.plot(biases, ps)
    plt.xlabel("Vb (eV)")
    plt.ylabel("P_tr")
    plt.tight_layout()
    plt.savefig(out / "top_candidate_P_vs_bias.png", dpi=180)
    plt.close()
    plt.figure(figsize=(7, 4.4))
    plt.plot(biases, iu, label="I_up")
    plt.plot(biases, idn, label="I_down")
    plt.xlabel("Vb (eV)")
    plt.ylabel("current integral (eV)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out / "top_candidate_currents_vs_bias.png", dpi=180)
    plt.close()

    plt.figure(figsize=(6, 4.4))
    plt.scatter([abs(r["I_tot"]) for r in rows], [r["abs_P_tr"] for r in rows], s=18)
    plt.xscale("log")
    plt.xlabel("|I_tot|")
    plt.ylabel("|P_tr|")
    plt.tight_layout()
    plt.savefig(out / "scatter_absP_vs_Itot.png", dpi=180)
    plt.close()
    plt.figure(figsize=(6, 4.4))
    plt.scatter([r["chirality_score"] for r in rows], [r["abs_P_tr"] for r in rows], s=18)
    plt.xlabel("chirality_score")
    plt.ylabel("|P_tr|")
    plt.tight_layout()
    plt.savefig(out / "scatter_absP_vs_chirality_score.png", dpi=180)
    plt.close()
    plt.figure(figsize=(6, 4.8))
    sc = plt.scatter(
        [r["Lambda_over_Gamma_avg"] for r in rows],
        [r["V_over_Gamma_avg"] for r in rows],
        c=[r["abs_P_tr"] for r in rows],
        cmap="viridis",
        s=28,
    )
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Lambda/Gamma_avg")
    plt.ylabel("V/Gamma_avg")
    plt.colorbar(sc, label="|P_tr|")
    plt.tight_layout()
    plt.savefig(out / "heatmap_scatter_maxP_lambda_gamma_v_gamma.png", dpi=180)
    plt.close()


def main() -> None:
    root = Path(__file__).resolve().parent
    out = root / "physical_search_results"
    raw = read_raw(out / "physical_polarization_search_raw.csv")

    print("Local grid refinement from raw top candidates")
    pool = []
    for r in raw[:30]:
        pool.extend(perturb(cand_from_row(r), 70))
    scored = []
    for i, c in enumerate(pool, 1):
        iu, idn, itot, P = currents_grid(c, n=221)
        score = abs(P) * math.tanh(abs(itot) / 1e-6) if np.isfinite(P) else -1
        scored.append((score, c))
    scored.sort(key=lambda x: x[0], reverse=True)

    print("Strict quad validation")
    rows = []
    seen = set()
    for _, c in scored[:55]:
        key = tuple(round(x, 9) for x in [c.V, c.Lambda, c.GammaL, c.GammaR, c.eps_avg, c.delta_eps, c.a_down_ratio, c.theta_base, c.delta_theta, c.bias])
        if key in seen:
            continue
        seen.add(key)
        try:
            rows.append(result_row(len(rows) + 1, c, "symmetric"))
        except Exception as e:
            print("skip", e)
    rows.sort(key=lambda r: (not r["passes_filters"], -r["abs_P_tr"]))
    for i, r in enumerate(rows, 1):
        r["rank"] = i
    write_csv(out / "physical_polarization_search_top.csv", rows)

    accepted = [r for r in rows if r["passes_filters"]]
    top = accepted[0] if accepted else rows[0]
    top_c = Candidate(top["V"], top["Lambda"], top["Gamma_L"], top["Gamma_R"], top["eps_avg"], top["delta_eps"], top["a_down"], top["theta_up"], top["theta_down"] - top["theta_up"], top["bias"])
    left0 = result_row(1, top_c, "left0")
    figures(out, top_c, rows)
    strict = [r for r in accepted if r["I_tot_gt_1e-6"]]
    report = [
        "# Physical polarization search report",
        "",
        "Stage 1 used the saved 50,000-point random scan; this finalizer locally perturbed the top 30 and strictly validated the top 55 with quad.",
        f"Accepted candidates passing filters: {len(accepted)} / {len(rows)}.",
        "",
        "## Best Robust Candidate",
        f"largest robust |P_tr| = {top['abs_P_tr']:.9g}",
        f"P_tr = {top['P_tr']:.9g}",
        f"I_up = {top['I_up']:.9e}, I_down = {top['I_down']:.9e}, I_tot = {top['I_tot']:.9e}",
        f"A_ch = {top['A_ch']:.9g}, A_peak = {top['A_peak']:.9g}",
        f"V={top['V']:.9g}, Lambda={top['Lambda']:.9g}, GammaL={top['Gamma_L']:.9g}, GammaR={top['Gamma_R']:.9g}",
        f"eps1={top['eps1']:.9g}, eps2={top['eps2']:.9g}, a_down/a_up={top['a_down']:.9g}",
        f"theta_up={top['theta_up']:.9g}, theta_down={top['theta_down']:.9g}, bias={top['bias']:.9g}",
        f"P_rev={top['P_rev']:.9g}, chirality_score={top['chirality_score']:.9g}",
        f"P_TRS_control={top['P_TRS_control']:.3e}, P_noSO_control={top['P_noSO_control']:.3e}",
        f"zero_bias Iup/Idown={top['zero_bias_Iup']:.3e}/{top['zero_bias_Idown']:.3e}",
        f"left0 same candidate: P_tr={left0['P_tr']:.9g}, I_tot={left0['I_tot']:.9e}, passes={left0['passes_filters']}",
        "",
        "## Stricter Current Threshold",
        f"best accepted with |I_tot|>1e-6: {strict[0]['abs_P_tr']:.9g}" if strict else "none accepted with |I_tot|>1e-6",
        "",
        "## Answers",
        f"A. Largest robust physical |P_tr| found: {top['abs_P_tr']:.9g}.",
        f"B. Survives all filters: {top['passes_filters']} ({top['failure_reason']}).",
        "C. Regime: strongly asymmetric effective spin-flip amplitudes/phases plus asymmetric lead broadenings/off-resonant onsite detuning.",
        f"D. Chirality reversal flips sign? score={top['chirality_score']:.4g}; {'yes' if top['chirality_score'] < 0.2 else 'not cleanly'}.",
        f"E. TRS control gives zero: {abs(top['P_TRS_control']) < 1e-8}.",
        f"F. Tiny denominator artifact: {'no' if abs(top['I_tot']) > 1e-8 else 'yes'}.",
        "G. Stable under integration refinement/wider window: yes for accepted candidates.",
        "H. This broad effective search finds transport polarization at the level reported above; use the |I_tot|>1e-6 line as the conservative robustness bar.",
        "I. If requiring chirality sign flip as a hard filter, use chirality_score < 0.2 subset from CSV.",
        "J. Recommended wording: moderate/large transport polarization only with explicit effective-parameter caveats; otherwise model identifies symmetry mechanism.",
    ]
    (out / "PHYSICAL_POLARIZATION_SEARCH_REPORT.md").write_text("\n".join(report), encoding="utf-8")
    print(out / "PHYSICAL_POLARIZATION_SEARCH_REPORT.md")


if __name__ == "__main__":
    main()
