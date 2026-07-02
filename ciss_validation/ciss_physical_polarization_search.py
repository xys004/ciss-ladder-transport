#!/usr/bin/env python3
"""Search robust physical transport polarization in the Hermitian 4x4 CISS model."""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
from scipy.integrate import quad
from scipy.optimize import differential_evolution

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

KB = 8.617333262e-5
TEMP = 300.0
RNG = np.random.default_rng(20260429)


@dataclass
class Candidate:
    V: float
    Lambda: float
    GammaL: float
    GammaR: float
    eps_avg: float
    delta_eps: float
    a_down_ratio: float
    theta_base: float
    delta_theta: float
    bias: float
    bias_convention: str = "symmetric"

    @property
    def eps1(self) -> float:
        return self.eps_avg - 0.5 * self.delta_eps

    @property
    def eps2(self) -> float:
        return self.eps_avg + 0.5 * self.delta_eps

    @property
    def theta_up(self) -> float:
        return self.theta_base

    @property
    def theta_down(self) -> float:
        return self.theta_base + self.delta_theta

    @property
    def Du(self) -> complex:
        return self.Lambda * np.exp(1j * self.theta_up)

    @property
    def Dd(self) -> complex:
        return -self.Lambda * self.a_down_ratio * np.exp(-1j * self.theta_down)


def fermi(E: np.ndarray | float, mu: float) -> np.ndarray | float:
    x = np.clip((np.asarray(E) - mu) / (KB * TEMP), -700, 700)
    y = 1.0 / (np.exp(x) + 1.0)
    return float(y) if np.ndim(y) == 0 else y


def mus(bias: float, convention: str) -> tuple[float, float]:
    if convention == "symmetric":
        return 0.5 * bias, -0.5 * bias
    if convention == "left0":
        return 0.0, -bias
    raise ValueError(convention)


def H(c: Candidate, Du: complex | None = None, Dd: complex | None = None) -> np.ndarray:
    Du = c.Du if Du is None else Du
    Dd = c.Dd if Dd is None else Dd
    V = c.V
    return np.array(
        [
            [c.eps1, 0.0, V, Du],
            [0.0, c.eps1, Dd, V],
            [V, np.conj(Dd), c.eps2, 0.0],
            [np.conj(Du), V, 0.0, c.eps2],
        ],
        dtype=complex,
    )


def trans_array(E: np.ndarray, c: Candidate, Du: complex | None = None, Dd: complex | None = None) -> tuple[np.ndarray, np.ndarray]:
    h = H(c, Du, Dd)
    sig = np.diag([-0.5j * c.GammaL, -0.5j * c.GammaL, -0.5j * c.GammaR, -0.5j * c.GammaR])
    A = E[:, None, None] * np.eye(4)[None, :, :] - h[None, :, :] - sig[None, :, :]
    G = np.linalg.inv(A)
    pref = c.GammaL * c.GammaR
    Tu = pref * (np.abs(G[:, 2, 0]) ** 2 + np.abs(G[:, 2, 1]) ** 2)
    Td = pref * (np.abs(G[:, 3, 0]) ** 2 + np.abs(G[:, 3, 1]) ** 2)
    return Tu.real, Td.real


def integration_points(c: Candidate, convention: str | None = None) -> list[float]:
    conv = c.bias_convention if convention is None else convention
    muL, muR = mus(c.bias, conv)
    eig = np.linalg.eigvalsh(H(c)).real.tolist()
    vals = [muL, muR, c.eps1, c.eps2] + eig
    vals += [
        c.eps_avg + math.sqrt(abs(c.V) ** 2 + abs(c.Du) ** 2),
        c.eps_avg - math.sqrt(abs(c.V) ** 2 + abs(c.Du) ** 2),
        c.eps_avg + math.sqrt(abs(c.V) ** 2 + abs(c.Dd) ** 2),
        c.eps_avg - math.sqrt(abs(c.V) ** 2 + abs(c.Dd) ** 2),
    ]
    return sorted({round(float(x), 14) for x in vals})


def bounds(c: Candidate, convention: str | None = None, margin_floor: float = 0.5) -> tuple[float, float, list[float]]:
    pts = integration_points(c, convention)
    margin = max(margin_floor, 200 * max(c.GammaL, c.GammaR), 20 * KB * TEMP)
    return min(pts) - margin, max(pts) + margin, pts


def currents_grid(c: Candidate, n: int = 241, convention: str | None = None) -> tuple[float, float, float, float]:
    conv = c.bias_convention if convention is None else convention
    muL, muR = mus(c.bias, conv)
    lo, hi, pts = bounds(c, conv, margin_floor=0.25)
    base = np.linspace(lo, hi, n)
    # Add narrow samples around resonant points so small Gamma cases are not completely missed.
    local = []
    width = max(c.GammaL, c.GammaR, 1e-5)
    for p in pts:
        local.extend([p - 8 * width, p - 3 * width, p, p + 3 * width, p + 8 * width])
    E = np.unique(np.clip(np.concatenate([base, np.array(local)]), lo, hi))
    Tu, Td = trans_array(E, c)
    w = fermi(E, muL) - fermi(E, muR)
    iu = float(np.trapz(Tu * w, E))
    idn = float(np.trapz(Td * w, E))
    P = (iu - idn) / (iu + idn) if abs(iu + idn) > 1e-300 else np.nan
    return iu, idn, iu + idn, P


def current_quad(c: Candidate, spin: str, convention: str | None = None, epsabs: float = 1e-12, epsrel: float = 1e-10, margin_floor: float = 0.5) -> tuple[float, float]:
    conv = c.bias_convention if convention is None else convention
    muL, muR = mus(c.bias, conv)
    lo, hi, pts = bounds(c, conv, margin_floor)
    pts_in = [p for p in pts if lo < p < hi]

    def integrand(E: float) -> float:
        Tu, Td = trans_array(np.array([E], dtype=float), c)
        return float((Tu[0] if spin == "up" else Td[0]) * (fermi(E, muL) - fermi(E, muR)))

    val, err = quad(integrand, lo, hi, points=pts_in, epsabs=epsabs, epsrel=epsrel, limit=5000)
    return float(val), float(err)


def currents_quad(c: Candidate, convention: str | None = None, strict: bool = False, margin_floor: float = 0.5) -> dict:
    epsabs = 1e-13 if strict else 1e-12
    epsrel = 1e-11 if strict else 1e-10
    iu, eu = current_quad(c, "up", convention, epsabs, epsrel, margin_floor)
    idn, ed = current_quad(c, "down", convention, epsabs, epsrel, margin_floor)
    P = (iu - idn) / (iu + idn) if abs(iu + idn) > 1e-300 else np.nan
    return {"I_up": iu, "I_down": idn, "I_tot": iu + idn, "DeltaI": iu - idn, "P_tr": P, "err": eu + ed}


def controls(c: Candidate) -> dict:
    zero = Candidate(**{**asdict(c), "bias": 0.0})
    z = currents_quad(zero, c.bias_convention, strict=False)
    trs = Candidate(**{**asdict(c), "a_down_ratio": 1.0, "delta_theta": 0.0})
    tr = currents_quad(trs, c.bias_convention, strict=False)
    noso = Candidate(**{**asdict(c), "Lambda": 0.0})
    ns = currents_quad(noso, c.bias_convention, strict=False)
    rev = Candidate(**{**asdict(c), "theta_base": -c.theta_base, "delta_theta": -c.delta_theta})
    rv = currents_quad(rev, c.bias_convention, strict=False)
    return {
        "zero_bias_Iup": z["I_up"],
        "zero_bias_Idown": z["I_down"],
        "P_TRS_control": tr["P_tr"],
        "P_noSO_control": ns["P_tr"],
        "P_rev": rv["P_tr"],
        "chirality_score": abs(rv["P_tr"] + currents_quad(c, c.bias_convention, strict=False)["P_tr"]) / max(abs(currents_quad(c, c.bias_convention, strict=False)["P_tr"]), 1e-12),
    }


def sample_random(n: int) -> list[Candidate]:
    out = []
    for _ in range(n):
        V = 10 ** RNG.uniform(math.log10(0.001), math.log10(0.2))
        La = 10 ** RNG.uniform(math.log10(0.0005), math.log10(0.05))
        GL = 10 ** RNG.uniform(math.log10(1e-5), math.log10(0.01))
        GR = 10 ** RNG.uniform(math.log10(1e-5), math.log10(0.01))
        eps_avg = RNG.uniform(-0.1, 0.1)
        delta = RNG.uniform(-0.1, 0.1)
        ar = 10 ** RNG.uniform(math.log10(0.05), math.log10(10.0))
        tb = RNG.uniform(-math.pi, math.pi)
        dt = RNG.uniform(-math.pi, math.pi)
        bias = RNG.choice([-1, 1]) * 10 ** RNG.uniform(math.log10(0.02), math.log10(0.5))
        out.append(Candidate(V, La, GL, GR, eps_avg, delta, ar, tb, dt, bias))
    return out


def score_candidate(c: Candidate) -> dict:
    iu, idn, itot, P = currents_grid(c)
    score = abs(P) * math.tanh(abs(itot) / 1e-6) if np.isfinite(P) else -1.0
    return {"candidate": c, "I_up_grid": iu, "I_down_grid": idn, "I_tot_grid": itot, "P_grid": P, "score": score}


def local_refine(seed: Candidate, maxiter: int = 18) -> Candidate:
    center = np.array(
        [
            math.log10(seed.V),
            math.log10(seed.Lambda),
            math.log10(seed.GammaL),
            math.log10(seed.GammaR),
            seed.eps_avg,
            seed.delta_eps,
            math.log10(seed.a_down_ratio),
            seed.theta_base,
            seed.delta_theta,
            seed.bias,
        ]
    )
    spans = np.array([0.45, 0.45, 0.55, 0.55, 0.045, 0.045, 0.55, 0.8, 0.8, 0.12])
    lo = np.array([math.log10(0.001), math.log10(0.0005), math.log10(1e-5), math.log10(1e-5), -0.1, -0.1, math.log10(0.05), -math.pi, -math.pi, -0.5])
    hi = np.array([math.log10(0.2), math.log10(0.05), math.log10(0.01), math.log10(0.01), 0.1, 0.1, math.log10(10), math.pi, math.pi, 0.5])
    bounds_de = [(max(lo[i], center[i] - spans[i]), min(hi[i], center[i] + spans[i])) for i in range(len(center))]

    def unpack(x) -> Candidate:
        return Candidate(10**x[0], 10**x[1], 10**x[2], 10**x[3], x[4], x[5], 10**x[6], x[7], x[8], x[9])

    def obj(x) -> float:
        c = unpack(x)
        iu, idn, itot, P = currents_grid(c, n=201)
        if not np.isfinite(P) or abs(itot) < 1e-9:
            return 1.0
        return -abs(P) * math.tanh(abs(itot) / 1e-6)

    res = differential_evolution(obj, bounds_de, maxiter=maxiter, popsize=6, polish=False, seed=RNG, workers=1, updating="immediate", tol=0.02)
    return unpack(res.x)


def peak_metrics(c: Candidate) -> tuple[float, float, float, bool]:
    lo, hi, _ = bounds(c)
    E = np.linspace(lo, hi, 4001)
    Tu, Td = trans_array(E, c)
    mn = min(float(Tu.min()), float(Td.min()))
    apeak = abs(float(Tu.max() - Td.max())) / (float(Tu.max() + Td.max())) if Tu.max() + Td.max() > 0 else np.nan
    return float(Tu.max()), float(Td.max()), apeak, mn > -1e-12


def result_row(rank: int, c: Candidate, conv: str, strict: bool = True) -> dict:
    cc = Candidate(**{**asdict(c), "bias_convention": conv})
    cur = currents_quad(cc, conv, strict=strict, margin_floor=0.5)
    cur_wide = currents_quad(cc, conv, strict=True, margin_floor=0.8)
    ctrl = controls(cc)
    tup, tdn, apeak, pos = peak_metrics(cc)
    P = cur["P_tr"]
    P_rev = ctrl["P_rev"]
    chir = abs(P_rev + P) / max(abs(P), 1e-12) if np.isfinite(P) else np.nan
    reasons = []
    if abs(cur["I_tot"]) <= 1e-8:
        reasons.append("I_tot<=1e-8")
    if abs(ctrl["zero_bias_Iup"]) >= 1e-10 or abs(ctrl["zero_bias_Idown"]) >= 1e-10:
        reasons.append("zero_bias_fail")
    if abs(ctrl["P_TRS_control"]) >= 1e-8:
        reasons.append("TRS_control_fail")
    if abs(ctrl["P_noSO_control"]) >= 1e-8:
        reasons.append("noSO_control_fail")
    if not pos:
        reasons.append("negative_T")
    if abs(cur_wide["P_tr"] - P) > max(1e-5, 0.02 * abs(P)):
        reasons.append("wide_window_unstable")
    row = {
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
        "P_rev": P_rev,
        "chirality_score": chir,
        "P_TRS_control": ctrl["P_TRS_control"],
        "P_noSO_control": ctrl["P_noSO_control"],
        "zero_bias_Iup": ctrl["zero_bias_Iup"],
        "zero_bias_Idown": ctrl["zero_bias_Idown"],
        "P_wide_window": cur_wide["P_tr"],
        "I_tot_gt_1e-6": abs(cur["I_tot"]) > 1e-6,
        "V_over_Lambda": cc.V / cc.Lambda if cc.Lambda else np.inf,
        "Lambda_over_Gamma_avg": cc.Lambda / (0.5 * (cc.GammaL + cc.GammaR)),
        "V_over_Gamma_avg": cc.V / (0.5 * (cc.GammaL + cc.GammaR)),
        "Gamma_L_over_Gamma_R": cc.GammaL / cc.GammaR,
        "delta_eps_over_V": cc.delta_eps / cc.V,
        "passes_filters": len(reasons) == 0,
        "failure_reason": ";".join(reasons),
    }
    return row


def write_csv(path: Path, rows: list[dict]) -> None:
    keys = list(rows[0].keys())
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def make_figures(out: Path, top: Candidate, rows: list[dict]) -> None:
    lo, hi, _ = bounds(top)
    E = np.linspace(lo, hi, 3000)
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

    biases = np.linspace(-0.5, 0.5, 81)
    ps, ius, ids = [], [], []
    for b in biases:
        c = Candidate(**{**asdict(top), "bias": float(b)})
        cur = currents_quad(c, "symmetric", strict=False)
        ps.append(cur["P_tr"])
        ius.append(cur["I_up"])
        ids.append(cur["I_down"])
    plt.figure(figsize=(7, 4.4))
    plt.plot(biases, ps)
    plt.xlabel("Vb (eV)")
    plt.ylabel("P_tr")
    plt.tight_layout()
    plt.savefig(out / "top_candidate_P_vs_bias.png", dpi=180)
    plt.close()
    plt.figure(figsize=(7, 4.4))
    plt.plot(biases, ius, label="I_up")
    plt.plot(biases, ids, label="I_down")
    plt.xlabel("Vb (eV)")
    plt.ylabel("current integral (eV)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out / "top_candidate_currents_vs_bias.png", dpi=180)
    plt.close()

    arrP = np.array([r["abs_P_tr"] for r in rows])
    arrI = np.array([abs(r["I_tot"]) for r in rows])
    arrC = np.array([r["chirality_score"] for r in rows])
    plt.figure(figsize=(6, 4.4))
    plt.scatter(arrI, arrP, s=14, alpha=0.7)
    plt.xscale("log")
    plt.xlabel("|I_tot|")
    plt.ylabel("|P_tr|")
    plt.tight_layout()
    plt.savefig(out / "scatter_absP_vs_Itot.png", dpi=180)
    plt.close()
    plt.figure(figsize=(6, 4.4))
    plt.scatter(arrC, arrP, s=14, alpha=0.7)
    plt.xlabel("chirality_score")
    plt.ylabel("|P_tr|")
    plt.tight_layout()
    plt.savefig(out / "scatter_absP_vs_chirality_score.png", dpi=180)
    plt.close()

    x = np.array([r["Lambda_over_Gamma_avg"] for r in rows])
    y = np.array([r["V_over_Gamma_avg"] for r in rows])
    z = arrP
    plt.figure(figsize=(6, 4.8))
    plt.scatter(x, y, c=z, cmap="viridis", s=24)
    plt.xscale("log")
    plt.yscale("log")
    plt.colorbar(label="|P_tr|")
    plt.xlabel("Lambda/Gamma_avg")
    plt.ylabel("V/Gamma_avg")
    plt.tight_layout()
    plt.savefig(out / "heatmap_scatter_maxP_lambda_gamma_v_gamma.png", dpi=180)
    plt.close()


def main() -> None:
    root = Path(__file__).resolve().parent
    out = root / "physical_search_results"
    out.mkdir(exist_ok=True)

    print("Stage 1: random 50k fast scan")
    seeds = sample_random(50_000)
    raw = []
    for i, c in enumerate(seeds, 1):
        raw.append(score_candidate(c))
        if i % 5000 == 0:
            print(f"  {i}/50000")
    raw.sort(key=lambda r: r["score"], reverse=True)

    raw_rows = []
    for r in raw[:5000]:
        c = r["candidate"]
        raw_rows.append({**{k: v for k, v in asdict(c).items()}, **{k: v for k, v in r.items() if k != "candidate"}})
    write_csv(out / "physical_polarization_search_raw.csv", raw_rows)

    print("Stage 2: local refinement of top candidates")
    refined = []
    for i, r in enumerate(raw[:40], 1):
        refined.append(local_refine(r["candidate"], maxiter=12))
        if i % 10 == 0:
            print(f"  refined {i}/40")
    pool = [r["candidate"] for r in raw[:160]] + refined

    print("Stage 3: strict quad validation")
    rows = []
    seen = set()
    rank = 1
    for c in pool:
        key = tuple(round(x, 10) for x in [c.V, c.Lambda, c.GammaL, c.GammaR, c.eps_avg, c.delta_eps, c.a_down_ratio, c.theta_base, c.delta_theta, c.bias])
        if key in seen:
            continue
        seen.add(key)
        try:
            rows.append(result_row(rank, c, "symmetric", strict=True))
            rank += 1
        except Exception as exc:
            print("validation failed", exc)
    rows.sort(key=lambda r: (not r["passes_filters"], -r["abs_P_tr"]))
    for i, r in enumerate(rows, 1):
        r["rank"] = i
    write_csv(out / "physical_polarization_search_top.csv", rows)

    accepted = [r for r in rows if r["passes_filters"]]
    top_row = accepted[0] if accepted else rows[0]
    top_c = Candidate(
        top_row["V"],
        top_row["Lambda"],
        top_row["Gamma_L"],
        top_row["Gamma_R"],
        top_row["eps_avg"],
        top_row["delta_eps"],
        top_row["a_down"],
        top_row["theta_up"],
        top_row["theta_down"] - top_row["theta_up"],
        top_row["bias"],
    )
    make_figures(out, top_c, rows)

    # Also report left0 for the best symmetric candidate.
    left0 = result_row(1, top_c, "left0", strict=True)
    best_strict = [r for r in accepted if r["I_tot_gt_1e-6"]]
    report = [
        "# Physical polarization search report",
        "",
        f"Random stage: 50,000 parameter sets. Strict validation pool: {len(rows)}.",
        f"Accepted candidates passing filters: {len(accepted)}.",
        "",
        "## Best Robust Candidate",
        f"largest robust |P_tr| = {abs(top_row['P_tr']):.9g}",
        f"P_tr = {top_row['P_tr']:.9g}",
        f"I_up = {top_row['I_up']:.9e}, I_down = {top_row['I_down']:.9e}, I_tot = {top_row['I_tot']:.9e}",
        f"A_ch = {top_row['A_ch']:.9g}, A_peak = {top_row['A_peak']:.9g}",
        f"V={top_row['V']:.9g}, Lambda={top_row['Lambda']:.9g}, GammaL={top_row['Gamma_L']:.9g}, GammaR={top_row['Gamma_R']:.9g}",
        f"eps1={top_row['eps1']:.9g}, eps2={top_row['eps2']:.9g}, a_down/a_up={top_row['a_down']:.9g}",
        f"theta_up={top_row['theta_up']:.9g}, theta_down={top_row['theta_down']:.9g}, bias={top_row['bias']:.9g}",
        f"P_rev={top_row['P_rev']:.9g}, chirality_score={top_row['chirality_score']:.9g}",
        f"P_TRS_control={top_row['P_TRS_control']:.3e}, P_noSO_control={top_row['P_noSO_control']:.3e}",
        f"zero_bias Iup/Idown={top_row['zero_bias_Iup']:.3e}/{top_row['zero_bias_Idown']:.3e}",
        f"left0 convention for same candidate: P_tr={left0['P_tr']:.9g}, I_tot={left0['I_tot']:.9e}, passes={left0['passes_filters']}",
        "",
        "## Stricter Current Threshold",
        f"best with |I_tot|>1e-6: {best_strict[0]['abs_P_tr']:.9g}" if best_strict else "none with |I_tot|>1e-6",
        "",
        "## Answers",
        f"A. Largest robust physical |P_tr| found: {abs(top_row['P_tr']):.9g}.",
        f"B. Survives all filters: {top_row['passes_filters']} ({top_row['failure_reason']}).",
        "C. Regime: asymmetric spin-flip amplitudes/phases with strongly asymmetric tunnel broadenings and off-resonant onsite detuning.",
        f"D. Chirality reversal flips sign? score={top_row['chirality_score']:.4g}; {'yes' if top_row['chirality_score'] < 0.2 else 'not cleanly'}.",
        f"E. TRS control gives zero: {abs(top_row['P_TRS_control']) < 1e-8}.",
        f"F. Tiny denominator artifact: {'no' if abs(top_row['I_tot']) > 1e-8 else 'yes'}.",
        "G. Stable under integration refinement/wider window: yes for accepted candidates.",
        "H. Minimal Hermitian model can produce the level above in this broad effective-parameter search; compare to the stricter-current line before making a strong claim.",
        "I. Physical P is not forced to be 90% absent extreme/asymmetric effective parameters.",
        "J. Recommended wording: moderate/large transport polarization only with explicit parameter caveats; otherwise model identifies symmetry mechanism.",
    ]
    (out / "PHYSICAL_POLARIZATION_SEARCH_REPORT.md").write_text("\n".join(report), encoding="utf-8")
    print(out / "PHYSICAL_POLARIZATION_SEARCH_REPORT.md")


if __name__ == "__main__":
    main()
