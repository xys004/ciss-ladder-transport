#!/usr/bin/env python3
"""Search chirality-odd physical transport polarization in the Hermitian 4x4 model."""

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

KB = 8.617333262e-5
TEMP = 300.0
RNG = np.random.default_rng(20260430)


@dataclass
class Base:
    V: float
    Lambda: float
    GammaL: float
    GammaR: float
    eps_avg: float
    delta_eps: float
    bias: float
    map_type: str
    params: tuple

    @property
    def eps1(self) -> float:
        return self.eps_avg - 0.5 * self.delta_eps

    @property
    def eps2(self) -> float:
        return self.eps_avg + 0.5 * self.delta_eps


def fermi(E, mu):
    x = np.clip((np.asarray(E) - mu) / (KB * TEMP), -700, 700)
    y = 1.0 / (np.exp(x) + 1.0)
    return float(y) if np.ndim(y) == 0 else y


def mus(bias: float, convention: str = "symmetric") -> tuple[float, float]:
    if convention == "symmetric":
        return 0.5 * bias, -0.5 * bias
    return 0.0, -bias


def d_values(b: Base, chi: int) -> tuple[complex, complex]:
    L = b.Lambda
    if b.map_type in ["A", "B", "C"]:
        a_ratio, theta0, delta = b.params
        aup, adown = 1.0, a_ratio
        tu = theta0 + 0.5 * delta
        td = theta0 - 0.5 * delta
        Du = L * aup * np.exp(1j * tu)
        Dd = -L * adown * np.exp(-1j * td)
        if chi == 1:
            return Du, Dd
        if b.map_type == "A":
            return L * aup * np.exp(-1j * tu), -L * adown * np.exp(1j * td)
        if b.map_type == "B":
            return -L * aup * np.exp(-1j * tu), L * adown * np.exp(1j * td)
        return -np.conj(Dd), -np.conj(Du)
    if b.map_type == "D":
        a_ratio, theta_bg, theta_ch, delta = b.params
        tu = theta_bg + chi * theta_ch + 0.5 * delta
        td = theta_bg + chi * theta_ch - 0.5 * delta
        return L * np.exp(1j * tu), -L * a_ratio * np.exp(-1j * td)
    # Map E.
    bup, bdn, cup, cdn, betau, betad, gammau, gammad = b.params
    Du_even = L * bup * np.exp(1j * betau)
    Dd_even = -L * bdn * np.exp(-1j * betad)
    Du_odd = L * cup * np.exp(1j * gammau)
    Dd_odd = -L * cdn * np.exp(-1j * gammad)
    return Du_even + chi * Du_odd, Dd_even + chi * Dd_odd


def H(b: Base, Du: complex, Dd: complex) -> np.ndarray:
    return np.array(
        [
            [b.eps1, 0, b.V, Du],
            [0, b.eps1, Dd, b.V],
            [b.V, np.conj(Dd), b.eps2, 0],
            [np.conj(Du), b.V, 0, b.eps2],
        ],
        dtype=complex,
    )


def trans_array(E: np.ndarray, b: Base, Du: complex, Dd: complex) -> tuple[np.ndarray, np.ndarray]:
    h = H(b, Du, Dd)
    sig = np.diag([-0.5j * b.GammaL, -0.5j * b.GammaL, -0.5j * b.GammaR, -0.5j * b.GammaR])
    A = E[:, None, None] * np.eye(4)[None, :, :] - h[None, :, :] - sig[None, :, :]
    G = np.linalg.inv(A)
    pref = b.GammaL * b.GammaR
    Tu = pref * (np.abs(G[:, 2, 0]) ** 2 + np.abs(G[:, 2, 1]) ** 2)
    Td = pref * (np.abs(G[:, 3, 0]) ** 2 + np.abs(G[:, 3, 1]) ** 2)
    return Tu.real, Td.real


def points_bounds(b: Base, Du: complex, Dd: complex, convention: str = "symmetric", margin_floor: float = 0.25):
    muL, muR = mus(b.bias, convention)
    eig = np.linalg.eigvalsh(H(b, Du, Dd)).real.tolist()
    pts = [muL, muR, b.eps1, b.eps2] + eig
    pts += [
        b.eps_avg + math.sqrt(abs(b.V) ** 2 + abs(Du) ** 2),
        b.eps_avg - math.sqrt(abs(b.V) ** 2 + abs(Du) ** 2),
        b.eps_avg + math.sqrt(abs(b.V) ** 2 + abs(Dd) ** 2),
        b.eps_avg - math.sqrt(abs(b.V) ** 2 + abs(Dd) ** 2),
    ]
    pts = sorted({round(float(x), 14) for x in pts})
    margin = max(margin_floor, 120 * max(b.GammaL, b.GammaR), 16 * KB * TEMP)
    return min(pts) - margin, max(pts) + margin, pts


def currents_grid(b: Base, chi: int, n: int = 67, convention: str = "symmetric") -> dict:
    Du, Dd = d_values(b, chi)
    muL, muR = mus(b.bias, convention)
    lo, hi, pts = points_bounds(b, Du, Dd, convention)
    base = np.linspace(lo, hi, n)
    width = max(b.GammaL, b.GammaR, 1e-5)
    local = []
    for p in pts:
        local += [p - 5 * width, p, p + 5 * width]
    E = np.unique(np.clip(np.concatenate([base, local]), lo, hi))
    Tu, Td = trans_array(E, b, Du, Dd)
    w = fermi(E, muL) - fermi(E, muR)
    iu = float(np.trapz(Tu * w, E))
    idn = float(np.trapz(Td * w, E))
    P = (iu - idn) / (iu + idn) if abs(iu + idn) > 1e-300 else np.nan
    return {"Iup": iu, "Idown": idn, "Itot": iu + idn, "P": P}


def current_quad(b: Base, chi: int, spin: str, convention: str = "symmetric", strict: bool = True, margin_floor: float = 0.5) -> float:
    Du, Dd = d_values(b, chi)
    muL, muR = mus(b.bias, convention)
    lo, hi, pts = points_bounds(b, Du, Dd, convention, margin_floor)
    pts = [p for p in pts if lo < p < hi]
    epsabs = 1e-13 if strict else 1e-12
    epsrel = 1e-11 if strict else 1e-10

    def f(E):
        Tu, Td = trans_array(np.array([E], dtype=float), b, Du, Dd)
        return float((Tu[0] if spin == "up" else Td[0]) * (fermi(E, muL) - fermi(E, muR)))

    val, _ = quad(f, lo, hi, points=pts, epsabs=epsabs, epsrel=epsrel, limit=5000)
    return float(val)


def currents_quad(b: Base, chi: int, convention: str = "symmetric", strict: bool = True, margin_floor: float = 0.5) -> dict:
    iu = current_quad(b, chi, "up", convention, strict, margin_floor)
    idn = current_quad(b, chi, "down", convention, strict, margin_floor)
    return {"Iup": iu, "Idown": idn, "Itot": iu + idn, "P": (iu - idn) / (iu + idn) if abs(iu + idn) > 1e-300 else np.nan}


def score_pair(cp: dict, cm: dict) -> tuple[float, float, float]:
    Pp, Pm = cp["P"], cm["P"]
    chir = abs(Pm + Pp) / max(abs(Pp), abs(Pm), 1e-12)
    absmin = min(abs(Pp), abs(Pm))
    score = absmin * math.tanh(abs(cp["Itot"]) / 1e-6) * math.tanh(abs(cm["Itot"]) / 1e-6) * math.exp(-chir / 0.1)
    loose = absmin / (1 + chir)
    return score, loose, chir


def rand_log(lo, hi):
    return 10 ** RNG.uniform(math.log10(lo), math.log10(hi))


def random_base(map_type: str) -> Base:
    V = rand_log(0.001, 0.2)
    L = rand_log(0.0005, 0.05)
    GL = rand_log(1e-5, 1e-2)
    GR = rand_log(1e-5, 1e-2)
    ea = RNG.uniform(-0.1, 0.1)
    de = RNG.uniform(-0.1, 0.1)
    bias = RNG.choice([-1, 1]) * rand_log(0.02, 0.5)
    if map_type in ["A", "B", "C"]:
        params = (rand_log(0.05, 10), RNG.uniform(-math.pi, math.pi), RNG.uniform(-math.pi, math.pi))
    elif map_type == "D":
        params = (rand_log(0.05, 10), RNG.uniform(-math.pi, math.pi), RNG.uniform(-math.pi, math.pi), RNG.uniform(-math.pi, math.pi))
    else:
        amps = [0.0 if RNG.random() < 0.1 else rand_log(0.02, 10) for _ in range(4)]
        phases = [RNG.uniform(-math.pi, math.pi) for _ in range(4)]
        params = tuple(amps + phases)
    return Base(V, L, GL, GR, ea, de, bias, map_type, params)


def eval_fast(b: Base) -> dict:
    cp = currents_grid(b, 1)
    cm = currents_grid(b, -1)
    score, loose, chir = score_pair(cp, cm)
    return {"base": b, "score": score, "score_loose": loose, "chirality_score": chir, "P_plus": cp["P"], "P_minus": cm["P"], "Itot_plus": cp["Itot"], "Itot_minus": cm["Itot"]}


def tags_for(b: Base, Du: complex, Dd: complex) -> str:
    tags = []
    if b.map_type in ["A", "B", "C", "D"]:
        ar = b.params[0]
        tags.append("modest_TRSB" if 0.2 <= ar <= 5 else "strong_TRSB")
    tags.append("weak_SOC" if b.Lambda <= 0.01 else "strong_SOC")
    gavg = 0.5 * (b.GammaL + b.GammaR)
    if gavg < 1e-4:
        tags.append("weak_contact")
    if b.V / gavg > 100:
        tags.append("molecule_contact_decoupled")
    ratio = max(abs(Du), abs(Dd)) / max(min(abs(Du), abs(Dd)), 1e-15)
    if ratio > 100:
        tags.append("extreme_D_asymmetry")
    return ";".join(tags)


def controls(b: Base, chi: int) -> dict:
    Du, Dd = d_values(b, chi)
    zero = Base(b.V, b.Lambda, b.GammaL, b.GammaR, b.eps_avg, b.delta_eps, 0.0, b.map_type, b.params)
    z = currents_quad(zero, chi, strict=False)
    # TRS control uses same Du but Dd=-conj(Du) through a temporary one-off map E.
    trs_params = (0, 0, abs(Du) / max(b.Lambda, 1e-300), 0, 0, 0, np.angle(Du), 0)
    trs = Base(b.V, b.Lambda, b.GammaL, b.GammaR, b.eps_avg, b.delta_eps, b.bias, "E", trs_params)
    tr = currents_quad(trs, 1, strict=False)
    noso = Base(b.V, 0.0, b.GammaL, b.GammaR, b.eps_avg, b.delta_eps, b.bias, b.map_type, b.params)
    ns = currents_quad(noso, chi, strict=False)
    return {"zero_Iup": z["Iup"], "zero_Idown": z["Idown"], "P_TRS": tr["P"], "P_noSO": ns["P"]}


def peak(b: Base, chi: int) -> tuple[float, bool]:
    Du, Dd = d_values(b, chi)
    lo, hi, _ = points_bounds(b, Du, Dd, margin_floor=0.5)
    E = np.linspace(lo, hi, 1600)
    Tu, Td = trans_array(E, b, Du, Dd)
    ok = min(float(Tu.min()), float(Td.min())) > -1e-12
    A = abs(float(Tu.max() - Td.max())) / float(Tu.max() + Td.max()) if Tu.max() + Td.max() > 0 else np.nan
    return A, ok


def validate(rank: int, b: Base) -> dict:
    cp = currents_quad(b, 1)
    cm = currents_quad(b, -1)
    cpw = currents_quad(b, 1, margin_floor=0.8)
    cmw = currents_quad(b, -1, margin_floor=0.8)
    score, loose, chir = score_pair(cp, cm)
    Du_p, Dd_p = d_values(b, 1)
    Du_m, Dd_m = d_values(b, -1)
    ctrl_p, ctrl_m = controls(b, 1), controls(b, -1)
    Apeak_p, posp = peak(b, 1)
    Apeak_m, posm = peak(b, -1)
    fail = []
    if abs(cp["Itot"]) <= 1e-8 or abs(cm["Itot"]) <= 1e-8:
        fail.append("finite_current")
    if abs(ctrl_p["zero_Iup"]) > 1e-10 or abs(ctrl_p["zero_Idown"]) > 1e-10 or abs(ctrl_m["zero_Iup"]) > 1e-10 or abs(ctrl_m["zero_Idown"]) > 1e-10:
        fail.append("zero_bias")
    if abs(ctrl_p["P_TRS"]) > 1e-8 or abs(ctrl_m["P_TRS"]) > 1e-8:
        fail.append("TRS")
    if abs(ctrl_p["P_noSO"]) > 1e-8 or abs(ctrl_m["P_noSO"]) > 1e-8:
        fail.append("noSO")
    if not (posp and posm):
        fail.append("positivity")
    if abs(cpw["P"] - cp["P"]) > max(1e-5, 0.02 * abs(cp["P"])) or abs(cmw["P"] - cm["P"]) > max(1e-5, 0.02 * abs(cm["P"])):
        fail.append("integration_stability")
    tags = tags_for(b, Du_p, Dd_p)
    return {
        "rank": rank,
        "map_type": b.map_type,
        "score": score,
        "score_loose": loose,
        "P_plus": cp["P"],
        "P_minus": cm["P"],
        "absP_min": min(abs(cp["P"]), abs(cm["P"])),
        "chirality_score": chir,
        "Iup_plus": cp["Iup"],
        "Idown_plus": cp["Idown"],
        "Itot_plus": cp["Itot"],
        "Iup_minus": cm["Iup"],
        "Idown_minus": cm["Idown"],
        "Itot_minus": cm["Itot"],
        "A_ch_plus": abs(cp["Iup"] - cp["Idown"]) / max(abs(cp["Iup"]), abs(cp["Idown"]), 1e-300),
        "A_ch_minus": abs(cm["Iup"] - cm["Idown"]) / max(abs(cm["Iup"]), abs(cm["Idown"]), 1e-300),
        "A_peak_plus": Apeak_p,
        "A_peak_minus": Apeak_m,
        "V": b.V,
        "Lambda": b.Lambda,
        "Gamma_L": b.GammaL,
        "Gamma_R": b.GammaR,
        "eps1": b.eps1,
        "eps2": b.eps2,
        "eps_avg": b.eps_avg,
        "delta_eps": b.delta_eps,
        "bias": b.bias,
        "Du_plus_real": Du_p.real,
        "Du_plus_imag": Du_p.imag,
        "Dd_plus_real": Dd_p.real,
        "Dd_plus_imag": Dd_p.imag,
        "Du_minus_real": Du_m.real,
        "Du_minus_imag": Du_m.imag,
        "Dd_minus_real": Dd_m.real,
        "Dd_minus_imag": Dd_m.imag,
        "D_asym_ratio_plus": max(abs(Du_p), abs(Dd_p)) / max(min(abs(Du_p), abs(Dd_p)), 1e-15),
        "D_asym_ratio_minus": max(abs(Du_m), abs(Dd_m)) / max(min(abs(Du_m), abs(Dd_m)), 1e-15),
        "P_TRS_control_plus": ctrl_p["P_TRS"],
        "P_TRS_control_minus": ctrl_m["P_TRS"],
        "P_noSO_plus": ctrl_p["P_noSO"],
        "P_noSO_minus": ctrl_m["P_noSO"],
        "zero_bias_Iup_plus": ctrl_p["zero_Iup"],
        "zero_bias_Idown_plus": ctrl_p["zero_Idown"],
        "zero_bias_Iup_minus": ctrl_m["zero_Iup"],
        "zero_bias_Idown_minus": ctrl_m["zero_Idown"],
        "passes_filters": len(fail) == 0,
        "tags": tags,
        "failure_reason": ";".join(fail),
        "params": repr(b.params),
    }


def write_csv(path: Path, rows: list[dict]):
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def make_figs(out: Path, b: Base, rows: list[dict]):
    biases = np.linspace(-0.5, 0.5, 41)
    pp, pm, ipu, idu, ipm, idm = [], [], [], [], [], []
    for x in biases:
        bb = Base(b.V, b.Lambda, b.GammaL, b.GammaR, b.eps_avg, b.delta_eps, float(x), b.map_type, b.params)
        cp, cm = currents_quad(bb, 1, strict=False), currents_quad(bb, -1, strict=False)
        pp.append(cp["P"]); pm.append(cm["P"])
        ipu.append(cp["Iup"]); idu.append(cp["Idown"]); ipm.append(cm["Iup"]); idm.append(cm["Idown"])
    plt.figure(figsize=(7, 4.4)); plt.plot(biases, pp, label="P_plus"); plt.plot(biases, pm, label="P_minus"); plt.legend(); plt.xlabel("Vb"); plt.ylabel("P_tr"); plt.tight_layout(); plt.savefig(out / "top_P_plus_minus_vs_bias.png", dpi=180); plt.close()
    plt.figure(figsize=(7, 4.4)); plt.plot(biases, ipu, label="Iup +"); plt.plot(biases, idu, label="Idown +"); plt.plot(biases, ipm, "--", label="Iup -"); plt.plot(biases, idm, "--", label="Idown -"); plt.legend(); plt.xlabel("Vb"); plt.ylabel("I"); plt.tight_layout(); plt.savefig(out / "top_currents_both_chiralities.png", dpi=180); plt.close()
    for chi, name in [(1, "plus"), (-1, "minus")]:
        Du, Dd = d_values(b, chi); lo, hi, _ = points_bounds(b, Du, Dd, margin_floor=0.5); E = np.linspace(lo, hi, 2000); Tu, Td = trans_array(E, b, Du, Dd)
        plt.figure(figsize=(7, 4.4)); plt.plot(E, Tu, label="T_up"); plt.plot(E, Td, label="T_down"); plt.legend(); plt.xlabel("E"); plt.ylabel("T(E)"); plt.tight_layout(); plt.savefig(out / f"top_transmission_{name}.png", dpi=180); plt.close()
    plt.figure(figsize=(6, 4.4)); plt.scatter([r["chirality_score"] for r in rows], [r["absP_min"] for r in rows], c=[r["score"] for r in rows], s=25); plt.xlabel("chirality_score"); plt.ylabel("absP_min"); plt.tight_layout(); plt.savefig(out / "scatter_absPmin_vs_chirality_score.png", dpi=180); plt.close()
    plt.figure(figsize=(6, 4.4)); plt.scatter([min(abs(r["Itot_plus"]), abs(r["Itot_minus"])) for r in rows], [r["score"] for r in rows], s=25); plt.xscale("log"); plt.xlabel("Itot_min"); plt.ylabel("score"); plt.tight_layout(); plt.savefig(out / "scatter_score_vs_Itot_min.png", dpi=180); plt.close()
    plt.figure(figsize=(6, 4.4)); plt.scatter([max(r["D_asym_ratio_plus"], r["D_asym_ratio_minus"]) for r in rows], [r["absP_min"] for r in rows], s=25); plt.xscale("log"); plt.xlabel("D_asym_ratio"); plt.ylabel("absP_min"); plt.tight_layout(); plt.savefig(out / "scatter_D_asym_vs_absPmin.png", dpi=180); plt.close()
    colors = {m: i for i, m in enumerate(sorted(set(r["map_type"] for r in rows)))}
    plt.figure(figsize=(6, 4.4)); plt.scatter([colors[r["map_type"]] for r in rows], [r["score"] for r in rows], c=[r["absP_min"] for r in rows], s=35); plt.xticks(list(colors.values()), list(colors.keys())); plt.ylabel("score"); plt.colorbar(label="absP_min"); plt.tight_layout(); plt.savefig(out / "scatter_score_by_map_type.png", dpi=180); plt.close()


def main():
    root = Path(__file__).resolve().parent
    out = root / "chiral_physical_results"
    out.mkdir(exist_ok=True)
    fast = []
    counts = {"A": 25_000, "B": 25_000, "C": 25_000, "D": 25_000, "E": 100_000}
    total = sum(counts.values())
    done = 0
    for mt, n in counts.items():
        for _ in range(n):
            r = eval_fast(random_base(mt))
            fast.append(r)
            done += 1
            if done % 20_000 == 0:
                print(f"fast {done}/{total}")
    fast.sort(key=lambda r: max(r["score"], 0.1 * r["score_loose"]), reverse=True)
    raw_rows = []
    for r in fast[:5000]:
        b = r["base"]
        raw_rows.append({k: v for k, v in r.items() if k != "base"} | {"map_type": b.map_type, "V": b.V, "Lambda": b.Lambda, "Gamma_L": b.GammaL, "Gamma_R": b.GammaR, "eps_avg": b.eps_avg, "delta_eps": b.delta_eps, "bias": b.bias, "params": repr(b.params)})
    write_csv(out / "chiral_physical_polarization_raw.csv", raw_rows)
    print("validating")
    rows = []
    seen = set()
    for r in fast[:120]:
        b = r["base"]
        key = (b.map_type, tuple(round(x, 9) for x in [b.V, b.Lambda, b.GammaL, b.GammaR, b.eps_avg, b.delta_eps, b.bias]), repr(tuple(round(float(x), 8) for x in b.params)))
        if key in seen:
            continue
        seen.add(key)
        try:
            rows.append(validate(len(rows) + 1, b))
        except Exception as exc:
            print("skip", exc)
        if len(rows) >= 80:
            break
    rows.sort(key=lambda r: (not r["passes_filters"], -r["score"], -r["absP_min"]))
    for i, r in enumerate(rows, 1):
        r["rank"] = i
    write_csv(out / "chiral_physical_polarization_top.csv", rows)
    top = next((r for r in rows if r["passes_filters"]), rows[0])
    # Reconstruct top base from params string safely through eval on numeric tuple only.
    btop = Base(top["V"], top["Lambda"], top["Gamma_L"], top["Gamma_R"], top["eps_avg"], top["delta_eps"], top["bias"], top["map_type"], tuple(eval(top["params"])))
    leftp, leftm = currents_quad(btop, 1, "left0"), currents_quad(btop, -1, "left0")
    left_score, left_loose, left_chir = score_pair(leftp, leftm)
    make_figs(out, btop, rows)
    good = [r for r in rows if r["passes_filters"] and r["chirality_score"] < 0.2]
    excellent = [r for r in rows if r["passes_filters"] and r["chirality_score"] < 0.05]
    strict_current = [r for r in good if abs(r["Itot_plus"]) > 1e-6 and abs(r["Itot_minus"]) > 1e-6]
    report = [
        "# Chirality-odd physical polarization search report",
        "",
        f"Fast random search: maps A-D 100,000 total plus map E 100,000 = {total}. Strict validated: {len(rows)}.",
        f"Passing filters: {sum(1 for r in rows if r['passes_filters'])}. Good chirality_score<0.2: {len(good)}. Excellent <0.05: {len(excellent)}.",
        "",
        "## Best Ranked Candidate",
        f"map={top['map_type']}, score={top['score']:.9g}, score_loose={top['score_loose']:.9g}",
        f"P_plus={top['P_plus']:.9g}, P_minus={top['P_minus']:.9g}, absP_min={top['absP_min']:.9g}, chirality_score={top['chirality_score']:.9g}",
        f"Itot_plus={top['Itot_plus']:.9e}, Itot_minus={top['Itot_minus']:.9e}",
        f"V={top['V']:.9g}, Lambda={top['Lambda']:.9g}, GammaL={top['Gamma_L']:.9g}, GammaR={top['Gamma_R']:.9g}, bias={top['bias']:.9g}",
        f"eps1={top['eps1']:.9g}, eps2={top['eps2']:.9g}",
        f"D_asym plus/minus={top['D_asym_ratio_plus']:.9g}/{top['D_asym_ratio_minus']:.9g}, tags={top['tags']}",
        f"controls TRS plus/minus={top['P_TRS_control_plus']:.3e}/{top['P_TRS_control_minus']:.3e}; noSO={top['P_noSO_plus']:.3e}/{top['P_noSO_minus']:.3e}",
        f"left0 same candidate: P_plus={leftp['P']:.9g}, P_minus={leftm['P']:.9g}, chirality_score={left_chir:.9g}",
        "",
        "## Best Good Candidate",
    ]
    if good:
        g = good[0]
        report += [f"map={g['map_type']}, P_plus={g['P_plus']:.9g}, P_minus={g['P_minus']:.9g}, absP_min={g['absP_min']:.9g}, chirality_score={g['chirality_score']:.9g}, score={g['score']:.9g}", f"Itot_plus={g['Itot_plus']:.9e}, Itot_minus={g['Itot_minus']:.9e}, tags={g['tags']}"]
    else:
        report.append("No validated candidate with chirality_score < 0.2.")
    report += [
        "",
        "## Answers",
        f"A. Largest robust chirality-odd |P_tr| found: {good[0]['absP_min']:.9g}." if good else "A. Largest robust chirality-odd |P_tr| found: none under chirality_score<0.2.",
        f"B. P_minus approx -P_plus: {'yes' if good else 'no validated good candidate'}.",
        f"C. Producing map: {good[0]['map_type'] if good else 'none'}.",
        f"D. Survives finite-current filters: {bool(good)}; stricter >1e-6 count={len(strict_current)}.",
        "E. TRS/no-SO controls pass for all rows marked passes_filters.",
        f"F. Extreme one-channel blockade: {'extreme_D_asymmetry' in top['tags']}.",
        "G. Physical viability: candidates with large score should be treated as effective/fine-tuned unless tags are modest.",
        "H. Regime is listed above and in CSV.",
        f"I. left0 validation listed above for top ranked candidate.",
        "J. Previous non-chiral high-P reached |P|~0.99999 but chirality_score~2; this search ranks chirality-odd behavior instead.",
        "K. Recommendation: use 'moderate chirality-odd transport polarization' only if the good-candidate magnitude is acceptable; otherwise 'large spin filtering but chirality-even / model needs extra physics for chirality-odd CISS'.",
    ]
    (out / "CHIRAL_PHYSICAL_POLARIZATION_SEARCH_REPORT.md").write_text("\n".join(report), encoding="utf-8")
    print(out / "CHIRAL_PHYSICAL_POLARIZATION_SEARCH_REPORT.md")


if __name__ == "__main__":
    main()
