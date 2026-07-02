#!/usr/bin/env python3
"""
Numerical benchmark for the two-site CISS effective Hamiltonian.

Computes spin-resolved currents using:
  A) Exact 4x4 Green function, Caroli/Landauer trace.
  B) Decoupled-channel analytical-limit formula used as a benchmark.

All energies are in eV. Convert meV -> eV with 1 meV = 1e-3 eV.

Run examples:
    python ciss_numeric_benchmark.py --quick
    python ciss_numeric_benchmark.py --scan-bias --outdir results
"""

from __future__ import annotations
import argparse
import math
import os
from dataclasses import dataclass
from typing import Iterable, Tuple

import numpy as np
from scipy.integrate import quad

try:
    import matplotlib.pyplot as plt
    HAS_MPL = True
except Exception:
    HAS_MPL = False

KB_EV = 8.617333262145e-5  # eV/K


def fermi(E: float, mu: float, T: float) -> float:
    x = (E - mu) / (KB_EV * T)
    if x > 700:
        return 0.0
    if x < -700:
        return 1.0
    return 1.0 / (math.exp(x) + 1.0)


@dataclass
class Params:
    eps1: float = 0.0
    eps2: float = 0.0
    V: float = 0.100       # 100 meV
    Lambda: float = 0.0055 # 5.5 meV
    GammaL: float = 0.00006
    GammaR: float = 0.00006
    T: float = 300.0
    phi: float = math.pi / 5
    a_down: float = 1.0
    alpha_down: float = 0.0

    @property
    def D_up(self) -> complex:
        # Gauge choice consistent with a chiral spin-flip amplitude.
        return self.Lambda * np.exp(-1j * self.phi)

    @property
    def D_down(self) -> complex:
        # TRS/Kramers limit is a_down=1 and alpha_down=0:
        # D_down = -conj(D_up).
        return -self.Lambda * self.a_down * np.exp(1j * (self.phi + self.alpha_down))


def H_central(p: Params) -> np.ndarray:
    V = p.V
    Du = p.D_up
    Dd = p.D_down
    return np.array(
        [
            [p.eps1, 0.0, V, Du],
            [0.0, p.eps1, Dd, V],
            [V, np.conj(Dd), p.eps2, 0.0],
            [np.conj(Du), V, 0.0, p.eps2],
        ],
        dtype=complex,
    )


def gamma_matrices(p: Params) -> Tuple[np.ndarray, np.ndarray]:
    GL = np.diag([p.GammaL, p.GammaL, 0.0, 0.0]).astype(complex)
    GR = np.diag([0.0, 0.0, p.GammaR, p.GammaR]).astype(complex)
    return GL, GR


def G_ret(E: float, p: Params) -> np.ndarray:
    H = H_central(p)
    GL, GR = gamma_matrices(p)
    Sigma = -0.5j * (GL + GR)
    return np.linalg.inv((E + 1e-16j) * np.eye(4) - H - Sigma)


def transmissions_exact(E: float, p: Params) -> Tuple[float, float, float]:
    """Return T_up_out, T_down_out, T_total."""
    G = G_ret(E, p)
    GL, GR = gamma_matrices(p)

    # Projectors onto outgoing spin at the right contact: basis 2=site2 up, 3=site2 down.
    PRu = np.diag([0.0, 0.0, 1.0, 0.0]).astype(complex)
    PRd = np.diag([0.0, 0.0, 0.0, 1.0]).astype(complex)

    Ga = G.conj().T
    Tu = np.trace(PRu @ GR @ G @ GL @ Ga).real
    Td = np.trace(PRd @ GR @ G @ GL @ Ga).real
    return max(Tu, 0.0), max(Td, 0.0), max(Tu + Td, 0.0)


def T_decoupled(E: float, p: Params, spin: str) -> float:
    D = p.D_up if spin == "up" else p.D_down
    a1 = E - p.eps1 + 0.5j * p.GammaL
    a2 = E - p.eps2 + 0.5j * p.GammaR
    K = p.V**2 + abs(D)**2
    return (p.GammaL * p.GammaR * K / abs(a1*a2 - K)**2).real


def integration_points(p: Params, muL: float, muR: float) -> list[float]:
    eig = np.linalg.eigvalsh(H_central(p)).real.tolist()
    return sorted(set([muL, muR] + eig))


def current_exact(muL: float, muR: float, p: Params, spin: str) -> Tuple[float, float]:
    pts = integration_points(p, muL, muR)
    width = max(p.GammaL, p.GammaR, KB_EV * p.T, 1e-6)
    Emin = min(pts) - 80 * width
    Emax = max(pts) + 80 * width

    def integrand(E: float) -> float:
        Tu, Td, _ = transmissions_exact(E, p)
        Tspin = Tu if spin == "up" else Td
        return Tspin * (fermi(E, muL, p.T) - fermi(E, muR, p.T))

    val, err = quad(integrand, Emin, Emax, points=pts, epsabs=1e-12, epsrel=1e-9, limit=1000)
    return val, err


def current_decoupled(muL: float, muR: float, p: Params, spin: str) -> Tuple[float, float]:
    # Include approximate resonance points of the decoupled denominator.
    D = p.D_up if spin == "up" else p.D_down
    K = p.V**2 + abs(D)**2
    res = math.sqrt(max(K, 0.0))
    pts = sorted(set([muL, muR, -res, res]))
    width = max(p.GammaL, p.GammaR, KB_EV * p.T, 1e-6)
    Emin = min(pts) - 80 * width
    Emax = max(pts) + 80 * width

    def integrand(E: float) -> float:
        return T_decoupled(E, p, spin) * (fermi(E, muL, p.T) - fermi(E, muR, p.T))

    val, err = quad(integrand, Emin, Emax, points=pts, epsabs=1e-12, epsrel=1e-9, limit=1000)
    return val, err


def polarization(Iu: float, Id: float) -> float:
    den = Iu + Id
    if abs(den) < 1e-30:
        return float("nan")
    return (Iu - Id) / den


def quick_report() -> None:
    print("=== Quick numerical benchmark ===")
    cases = [
        ("TRS limit", Params(a_down=1.0, alpha_down=0.0)),
        ("TRSB amplitude asymmetry", Params(a_down=0.80, alpha_down=0.0)),
        ("TRSB phase asymmetry", Params(a_down=1.0, alpha_down=0.3 * math.pi / 5)),
        ("TRSB amp+phase asymmetry", Params(a_down=0.80, alpha_down=0.3 * math.pi / 5)),
    ]
    muL = 0.0
    muR = 0.2  # eV, as used in the contour plot text
    for name, p in cases:
        Iue, _ = current_exact(muL, muR, p, "up")
        Ide, _ = current_exact(muL, muR, p, "down")
        Iud, _ = current_decoupled(muL, muR, p, "up")
        Idd, _ = current_decoupled(muL, muR, p, "down")
        print(f"\n{name}")
        print(f"  D_up   = {p.D_up:.6g}")
        print(f"  D_down = {p.D_down:.6g}")
        print(f"  Exact:     Iu={Iue:.8e}, Id={Ide:.8e}, P={polarization(Iue, Ide):+.6f}")
        print(f"  Decoupled: Iu={Iud:.8e}, Id={Idd:.8e}, P={polarization(Iud, Idd):+.6f}")


def scan_bias(outdir: str) -> None:
    os.makedirs(outdir, exist_ok=True)
    # Bias range comparable to manuscript plots. muL=0, muR=Vbias.
    biases = np.linspace(0.0, 0.25, 101)
    p = Params(a_down=0.80, alpha_down=0.3 * math.pi / 5)

    rows = []
    for b in biases:
        Iue, _ = current_exact(0.0, float(b), p, "up")
        Ide, _ = current_exact(0.0, float(b), p, "down")
        Iud, _ = current_decoupled(0.0, float(b), p, "up")
        Idd, _ = current_decoupled(0.0, float(b), p, "down")
        rows.append([b, Iue, Ide, polarization(Iue, Ide), Iud, Idd, polarization(Iud, Idd)])

    arr = np.array(rows, dtype=float)
    csv_path = os.path.join(outdir, "bias_scan_exact_vs_decoupled.csv")
    np.savetxt(
        csv_path,
        arr,
        delimiter=",",
        header="bias_eV,Iu_exact,Id_exact,P_exact,Iu_decoupled,Id_decoupled,P_decoupled",
        comments="",
    )
    print(f"Wrote {csv_path}")

    if HAS_MPL:
        plt.figure()
        plt.plot(arr[:, 0], arr[:, 3], label="Exact 4x4")
        plt.plot(arr[:, 0], arr[:, 6], "--", label="Decoupled limit")
        plt.xlabel("bias chemical-potential shift (eV)")
        plt.ylabel("spin polarization P")
        plt.legend()
        plt.tight_layout()
        fig_path = os.path.join(outdir, "bias_scan_polarization.png")
        plt.savefig(fig_path, dpi=200)
        print(f"Wrote {fig_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="Run a quick benchmark at muR=0.2 eV")
    parser.add_argument("--scan-bias", action="store_true", help="Run a bias scan and write CSV/PNG")
    parser.add_argument("--outdir", default="/mnt/data/ciss_validation/results", help="Output directory")
    args = parser.parse_args()

    if args.quick or not args.scan_bias:
        quick_report()
    if args.scan_bias:
        scan_bias(args.outdir)


if __name__ == "__main__":
    main()
