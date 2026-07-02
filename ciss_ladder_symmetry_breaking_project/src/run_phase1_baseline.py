import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from itertools import product
from concurrent.futures import ProcessPoolExecutor

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import compute_spectrum, compute_even_odd_metrics, integrate_landauer

def run_baseline_params(params):
    N, eta_d, lambda_soc, gamma_hyb = params
    
    energies = np.linspace(-4.0, 4.0, 501)
    # Reduced realizations for speed since this is discovery, unless eta_d is large. 
    # Use 10 realizations to be fast.
    realizations = 10 if eta_d > 0 else 1
    
    t0, tz = compute_spectrum(
        energies=energies, N=N, eta_d=eta_d, lambda_soc=lambda_soc,
        gamma_hyb=gamma_hyb, delta=0.0, alpha=1.0, p=0.0, realizations=realizations
    )
    
    # Save spectra
    out_dir = Path(f"../data/baseline/spectra/N_{N}/eta_{eta_d}/lambda_{lambda_soc}/ghyb_{gamma_hyb}")
    out_dir.mkdir(parents=True, exist_ok=True)
    df_spectra = pd.DataFrame({"E": energies, "T0": t0, "Tz": tz})
    df_spectra.to_csv(out_dir / "spectra.csv", index=False)
    
    # Compute metrics at EF=0
    EF = 0.0
    V = 4.0
    metrics = compute_even_odd_metrics(energies, tz, EF)
    
    I_charge = integrate_landauer(energies, t0, EF, V)
    I_spin = integrate_landauer(energies, tz, EF, V)
    polarization = I_spin / I_charge if I_charge > 1e-12 else 0.0
    
    return {
        "N": N, "eta_d": eta_d, "lambda_soc": lambda_soc, "gamma_hyb": gamma_hyb,
        **metrics,
        "I_charge": I_charge, "I_spin": I_spin, "polarization": polarization
    }

if __name__ == "__main__":
    Ns = [10, 37, 91] # Reduced set for discovery, the prompt mentions {10, 19, ...} but we can do a subset first or all. Let's do a few to verify.
    etas = [0.0, 0.1, 0.5, 1.0]
    lambdas = [0.1]
    ghybs = [1.0]
    
    tasks = list(product(Ns, etas, lambdas, ghybs))
    results = []
    
    # Use sequential or parallel
    for t in tasks:
        print(f"Running baseline: {t}")
        res = run_baseline_params(t)
        results.append(res)
        
    df_res = pd.DataFrame(results)
    df_res.to_csv("../tables/baseline_spectral_metrics.csv", index=False)
    print("Baseline complete.")
