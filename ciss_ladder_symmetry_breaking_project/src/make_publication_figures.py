from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from campaign_core import FIGURE_POINTS, PROJECT_ROOT, compute_spectra, even_odd_decomposition, load_table


TABLES = PROJECT_ROOT / "tables"
FIGS_PAPER = PROJECT_ROOT / "figs" / "paper"
FIGS_PRESENTATION = PROJECT_ROOT / "figs" / "presentation"


def apply_style(variant: str) -> None:
    font_size = 10 if variant == "paper" else 14
    plt.rcParams.update(
        {
            "font.family": "DejaVu Serif",
            "font.size": font_size,
            "axes.labelsize": font_size + 1,
            "axes.titlesize": font_size + 1,
            "xtick.labelsize": font_size,
            "ytick.labelsize": font_size,
            "legend.fontsize": font_size - 1,
            "figure.dpi": 600,
            "savefig.dpi": 600,
            "axes.grid": False,
            "lines.linewidth": 1.8 if variant == "paper" else 2.2,
        }
    )


def save_figure(fig: plt.Figure, stem: str, variant: str) -> None:
    root = FIGS_PAPER if variant == "paper" else FIGS_PRESENTATION
    root.mkdir(parents=True, exist_ok=True)
    for extension in ["pdf", "png", "svg"]:
        fig.savefig(root / f"{stem}.{extension}", bbox_inches="tight")


def highres_frame(**kwargs) -> pd.DataFrame:
    return compute_spectra(num_points=FIGURE_POINTS, **kwargs)


def plot_baseline(variant: str) -> None:
    apply_style(variant)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.4))
    for eta_d in [0.0, 0.25, 0.5, 1.0]:
        frame = highres_frame(N=37, eta_d=eta_d, lambda_soc=0.1, gamma_hyb=1.0)
        axes[0].plot(frame["E"], frame["Tz"], label=fr"$\eta_d={eta_d}$")
        axes[1].plot(frame["E"], frame["T0"], label=fr"$\eta_d={eta_d}$")
    axes[0].set_xlabel("Energy (meV)")
    axes[0].set_ylabel(r"$T^z(E)$")
    axes[1].set_xlabel("Energy (meV)")
    axes[1].set_ylabel(r"$T^0(E)$")
    axes[0].legend(frameon=False)
    fig.tight_layout()
    save_figure(fig, "Fig01_baseline_spectra", variant)
    plt.close(fig)


def plot_even_odd_baseline(variant: str) -> None:
    apply_style(variant)
    baseline = load_table(TABLES / "baseline_spectral_metrics.csv")
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.4))

    frame = highres_frame(N=37, eta_d=0.5, lambda_soc=0.1, gamma_hyb=1.0)
    eps_values, even_values, odd_values = even_odd_decomposition(frame["E"].to_numpy(), frame["Tz"].to_numpy(), EF=0.0)
    axes[0].plot(eps_values, even_values, label="Even part")
    axes[0].plot(eps_values, odd_values, label="Odd part")
    axes[0].set_xlabel(r"$\epsilon$ (meV)")
    axes[0].set_ylabel(r"Decomposition of $T^z$")
    axes[0].legend(frameon=False)

    subset = baseline[(baseline["lambda_soc"] == 0.1) & (baseline["gamma_hyb"] == 1.0) & (baseline["N"].isin([10, 37, 91]))]
    for N in [10, 37, 91]:
        block = subset[subset["N"] == N].sort_values("eta_d")
        axes[1].plot(block["eta_d"], block["R_even"], marker="o", label=f"N={N}")
    axes[1].set_xlabel(r"Dephasing $\eta_d$ (meV)")
    axes[1].set_ylabel(r"$R_{\mathrm{even}}$")
    axes[1].legend(frameon=False)

    fig.tight_layout()
    save_figure(fig, "Fig02_even_odd_baseline", variant)
    plt.close(fig)


def plot_detuning(variant: str) -> None:
    apply_style(variant)
    detuning = load_table(TABLES / "detuning_phase_map.csv")
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.4))

    for Delta in [0.0, 0.1, 0.25, 0.5]:
        frame = highres_frame(N=37, eta_d=0.5, lambda_soc=0.1, gamma_hyb=1.0, Delta=Delta)
        axes[0].plot(frame["E"], frame["Tz"], label=fr"$\Delta={Delta}$")
    axes[0].set_xlabel("Energy (meV)")
    axes[0].set_ylabel(r"$T^z(E)$")
    axes[0].legend(frameon=False)

    for eta_d in [0.0, 0.25, 0.5, 1.0]:
        block = detuning[(detuning["N"] == 37) & (detuning["eta_d"] == eta_d)].sort_values("Delta")
        axes[1].plot(block["Delta"], block["I_spin"], marker="o", label=fr"$\eta_d={eta_d}$")
    axes[1].set_xlabel(r"Detuning $\Delta$ (meV)")
    axes[1].set_ylabel(r"$I_{\mathrm{spin}}$")
    axes[1].legend(frameon=False)

    fig.tight_layout()
    save_figure(fig, "Fig03_detuning_unlocks_even_component", variant)
    plt.close(fig)


def plot_detuning_phase_map(variant: str) -> None:
    apply_style(variant)
    detuning = load_table(TABLES / "detuning_phase_map.csv")
    if detuning.empty:
        return

    plot_slice = detuning[detuning["N"] == 37].copy()
    phase_map = plot_slice.pivot_table(index="eta_d", columns="Delta", values="I_spin", aggfunc="mean")
    pol_map = plot_slice.pivot_table(index="eta_d", columns="Delta", values="polarization", aggfunc="mean")

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.4))
    im0 = axes[0].imshow(phase_map.sort_index().to_numpy(), aspect="auto", origin="lower", cmap="coolwarm")
    axes[0].set_xticks(range(len(phase_map.columns)), [f"{value:g}" for value in phase_map.columns], rotation=45)
    axes[0].set_yticks(range(len(phase_map.index)), [f"{value:g}" for value in phase_map.index])
    axes[0].set_xlabel(r"$\Delta$ (meV)")
    axes[0].set_ylabel(r"$\eta_d$ (meV)")
    axes[0].set_title(r"$I_{\mathrm{spin}}$")
    fig.colorbar(im0, ax=axes[0], fraction=0.046)

    im1 = axes[1].imshow(pol_map.sort_index().to_numpy(), aspect="auto", origin="lower", cmap="coolwarm")
    axes[1].set_xticks(range(len(pol_map.columns)), [f"{value:g}" for value in pol_map.columns], rotation=45)
    axes[1].set_yticks(range(len(pol_map.index)), [f"{value:g}" for value in pol_map.index])
    axes[1].set_xlabel(r"$\Delta$ (meV)")
    axes[1].set_ylabel(r"$\eta_d$ (meV)")
    axes[1].set_title("Polarization")
    fig.colorbar(im1, ax=axes[1], fraction=0.046)

    fig.tight_layout()
    save_figure(fig, "Fig04_phase_map_eta_delta_Ispin", variant)
    plt.close(fig)


def plot_contact_vs_detuning(variant: str) -> None:
    apply_style(variant)
    detuning = load_table(TABLES / "detuning_phase_map.csv")
    contact = load_table(TABLES / "contact_asymmetry_phase_map.csv")
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.4))

    for eta_d in [0.0, 0.25, 0.5, 1.0]:
        block = contact[(contact["N"] == 37) & (contact["eta_d"] == eta_d)].sort_values("alpha")
        axes[0].plot(block["alpha"], block["I_spin"], marker="o", label=fr"$\eta_d={eta_d}$")
    axes[0].set_xlabel(r"Contact asymmetry $\alpha$")
    axes[0].set_ylabel(r"$I_{\mathrm{spin}}$")
    axes[0].legend(frameon=False)

    det_best = detuning.groupby("eta_d", as_index=False).agg(best_Reven=("R_even", "max"))
    con_best = contact.groupby("eta_d", as_index=False).agg(best_Reven=("R_even", "max"))
    axes[1].plot(det_best["eta_d"], det_best["best_Reven"], marker="o", label="Detuning")
    axes[1].plot(con_best["eta_d"], con_best["best_Reven"], marker="s", label="Contact asym.")
    axes[1].set_xlabel(r"Dephasing $\eta_d$ (meV)")
    axes[1].set_ylabel(r"Best $R_{\mathrm{even}}$")
    axes[1].legend(frameon=False)

    fig.tight_layout()
    save_figure(fig, "Fig05_contact_asymmetry_vs_detuning", variant)
    plt.close(fig)


def plot_bias_gate(variant: str) -> None:
    apply_style(variant)
    bias_gate = load_table(TABLES / "bias_gate_response.csv")
    if bias_gate.empty:
        fig, ax = plt.subplots(figsize=(6.5, 4.0))
        ax.axis("off")
        ax.text(
            0.5,
            0.5,
            "No bias/gate map was generated because neither detuning\nnor scalar contact asymmetry produced a significant candidate.",
            ha="center",
            va="center",
        )
        fig.tight_layout()
        save_figure(fig, "Fig06_bias_gate_polarization", variant)
        plt.close(fig)
        return

    strongest = bias_gate.sort_values("I_spin", key=lambda series: series.abs(), ascending=False).iloc[0]
    block = bias_gate[
        (bias_gate["stage"] == strongest["stage"])
        & (bias_gate["N"] == strongest["N"])
        & np.isclose(bias_gate["eta_d"], strongest["eta_d"])
    ].copy()
    spin_map = block.pivot_table(index="EF", columns="V", values="I_spin", aggfunc="mean").sort_index()
    pol_map = block.pivot_table(index="EF", columns="V", values="polarization", aggfunc="mean").sort_index()

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.4))
    im0 = axes[0].imshow(spin_map.to_numpy(), aspect="auto", origin="lower", cmap="coolwarm")
    axes[0].set_xticks(range(len(spin_map.columns)), [f"{value:g}" for value in spin_map.columns])
    axes[0].set_yticks(range(len(spin_map.index)), [f"{value:g}" for value in spin_map.index])
    axes[0].set_xlabel("Bias V (meV)")
    axes[0].set_ylabel("Fermi level EF (meV)")
    axes[0].set_title(r"$I_{\mathrm{spin}}$")
    fig.colorbar(im0, ax=axes[0], fraction=0.046)

    im1 = axes[1].imshow(pol_map.to_numpy(), aspect="auto", origin="lower", cmap="coolwarm")
    axes[1].set_xticks(range(len(pol_map.columns)), [f"{value:g}" for value in pol_map.columns])
    axes[1].set_yticks(range(len(pol_map.index)), [f"{value:g}" for value in pol_map.index])
    axes[1].set_xlabel("Bias V (meV)")
    axes[1].set_ylabel("Fermi level EF (meV)")
    axes[1].set_title("Polarization")
    fig.colorbar(im1, ax=axes[1], fraction=0.046)

    fig.tight_layout()
    save_figure(fig, "Fig06_bias_gate_polarization", variant)
    plt.close(fig)


def main() -> None:
    for variant in ["paper", "presentation"]:
        plot_baseline(variant)
        plot_even_odd_baseline(variant)
        plot_detuning(variant)
        plot_detuning_phase_map(variant)
        plot_contact_vs_detuning(variant)
        plot_bias_gate(variant)


if __name__ == "__main__":
    main()
