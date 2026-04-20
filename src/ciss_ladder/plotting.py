"""Minimal plotting helpers for figure-ready transport outputs."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def plot_energy_kernel(input_csv: str | Path, x_column: str, y_column: str, ylabel: str, output_path: str | Path, title: str = "") -> Path:
    """Plot one spectral kernel as a simple manuscript-facing figure."""

    frame = pd.read_csv(input_csv)
    figure, axis = plt.subplots(figsize=(6.0, 4.0))
    axis.plot(frame[x_column], frame[y_column], lw=1.5)
    axis.set_xlabel("Energy [meV]")
    axis.set_ylabel(ylabel)
    if title:
        axis.set_title(title)
    axis.grid(alpha=0.2)
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(target, dpi=200)
    plt.close(figure)
    return target


def plot_length_scan(input_csv: str | Path, x_column: str, y_columns: list[str], output_path: str | Path, ylabel: str) -> Path:
    """Plot one integrated current scan versus chain length."""

    frame = pd.read_csv(input_csv)
    figure, axis = plt.subplots(figsize=(6.0, 4.0))
    for column in y_columns:
        axis.plot(frame[x_column], frame[column], marker="o", lw=1.2, label=column)
    axis.set_xlabel("N")
    axis.set_ylabel(ylabel)
    axis.grid(alpha=0.2)
    if len(y_columns) > 1:
        axis.legend()
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(target, dpi=200)
    plt.close(figure)
    return target
