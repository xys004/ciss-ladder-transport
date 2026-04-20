"""Input/output helpers for raw referee-facing datasets."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


def save_spectral_kernel_csv(path: str | Path, frame: pd.DataFrame) -> Path:
    """Write one spectral kernel dataset as CSV."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(target, index=False)
    return target


def save_metadata_json(path: str | Path, metadata: dict) -> Path:
    """Write a metadata sidecar describing parameters and provenance."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
    return target
