# Release Notes v0.1.1

This release updates the initial archive with a documented Google Colab workflow.

## Added

- Colab-ready notebook for the post-processing workflow:
  `notebooks/ciss_ladder_transport_colab.ipynb`
- helper script to stage Google Drive transport CSV files into the repository layout:
  `scripts/prepare_colab_data.py`
- Colab and Google Cloud execution guide:
  `docs/google_colab.md`
- direct Colab launch link in the main `README.md`

## Purpose

This release makes the repository easier to run in a cloud environment while preserving the legacy research scripts unchanged under `legacy/`.
