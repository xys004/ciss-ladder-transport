# Google Colab Guide

This repository includes a Colab-ready workflow for the current-vs-length post-processing stage. It is designed for users who already have transmission CSV files from the transport calculations and want a reproducible notebook-based environment.

## Recommended Colab entry point

Open:

- `notebooks/ciss_ladder_transport_colab.ipynb`

The notebook:

1. clones the GitHub repository,
2. installs dependencies,
3. optionally mounts Google Drive,
4. copies transmission CSV files into the repository layout expected by the clean post-processing scripts,
5. runs the current integration for dephasing and/or disorder cases.

## Expected Google Drive source layout

The helper script assumes a Drive folder with the same structure used by the original Colab workflow:

```text
MyDrive/
  Rashba/
    trans_vs_N/
      data_N10.csv
      ...
    trans_vs_N_decoherencia1/
      w05/data_decoheren_N10.csv
      w/data_decoheren_N10.csv
      w2/data_decoheren_N10.csv
    desorden10000/
      w05/data_disorder_N10.csv
      w/data_disorder_N10.csv
      w2/data_disorder_N10.csv
```

If your Drive folders use a different base path, adjust only the `SOURCE_ROOT` variable in the notebook.

## Helper script

Use:

```bash
python scripts/prepare_colab_data.py --source-root /content/drive/MyDrive/Rashba --mode both --disorder-realizations 10000
```

This copies the relevant CSV files into:

```text
data/raw/dephasing/
data/raw/disorder/
```

without changing the preserved legacy scripts.

## Running the clean post-processing step

Dephasing:

```bash
python scripts/current_from_transmission.py --config configs/dephasing_current_example.json
```

Disorder:

```bash
python scripts/current_from_transmission.py --config configs/disorder_current_example.json
```

## About the legacy scripts

The original transport scripts in `legacy/` are preserved for archival integrity. They are not yet packaged as polished Colab modules, and their runtime can be substantial for large disorder averages such as `M = 10000`.

For public reproducibility, the recommended Colab route in this repository is:

- use archived transmission CSV files as inputs,
- run the clean local post-processing and documentation-backed workflow,
- keep the original scripts as reference implementations.

## Google Cloud VM usage

The same cleaned workflow also runs on a Google Cloud VM or any Linux machine:

```bash
git clone https://github.com/xys004/ciss-ladder-transport.git
cd ciss-ladder-transport
pip install -r requirements.txt
python scripts/current_from_transmission.py --config configs/dephasing_current_example.json
```

In that case, place the CSV files directly under `data/raw/` instead of mounting Google Drive.
