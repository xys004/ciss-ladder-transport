import os
from pathlib import Path

proj_dir = Path("ciss_ladder_symmetry_breaking_project/src")

# Fix utils.py
utils_file = proj_dir / "utils.py"
text = utils_file.read_text(encoding="utf-8")
text = text.replace("parents[3]", "parents[2]")
text = text.replace("dresselhaus_chain_1=0.0,\n        dresselhaus_chain_2=0.0,", "dresselhaus=0.0,")
utils_file.write_text(text, encoding="utf-8")

# Fix others
for fname in ["generate_spectra_baseline.py", "generate_spectra_detuning.py", 
              "generate_spectra_contact_asymmetry.py", "generate_spectra_spin_active_contacts.py",
              "integrate_observables.py", "make_publication_figures.py", "build_master_report.py"]:
    f = proj_dir / fname
    if not f.exists(): continue
    text = f.read_text(encoding="utf-8")
    
    # We want PROJ_ROOT = parents[1], and REPO_ROOT = parents[2]
    if "REPO_ROOT = Path(__file__).resolve().parents[2]" in text:
        text = text.replace(
            "REPO_ROOT = Path(__file__).resolve().parents[2]\nsys.path.insert(0, str(REPO_ROOT / \"src\"))",
            "PROJ_ROOT = Path(__file__).resolve().parents[1]\nREPO_ROOT = Path(__file__).resolve().parents[2]\nsys.path.insert(0, str(REPO_ROOT / \"src\"))"
        )
        text = text.replace("REPO_ROOT / \"data\"", "PROJ_ROOT / \"data\"")
        text = text.replace("REPO_ROOT / \"tables\"", "PROJ_ROOT / \"tables\"")
        text = text.replace("REPO_ROOT / \"reports\"", "PROJ_ROOT / \"reports\"")
        text = text.replace("REPO_ROOT / \"figs\"", "PROJ_ROOT / \"figs\"")
        f.write_text(text, encoding="utf-8")
print("Fixes applied successfully.")
