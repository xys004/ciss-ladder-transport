from __future__ import annotations

from build_master_report import build_report
from campaign_core import default_workers, ensure_layout, write_contact_audit
from generate_controls import main as run_controls
from generate_spectra_baseline import main as run_baseline
from generate_spectra_contact_asymmetry import main as run_contact_asymmetry
from generate_spectra_detuning import main as run_detuning
from generate_spectra_spin_active_contacts import main as run_spin_active
from integrate_observables import main as run_bias_gate
from make_publication_figures import main as make_figures


def main() -> None:
    workers = default_workers()
    ensure_layout()
    write_contact_audit()
    run_baseline(max_workers=workers)
    run_detuning(max_workers=workers)
    run_contact_asymmetry(max_workers=workers)
    run_spin_active(max_workers=workers)
    run_bias_gate(max_workers=workers)
    run_controls(max_workers=workers)
    make_figures()
    build_report()


if __name__ == "__main__":
    main()
