from __future__ import annotations

import argparse
import shutil
from pathlib import Path


DEPHASING_MAP = {
    "w05": "w05",
    "w": "w",
    "w2": "w2",
}

DISORDER_MAP = {
    "w05": "w05",
    "w": "w",
    "w2": "w2",
}


def _copy_tree(src: Path, dst: Path) -> None:
    if not src.exists():
        raise FileNotFoundError(f"Source path does not exist: {src}")
    dst.mkdir(parents=True, exist_ok=True)
    for item in src.iterdir():
        target = dst / item.name
        if item.is_dir():
            shutil.copytree(item, target, dirs_exist_ok=True)
        else:
            shutil.copy2(item, target)


def stage_dephasing(source_root: Path, repo_root: Path) -> None:
    raw_root = repo_root / "data" / "raw" / "dephasing"
    for src_name, dst_name in DEPHASING_MAP.items():
        _copy_tree(source_root / "trans_vs_N_decoherencia1" / src_name, raw_root / dst_name)
    _copy_tree(source_root / "trans_vs_N", raw_root / "coherent")


def stage_disorder(source_root: Path, repo_root: Path, realizations: str = "10000") -> None:
    raw_root = repo_root / "data" / "raw" / "disorder"
    for src_name, dst_name in DISORDER_MAP.items():
        _copy_tree(source_root / f"desorden{realizations}" / src_name, raw_root / dst_name)
    _copy_tree(source_root / "trans_vs_N", raw_root / "coherent")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Stage Google Drive transport data into the repository layout used by the Colab workflow.")
    parser.add_argument("--repo-root", default=".", help="Path to the repository root.")
    parser.add_argument("--source-root", required=True, help="Drive folder containing Rashba transport subdirectories.")
    parser.add_argument("--mode", choices=["dephasing", "disorder", "both"], default="both")
    parser.add_argument("--disorder-realizations", default="10000", help="Disorder batch to copy, e.g. 100, 1000, or 10000.")
    args = parser.parse_args(argv)

    repo_root = Path(args.repo_root).resolve()
    source_root = Path(args.source_root).resolve()

    if args.mode in {"dephasing", "both"}:
        stage_dephasing(source_root, repo_root)
    if args.mode in {"disorder", "both"}:
        stage_disorder(source_root, repo_root, realizations=args.disorder_realizations)

    print(f"Staged data under {repo_root / 'data' / 'raw'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
