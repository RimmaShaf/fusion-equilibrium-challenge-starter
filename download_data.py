#!/usr/bin/env python3
"""
Pre-download the Fusion Equilibrium Challenge dataset for fast, offline, repeatable runs.

By default the starter scripts STREAM the data from the Hugging Face Hub (no disk
commitment, but re-downloaded every run and with a limited shuffle buffer). For real
modeling — many epochs, hyperparameter sweeps, full-split shuffling, working offline —
you'll want a local copy. This script materializes one or more configs into the standard
Hugging Face datasets cache, where `experiments.py --local` (and any
`load_dataset(..., streaming=False)` call) will pick them up automatically.

The HF cache handles resume and de-duplication: re-running is cheap and won't store a
second copy. To relocate the cache, set HF_HOME (or HF_DATASETS_CACHE) before running.

Usage:
    python download_data.py                       # download diii_d_train (the training split)
    python download_data.py --config all          # train + both public test configs
    python download_data.py --config mast_public_test
    python download_data.py --config all --yes     # skip the size confirmation prompt
    python download_data.py --list                # show configs + estimated sizes, download nothing
"""
from __future__ import annotations

import argparse
import sys

from datasets import load_dataset, load_dataset_builder

REPO_ID = "Sophelio/fusion-equilibrium-challenge"

# config -> splits to fetch. diii_d_train carries the target flux maps; the *_public_test
# configs are inputs-only (efit_psirz withheld).
CONFIG_SPLITS = {
    "diii_d_train": ["train"],
    "diii_d_public_test": ["public_test"],
    "mast_public_test": ["public_test"],
}


def _fmt_bytes(n: int | None) -> str:
    if not n:
        return "unknown size"
    val = float(n)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if val < 1024 or unit == "TB":
            return f"{val:.1f} {unit}"
        val /= 1024
    return f"{val:.1f} TB"


def estimate(config: str) -> tuple[int, int]:
    """Return (num_examples, num_bytes) summed over the config's splits, best-effort.

    Reads the dataset metadata only — no shot data is downloaded here."""
    builder = load_dataset_builder(REPO_ID, config)
    splits_info = builder.info.splits or {}
    n_rows = n_bytes = 0
    for split in CONFIG_SPLITS[config]:
        info = splits_info.get(split)
        if info is not None:
            n_rows += info.num_examples or 0
            n_bytes += info.num_bytes or 0
    return n_rows, n_bytes


def print_estimates(configs: list[str]) -> int:
    total_bytes = 0
    print(f"Repo: {REPO_ID}")
    print(f"{'config':<22} {'shots':>8} {'est. size':>14}")
    print("-" * 46)
    for config in configs:
        try:
            n_rows, n_bytes = estimate(config)
        except Exception as exc:  # metadata fetch failed — show what we can, keep going
            print(f"{config:<22} {'?':>8} {'(metadata unavailable: ' + type(exc).__name__ + ')':>14}")
            continue
        total_bytes += n_bytes
        print(f"{config:<22} {n_rows:>8} {_fmt_bytes(n_bytes):>14}")
    print("-" * 46)
    print(f"{'TOTAL':<22} {'':>8} {_fmt_bytes(total_bytes):>14}")
    return total_bytes


def download(config: str) -> None:
    for split in CONFIG_SPLITS[config]:
        print(f"\n→ {config} [{split}] ...")
        ds = load_dataset(REPO_ID, config, split=split)  # non-streaming = cache to disk
        print(f"  cached {len(ds)} shots ({config}/{split})")


def main() -> None:
    ap = argparse.ArgumentParser(description="Download the challenge dataset to the local HF cache")
    ap.add_argument(
        "--config",
        default="diii_d_train",
        help="config to download: a name from %s, or 'all' (default: diii_d_train)"
        % list(CONFIG_SPLITS),
    )
    ap.add_argument("--list", action="store_true", help="show configs + estimated sizes, then exit")
    ap.add_argument("--yes", action="store_true", help="skip the size confirmation prompt")
    args = ap.parse_args()

    if args.config == "all":
        configs = list(CONFIG_SPLITS)
    elif args.config in CONFIG_SPLITS:
        configs = [args.config]
    else:
        ap.error(f"unknown config {args.config!r}; choose from {list(CONFIG_SPLITS)} or 'all'")

    print("Estimating download size (reads metadata only)...\n")
    print_estimates(configs)

    if args.list:
        return

    if not args.yes:
        reply = input("\nDownload these to the HF cache? [y/N] ").strip().lower()
        if reply not in ("y", "yes"):
            print("Aborted. (Nothing downloaded.)")
            sys.exit(0)

    for config in configs:
        download(config)

    print(
        "\nDone. The data is in your Hugging Face datasets cache.\n"
        "Train from the local copy with:  python experiments.py --local --n-shots 50"
    )


if __name__ == "__main__":
    main()
