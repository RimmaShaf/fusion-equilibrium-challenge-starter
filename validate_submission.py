#!/usr/bin/env python3
"""
Submission validator for the Fusion Equilibrium Challenge.

Checks that a submission .npz has the right SHAPE before you upload it to Codabench — it does
NOT score (the scorer is held by the organizers and runs on the platform). Catching a malformed
file here saves you a wasted submission slot.

For a config it streams the public-test inputs from Hugging Face (no ground truth needed) and
verifies, for every shot in stream order:
  - the key `shot_0000`, `shot_0001`, … is present,
  - the array is `(T, H, W)` with T = number of `efit_times` and the machine's native grid
    (DIII-D 65×65, MAST 65×129),
  - the dtype is floating point,
  - (DIII-D) no NaN/Inf, since the DIII-D ground truth is fully finite.

Usage:
    python validate_submission.py submission/diii_d_public_test.npz --config diii_d_public_test
    python validate_submission.py submission/mast_public_test.npz  --config mast_public_test
    python validate_submission.py submission/diii_d_public_test.npz --config diii_d_public_test --max-shots 5
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

REPO_ID = "Sophelio/fusion-equilibrium-challenge"
# Native target grid per machine (rows = Z, cols = R).
GRID = {"DIII-D": (65, 65), "MAST": (65, 129)}
# config -> (split, machine). One public-test config = one machine.
CONFIG_INFO = {
    "diii_d_public_test": ("public_test", "DIII-D"),
    "mast_public_test": ("public_test", "MAST"),
}


def validate(npz_path: Path, config: str, max_shots: int) -> int:
    split, machine = CONFIG_INFO[config]
    H, W = GRID[machine]

    if not npz_path.exists():
        print(f"ERROR: {npz_path} not found", file=sys.stderr)
        return 2

    from datasets import load_dataset

    sub = dict(np.load(npz_path, allow_pickle=False))
    print(f"Validating {npz_path.name} against {config} (expects {machine} grid {(H, W)})")
    print(f"  {len(sub)} arrays in the .npz; streaming reference inputs from the Hub …")

    ds = load_dataset(REPO_ID, config, split=split, streaming=True)
    errors: list[str] = []
    n = 0
    capped = False
    for i, row in enumerate(ds):
        if max_shots and i >= max_shots:
            capped = True
            print(f"  (stopped at --max-shots {max_shots}; full validation needs --max-shots 0)")
            break
        n += 1
        key = f"shot_{i:04d}"
        T = len(np.asarray(row["efit_times"]))
        if key not in sub:
            errors.append(f"{key}: MISSING (expected shape ({T}, {H}, {W}))")
            continue
        arr = sub[key]
        if arr.shape != (T, H, W):
            errors.append(f"{key}: shape {arr.shape}, expected ({T}, {H}, {W})")
        if not np.issubdtype(arr.dtype, np.floating):
            errors.append(f"{key}: dtype {arr.dtype}, expected float")
        if machine == "DIII-D" and not np.isfinite(arr).all():
            errors.append(f"{key}: contains NaN/Inf (DIII-D is fully finite; those pixels score as error)")

    if not capped:
        expected_keys = {f"shot_{i:04d}" for i in range(n)}
        extra = [k for k in sub if k not in expected_keys]
        if extra:
            errors.append(f"unexpected keys not in stream order: {extra[:5]}{' …' if len(extra) > 5 else ''}")

    if errors:
        print(f"\n✗ {len(errors)} problem(s):")
        for e in errors[:40]:
            print(f"    - {e}")
        if len(errors) > 40:
            print(f"    … and {len(errors) - 40} more")
        return 1
    print(f"\n✓ OK — {n} shots, all shapes/keys/dtype valid for {config}.")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="Validate a submission .npz's shape (no scoring)")
    ap.add_argument("npz", type=Path, help="submission .npz to validate")
    ap.add_argument("--config", required=True, choices=list(CONFIG_INFO),
                    help="which public-test config this .npz targets")
    ap.add_argument("--max-shots", type=int, default=0, help="cap shots checked (0 = all)")
    args = ap.parse_args()
    return validate(args.npz, args.config, args.max_shots)


if __name__ == "__main__":
    raise SystemExit(main())
