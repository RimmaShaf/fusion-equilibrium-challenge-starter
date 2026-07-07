#!/usr/bin/env python3
"""
Submission validator for the Fusion Equilibrium Challenge.

Checks that a submission .npz has the right STRUCTURE before you upload it to Codabench — it does
NOT score (the scorer is held by the organizers and runs on the platform). Catching a malformed
file here saves you a wasted submission slot.

A submission predicts the flux map and six composite scalars (the sixth is dsep) per shot
(grouped per shot; see README →
"Output & Submission Format"). The LCFS is NOT submitted — the scorer derives it from the flux
map. For a config this streams the public-test inputs from Hugging Face (no ground truth needed)
and verifies, for every shot in stream order, with T = number of `efit_times`:
  - `shot_XXXX_psirz`  present, shape `(T, H, W)` in the machine's native grid (both dense 65×65:
    DIII-D 65×65, MAST 65×65), floating dtype, and (DIII-D only) no NaN/Inf since its GT is finite,
  - `shot_XXXX_<scalar>` present for each of betaN, li, q95, R_axis, Z_axis, shape `(T,)`, floating,
  - `shot_XXXX_dsep` present, shape `(T,)`, floating — the x-point gap, the sixth composite scalar
    (also reported on its own as R²_dsep).

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
# Native flux grid per machine (rows = Z, cols = R).
GRID = {"DIII-D": (65, 65), "MAST": (65, 65)}
SCALARS = ["betaN", "li", "q95", "R_axis", "Z_axis"]  # five of the six composite scalars
DSEP = "dsep"  # the sixth composite scalar in R2_scalars (also reported on its own as R2_dsep)
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
        prefix = f"shot_{i:04d}"
        T = len(np.asarray(row["efit_times"]))

        # flux map: (T, H, W)
        k = f"{prefix}_psirz"
        if k not in sub:
            errors.append(f"{k}: MISSING (expected ({T}, {H}, {W}))")
        else:
            arr = sub[k]
            if arr.shape != (T, H, W):
                errors.append(f"{k}: shape {arr.shape}, expected ({T}, {H}, {W})")
            if not np.issubdtype(arr.dtype, np.floating):
                errors.append(f"{k}: dtype {arr.dtype}, expected float")
            if machine == "DIII-D" and not np.isfinite(arr).all():
                errors.append(f"{k}: contains NaN/Inf (DIII-D is fully finite; those pixels score as error)")

        # scalars (+ dsep): one (T,) array each, named to prevent column mix-ups
        for name in (*SCALARS, DSEP):
            k = f"{prefix}_{name}"
            if k not in sub:
                errors.append(f"{k}: MISSING (expected ({T},))")
                continue
            arr = sub[k]
            if arr.shape != (T,):
                errors.append(f"{k}: shape {arr.shape}, expected ({T},)")
            if not np.issubdtype(arr.dtype, np.floating):
                errors.append(f"{k}: dtype {arr.dtype}, expected float")

    if not capped:
        expected_keys = {f"shot_{i:04d}_{s}" for i in range(n) for s in ("psirz", *SCALARS, DSEP)}
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
