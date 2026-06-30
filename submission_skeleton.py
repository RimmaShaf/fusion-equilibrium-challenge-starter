#!/usr/bin/env python3
"""
Submission skeleton — a runnable, format-correct example of a challenge submission.

> Status: PROPOSED DRAFT format (scoring not yet finalized). See README → "Output &
> Submission Format". This script is the executable version of that spec: run it and read
> the printed shapes to see exactly what a submission looks like — no Git LFS / sample data
> needed (it streams the public test inputs from Hugging Face).

What it does:
  - For each test config, streams the input-only shots, and for every shot produces a
    prediction of `efit_psirz` at the shot's `efit_times`, in the machine's NATIVE grid:
        DIII-D -> (T, 65, 65)        MAST -> (T, 65, 129)   (central NaN block allowed)
  - Saves one .npz per config with arrays keyed `shot_0000`, `shot_0001`, … in stream order.

To make a real submission: replace `your_model_predict()` with your trained model. The
placeholder here just emits zeros of the correct shape (a valid-but-useless submission), so
you can confirm the plumbing/shape before plugging in a model.

    uv run python submission_skeleton.py --max-shots 5      # quick demo (few shots)
    uv run python submission_skeleton.py --max-shots 0      # all shots (full submission)
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
from datasets import load_dataset

REPO_ID = "Sophelio/fusion-equilibrium-challenge"
TEST_CONFIGS = [("diii_d_public_test", "public_test"), ("mast_public_test", "public_test")]
# Native target grid per machine (rows=Z, cols=R). MAST's inner ~64 columns are NaN in the
# ground truth (central-column hardware); the proposed scorer ignores non-finite GT pixels.
GRID = {"DIII-D": (65, 65), "MAST": (65, 129)}


def your_model_predict(row: dict, source: str) -> np.ndarray:
    """REPLACE ME. Return predicted efit_psirz as (T, H, W) for this shot, aligned to
    row['efit_times'].

    Inputs in `row` (NO magnetic-diagnostic array — that's the point of the challenge):
      - `magnetics_*` COIL CURRENTS = commanded actuators (F-coils, ECOILA/bcoil, MAST P-coils,
        Solenoid, TF, …), not measurements of the plasma's field.
      - `thomson_*` = kinetic profiles (electron temperature & density).
      - `efit_times` (+ MAST: `efit_grid_R/Z`).
    The target `efit_psirz` is withheld in test configs. `magnetics_dsep` is EFIT-DERIVED (from the
    target) and is kept in the data but considered OUT-OF-SPIRIT — don't depend on it; see
    README → "What 'blind' really means"."""
    T = len(np.asarray(row["efit_times"]))
    H, W = GRID[source]
    return np.zeros((T, H, W), dtype=np.float32)   # placeholder baseline


def build_submission(config: str, split: str, out_dir: Path, max_shots: int) -> Path:
    ds = load_dataset(REPO_ID, config, split=split, streaming=True)
    preds: dict[str, np.ndarray] = {}
    for i, row in enumerate(ds):
        if max_shots and i >= max_shots:
            break
        source = row.get("source", "DIII-D")
        pred = your_model_predict(row, source)
        T, H, W = pred.shape
        assert (H, W) == GRID[source], f"{config} shot {i}: got {(H, W)}, expected {GRID[source]}"
        assert T == len(np.asarray(row["efit_times"])), f"{config} shot {i}: T mismatch"
        preds[f"shot_{i:04d}"] = pred.astype(np.float32)
        if (i + 1) % 25 == 0:
            print(f"  {config}: {i + 1} shots")
    out = out_dir / f"{config}.npz"
    np.savez_compressed(out, **preds)
    shapes = {k: v.shape for k, v in list(preds.items())[:3]}
    print(f"  {config}: {len(preds)} shots -> {out.name}  (e.g. {shapes})")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate a format-correct submission skeleton")
    ap.add_argument("--max-shots", type=int, default=5, help="cap shots per config (0 = all)")
    ap.add_argument("--out", type=Path, default=Path("submission"))
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    print("Building proposed-format submission (placeholder zeros — swap in your_model_predict):")
    for config, split in TEST_CONFIGS:
        build_submission(config, split, args.out, args.max_shots)
    print(f"\nWrote submission/*.npz to {args.out.resolve()}")
    print("Each .npz: one array per shot (shot_0000, …) of shape (T, H, W) at efit_times.")


if __name__ == "__main__":
    main()
