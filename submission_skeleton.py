#!/usr/bin/env python3
"""
Submission skeleton — a runnable, format-correct example of a challenge submission.

This is the executable version of README → "Output & Submission Format". Run it and read the
printed shapes to see exactly what a submission looks like — no Git LFS / sample data needed
(it streams the public test inputs from Hugging Face).

What it does:
  - For each test config, streams the input-only shots, and for every shot produces predictions
    of the THREE targets at the shot's `efit_times`, grouped per shot in one .npz per config:
        shot_0000_psirz   (T, H, W)   flux map   DIII-D 65x65 / MAST 65x129 (central NaN allowed)
        shot_0000_lcfs    (T, N, 2)   LCFS contour — N ordered (R,Z) boundary points (meters)
        shot_0000_scalars (T, 5)      [betaN, li, q95, R_axis, Z_axis]
  - Writes one `<config>.npz` per config plus a `manifest.json` declaring the harmonization layer
    (required by the rules; fill in your own H identifier).

The leaderboard score is the composite  S = 0.6*R2_psi + 0.25*R2_scalars + 0.15*(1 - D_LCFS).
To make a real submission, replace `your_model_predict()` with your trained model. The placeholder
here emits zeros of the correct shape (a valid-but-useless submission) so you can confirm the
plumbing/shapes before plugging in a model, then validate with `validate_submission.py`.

    uv run python submission_skeleton.py --max-shots 5      # quick demo (few shots)
    uv run python submission_skeleton.py --max-shots 0      # all shots (full submission)
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
from datasets import load_dataset

REPO_ID = "Sophelio/fusion-equilibrium-challenge"
TEST_CONFIGS = [("diii_d_public_test", "public_test"), ("mast_public_test", "public_test")]
# Native flux grid per machine (rows=Z, cols=R). MAST's inner ~64 columns are NaN in the ground
# truth (central-column hardware); the scorer's R2_psi ignores non-finite GT pixels.
GRID = {"DIII-D": (65, 65), "MAST": (65, 129)}
# Scalar column order — MUST match this for R2_scalars to line up.
SCALARS = ["betaN", "li", "q95", "R_axis", "Z_axis"]
# LCFS contour density used by this skeleton. Your choice — Hausdorff is point-count agnostic.
LCFS_N_POINTS = 128


def your_model_predict(row: dict, source: str) -> dict:
    """REPLACE ME. Return predictions for this shot, aligned to row['efit_times'], as a dict:
        {"psirz":   (T, H, W) flux map,
         "lcfs":    (T, N, 2) ordered (R,Z) LCFS boundary points in meters,
         "scalars": (T, 5)    [betaN, li, q95, R_axis, Z_axis]}

    Inputs in `row` (NO magnetic-diagnostic array — that's the point of the challenge):
      - `magnetics_*` COIL CURRENTS = commanded actuators (F-coils, ECOILA/bcoil, MAST P-coils,
        Solenoid, TF, …), not measurements of the plasma's field.
      - `thomson_*` = kinetic profiles (electron temperature & density).
      - `efit_times` (+ MAST: `efit_grid_R/Z`).
    The targets are withheld in test configs. `magnetics_dsep` is EFIT-DERIVED (from the target)
    and is kept in the data but considered OUT-OF-SPIRIT — don't depend on it; see README →
    "What 'blind' really means"."""
    T = len(np.asarray(row["efit_times"]))
    H, W = GRID[source]
    return {
        "psirz": np.zeros((T, H, W), dtype=np.float32),                # placeholder baseline
        "lcfs": np.zeros((T, LCFS_N_POINTS, 2), dtype=np.float32),     # placeholder boundary
        "scalars": np.zeros((T, len(SCALARS)), dtype=np.float32),      # placeholder scalars
    }


def build_submission(config: str, split: str, out_dir: Path, max_shots: int) -> Path:
    ds = load_dataset(REPO_ID, config, split=split, streaming=True)
    preds: dict[str, np.ndarray] = {}
    n = 0
    for i, row in enumerate(ds):
        if max_shots and i >= max_shots:
            break
        source = row.get("source", "DIII-D")
        T = len(np.asarray(row["efit_times"]))
        H, W = GRID[source]
        out = your_model_predict(row, source)

        assert out["psirz"].shape == (T, H, W), f"{config} shot {i}: psirz {out['psirz'].shape} != {(T, H, W)}"
        assert out["scalars"].shape == (T, len(SCALARS)), f"{config} shot {i}: scalars {out['scalars'].shape} != {(T, len(SCALARS))}"
        assert out["lcfs"].ndim == 3 and out["lcfs"].shape[0] == T and out["lcfs"].shape[2] == 2, \
            f"{config} shot {i}: lcfs {out['lcfs'].shape} != (T, N, 2)"

        preds[f"shot_{i:04d}_psirz"] = out["psirz"].astype(np.float32)
        preds[f"shot_{i:04d}_lcfs"] = out["lcfs"].astype(np.float32)
        preds[f"shot_{i:04d}_scalars"] = out["scalars"].astype(np.float32)
        n = i + 1
        if n % 25 == 0:
            print(f"  {config}: {n} shots")

    out_path = out_dir / f"{config}.npz"
    np.savez_compressed(out_path, **preds)
    print(f"  {config}: {n} shots -> {out_path.name}  "
          f"(e.g. shot_0000_psirz {preds.get('shot_0000_psirz', np.empty(0)).shape}, "
          f"_lcfs {preds.get('shot_0000_lcfs', np.empty(0)).shape}, "
          f"_scalars {preds.get('shot_0000_scalars', np.empty(0)).shape})")
    return out_path


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate a format-correct submission skeleton")
    ap.add_argument("--max-shots", type=int, default=5, help="cap shots per config (0 = all)")
    ap.add_argument("--out", type=Path, default=Path("submission"))
    ap.add_argument("--harmonization", default="default-H",
                    help="harmonization-layer id recorded in manifest.json")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    print("Building submission (placeholder zeros — swap in your_model_predict):")
    written = []
    for config, split in TEST_CONFIGS:
        written.append(build_submission(config, split, args.out, args.max_shots).name)

    # Manifest declaring the harmonization layer (required by the rules).
    manifest = {
        "harmonization_layer": args.harmonization,
        "scalar_order": SCALARS,
        "configs": written,
    }
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2))

    print(f"\nWrote {', '.join(written)} + manifest.json to {args.out.resolve()}")
    print("Each .npz: per shot, keys shot_XXXX_psirz (T,H,W), _lcfs (T,N,2), _scalars (T,5).")
    print("Next: python validate_submission.py <config>.npz --config <config>")


if __name__ == "__main__":
    main()
