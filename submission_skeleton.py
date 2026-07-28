#!/usr/bin/env python3
"""
Submission skeleton — a runnable, format-correct example of a challenge submission.

This is the executable version of README → "Output & Submission Format". Run it and read the
printed shapes to see exactly what a submission looks like — no Git LFS / sample data needed
(it streams the public test inputs from Hugging Face).

What you submit (per shot, at each `efit_times` timestamp), grouped per shot in one .npz per config:
    shot_0000_psirz    (T, H, W)   flux map   DIII-D 65x65 / MAST 65x65 (both dense, no NaN region)
    shot_0000_q95      (T,)        edge safety factor
    shot_0000_betaN    (T,)        normalized beta

That is the WHOLE contract (metric v2): the flux map plus the only two scalars a flux map cannot
contain (q95 needs the toroidal field function F(psi), betaN needs the pressure profile p(psi)).
Everything else — the LCFS boundary, magnetic axis (R_axis/Z_axis), elongation, triangularity,
volume and internal inductance (li) — is DERIVED from your submitted
flux map by the scorer, with the same published functionals it applies to the ground-truth flux.
A scalar can only be earned by a psi that implies it: there is no separate scalar head to tune.

The leaderboard score is the composite
    S = 0.55*R2_psi + 0.15*R2_{q95,betaN} + 0.10*(1 - D_LCFS) + 0.20*Consistency
where Consistency is the mean agreement of the seven psi-derived scalars, f(psi_pred) vs
f(psi_gt). A perfect flux map scores D_LCFS = 0 and Consistency = 1 by construction.

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
# Native flux grid per machine (rows=Z, cols=R). Both machines are a dense, fully finite 65x65
# grid — MAST's upstream EFIT 65x129 grid (65 real R columns interleaved with 64 empty ones) is
# collapsed to a dense 65x65 in the corrected dataset, so there is no central NaN region.
GRID = {"DIII-D": (65, 65), "MAST": (65, 65)}
# The only two submitted scalars (metric v2), each under its own per-shot key (named, not
# positional, to make column mix-ups impossible). Everything else is derived from your flux map.
SCALARS = ["q95", "betaN"]


def your_model_predict(row: dict, source: str) -> dict:
    """REPLACE ME. Return predictions for this shot, aligned to row['efit_times'], as a dict:
        {"psirz":  (T, H, W) flux map,
         "q95":    (T,),
         "betaN":  (T,)}

    Inputs in `row` (NO magnetic-diagnostic array — that's the point of the challenge):
      - `magnetics_*` COIL CURRENTS = commanded actuators (F-coils, ECOILA/bcoil, MAST P-coils,
        Solenoid, TF, …), not measurements of the plasma's field.
      - `thomson_*` = kinetic profiles (electron temperature & density).
      - `efit_times` (+ MAST: `efit_grid_R/Z`).
    The targets are withheld in test configs. `magnetics_dsep` is EFIT-DERIVED (computed from
    the target equilibrium) and is neither an input nor a scored target. Also available as
    inputs: `coil_*` (PF-coil positions/turns, joined to the current columns by
    `coil_input_column`) and `thomson_chord_R/Z` (chord positions). Focus on making psi(R,Z)
    right; the geometry terms (boundary, axis, shape, li) all follow from it."""
    T = len(np.asarray(row["efit_times"]))
    H, W = GRID[source]
    out = {"psirz": np.zeros((T, H, W), dtype=np.float32)}      # placeholder baseline
    for name in SCALARS:
        out[name] = np.zeros(T, dtype=np.float32)               # placeholder scalars
    return out


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
        preds[f"shot_{i:04d}_psirz"] = out["psirz"].astype(np.float32)
        for name in SCALARS:   # the two submitted scalars
            arr = np.asarray(out[name])
            assert arr.shape == (T,), f"{config} shot {i}: {name} {arr.shape} != ({T},)"
            preds[f"shot_{i:04d}_{name}"] = arr.astype(np.float32)

        n = i + 1
        if n % 25 == 0:
            print(f"  {config}: {n} shots")

    out_path = out_dir / f"{config}.npz"
    np.savez_compressed(out_path, **preds)
    psh = preds.get("shot_0000_psirz", np.empty(0)).shape
    print(f"  {config}: {n} shots -> {out_path.name}  (e.g. shot_0000_psirz {psh}, "
          f"shot_0000_q95 {preds.get('shot_0000_q95', np.empty(0)).shape})")
    return out_path


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate a format-correct submission skeleton")
    ap.add_argument("--max-shots", type=int, default=5, help="cap shots per config (0 = all)")
    ap.add_argument("--out", type=Path, default=Path("submission"))
    ap.add_argument("--harmonization", default="default-H",
                    help="harmonization-layer id recorded in manifest.json (an optional label — "
                         "it is not scored and gates nothing)")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    print("Building submission (placeholder zeros — swap in your_model_predict):")
    written = []
    for config, split in TEST_CONFIGS:
        written.append(build_submission(config, split, args.out, args.max_shots).name)

    # Manifest with the optional harmonization-layer label (descriptive metadata only).
    manifest = {
        "harmonization_layer": args.harmonization,
        "scalars": SCALARS,
        "configs": written,
    }
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2))

    print(f"\nWrote {', '.join(written)} + manifest.json to {args.out.resolve()}")
    print("Each .npz: per shot, key shot_XXXX_psirz (T,H,W) + one (T,) key per submitted scalar "
          f"({', '.join(SCALARS)}). Everything else is derived from psirz by the scorer.")
    print("Next: python validate_submission.py <config>.npz --config <config>")


if __name__ == "__main__":
    main()
