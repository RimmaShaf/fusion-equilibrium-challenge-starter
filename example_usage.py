#!/usr/bin/env python3
"""
Example: load the Fusion Equilibrium Challenge dataset from the Hugging Face Hub.

    pip install -r requirements.txt
    python example_usage.py
"""

import numpy as np
from datasets import load_dataset

REPO_ID = "Sophelio/fusion-equilibrium-challenge"


def main() -> None:
    # --- DIII-D training data (includes the target flux maps) ---
    train = load_dataset(REPO_ID, "diii_d_train", split="train", streaming=True)
    shot = next(iter(train))

    times = np.asarray(shot["efit_times"])      # (T,) ms
    psirz = np.asarray(shot["efit_psirz"])      # (T, 65, 65) — the target
    ip = np.asarray(shot["magnetics_plasma_current"])

    print("DIII-D train example")
    print(f"  source              : {shot['source']}")
    print(f"  EFIT timesteps (T)  : {len(times)}")
    print(f"  flux map shape      : {psirz.shape}")
    print(f"  |Ip| peak (A)       : {np.nanmax(np.abs(ip)):.0f}")

    # --- Public test (inputs only — efit_psirz is withheld) ---
    test = load_dataset(REPO_ID, "diii_d_public_test", split="public_test", streaming=True)
    t0 = next(iter(test))
    assert "efit_psirz" not in t0, "test split should not contain the target"
    print("\nDIII-D public_test example")
    print(f"  has efit_psirz?     : {'efit_psirz' in t0}  (expected False)")
    print(f"  predict flux maps at: {len(t0['efit_times'])} timesteps")

    # --- MAST is zero-shot: test config only ---
    mast = load_dataset(REPO_ID, "mast_public_test", split="public_test", streaming=True)
    m0 = next(iter(mast))
    print("\nMAST public_test example")
    print(f"  source              : {m0['source']}")
    print(f"  grid R / Z present  : {'efit_grid_R' in m0} / {'efit_grid_Z' in m0}")


if __name__ == "__main__":
    main()
