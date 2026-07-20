# ⚛️ The Fusion Equilibrium Challenge: A Hacker's Guide

Matthew Waller1, Craig Michoski1, Tapan Ganatma Nakkina1, Brian Sammuli2, William Boyes2, Mitchell Clark2, Sterling Smith2, Raffi Nazikian2

1. Sophelio  
2. General Atomics
3. UT Austin

## Getting Started

This repository is the **public starter kit** for the Fusion Equilibrium Challenge. It gives you:

- **Sample parquet shots** (3 DIII-D + 3 MAST) for offline exploration and dFL visualization
- **Full training data** on Hugging Face: [`Sophelio/fusion-equilibrium-challenge`](https://huggingface.co/datasets/Sophelio/fusion-equilibrium-challenge)
- **Baseline models** in `experiments.py` that load from Hugging Face (same data hackathoners use), and can also read a fully **downloaded** copy of the dataset with `--source local`

> **What you predict (metric v2):** the primary target is the 2-D flux map `efit_psirz`; you
> submit it plus only **`q95` and `betaN`** — the two scalars a flux map cannot contain. The
> training data still ships all EFIT scalar labels (`efit_beta_n`, `efit_li`, `efit_q95`,
> `efit_r_axis`, `efit_z_axis`, `magnetics_dsep`) as supervision, and all are **withheld on the
> test splits** — but at scoring time the axis, shape, `li`, and `dsep` are *derived from your
> submitted flux map* and scored for consistency against the same derivation on the true flux.
> A good ψ is what earns them.



### Get the demo shots (Git LFS)

The six files in `parquet_data/` are tracked with **Git LFS**. If you cloned without LFS
installed you'll see ~130-byte pointer files instead of real data, and the dFL visualizer
won't load. Install LFS once, then pull the blobs:

```bash
git lfs install
git lfs pull
```

The **training/evaluation data on Hugging Face does not need LFS** — only the local demo shots do.

### Environment setup

Pick **uv** (recommended) or **conda/mamba**. Both create a self-contained environment for this repo — no need for a pre-existing Sophelio conda env.

**Option A — uv**

```bash
cd fusion-equilibrium-challenge-starter   # or clone this repo
uv sync                                   # core deps → .venv/
uv sync --group pytorch                   # add PyTorch for neural-net baselines
uv run python example_usage.py
uv run python experiments.py --quick
```

**Option B — conda / mamba**

```bash
mamba env create -f environment.yml       # core + PyTorch
mamba activate fusion-equilibrium-starter
python example_usage.py
python experiments.py --quick
```

**Option C — plain pip**

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install -r requirements-pytorch.txt   # optional, for neural-net baselines
python example_usage.py
```

Sklearn baselines in `experiments.py` work with the core install only. PyTorch models need the `pytorch` group / `requirements-pytorch.txt`.

### Run the baselines

```bash
# Inspect the Hugging Face dataset
python example_usage.py

# Train baseline models (loads diii_d_train from the Hub)
python experiments.py --quick
python experiments.py --n-shots 50 --epochs 50

# Or train from a fully downloaded copy of the dataset (offline; no Hub streaming).
# Expects the Hub layout: <local-data-dir>/data/<config>/*.parquet
python experiments.py --source local --n-shots 50
python experiments.py --source local --local-data-dir /path/to/hf_dataset
```

The baselines predict the flux map **and** the EFIT scalar targets: after the flux-map
models, a Ridge baseline reports per-scalar R² for `efit_beta_n`, `efit_li`, `efit_q95`,
`efit_r_axis`, `efit_z_axis`, and `magnetics_dsep` (see `results/scalar_r2.png`).

PyTorch baselines auto-detect the best device (CUDA → MPS → CPU); override with
`--device cuda|mps|cpu`.

See `MODELING_GUIDE.md` for the ML walkthrough.

### Your own experiments → `my_experiments/`

Keep your custom models, scratch scripts, cached shots, and result images in a
`my_experiments/` folder at the repo root. It's listed in `.gitignore`, so your work
stays local and never collides with the starter kit when you `git pull` updates. The starter
modules are importable from there:

```python
import sys; from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from experiments import load_shot_from_hf_row, interpolate_magnetics_to_efit  # etc.
```

---



## Welcome to the Machine

You are about to work with data from two large nuclear fusion research devices:

- **DIII-D** - General Atomics tokamak (San Diego, USA)
- **MAST** - Mega Ampere Spherical Tokamak (Culham, UK)

Your goal is to solve a control theory problem that is critical for the future of clean energy: **Predicting the shape of the plasma.**

### The "Jelly Donut" Analogy (Fusion 101)

Imagine you have a donut made of super-hot, invisible jelly (the plasma). This jelly is 100 million degrees, so you can't touch it. Instead, you hold it in place using powerful, invisible magnetic fields (the "magnetic bottle").

Usually, we use magnetic sensors to "feel" where the jelly is. **But in this challenge, you are blind.** The magnetic sensors are broken or unavailable.

**Your Mission:** You must infer the exact shape of the jelly in the donut using only:

1. **The Knobs:** How much current you are sending to the electromagnets.
2. **The Thermometer:** Lasers that measure how hot and dense the jelly is at specific points.



### What "blind" really means — and why it matters

Conventional EFIT reconstructs the 2D equilibrium from a dedicated suite of **magnetic
diagnostics**: an array of magnetic field probes and flux loops mounted around the vessel
that "feel" the plasma's own field. **This challenge withholds that diagnostic array.** Your
inputs are only:

- **Actuators** — the *commanded* coil currents (`magnetics_F`*, `ECOILA`/`bcoil`, MAST P-coils,
`Solenoid`, `TF`, …). These are knobs you *drive*, not measurements of the plasma's field.
- **Kinetic profiles** — Thomson scattering electron temperature & density (`thomson_`*).

The motivation is concrete and physical: if a model can reconstruct the equilibrium **without a
magnetic-diagnostic suite**, a tokamak could be built and operated more cheaply without that
instrumentation — or keep reconstructing when those sensors degrade or fail. This is the *proper
zero-shot* goal: equilibrium reconstruction from actuators + kinetic profiles alone. The
**MAST track** pushes it one step further — can the learned physics reconstruct a machine the
model has **never seen** (different size, shape, and coil set)?

> ⚠️ **`magnetics_dsep` is a target quantity, not an input.** It is **EFIT-derived** — computed
> *from the target equilibrium itself* — so it encodes x-point/divertor geometry straight from the
> label. It is **withheld on the test splits** (alongside `efit_psirz` and the scalar labels), and
> under metric v2 you don't submit it either: the scorer **derives `dsep` from your predicted flux
> map** and scores its agreement with the same derivation on the true flux. Never read `dsep` as a
> model input — make your ψ imply it. (`magnetics_plasma_current` (Ip) *is* an allowed input: a
> single legitimate global magnetic scalar, not label-derived.)

The data isn't *purely* zero-shot — Ip is a real global measurement you may use — but that's fine:
the setup is a strong starting point for the two things that matter most here: **cross-machine
robustness** (models that learn physics, not one machine's wiring) and **synthetic diagnostics**
(deriving machine-agnostic, physics-meaningful inputs from actuators + kinetic profiles). Treat
those as the real targets.

---



## 📁 Data Organization

Record IDs clearly indicate the data source:

- `DIII-D_182494` - DIII-D shot 182494
- `MAST_25607` - MAST shot 25607

Signal names are prefixed with the source:

- `DIII-D: F1A` - DIII-D F1A coil current
- `MAST: P2L` - MAST P2L coil current

**Training & evaluation data** live on Hugging Face (`diii_d_train`, `diii_d_public_test`, `mast_public_test`). The `parquet_data/` folder here holds six **demo shots only** (3 DIII-D + 3 MAST) for local inspection and the dFL visualizer.

---



## 🎯 The Target: What you are predicting

In physics terms, you are predicting the **Magnetic Equilibrium** — the 2D poloidal flux map
$\psi(R,Z)$ plus the **two scalars a flux map cannot contain** ($q_{95}$ and $\beta_N$). The
plasma boundary (**LCFS**), the magnetic axis, the shape scalars, $l_i$, and the x-point gap
`dsep` are all scored too, but you don't submit any of them — the scorer derives each one from
your flux map, the same way it derives them from the true flux map. See **Output & Submission
Format** for exactly what to submit and how it's scored.

### `efit/` (The Ground Truth)

This data comes from a reconstruction code called "EFIT" (equilibrium fitting). 

**Primary target — the flux map:**

| Key | Shape | Description |
|-----|-------|-------------|
| `efit_psirz` | (T, 65, 65) (both machines) | Poloidal flux map - a 2D image at each timestep. Think of it like a topographical map where contour lines show the magnetic cage shape. *Withheld on test.* |
| `efit_times` | (T,) | Timestamps (ms) for the target images. Align all inputs to these times. |
| `efit_grid_R` / `efit_grid_Z` | (65,) | Physical R/Z (m) labelling the flux-map columns/rows (both machines). Kept on every split. |

**EFIT scalar labels** (one value per `efit_times` step; present in `train`, withheld on test).
Under metric v2 you *submit* only `q95` and `betaN`; the rest are training supervision — at
scoring time the axis, shape, `li`, and `dsep` are derived from your submitted flux map:

| Key | Shape | Description |
|-----|-------|-------------|
| `efit_q95` | (T,) | Safety factor at the 95% flux surface — **submitted scalar** |
| `efit_beta_n` | (T,) | Normalised beta β_N — **submitted scalar** |
| `efit_li` | (T,) | Internal inductance ℓi (supervision; derived from your ψ at scoring) |
| `efit_r_axis` / `efit_z_axis` | (T,) | Magnetic-axis R/Z (m) (supervision; derived from your ψ at scoring) |
| `magnetics_dsep` | (T,) | X-point gap (m); `>0` diverted, `<0` limited. EFIT-derived, despite the `magnetics_` prefix (supervision; derived from your ψ at scoring). Undefined on limited/startup frames (NaN or `-1.0` sentinel); mask those before training. |
| *(bonus, train only)* `efit_lcfs_n`, `efit_lcfs_r`, `efit_lcfs_z` | (T,) / (T, N) | Last-closed-flux-surface boundary contour + valid-point count. Provided as context. |

Reconstructions often include electrical currents present in major conductors such as the vacuum vessel, which for simplicity are omitted here. 

---



## 📤 Output & Submission Format

You submit **two things** for every shot, at each provided `efit_times` timestamp:

1. **Flux map** `efit_psirz` — the 2D poloidal flux ψ(R,Z) in the machine's native grid.
2. **Two scalars** — `q95` and `betaN`: the edge safety factor and normalized beta, the only two
   scalars that are not functions of ψ(R,Z) (they need `F(ψ)` / `p(ψ)`, which a flux map does not
   contain).

You do **not** submit anything else. The LCFS boundary, magnetic axis, elongation, triangularity,
volume, `li`, and `dsep` are all **derived from your submitted flux map by the scorer**, with the
same published functionals it applies to the ground-truth flux — so getting ψ right is what earns
every geometry term, and there is no separate scalar head to tune (or to game).

**Per-shot keys** (variable `T` = number of `efit_times`; grouped per shot in one `.npz` per
config). Each scalar is its own named key — no positional column order to get wrong:


| Key suffix        | Shape             | Notes                                                          |
| ----------------- | ----------------- | -------------------------------------------------------------- |
| `_psirz`          | `(T, H, W)` float | Both machines dense `H,W = 65,65` (DIII-D `65,65`; MAST `65,65`) |
| `_q95` `_betaN`   | `(T,)` float each | the two submitted scalars, one per key                          |


So a DIII-D submission `.npz` holds `shot_0000_psirz`, `shot_0000_q95`, `shot_0000_betaN`,
`shot_0001_psirz`, … in test-stream order. The skeleton writes a `manifest.json` alongside (with
the optional harmonization-layer label — descriptive metadata; it is not scored and gates
nothing).

- **Align to** `efit_times` — one prediction per target timestamp. Resample your *inputs* to these
times; never resample/interpolate the target grid itself.
- **Preserve shot order** — emit predictions in the same order the test split streams rows.

**Flux map is dense on both machines.** The corrected MAST `_psirz` is a dense 65×65 grid — the
upstream EFIT stored it on a doubled 65×129 R grid (65 real columns interleaved with 64 empty
ones), which the dataset collapses to the 65 real columns. So, like DIII-D, MAST has **no
central-column NaN region** to skip. The scorer's R²_ψ is still computed only over finite
ground-truth pixels, so any occasional non-finite frame is handled for you.

### How you're scored

The leaderboard score is the **composite intra-machine score** (Award #1):

```
S_model = 0.55 · R²_ψ  +  0.15 · R²_{q95,βN}  +  0.10 · (1 − D_LCFS)  +  0.20 · Consistency
```

| Term           | What it measures                                                                              |
| -------------- | --------------------------------------------------------------------------------------------- |
| `R²_ψ`         | Global R² of the flux map over all (R,Z) points × timesteps × shots. Clipped to ≥ 0.          |
| `R²_{q95,βN}`  | Mean pooled R² of the two submitted scalars vs the stored EFIT values. Clipped to ≥ 0.        |
| `D_LCFS`       | Symmetric Hausdorff distance between the LCFS contours the scorer extracts from your ψ and the true ψ, normalized by mean true LCFS R. Clipped to ≤ 1. |
| `Consistency`  | Mean agreement of the eight ψ-derived scalars — `R_axis, Z_axis, κ, δ_top, δ_bot, V, li, dsep` — computed from your ψ vs the same derivation on the true ψ (pooled R² per scalar, clipped to ≥ 0, averaged). |

Each consistency scalar is scored only on frames where its ground-truth derivation is well-posed:
the shape scalars and `dsep` on **diverted** frames only (a limited plasma's boundary is limiter
contact, which no flux-only functional can see), `dsep` additionally where a clean x-point pair
exists. A perfect flux map scores `D_LCFS = 0` and `Consistency = 1` **by construction** — the
same code runs on both sides. Per-scalar breakdowns appear in your submission's detailed results.

**Cross-machine (Award #2):** `G_ratio = S_model(MAST) / S_model(DIII-D)`, among entries with
`R²_ψ > 0.6` on DIII-D. DIII-D and MAST are scored separately. The scorer runs **on Codabench
against held-out ground truth** — it is not part of this starter kit. See `MODELING_GUIDE.md →
Evaluation Metrics`.

**Validate your file before uploading (`validate_submission.py`).** This checks the *structure* of
your submission (no ground truth, no score) so a malformed `.npz` doesn't burn a submission slot:

```bash
python validate_submission.py submission/diii_d_public_test.npz --config diii_d_public_test
python validate_submission.py submission/mast_public_test.npz  --config mast_public_test
```

It confirms the per-shot keys (`_psirz`, `_q95`, `_betaN`), shot order/count, per-shot `T`,
native grid, and dtypes against the streamed public-test inputs — the errors that otherwise only
surface after you submit.

## 📮 How to Submit

Submit on Codabench: **https://www.codabench.org/competitions/17456/** (register first — the
"Register" button on the competition page).

1. **Generate** predictions in the proposed format: `python submission_skeleton.py --max-shots 0`
   (after swapping in your model). Produces `submission/diii_d_public_test.npz` and
   `submission/mast_public_test.npz`.
2. **Validate** each file with `validate_submission.py` (above) — a malformed `.npz` is the most
   common cause of a failed submission.
3. **Zip** the two `.npz` files plus a `manifest.json` (see `submission_skeleton.py`) at the root of
   the archive, and **upload the zip** on the competition's **Submit** tab. The platform scores your
   predictions against the held-out ground truth and updates the leaderboard. You enter both
   challenges with one submission — **Challenge 1 (DIII-D)** needs only the DIII-D file; **Challenge 2
   (cross-machine `G_ratio`)** additionally needs the MAST file.
4. **Development phase deadline:** **October 18, 2026**. **Submission limits:** **5 per day, 100
   total** per participant during Development (the blind Final phase, Oct 19–26, allows 3 total).

By submitting you agree to the official competition rules and the dataset terms (see the dataset
card on Hugging Face and the Disclaimer below). Starter-kit code is MIT-licensed (`LICENSE`).

---



## 🔌 DIII-D: The Actuators (Magnets)

DIII-D uses a set of shaping coils (F-coils) and main field coils to control the plasma.

### Shaping Coils (F-coils)

These 18 coils act like invisible hands that mold the plasma:


| Signal                              | Description         | Range     |
| ----------------------------------- | ------------------- | --------- |
| `DIII-D: F1A` through `DIII-D: F9A` | Upper shaping coils | ±10,000 A |
| `DIII-D: F1B` through `DIII-D: F9B` | Lower shaping coils | ±10,000 A |




### Main Coils


| Signal           | Description                                                         |
| ---------------- | ------------------------------------------------------------------- |
| `DIII-D: ECOILA` | Ohmic heating coil - central solenoid that drives plasma current    |
| `DIII-D: bcoil`  | Toroidal field coil - main stability field going "around the track" |


---



### Additional Quantities

| Signal | Description |
|--------|-------------|
| `DIII-D: ip` | Integrated electrical current carried in the plasma bulk (an **input**) |
| `DIII-D: dsep` | X-point gap: `>0` diverted (magnetic null "xpoint"), `<0` limited. **EFIT-derived, so a prediction target** — present in `train`, withheld on test. |

---



## 🔌 MAST: The Actuators (Magnets)

MAST is a spherical tokamak with a different coil configuration than conventional tokamaks.

### Poloidal Field Coils (P-coils)

MAST has 10 poloidal field coils (P2-P6, Lower and Upper):


| Signal                          | Description                |
| ------------------------------- | -------------------------- |
| `MAST: P2L` through `MAST: P6L` | Lower poloidal field coils |
| `MAST: P2U` through `MAST: P6U` | Upper poloidal field coils |


**Note:** MAST has no P1, P7, P8, or P9 coils (different machine geometry).

### Main Coils


| Signal           | Description                             |
| ---------------- | --------------------------------------- |
| `MAST: Solenoid` | Central solenoid (equivalent to ECOILA) |
| `MAST: TF`       | Toroidal field coil                     |
| `MAST: Ip`       | Plasma current measurement              |
| `MAST: EFPS`     | Error field protection system coil      |


---



## 🌡️ Thomson Scattering (Both Machines)

Both DIII-D and MAST use **Thomson Scattering** - lasers that bounce off electrons to measure:

1. **Temperature ($T_e$):** How hot are the electrons? (eV)
2. **Density ($n_e$):** How crowded are the electrons? (m⁻³)

Each system ships **one** spatial coordinate array (not an R/Z pair); which axis it
represents differs by machine, as noted below.

### Vertical "Core" System (Looking Down) — `thomson_core_*`

Vertical view, named core for purely historical reasons.


| Parquet column       | Description                                                                                                                                                      |
| -------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `thomson_core_Te`    | Electron temperature (eV), one profile per timestep                                                                                                              |
| `thomson_core_ne`    | Electron density (m⁻³)                                                                                                                                           |
| `thomson_core_R`     | Channel radial position(s) (m). **DIII-D:** constant ≈ 1.94 (vertical chord — channels vary in Z, which is not provided). **MAST:** per-channel R (≈ 0.25–1.5 m) |
| `thomson_core_times` | Timestamps (ms)                                                                                                                                                  |




### Horizontal "Tangential" / Edge System (Looking Sideways) — `thomson_edge_*`

Horizontal view, named tangential for purely historical reasons.


| Parquet column         | Description                                                                               |
| ---------------------- | ----------------------------------------------------------------------------------------- |
| `thomson_edge_Te`      | Electron temperature (eV)                                                                 |
| `thomson_edge_ne`      | Electron density (m⁻³)                                                                    |
| `thomson_edge_spatial` | Channel positions (m). **DIII-D:** Z (≈ −0.05 m near midplane). **MAST:** R (≈ 1.3–1.5 m) |
| `thomson_edge_times`   | Timestamps (ms)                                                                           |


There is no `tan_z`/`core_z` column: each machine provides a single spatial axis per
system (DIII-D edge = Z, MAST edge = R; core radius in `thomson_core_R`).

---



## ⚡ Complete Signal Dictionary

Each Parquet file holds **one shot per row**. All time-series, profiles, and
flux maps are stored as nested arrays within that row. Every time array is
in **milliseconds** (both machines).

Diagnostic groups:

- `efit/*` — magnetic equilibrium reconstruction (the *target*).
- `magnetics/*` — coil currents (the *actuators*) and one EFIT-derived
scalar (`dsep`, the x-point gap; see "Equilibrium-Derived Quantities").
- `thomson/*` — electron temperature & density profiles (the *sensors*).



### DIII-D columns

| Display name | Parquet column | Shape | Notes |
|---|---|---|---|
| — | `source` | scalar string | `"DIII-D"` |
| **EFIT (targets)** | | | |
| EFIT times | `efit_times` | `(T,)` float64 | ms; T ≈ 50–445 across the dataset (median ~260) |
| EFIT psirz | `efit_psirz` | `(T,)` of `(65, 65)` | Poloidal flux maps (V·s/rad). *Primary target; withheld on test.* |
| EFIT grid R/Z | `efit_grid_R`, `efit_grid_Z` | `(65,)` float64 | Physical R/Z (m) of the flux grid. Kept on every split. |
| EFIT scalars | `efit_beta_n`, `efit_li`, `efit_q95`, `efit_r_axis`, `efit_z_axis` | `(T,)` float64 each | Scored scalar targets (β_N, ℓi, q95, axis R/Z). *Withheld on test.* |
| EFIT boundary | `efit_lcfs_n`, `efit_lcfs_r`, `efit_lcfs_z` | `(T,)` / `(T, N)` | LCFS contour + valid-point count. Bonus context in `train`. *Withheld on test.* |
| **Magnetics time bases** | | | |
| — | `magnetics_time` | `(49152,)` float32 | ms; shared by every magnetics signal at 49 kHz (ECOILA, bcoil, all F-coils) |
| — | `magnetics_plasma_current_times` | `(30719,)` float32 | ms; Ip is on its own ADC at a different sample rate |
| — | `magnetics_dsep_times` | `(T,)` float32 | ms; identical to `efit_times` since dsep is EFIT-derived |
| **Main coils** | | | |
| `DIII-D: ECOILA` | `magnetics_ECOILA` | `(49152,)` float64 | Ohmic / central solenoid (A). Uses `magnetics_time`. |
| `DIII-D: bcoil` | `magnetics_bcoil` | `(49152,)` float64 | Toroidal field (A). Uses `magnetics_time`. |
| `DIII-D: Ip` | `magnetics_plasma_current` | `(30719,)` float32 | Plasma current (A). Uses `magnetics_plasma_current_times`. |
| **Shaping coils (18)** | | | |
| `DIII-D: F1A`–`F9B` | `magnetics_F{1-9}{A,B}` | `(49152,)` float64 each | Upper (A) / lower (B) shaping coils, ±10 kA. All use `magnetics_time`. |
| **EFIT-derived (target)** | | | |
| `DIII-D: dsep` | `magnetics_dsep` | `(T,)` float32 | X-point gap (m). `>0` diverted, `<0` limited. **Prediction target** (EFIT-derived), withheld on test. Uses `magnetics_dsep_times`. |
| **Thomson core** (vertical chord, ~R = 1.94 m, looks down) | | | |
| — | `thomson_core_times` | `(~1300–1900,)` float64 | ms |
| — | `thomson_core_Te` | `(~T_c,)` of `(44,)` | Electron temperature (eV) per profile |
| — | `thomson_core_ne` | `(~T_c,)` of `(44,)` | Electron density (m⁻³) per profile |
| — | `thomson_core_R` | `(44,)` float64 | Radial positions (m) — constant ≈ 1.94 since this is a vertical chord |
| **Thomson edge** (horizontal tangential view, ~Z ≈ −0.05 m) | | | |
| — | `thomson_edge_times` | `(~200–500,)` float64 | ms |
| — | `thomson_edge_Te` | `(~T_e,)` of `(10,)` | Electron temperature (eV) |
| — | `thomson_edge_ne` | `(~T_e,)` of `(10,)` | Electron density (m⁻³) |
| — | `thomson_edge_spatial` | `(10,)` float64 | Z positions (m) of the 10 tangential channels |

### MAST columns

MAST is **zero-shot: a test split only**, so its EFIT targets (`efit_psirz` and the
scalars) are not distributed. The three MAST demo shots in `parquet_data/` do include a
`efit_psirz` (clean 65×65) purely for dFL visualization.

| Display name | Parquet column | Shape | Notes |
|---|---|---|---|
| — | `source` | scalar string | `"MAST"` |
| **EFIT (targets — withheld on the MAST test split)** | | | |
| EFIT times | `efit_times` | `(T,)` float64 | ms; T ≈ 50–139 across the dataset (median ~82) |
| EFIT psirz | `efit_psirz` | `(T,)` of `(65, 65)` | Clean 65×65 flux map (no NaNs). Upstream MAST stores psirz on a doubled 129-column R grid — 65 real columns interleaved with 64 empty ones — which we drop to recover the dense grid. |
| EFIT grid R | `efit_grid_R` | `(65,)` float64 | Physical R (m) for the flux grid (≈ 0.06–2.0 m) |
| EFIT grid Z | `efit_grid_Z` | `(65,)` float64 | Physical Z (m) for the flux grid (≈ −2.0–2.0 m) |
| **Magnetics (shared time base)** | | | |
| — | `magnetics_time` | `(15482,)` float64 | ms; shared by every MAST magnetics signal |
| `MAST: Ip` | `magnetics_plasma_current` | `(15482,)` float64 | Plasma current (A) |
| `MAST: TF` | `magnetics_tf_current` | `(15482,)` float64 | Toroidal field coil (A) |
| `MAST: Solenoid` | `magnetics_sol_current` | `(15482,)` float64 | Central solenoid (A) |
| `MAST: EFPS` | `magnetics_efps_current` | `(15482,)` float64 | Error field protection system (A) |
| **Poloidal field coils (10)** | | | |
| `MAST: P{2-6}{L,U}` | `magnetics_p{2-6}{l,u}_current` | `(15482,)` float64 each | P2–P6 lower/upper, no P1/P7/P8/P9 |
| **EFIT-derived** | | | |
| `MAST: dsep` | `magnetics_dsep` (+ `_times`) | `(T,)` float32 | X-point gap (m), derived from upstream `esm/dr_sep_out`. Same definition and sign convention as DIII-D `dsep`. Aligned to `efit_times`. |
| **Thomson core** | | | |
| — | `thomson_core_times` | `(~50–112,)` float64 | ms |
| — | `thomson_core_Te` | `(~T_c,)` of `(~130,)` | Electron temperature (eV) |
| — | `thomson_core_ne` | `(~T_c,)` of `(~130,)` | Electron density (m⁻³) |
| — | `thomson_core_R` | `(~130,)` float64 | Radial positions (m) of each core channel |
| **Thomson edge** | | | |
| — | `thomson_edge_times` | `(~50–112,)` float64 | ms |
| — | `thomson_edge_Te` | `(~T_e,)` of `(~16,)` | Electron temperature (eV) |
| — | `thomson_edge_ne` | `(~T_e,)` of `(~16,)` | Electron density (m⁻³) |
| — | `thomson_edge_spatial` | `(~16,)` float64 | Spatial positions (m) of edge channels |

### Cross-machine convention notes

- **Time units are ms everywhere**, including MAST (`magnetics_time`, `efit_times`, `magnetics_dsep_times`). MAST upstream stores some signals in seconds; the conversion is applied at parquet build time so participants don't have to think about it.
- **Magnetics time base is shared per machine**: both DIII-D and MAST expose one `magnetics_time` array used by every coil signal at the primary sampling rate. On DIII-D, `magnetics_plasma_current` (Ip) sits on its own ADC at a different rate and therefore has its own `magnetics_plasma_current_times` companion.
- `dsep` **is on the EFIT time base**: `magnetics_dsep_times` is identical to `efit_times` on every shot for both machines. It's grouped under `magnetics_`* only for column-naming consistency; physically it's an EFIT-derived geometric quantity, not a magnetic measurement.
- `magnetics_time` **spans cover the full DAQ window** (pre-shot baseline through post-shot ringdown), so they extend well beyond the plasma's actual lifetime. The plasma window is bounded by `efit_times`.

---



## 🗂️ Repository Layout

```
fusion-equilibrium-challenge-starter/
├── parquet_data/                  # 6 demo shots (3 DIII-D + 3 MAST) for dFL / offline peek
│   ├── d3d_shot_203702.parquet
│   ├── d3d_shot_203703.parquet
│   ├── d3d_shot_203704.parquet
│   ├── mast_shot_28348.parquet
│   ├── mast_shot_28350.parquet
│   └── mast_shot_28351.parquet
├── fusion_data_provider.py        # dFL data provider (reads parquet_data/)
├── MODELING_GUIDE.md              # ML walkthrough
├── example_usage.py               # Load the Hugging Face dataset
├── experiments.py
├── experiments_torch.py           # Baseline models (train from Hugging Face)
├── submission_skeleton.py         # Produce a format-correct submission .npz
├── validate_submission.py         # Shape-check a submission before uploading (no scoring)
├── pyproject.toml                 # uv / pip dependency source of truth
├── environment.yml                # conda / mamba alternative
├── requirements.txt               # core deps (plain pip)
├── requirements-pytorch.txt       # optional PyTorch baselines
├── uv.lock                        # pinned deps (uv)
├── my_experiments/                # YOUR custom work (gitignored — create as needed)
├── LICENSE                        # MIT (starter-kit code; dataset has its own terms)
└── README.md                      # This file
```

Each demo parquet file is **one row per shot** with nested array columns (`efit_psirz`, coil currents, Thomson profiles, etc.). The full challenge dataset uses the same schema on [Hugging Face](https://huggingface.co/datasets/Sophelio/fusion-equilibrium-challenge).

---



## 🔬 Key Differences Between Machines

| Feature | DIII-D | MAST |
|---------|--------|------|
| **Location** | San Diego, USA | Culham, UK |
| **Type** | Conventional tokamak | Spherical tokamak |
| **Flux grid shape** | 65×65 | 65×65 (dense; upstream 65×129 empty columns dropped) |
| **R coordinates** | Normalized 0-1 | Physical: 0.12 - 2.0 m |
| **Z coordinates** | Normalized 0-1 | Physical: -2.0 to 2.0 m |
| **Shaping coils** | 18 (F1A-F9B) | 10 (P2L-P6U) |
| **Thomson data orientation** | (spatial, time) | (time, spatial) |
| **Tangential axis** | Z (vertical) | R (radial) |

---



## 🔍 Understanding the Flux Data



### Geometry Differences

**DIII-D (Conventional Tokamak):**

- Large central solenoid with substantial magnetic core
- Plasma forms a "D" shape around the center
- Flux data covers the full computational domain
- Contours form a pattern concentric to the plasma axis that will appear more "shaped" at the boundary, becoming elliptic then circular close to the axis.

**MAST (Spherical Tokamak):**

- Very narrow central column (the "cored apple" design)
- Plasma wraps tightly around a thin central post
- More compact geometry but with unique measurement challenges
- Contours wrap around the hollow center, forming an asymmetric kidney-bean shape

### MAST's 65×65 grid (and why the raw grid was 65×129)

The corrected dataset ships MAST `efit_psirz` as a clean **65×65** grid with **no NaNs**,
matching DIII-D's dimensions. This is not a physical hole — it's a grid artifact:
MAST's upstream EFIT stores `psirz` on a doubled 129-column R grid, where **65 real R
columns are interleaved with 64 empty ones**. We drop the empty columns to recover the
dense grid the data is actually defined on (MAST R ∈ [0.06, 2.0] m, Z ∈ [−2.0, 2.0] m).

The dFL flux grapher in `fusion_data_provider.py` additionally filters any remaining
all-NaN columns defensively, so it renders both the corrected (65×65) and any legacy
(65×129) shots correctly.

### Flux Pattern Interpretation

The `psirz` flux map is like a topographical map:

- **Contour lines** = magnetic flux surfaces where plasma particles travel
- **Innermost contours** = plasma core (hottest, densest region)
- **Outermost contours** = plasma edge (separatrix boundary)
- **Color gradient** = flux magnitude (V·s/rad)



### Checking out the data in the dFL (Data Fusion Labeler)

The dFL can help you visualize any data (fusion or any other kind of dataset), and label the data for downstream ML/AI tasks.
You can download the dFL here:

Mac (Apple Silicon): [https://github.com/Sophelio/dFL/releases/latest/download/Labeler-mac-arm64.dmg](https://github.com/Sophelio/dFL/releases/latest/download/Labeler-mac-arm64.dmg)  
Windows: [https://github.com/Sophelio/dFL/releases/latest/download/Labeler-windows.exe](https://github.com/Sophelio/dFL/releases/latest/download/Labeler-windows.exe)  
Linux: [https://github.com/Sophelio/dFL/releases/latest/download/Labeler-linux.AppImage](https://github.com/Sophelio/dFL/releases/latest/download/Labeler-linux.AppImage)

Once you open the dFL, select a "custom script" and point it at `fusion_data_provider.py`. It will load the demo shots from `parquet_data/` in this repository.

# Disclaimer

Work supported by the U.S. Department of Energy, Office of Science, Office of Fusion Energy Sciences, using the DIII-D National Fusion Facility, a DOE Office of Science user facility, under Award No. DE-FC02-04ER54698, along with Office of Fusion Energy Sciences Awards No. DE-SC0024426, DE-SC0024499, DE-SC0024409, and DE-SC0024571.

Disclaimer: This report was prepared as an account of work sponsored by an agency of the United States Government. Neither the United States Government nor any agency thereof, nor any of their employees, makes any warranty, express or implied, or assumes any legal liability or responsibility for the accuracy, completeness, or usefulness of any information, apparatus, product, or process disclosed, or represents that its use would not infringe privately owned rights. Reference herein to any specific commercial product, process, or service by trade name, trademark, manufacturer, or otherwise does not necessarily constitute or imply its endorsement, recommendation, or favoring by the United States Government or any agency thereof. The views and opinions of authors expressed herein do not necessarily state or reflect those of the United States Government or any agency thereof.