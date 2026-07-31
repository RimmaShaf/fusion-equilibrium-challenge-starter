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

> **What you predict:** the primary target is the 2-D flux map `efit_psirz`; you
> submit it plus only **`q95` and `betaN`** — the two scalars a flux map cannot contain. The
> training data still ships all EFIT scalar labels (`efit_beta_n`, `efit_li`, `efit_q95`,
> `efit_r_axis`, `efit_z_axis`, `magnetics_dsep`) as supervision, and all are **withheld on the
> test splits** — but at scoring time the axis, shape and `li` are *derived from your
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

The baselines predict the flux map **and** regress the EFIT scalar labels: after the flux-map
models, a Ridge baseline reports per-scalar R² for `efit_beta_n`, `efit_li`, `efit_q95`,
`efit_r_axis`, `efit_z_axis`, and `magnetics_dsep` (see `results/scalar_r2.png`).

> Those six are **supervision, not the scored set.** Only `q95` and `betaN` are submitted and
> scored as values; the axis, `li` and the shape scalars are *derived from your submitted flux map*
> at scoring time; and `magnetics_dsep` is not scored at all. The Ridge numbers are a demo of what
> the inputs support, not a leaderboard preview.

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
**MAST leg — Challenge 2** pushes it one step further: can the learned physics reconstruct a
machine the model has **never seen** (different size, shape, and coil set)?

> ⚠️ **`magnetics_dsep` is not an input.** It is **EFIT-derived** — computed *from the target
> equilibrium itself* — so it encodes x-point/divertor geometry straight from the label, and is
> **withheld on the test splits** alongside `efit_psirz` and the scalar labels. It is not a scored
> target either (it was dropped from the metric: DIII-D's `dsep` is a separatrix↔limiter clearance
> while MAST's is divertor balance — different physical quantities that one functional cannot score
> coherently). Never read `dsep` as a model input. (`magnetics_plasma_current` (Ip) *is* an allowed
> input: a single legitimate global magnetic scalar, not label-derived.)

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
plasma boundary (**LCFS**), the magnetic axis, the shape scalars and $l_i$ are all scored too,
but you don't submit any of them — the scorer derives each one from your flux map, the same way
it derives them from the true flux map. See **Output & Submission
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
You *submit* only `q95` and `betaN`; the rest are training supervision — at
scoring time the axis, shape and `li` are derived from your submitted flux map:

| Key | Shape | Description |
|-----|-------|-------------|
| `efit_q95` | (T,) | Safety factor at the 95% flux surface — **submitted scalar** |
| `efit_beta_n` | (T,) | Normalised beta β_N — **submitted scalar** |
| `efit_li` | (T,) | Internal inductance ℓi (supervision; derived from your ψ at scoring) |
| `efit_r_axis` / `efit_z_axis` | (T,) | Magnetic-axis R/Z (m) (supervision; derived from your ψ at scoring) |
| `magnetics_dsep` | (T,) | **Context only — not scored.** EFIT-derived despite the `magnetics_` prefix. On DIII-D it is the a-file `DSEP` separatrix↔limiter clearance, `>0` on every shipped frame (its sign defined the diverted filter) and free of NaN. On MAST it is a *different* quantity — divertor balance δR_sep, which straddles zero on ordinary diverted plasmas, so **its sign is not a limited/diverted flag**. |
| *(bonus, train only)* `efit_lcfs_n`, `efit_lcfs_r`, `efit_lcfs_z` | (T,) / (T, N) | Last-closed-flux-surface boundary contour + valid-point count. Provided as context. |

> **Where you will actually see these.** Every column in this table is EFIT-derived, so all of them
> are withheld on `diii_d_public_test` and `mast_public_test`. Since there is no `mast_train`
> either, **MAST's labels appear in no released config at all** — the only MAST shots that carry
> `efit_psirz`, the scalars or `magnetics_dsep` are the three demo files in `parquet_data/`. Build
> your pipeline so it does not expect them on MAST.

Reconstructions often include electrical currents present in major conductors such as the vacuum vessel, which for simplicity are omitted here. 

---



## 📤 Output & Submission Format

You submit **two things** for every shot, at each provided `efit_times` timestamp:

1. **Flux map** `efit_psirz` — the 2D poloidal flux ψ(R,Z) in the machine's native grid.
2. **Two scalars** — `q95` and `betaN`: the edge safety factor and normalized beta, the only two
   scalars that are not functions of ψ(R,Z) (they need `F(ψ)` / `p(ψ)`, which a flux map does not
   contain).

You do **not** submit anything else. The LCFS boundary, magnetic axis, elongation, triangularity,
volume and `li` are all **derived from your submitted flux map by the scorer**, with the
same published functionals it applies to the ground-truth flux — so getting ψ right is what earns
every geometry term, and there is no separate scalar head to tune (or to game).

**Per-shot keys** (variable `T` = number of `efit_times`; grouped per shot in one `.npz` per
config). Each scalar is its own named key — no positional column order to get wrong:


| Key suffix        | Shape             | Notes                                                          |
| ----------------- | ----------------- | -------------------------------------------------------------- |
| `_psirz`          | `(T, H, W)` float | Both machines dense `H,W = 65,65` (DIII-D `65,65`; MAST `65,65`) |
| `_q95` `_betaN`   | `(T,)` float each | the two submitted scalars, one per key                          |


So a DIII-D submission `.npz` holds `shot_0000_psirz`, `shot_0000_q95`, `shot_0000_betaN`,
`shot_0001_psirz`, … in test-stream order. The skeleton writes a small `manifest.json` alongside;
it is descriptive only — the scorer locates your predictions by **filename**, so the two `.npz`
must keep the exact config names.

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
| `Consistency`  | Mean agreement of the seven ψ-derived scalars — `R_axis, Z_axis, κ, δ_top, δ_bot, V, li` — computed from your ψ vs the same derivation on the true ψ (pooled R² per scalar, clipped to ≥ 0, averaged). |

**No scalar is masked by boundary type any more.** Earlier versions restricted the shape scalars
to diverted frames; the dataset is now diverted-only, so that mask is gone. Measured on the
reference folds, six of the seven scalars are well-posed on **100%** of frames and `li` on 99.99%
(DIII-D) / 99.5% (MAST) — a frame leaves a scalar's average only where the *ground-truth*
derivation itself fails. If your ψ fails to yield a scalar the truth does have, the frame is not
skipped: it is mean-substituted and earns ~0. A perfect flux map scores `D_LCFS = 0` and
`Consistency = 1` **by construction** — the same code runs on both sides. Per-scalar breakdowns
and your derivation-failure rate appear in your submission's detailed results.

**The flux sign is normalized for you.** DIII-D and MAST store ψ with opposite sign conventions
(see *Cross-machine convention notes*), so a model that transfers correctly still lands
sign-inverted. The scorer determines the global sign of your submitted flux map per machine,
scores you under it, and reports which it used (`psi_sign`: `+1` as submitted, `−1` normalized).
It is one bit per machine over the whole fold, and only the sign — not the amplitude — is
normalized. **You do not need to guess the convention.**

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
   `submission/mast_public_test.npz` — about **1.9 GB** together in `float16`.
2. **Validate** each file with `validate_submission.py` (above) — a malformed `.npz` is the most
   common cause of a failed submission.
3. **Submit** by one of the two routes below. Both score identically; you enter both challenges
   with one submission — **Challenge 1 (DIII-D)** needs only the DIII-D file; **Challenge 2
   (cross-machine `G_ratio`)** additionally needs the MAST file.
4. **Development phase deadline:** **October 18, 2026**. **Submission limits:** **5 per day, 100
   total** per participant during Development (the blind Final phase, Oct 19–26, allows 3 total).

### Route A — Hugging Face pointer (recommended)

Push the `.npz` to a **private** Hugging Face *dataset* repo and submit a small `manifest.json`
naming a pinned commit plus a read token scoped to that one repo. Your predictions stay private.
`push_predictions.py` does all of it:

```bash
uv run huggingface-cli login                    # once, with a WRITE token — stays on your machine
uv run python push_predictions.py \
    --repo your-username/fusion-eq-predictions \
    --read-token hf_...                         # scoped READ token, see below
```

That writes `submission_pointer.zip` — upload **that** on the Submit tab. It verifies with the read
token that the scorer will actually be able to see your files, before you spend a submission slot.

Why this is the default advice, measured from the scoring machine: **Codabench's file storage
sustains ~0.5 MB/s and the Hugging Face CDN ~50 MB/s.** A 1.9 GB direct upload spends about an
hour in transfer before scoring starts; the pointer takes under a minute. The scoring worker
runs one job at a time, so that hour is queue time everyone shares.

**Two tokens, different jobs.** The **write** token uploads your files and never leaves your
machine (`huggingface-cli login`). The **fine-grained read** token goes inside `manifest.json` and
is submitted, so the scorer can read your private repo. Create it at
[huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) → *New token* →
**Fine-grained** → select your predictions repo → tick **only** "Read access to contents of
selected repos". `push_predictions.py` refuses a write token or a classic read token, since both
grant more than the scorer needs. Revoke it when the competition ends.

**Submitting predictions you did not produce is plagiarism and disqualifies the whole team.** The
organizers re-score leading entries from source before prizes; the pinned commit SHA exists so
that what was scored cannot be changed afterwards.

### Route B — direct upload

Zip the two `.npz` at the **root** of the archive and upload it on the **Submit** tab:

```bash
cd submission && zip -0 -r ../submission.zip .   # -0 = stored; the .npz are already compressed
```

Nothing external is involved, but expect ~1 h of transfer per submission, and it counts against
your 15 GB Codabench storage quota (~7 full submissions — delete superseded ones from the
Resources tab).

By submitting you agree to the official competition rules and the dataset terms (see the dataset
card on Hugging Face and the Disclaimer below). Starter-kit code is MIT-licensed (`LICENSE`).

---



## 🔌 DIII-D: The Actuators (Magnets)

DIII-D uses a set of shaping coils (F-coils) and main field coils to control the plasma.

### Shaping Coils (F-coils)

These 18 coils act like invisible hands that mold the plasma:


| Signal                              | Description         | Range     |
| ----------------------------------- | ------------------- | --------- |
| `DIII-D: F1A` through `DIII-D: F9A` | Upper shaping coils | ~±600 kA·turn in-window |
| `DIII-D: F1B` through `DIII-D: F9B` | Lower shaping coils | ~±600 kA·turn in-window |




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
| `DIII-D: dsep` | EFIT a-file `DSEP`: minimum separatrix↔limiter clearance (m); `>0` diverted, `<0` limited. **EFIT-derived**, so present in `train` and withheld on test. Context only — not scored. |

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

## 🆕 Machine geometry (both machines, every split)

A coil current means little until you know where the coil is, and a Thomson profile localises
nothing until you know where the chord is. Both now ship on every row — they are inputs, so
nothing is withheld:

| Column | Shape | Description |
|--------|-------|-------------|
| `coil_name` | (C,) str | Coil (or conductor element) identifier |
| `coil_input_column` | (C,) str | **Join key** — the current column this geometry belongs to, e.g. `magnetics_F1A` |
| `coil_R`, `coil_Z` | (C,) | Conductor-rectangle centre (m) |
| `coil_width`, `coil_height` | (C,) | Radial / vertical extent (m) |
| `coil_angle1`, `coil_angle2` | (C,) | Parallelogram skew angles (degrees) |
| `thomson_chord_name` | (N,) str | `TS_core_*`, `TS_tangential_*`, `TS_divertor_*` |
| `thomson_chord_R`, `thomson_chord_Z` | (N,) | Chord position (m) |

**The two machines describe their coils at different granularity — that is not a bug.** DIII-D
ships **C = 19 lumped rectangles** (18 F-coils + `ECOILA`), the representation EFIT's own
`mhdin.dat` uses. MAST ships **C = 812 individual conductor elements** from the FAIR level-2
`pf_active` IDS — one row per conductor turn, the solenoid alone being 656 of them, so a MAST
coil's turn count is how many rows share its `coil_input_column` (P2 = 20, P3 = 8, P4 = 23,
P5 = 23, P6 = 4). **Neither machine ships a `coil_turns` column**: the turn counts are folded
into the current values (see the units note).
Join either back to the currents with `coil_input_column`; several MAST elements share one column
(P2 inner/outer are parallel-fed; all 656 solenoid elements share `magnetics_sol_current`).

**Six DIII-D F-coils are parallelograms.** `coil_angle1` / `coil_angle2` are EFIT's `AF` / `AF2`
shear angles: `F5A`/`F5B` ±45°, `F6A`/`F6B` ±92.4°, `F7A`/`F7B` ±108.06°; everything else 0.0.
They only matter if you integrate over the conductor cross-section rather than treating each coil
as a filament. MAST's are structurally 0.0 — IMAS rectangles carry no skew.

> `0.0` means **no skew (plain axis-aligned rectangle)**, not "sides lie flat". EFIT branches on
> `angle1 == 0 and angle2 == 0` to emit a rectangle, and normalises `90 → 0` on write-out, so 90°
> and 0° denote the same unskewed coil — we ship the canonical `0`. MAST's zeros mean exactly what
> the 13 DIII-D zeros mean, so one code path covers both machines.

Every DIII-D F-coil row (R, Z, width, height, turns) matches EFIT's own `mhdin.dat` machine file
exactly — and those turn counts (58 or 55) are now folded into `magnetics_F*`, which therefore
carries **total ampere-turns per rectangle**, the quantity a Green's-function calculation wants.
`ECOILA` is the exception: EFIT models that group as 48 single-turn elements over the same
envelope and `ECOILB` is a second co-located group not shipped here, so its turn convention is
ambiguous. `magnetics_ECOILA` is in **kA**, not kA·turn — don't apply an ampere-turn multiplier.

*Not covered:* `magnetics_bcoil` (DIII-D TF) and MAST's `magnetics_tf_current` /
`magnetics_efps_current` have no poloidal-plane rectangle — 19 of 21 DIII-D current columns and
11 of 14 MAST ones are covered.

**Chord positions close a real gap.** DIII-D's core Thomson is a *vertical* laser, so
`thomson_core_R` is a constant ≈ 1.94 m and the informative coordinate — Z — was never shipped;
the tangential system is the mirror image; and the **divertor** subsystem had no shipped
counterpart at all. `thomson_chord_R`/`_Z` carry all three subsystems with both coordinates.

> ⚠️ **DIII-D chord positions are per shot and they genuinely vary** — 22 distinct channel-name
> layouts in train (19 in public test) over 6 distinct subsystem
> layouts, channel counts from 59 to 138, because the divertor system was reconfigured between
> campaigns. Do not cache one shot's chord array and reuse it.

MAST's Thomson is a *horizontal midplane* laser: per-channel R (same values as
`thomson_core_R` / `thomson_edge_spatial`) with `thomson_chord_Z = 0`. That zero is not a
placeholder — FAIR level-2 exposes only `thomson_scattering.channel[:].position.r`.

A worked use: build each coil's vacuum Green's function from `coil_R/Z/turns` and you get the
coil-driven part of ψ analytically rather than learning it. Fitting ψ outside the plasma envelope
to those Green's functions, with the shipped currents and turn counts, reaches R² ≈ 0.94.

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
| EFIT times | `efit_times` | `(T,)` float64 | ms; T ranges **1–445** across the dataset (median 241); 42 shots have T < 10 and 129 have T < 50 — do not assume a long record |
| EFIT psirz | `efit_psirz` | `(T,)` of `(65, 65)` | Poloidal flux maps (V·s/rad). *Primary target; withheld on test.* |
| EFIT grid R/Z | `efit_grid_R`, `efit_grid_Z` | `(65,)` float64 | Physical R/Z (m) of the flux grid. Kept on every split. |
| EFIT scalars | `efit_beta_n`, `efit_li`, `efit_q95`, `efit_r_axis`, `efit_z_axis` | `(T,)` float64 each | Scored scalar targets (β_N, ℓi, q95, axis R/Z). *Withheld on test.* |
| EFIT boundary | `efit_lcfs_n`, `efit_lcfs_r`, `efit_lcfs_z` | `(T,)` / `(T, N)` | LCFS contour + valid-point count. Bonus context in `train`. *Withheld on test.* |
| **Magnetics time bases** | | | |
| — | `magnetics_time` | `(M,)` float32 | ms; shared by every DIII-D magnetics signal. **M varies by shot**: 70.0% of train shots (67.0% of public test) are `(480256,)` at ~20 kHz, the rest mostly `(49152,)` or `(50176,)` at 2 kHz. Six distinct lengths occur in all. Both rates span the full ~24 s record — do not hard-code the length. |
| — | `magnetics_plasma_current_times` | `(30719,)` float32 | ms; Ip is on its own ADC at a different sample rate |
| — | `magnetics_dsep_times` | `(T,)` float32 | ms; identical to `efit_times` since dsep is EFIT-derived |
| **Main coils** | | | |
| `DIII-D: ECOILA` | `magnetics_ECOILA` | `(M,)` float64 | Ohmic / central solenoid — **kA** (not kA·turn; turn convention unresolved). Uses `magnetics_time`. |
| `DIII-D: bcoil` | `magnetics_bcoil` | `(M,)` float64 | Toroidal field — **kA** (toroidal coil: no PF turn count). Uses `magnetics_time`. |
| `DIII-D: Ip` | `magnetics_plasma_current` | `(30719,)` float32 | Plasma current — **kA** (matches MAST). Uses `magnetics_plasma_current_times`. |
| **Shaping coils (18)** | | | |
| `DIII-D: F1A`–`F9B` | `magnetics_F{1-9}{A,B}` | `(M,)` float64 each | Upper (A) / lower (B) shaping coils — **kA·turn** (58 or 55 turns folded in). All use `magnetics_time`. |
| **EFIT-derived (target)** | | | |
| `DIII-D: dsep` | `magnetics_dsep` | `(T,)` float32 | EFIT a-file `DSEP`: separatrix↔limiter clearance (m); `>0` diverted, `<0` limited. EFIT-derived, withheld on test, not scored. Uses `magnetics_dsep_times`. |
| **Thomson core** (vertical chord, ~R = 1.94 m, looks down) | | | |
| — | `thomson_core_times` | `(~1300–1900,)` float64 | ms |
| — | `thomson_core_Te` | `(~T_c,)` of `(C_c,)` | Electron temperature (eV) per profile; `C_c` varies by shot |
| — | `thomson_core_ne` | `(~T_c,)` of `(C_c,)` | Electron density (m⁻³) per profile; `C_c` varies by shot |
| — | `thomson_core_R` | `(C_c,)` float64 | **varies by shot: 40, 42, 43, 44 or 54 channels** (44 on 6,327 of 7,041 train shots). Radial positions (m) — constant ≈ 1.94 since this is a vertical chord |
| **Thomson edge** (horizontal tangential view, ~Z ≈ −0.05 m) | | | |
| — | `thomson_edge_times` | `(~200–500,)` float64 | ms |
| — | `thomson_edge_Te` | `(~T_e,)` of `(C_e,)` | Electron temperature (eV); `C_e` varies by shot |
| — | `thomson_edge_ne` | `(~T_e,)` of `(C_e,)` | Electron density (m⁻³); `C_e` varies by shot |
| — | `thomson_edge_spatial` | `(C_e,)` float64 | **varies by shot: 10 channels (6,385 shots) or 14 (656)**. Z positions (m) of the tangential channels |

### MAST columns

MAST is **zero-shot: a test split only**, so its EFIT targets (`efit_psirz` and the
scalars) are not distributed. The three MAST demo shots in `parquet_data/` do include a
`efit_psirz` (clean 65×65) purely for dFL visualization.

| Display name | Parquet column | Shape | Notes |
|---|---|---|---|
| — | `source` | scalar string | `"MAST"` |
| **EFIT (targets — withheld on the MAST test split)** | | | |
| EFIT times | `efit_times` | `(T,)` float64 | ms; T ranges **5–98** across the dataset (median 58); 397 of 1,206 shots have T < 50 |
| EFIT psirz | `efit_psirz` | `(T,)` of `(65, 65)` | Clean 65×65 flux map (no NaNs). Upstream MAST stores psirz on a doubled 129-column R grid — 65 real columns interleaved with 64 empty ones — which we drop to recover the dense grid. |
| EFIT grid R | `efit_grid_R` | `(65,)` float64 | Physical R (m) for the flux grid (≈ 0.06–2.0 m) |
| EFIT grid Z | `efit_grid_Z` | `(65,)` float64 | Physical Z (m) for the flux grid (≈ −2.0–2.0 m) |
| **Magnetics (shared time base)** | | | |
| — | `magnetics_time` | `(30000,)` or `(15482,)` float64 | ms; shared by every MAST magnetics signal. **Two populations** — see note below |
| `MAST: Ip` | `magnetics_plasma_current` | same as `magnetics_time` | Plasma current — **kA** |
| `MAST: TF` | `magnetics_tf_current` | same as `magnetics_time` | Toroidal field coil feed — **kA** |
| `MAST: Solenoid` | `magnetics_sol_current` | same as `magnetics_time` | Central solenoid feed — **kA** |
| `MAST: EFPS` | `magnetics_efps_current` | same as `magnetics_time` | Error field protection system — **kA** |
| **Poloidal field coils (10)** | | | |
| `MAST: P{2-6}{L,U}` | `magnetics_p{2-6}{l,u}_current` | same as `magnetics_time` | P2–P6 lower/upper (no P1/P7/P8/P9) — **kA·turn**, not kA. See the units note |

> **⚠️ MAST magnetics come in two time-base populations, and one of them has gaps.**
>
> - **1,092 of 1,206 test shots** (shot numbers above ~23,750): `magnetics_time` is `(30000,)`,
>   a uniform 0.2 ms / 5 kHz grid spanning −2,000 → +3,999.8 ms, every column finite.
> - **114 shots** (the early campaign): `magnetics_time` is `(15482,)` spanning −2,500 → +5,499
>   ms and is the **union of two acquisition grids** — the poloidal set (P-coils, Ip, solenoid,
>   EFPS) at 0.2 ms / 5 kHz over −150 → +1,349.8 ms (7,500 samples), and the toroidal field coil
>   at 1.0 ms / 1 kHz over the full record (8,000 samples). Each column is therefore **null on
>   the samples belonging to the other grid**: ~52% of rows for the poloidal set, ~48% for TF.
>
> **No data is missing** — the nulls are holes in a union axis, not dropouts, and both native
> grids fully cover the plasma window on every one of the 114 shots. This is the layout
> FAIR-MAST's own `amc` group ships for those shots.
>
> The gaps are plain `NaN` (no parquet nulls anywhere in this release), so mask per column
> instead of assuming a dense array:
>
> ```python
> t   = np.asarray(row["magnetics_time"], dtype=float)
> tf  = np.asarray(row["magnetics_tf_current"], dtype=float)
> ok  = np.isfinite(tf)
> tf_t, tf_v = t[ok], tf[ok]          # native 1 kHz TF trace, gap-free
> ```
| **EFIT-derived** | | | |
| `MAST: dsep` | `magnetics_dsep` (+ `_times`) | `(T,)` float32 | **δR_sep — divertor *balance*, NOT the same quantity as DIII-D's `dsep`.** From `esm/dr_sep_out`: the radial gap between the upper and lower separatrices at the outboard midplane, so it straddles zero on ordinary diverted plasmas and **its sign is not a limited/diverted flag**. Not scored, and — like every EFIT-derived MAST column — present only in the `parquet_data/` demo shots, never in a released config. |
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

- **⚠️ The two machines store `efit_psirz` with OPPOSITE SIGN CONVENTIONS.** On **DIII-D** the
  magnetic axis is the **minimum** of ψ; on **MAST** it is the **maximum**. Measured on the
  shipped corpus: DIII-D 99.98% of 1,559,340 frames with zero counter-examples, MAST 100%. This
  is a provenance difference between two EFIT implementations (DIII-D's EFIT vs
  EFIT++/FAIR-MAST), **not physics** — both machines run positive plasma current here.
  - A DIII-D-trained model applied zero-shot to MAST predicts a correct equilibrium with the
    wrong overall sign. A naive R² then reports a large *negative* number and looks like total
    failure. **Check `R²(−ψ_pred)` before concluding your transfer failed.**
  - Anything assuming "axis = maximum" (O-point search, contouring, ψ_N normalization) needs the
    per-machine sign or must detect it.
  - **The official scorer is sign-invariant**: it determines the global sign of your submitted
    flux map per machine, scores you under it, and reports which sign it used. You are not being
    tested on guessing a storage convention. Amplitude is *not* normalized.
- **⚠️ Current units differ between the machines — and not by a single factor.** Confirmed
  against FAIR-MAST's own metadata and cross-checked against its IMAS level-2 store (SI, amperes):

  | Columns | Units as shipped | To amperes-per-turn (DIII-D's convention) |
  | :--- | :--- | :--- |
  | DIII-D `magnetics_F*`, `ECOILA`, `bcoil`, `plasma_current` | **A** | already A |
  | MAST `magnetics_plasma_current`, `_tf_current`, `_sol_current`, `_efps_current` | **kA** | `× 1000` |
  | MAST `magnetics_p{2-6}{l,u}_current` | **kA·turn** | `× 1000 / turns` |

  The ten MAST P-coil columns are **ampere-turns** — upstream labels them `kA * turn` — which is
  a different quantity from a coil current. Turn counts come from this dataset's own `coil_*`
  geometry columns (elements per coil): **P2 = 20**, **P3 = 8**, **P4 = 23**, **P5 = 23**,
  **P6 = 4**. A naive "×1000" therefore fixes Ip/TF/solenoid/EFPS but leaves every P-coil wrong
  by 8–23×. Normalizing per machine (recommended) absorbs all of it.
- **Time units are ms everywhere**, including MAST (`magnetics_time`, `efit_times`, `magnetics_dsep_times`). MAST upstream stores some signals in seconds; the conversion is applied at parquet build time so participants don't have to think about it.
- **Magnetics time base is shared per machine**: both DIII-D and MAST expose one `magnetics_time` array used by every coil signal at the primary sampling rate. On DIII-D, `magnetics_plasma_current` (Ip) sits on its own ADC at a different rate and therefore has its own `magnetics_plasma_current_times` companion. On MAST, 114 early-campaign shots use a two-grid union base with per-column nulls — see the MAST magnetics note above.
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