"""
Fusion Equilibrium Challenge - Baseline Model Experiments

End-to-end demo: load DIII-D training shots from the Hugging Face Hub, train models
to predict plasma shape (2D flux maps) from coil currents and Thomson scattering,
evaluate and plot results.

Training data: https://huggingface.co/datasets/Sophelio/fusion-equilibrium-challenge
(config `diii_d_train`). Demo parquet shots in `parquet_data/` are for dFL only.

Usage:
    python experiments.py --quick              # 3 shots, fast demo
    python experiments.py --n-shots 10         # 10 shots, full pipeline
    python experiments.py --include-thomson    # add Thomson scattering features
    python experiments.py --skip-sklearn       # only run PyTorch models
    python experiments.py --skip-pytorch       # only run sklearn models
"""

from __future__ import annotations

import argparse
import time
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, RidgeCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from datasets import load_dataset

try:
    from skimage.metrics import structural_similarity as ssim
    HAS_SSIM = True
except ImportError:
    HAS_SSIM = False

warnings.filterwarnings("ignore", category=UserWarning)

# ---------------------------------------------------------------------------
# 1. Configuration
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parent
HF_REPO_ID = "Sophelio/fusion-equilibrium-challenge"
HF_TRAIN_CONFIG = "diii_d_train"


@dataclass
class ExperimentConfig:
    hf_repo_id: str = HF_REPO_ID
    hf_config: str = HF_TRAIN_CONFIG
    n_shots: int = 10
    n_pca_components: int = 50
    test_size: float = 0.2
    val_size: float = 0.1
    include_thomson: bool = False
    random_state: int = 42
    output_dir: Path = field(default_factory=lambda: REPO_ROOT / "results")
    # PyTorch settings
    batch_size: int = 256
    epochs: int = 50
    learning_rate: float = 1e-3
    device: str = "auto"  # "auto", "mps", "cpu" — used by experiments_torch.py


# ---------------------------------------------------------------------------
# 2. Signal definitions
# ---------------------------------------------------------------------------

EFIT_GRID_SIZE = 65

# 21 DIII-D magnetics signals: each has magnetics_{name} and magnetics_{name}_times
D3D_MAGNETICS_SIGNALS = [
    "ECOILA", "bcoil",
    "F1A", "F1B", "F2A", "F2B", "F3A", "F3B",
    "F4A", "F4B", "F5A", "F5B", "F6A", "F6B",
    "F7A", "F7B", "F8A", "F8B", "F9A", "F9B",
    "plasma_current",
]


# ---------------------------------------------------------------------------
# 3. Data loading (Hugging Face Hub)
# ---------------------------------------------------------------------------

def _as_psirz_stack(psirz_raw) -> np.ndarray:
    """Normalize efit_psirz to (T, 65, 65) whether from HF arrays or nested lists."""
    psirz = np.asarray(psirz_raw, dtype=np.float32)
    if psirz.ndim == 3:
        return psirz
    return np.stack(
        [np.stack([np.asarray(r, dtype=np.float32) for r in grid]) for grid in psirz_raw]
    )


def load_shot_from_hf_row(row: dict) -> dict:
    """Convert one Hugging Face dataset row into the internal shot dict."""
    shot: dict = {"source": row.get("source", "DIII-D")}

    shot["efit_times"] = np.asarray(row["efit_times"], dtype=np.float64)
    shot["efit_psirz"] = _as_psirz_stack(row["efit_psirz"])

    shared_mag_time = np.asarray(row["magnetics_time"], dtype=np.float64)
    ip_times = np.asarray(row["magnetics_plasma_current_times"], dtype=np.float64)

    magnetics = {}
    for sig in D3D_MAGNETICS_SIGNALS:
        data_col = f"magnetics_{sig}"
        if data_col not in row:
            continue
        times = ip_times if sig == "plasma_current" else shared_mag_time
        magnetics[sig] = {
            "values": np.asarray(row[data_col], dtype=np.float32),
            "times": times,
        }
    shot["magnetics"] = magnetics

    if "thomson_core_times" in row:
        try:
            shot["thomson_core"] = {
                "times": np.asarray(row["thomson_core_times"], dtype=np.float64),
                "Te": np.asarray(row["thomson_core_Te"], dtype=np.float32),
                "ne": np.asarray(row["thomson_core_ne"], dtype=np.float32),
            }
        except Exception:
            pass

    if "thomson_edge_times" in row:
        try:
            shot["thomson_edge"] = {
                "times": np.asarray(row["thomson_edge_times"], dtype=np.float64),
                "Te": np.asarray(row["thomson_edge_Te"], dtype=np.float32),
                "ne": np.asarray(row["thomson_edge_ne"], dtype=np.float32),
            }
        except Exception:
            pass

    return shot


def load_shots_from_hf(
    n_shots: int,
    seed: int = 42,
    repo_id: str = HF_REPO_ID,
    config: str = HF_TRAIN_CONFIG,
) -> list[dict]:
    """Load a random sample of DIII-D training shots from the Hugging Face Hub."""
    print(f"  Hub: {repo_id} ({config})")
    ds = load_dataset(repo_id, config, split="train", streaming=True)
    ds = ds.shuffle(seed=seed, buffer_size=10_000)

    shots = []
    for i, row in enumerate(ds):
        if i >= n_shots:
            break
        shot = load_shot_from_hf_row(row)
        shot["shot_index"] = i
        print(f"  Loading shot {i + 1}/{n_shots} ({shot['source']}, T={len(shot['efit_times'])})")
        shots.append(shot)

    if not shots:
        raise RuntimeError(f"No shots loaded from {repo_id}/{config}")
    return shots


# ---------------------------------------------------------------------------
# 4. Interpolation & feature engineering
# ---------------------------------------------------------------------------

def interpolate_magnetics_to_efit(shot: dict) -> np.ndarray:
    """Interpolate all magnetics signals to EFIT times. Returns (T, n_signals)."""
    efit_times = shot["efit_times"]
    features = []
    for sig in D3D_MAGNETICS_SIGNALS:
        if sig not in shot["magnetics"]:
            features.append(np.zeros(len(efit_times), dtype=np.float32))
            continue
        mag = shot["magnetics"][sig]
        f = interp1d(
            mag["times"], mag["values"],
            kind="linear", fill_value="extrapolate", bounds_error=False,
        )
        features.append(f(efit_times).astype(np.float32))
    return np.column_stack(features)  # (T, 21)


def interpolate_thomson_to_efit(shot: dict) -> Optional[np.ndarray]:
    """Interpolate Thomson scattering to EFIT times. Returns (T, n_channels) or None."""
    efit_times = shot["efit_times"]
    all_features = []

    for system_key in ["thomson_core", "thomson_edge"]:
        if system_key not in shot:
            continue
        system = shot[system_key]
        for measure in ["Te", "ne"]:
            if measure not in system:
                continue
            profiles = system[measure]  # (n_thomson_times, n_channels)
            thomson_times = system["times"]
            n_channels = profiles.shape[1] if profiles.ndim == 2 else 1

            for ch in range(n_channels):
                if profiles.ndim == 2:
                    ch_data = profiles[:, ch]
                else:
                    ch_data = profiles
                mask = np.isfinite(ch_data)
                if mask.sum() < 2:
                    all_features.append(np.zeros(len(efit_times), dtype=np.float32))
                    continue
                f = interp1d(
                    thomson_times[mask], ch_data[mask],
                    kind="linear", fill_value=0.0, bounds_error=False,
                )
                all_features.append(f(efit_times).astype(np.float32))

    if not all_features:
        return None
    return np.column_stack(all_features)


def build_feature_matrix(
    shots: list[dict], include_thomson: bool = False
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build X (features) and Y (targets) from all shots.
    Returns X (total_T, n_features), Y (total_T, 65, 65), shot_ids (total_T,).
    """
    X_parts, Y_parts, id_parts = [], [], []

    for shot in shots:
        mag_features = interpolate_magnetics_to_efit(shot)  # (T, 21)
        features = mag_features

        if include_thomson:
            thomson_features = interpolate_thomson_to_efit(shot)
            if thomson_features is not None:
                features = np.hstack([features, thomson_features])

        psirz = shot["efit_psirz"]  # (T, 65, 65)
        n_times = min(len(features), len(psirz))

        X_parts.append(features[:n_times])
        Y_parts.append(psirz[:n_times])
        id_parts.append(np.full(n_times, shot["shot_index"], dtype=np.int32))

    X = np.concatenate(X_parts, axis=0)
    Y = np.concatenate(Y_parts, axis=0)
    shot_ids = np.concatenate(id_parts, axis=0)

    # Drop rows with NaN/Inf
    valid = np.isfinite(X).all(axis=1) & np.isfinite(Y.reshape(len(Y), -1)).all(axis=1)
    if not valid.all():
        n_dropped = (~valid).sum()
        print(f"  Dropped {n_dropped} time slices with NaN/Inf values")
        X, Y, shot_ids = X[valid], Y[valid], shot_ids[valid]

    return X, Y, shot_ids


# ---------------------------------------------------------------------------
# 5. PCA target compression
# ---------------------------------------------------------------------------

class TargetPCA:
    """PCA compression for 65x65 flux maps."""

    def __init__(self, n_components: int = 50):
        self.n_components = n_components
        self.pca = PCA(n_components=n_components)
        self._fitted = False

    def fit(self, Y: np.ndarray) -> "TargetPCA":
        Y_flat = Y.reshape(len(Y), -1).astype(np.float64)
        self.pca.fit(Y_flat)
        self._fitted = True
        return self

    def transform(self, Y: np.ndarray) -> np.ndarray:
        Y_flat = Y.reshape(len(Y), -1).astype(np.float64)
        return self.pca.transform(Y_flat).astype(np.float32)

    def inverse_transform(self, coefficients: np.ndarray) -> np.ndarray:
        reconstructed = self.pca.inverse_transform(coefficients.astype(np.float64))
        return reconstructed.reshape(-1, EFIT_GRID_SIZE, EFIT_GRID_SIZE).astype(np.float32)

    @property
    def explained_variance_ratio(self) -> np.ndarray:
        return self.pca.explained_variance_ratio_


# ---------------------------------------------------------------------------
# 6. Data pipeline
# ---------------------------------------------------------------------------

class FusionDataPipeline:
    """Orchestrates data loading, feature engineering, splitting, and scaling."""

    def __init__(self, config: ExperimentConfig):
        self.config = config

    def prepare(self) -> dict:
        print("\n=== Loading Data (Hugging Face) ===")
        shots = load_shots_from_hf(
            self.config.n_shots,
            self.config.random_state,
            self.config.hf_repo_id,
            self.config.hf_config,
        )

        print("\n=== Building Feature Matrix ===")
        X, Y, shot_ids = build_feature_matrix(shots, self.config.include_thomson)
        print(f"  X: {X.shape}, Y: {Y.shape}, unique shots: {len(np.unique(shot_ids))}")

        # Split by shot to avoid data leakage
        print("\n=== Splitting Data (by shot) ===")
        unique_shots = np.unique(shot_ids)
        n_test = max(1, int(len(unique_shots) * self.config.test_size))
        n_val = max(1, int(len(unique_shots) * self.config.val_size))
        n_train = len(unique_shots) - n_test - n_val

        rng = np.random.RandomState(self.config.random_state)
        perm = rng.permutation(unique_shots)
        train_shots = set(perm[:n_train])
        val_shots = set(perm[n_train:n_train + n_val])
        test_shots = set(perm[n_train + n_val:])

        train_mask = np.isin(shot_ids, list(train_shots))
        val_mask = np.isin(shot_ids, list(val_shots))
        test_mask = np.isin(shot_ids, list(test_shots))

        X_train, Y_train = X[train_mask], Y[train_mask]
        X_val, Y_val = X[val_mask], Y[val_mask]
        X_test, Y_test = X[test_mask], Y[test_mask]
        print(f"  Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

        # Feature scaling
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_val = scaler.transform(X_val)
        X_test = scaler.transform(X_test)

        # PCA on targets (fit on train only)
        print("\n=== Fitting PCA on targets ===")
        n_components = min(self.config.n_pca_components, len(X_train), EFIT_GRID_SIZE ** 2)
        target_pca = TargetPCA(n_components=n_components)
        target_pca.fit(Y_train)
        cumvar = np.cumsum(target_pca.explained_variance_ratio)
        print(f"  {n_components} components capture {cumvar[-1] * 100:.1f}% variance")

        Y_train_pca = target_pca.transform(Y_train)
        Y_val_pca = target_pca.transform(Y_val)
        Y_test_pca = target_pca.transform(Y_test)

        return {
            "X_train": X_train, "X_val": X_val, "X_test": X_test,
            "Y_train": Y_train, "Y_val": Y_val, "Y_test": Y_test,
            "Y_train_pca": Y_train_pca, "Y_val_pca": Y_val_pca, "Y_test_pca": Y_test_pca,
            "scaler": scaler, "target_pca": target_pca,
            "n_features": X_train.shape[1],
        }


# ---------------------------------------------------------------------------
# 7. Models - sklearn baselines
# ---------------------------------------------------------------------------

class SklearnModel:
    """Wrapper for sklearn models predicting PCA coefficients."""

    def __init__(self, name: str, model):
        self.name = name
        self.model = model

    def fit(self, X: np.ndarray, Y_pca: np.ndarray):
        self.model.fit(X, Y_pca)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X).astype(np.float32)


def get_sklearn_models() -> list[SklearnModel]:
    return [
        SklearnModel("Linear Regression", LinearRegression()),
        SklearnModel("Ridge (CV)", RidgeCV(alphas=np.logspace(-3, 3, 10))),
        SklearnModel("Random Forest", RandomForestRegressor(
            n_estimators=100, max_depth=20, n_jobs=-1, random_state=42,
        )),
        SklearnModel("MLP (sklearn)", MLPRegressor(
            hidden_layer_sizes=(256, 128), max_iter=500,
            early_stopping=True, random_state=42,
        )),
    ]


# ---------------------------------------------------------------------------
# 8. Evaluation metrics
# ---------------------------------------------------------------------------

def compute_metrics(Y_true: np.ndarray, Y_pred: np.ndarray) -> dict:
    """Compute regression metrics on 65x65 flux maps."""
    Y_true_flat = Y_true.reshape(len(Y_true), -1)
    Y_pred_flat = Y_pred.reshape(len(Y_pred), -1)

    mse = mean_squared_error(Y_true_flat, Y_pred_flat)
    mae = mean_absolute_error(Y_true_flat, Y_pred_flat)
    r2 = r2_score(Y_true_flat, Y_pred_flat)

    residual_map = np.mean(np.abs(Y_true - Y_pred), axis=0)  # (65, 65)

    avg_ssim = None
    if HAS_SSIM:
        ssim_vals = []
        for i in range(min(len(Y_true), 200)):
            s = ssim(Y_true[i], Y_pred[i], data_range=Y_true[i].max() - Y_true[i].min())
            ssim_vals.append(s)
        avg_ssim = float(np.mean(ssim_vals))

    true_norm = np.linalg.norm(Y_true_flat, axis=1)
    diff_norm = np.linalg.norm(Y_true_flat - Y_pred_flat, axis=1)
    rel_error = float(np.mean(diff_norm / (true_norm + 1e-10)))

    return {
        "MSE": mse,
        "MAE": mae,
        "R2": r2,
        "SSIM": avg_ssim,
        "Relative Error": rel_error,
        "residual_map": residual_map,
    }


# ---------------------------------------------------------------------------
# 9. Plotting utilities
# ---------------------------------------------------------------------------

def plot_flux_comparison(Y_true: np.ndarray, Y_pred: np.ndarray,
                         sample_idx: int, title: str, output_path: Path):
    """Plot actual vs predicted vs residual for a single flux map."""
    true_map = Y_true[sample_idx]
    pred_map = Y_pred[sample_idx]
    diff_map = true_map - pred_map

    # Use the actual map's range so ground truth looks consistent across models
    vmin = true_map.min()
    vmax = true_map.max()

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    im0 = axes[0].imshow(true_map, cmap="RdBu_r", vmin=vmin, vmax=vmax, origin="lower")
    axes[0].set_title("Actual")
    plt.colorbar(im0, ax=axes[0], fraction=0.046)

    im1 = axes[1].imshow(pred_map, cmap="RdBu_r", vmin=vmin, vmax=vmax, origin="lower")
    axes[1].set_title("Predicted")
    plt.colorbar(im1, ax=axes[1], fraction=0.046)

    diff_max = max(abs(diff_map.min()), abs(diff_map.max()))
    im2 = axes[2].imshow(diff_map, cmap="RdBu_r", vmin=-diff_max, vmax=diff_max, origin="lower")
    axes[2].set_title("Residual (True - Pred)")
    plt.colorbar(im2, ax=axes[2], fraction=0.046)

    fig.suptitle(title, fontsize=14)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_model_comparison(results: dict, output_path: Path):
    """Bar chart comparing metrics across models."""
    names = list(results.keys())
    metrics_to_plot = ["MSE", "R2", "SSIM"]
    available = [m for m in metrics_to_plot if results[names[0]].get(m) is not None]

    fig, axes = plt.subplots(1, len(available), figsize=(6 * len(available), 5))
    if len(available) == 1:
        axes = [axes]

    for ax, metric in zip(axes, available):
        vals = [results[n][metric] for n in names]
        colors = plt.cm.Set2(np.linspace(0, 1, len(names)))
        bars = ax.bar(range(len(names)), vals, color=colors)
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, rotation=45, ha="right", fontsize=9)
        ax.set_ylabel(metric)
        ax.set_title(metric)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                    f"{val:.4f}", ha="center", va="bottom", fontsize=8)

    plt.suptitle("Model Comparison", fontsize=14)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_residual_heatmap(residual_map: np.ndarray, model_name: str, output_path: Path):
    """Plot average residual heatmap (65x65)."""
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(residual_map, cmap="hot", origin="lower")
    ax.set_title(f"Mean Absolute Error Map - {model_name}")
    ax.set_xlabel("R index")
    ax.set_ylabel("Z index")
    plt.colorbar(im, ax=ax, label="MAE")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_pca_explained_variance(target_pca: TargetPCA, output_path: Path):
    """Plot cumulative explained variance from PCA."""
    cumvar = np.cumsum(target_pca.explained_variance_ratio)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(range(1, len(cumvar) + 1), cumvar * 100, "b-o", markersize=3)
    ax.set_xlabel("Number of PCA Components")
    ax.set_ylabel("Cumulative Explained Variance (%)")
    ax.set_title("PCA Target Compression")
    ax.axhline(y=99, color="r", linestyle="--", alpha=0.7, label="99%")
    ax.axhline(y=95, color="orange", linestyle="--", alpha=0.7, label="95%")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_training_curves(train_losses: list, val_losses: list,
                         model_name: str, output_path: Path):
    """Plot PyTorch training and validation loss curves."""
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(train_losses, label="Train Loss")
    ax.plot(val_losses, label="Val Loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE Loss")
    ax.set_title(f"Training Curves - {model_name}")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_pca_reconstruction(Y_true: np.ndarray, target_pca: TargetPCA,
                            sample_idx: int, output_path: Path):
    """Show PCA-only reconstruction quality (no model involved)."""
    Y_pca = target_pca.transform(Y_true[sample_idx:sample_idx + 1])
    Y_recon = target_pca.inverse_transform(Y_pca)
    plot_flux_comparison(
        Y_true[sample_idx:sample_idx + 1],
        Y_recon,
        sample_idx=0,
        title=f"PCA Reconstruction ({target_pca.n_components} components)",
        output_path=output_path,
    )


# ---------------------------------------------------------------------------
# 10. Experiment runner
# ---------------------------------------------------------------------------

def run_all_experiments(config: ExperimentConfig, skip_sklearn=False, skip_pytorch=False):
    """Run the full experiment pipeline."""
    config.output_dir.mkdir(parents=True, exist_ok=True)

    pipeline = FusionDataPipeline(config)
    data = pipeline.prepare()

    # Plot PCA variance
    plot_pca_explained_variance(data["target_pca"], config.output_dir / "pca_variance.png")
    print(f"  Saved: pca_variance.png")

    # Plot PCA reconstruction example
    plot_pca_reconstruction(
        data["Y_test"], data["target_pca"], sample_idx=0,
        output_path=config.output_dir / "pca_reconstruction.png",
    )
    print(f"  Saved: pca_reconstruction.png")

    all_results = {}

    # --- Sklearn models ---
    if not skip_sklearn:
        print("\n=== Training Sklearn Models ===")
        sklearn_models = get_sklearn_models()

        for sm in sklearn_models:
            print(f"\n  [{sm.name}]")
            t0 = time.time()
            sm.fit(data["X_train"], data["Y_train_pca"])
            fit_time = time.time() - t0

            Y_pred_pca = sm.predict(data["X_test"])
            Y_pred = data["target_pca"].inverse_transform(Y_pred_pca)
            metrics = compute_metrics(data["Y_test"], Y_pred)
            metrics["fit_time"] = fit_time
            all_results[sm.name] = metrics

            print(f"    MSE={metrics['MSE']:.6f}, R2={metrics['R2']:.4f}, "
                  f"SSIM={metrics.get('SSIM', 'N/A')}, time={fit_time:.1f}s")

            safe_name = sm.name.replace(" ", "_").replace("(", "").replace(")", "")
            plot_flux_comparison(
                data["Y_test"], Y_pred, sample_idx=0,
                title=f"{sm.name} - Test Sample",
                output_path=config.output_dir / f"flux_{safe_name}.png",
            )
            plot_residual_heatmap(
                metrics["residual_map"], sm.name,
                output_path=config.output_dir / f"residual_{safe_name}.png",
            )

    # --- PyTorch models ---
    if not skip_pytorch:
        try:
            from experiments_torch import run_pytorch_experiments
            all_results.update(run_pytorch_experiments(config, data))
        except ImportError:
            print(
                "\n=== Skipping PyTorch models (torch not installed) ===\n"
                "  Install with: uv sync --group pytorch   OR   pip install -r requirements-pytorch.txt"
            )

    # --- Summary ---
    if all_results:
        print("\n" + "=" * 80)
        print("RESULTS SUMMARY")
        print("=" * 80)
        header = f"{'Model':<30} {'MSE':>12} {'MAE':>12} {'R2':>10} {'SSIM':>10} {'Time':>8}"
        print(header)
        print("-" * len(header))
        for name, m in all_results.items():
            ssim_str = f"{m['SSIM']:.4f}" if m.get("SSIM") is not None else "N/A"
            print(f"{name:<30} {m['MSE']:>12.6f} {m['MAE']:>12.6f} "
                  f"{m['R2']:>10.4f} {ssim_str:>10} {m['fit_time']:>7.1f}s")

        plot_model_comparison(all_results, config.output_dir / "model_comparison.png")
        print(f"\nPlots saved to: {config.output_dir}")

    return all_results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fusion Equilibrium - Model Experiments")
    parser.add_argument("--n-shots", type=int, default=10, help="Number of shots to load")
    parser.add_argument("--n-pca", type=int, default=50, help="PCA components for target compression")
    parser.add_argument("--include-thomson", action="store_true", help="Include Thomson scattering features")
    parser.add_argument("--epochs", type=int, default=50, help="PyTorch training epochs")
    parser.add_argument("--batch-size", type=int, default=256, help="PyTorch batch size")
    parser.add_argument("--quick", action="store_true", help="Quick mode: 3 shots, 20 PCA, 10 epochs")
    parser.add_argument("--skip-sklearn", action="store_true", help="Skip sklearn models")
    parser.add_argument("--skip-pytorch", action="store_true", help="Skip PyTorch models")
    parser.add_argument("--device", default="auto", choices=["auto", "mps", "cpu"])
    args = parser.parse_args()

    config = ExperimentConfig(
        n_shots=args.n_shots,
        n_pca_components=args.n_pca,
        include_thomson=args.include_thomson,
        epochs=args.epochs,
        batch_size=args.batch_size,
        device=args.device,
    )

    if args.quick:
        config.n_shots = 3
        config.n_pca_components = 20
        config.epochs = 10

    print("Fusion Equilibrium Challenge - Baseline Experiments")
    print(f"  Data: {config.hf_repo_id} / {config.hf_config}")
    print(f"  Shots: {config.n_shots}")
    print(f"  PCA components: {config.n_pca_components}")
    print(f"  Thomson: {config.include_thomson}")
    print(f"  PyTorch: {'enabled' if not args.skip_pytorch else 'skipped'}")
    print(f"  Epochs: {config.epochs}")

    if args.quick:
        print("\n  NOTE: Running in --quick mode (3 shots, 10 epochs). This uses very")
        print("  little data and is only meant to verify the pipeline works. Neural")
        print("  network results will be poor due to extreme overfitting (~264 training")
        print("  samples for millions of parameters). Run with --n-shots 50+ and")
        print("  --epochs 50+ for meaningful neural network results.")

    results = run_all_experiments(config, skip_sklearn=args.skip_sklearn, skip_pytorch=args.skip_pytorch)
