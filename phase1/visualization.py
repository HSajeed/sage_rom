"""
Render figure PNGs from the Phase 1 result artifacts written by
run_phase1.py. Pure read-only: never re-runs the data pipeline and never
modifies any results folder. Each plot function reads its own source file
and is skipped if that file is absent, so the script works against any run
directory.

Usage:
    python visualization.py                          # results_dmdpod -> figures/
    python visualization.py --dir results            # historical dir
    python visualization.py --out figures_cpu        # custom output dir
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _load_json(path: Path) -> dict | None:
    if not path.exists():
        print(f"  [skip] missing {path.name}")
        return None
    with open(path) as f:
        return json.load(f)


def _load_csv(path: Path) -> list[dict] | None:
    if not path.exists():
        print(f"  [skip] missing {path.name}")
        return None
    with open(path) as f:
        return list(csv.DictReader(f))


def plot_dmd_error_vs_rank(results_dir: Path, out_dir: Path) -> None:
    rows = _load_csv(results_dir / "dmd_error_vs_rank.csv")
    if not rows:
        return
    metadata = _load_json(results_dir / "run_metadata.json")
    ranks = [int(r["rank"]) for r in rows]
    recon = [float(r["reconstruction_err"]) for r in rows]
    extrap = [float(r["extrapolation_err"]) for r in rows]

    fig, ax = plt.subplots()
    ax.plot(ranks, recon, "o-", label="reconstruction (train)")
    ax.plot(ranks, extrap, "s-", label="extrapolation (test)")
    ax.set_yscale("log")
    ax.set_xlabel("DMD rank")
    ax.set_ylabel("relative L2 error (log)")
    if metadata and "rank_99" in metadata:
        r99 = metadata["rank_99"]
        ax.axvline(r99, color="red", ls="--", lw=1, alpha=0.6,
                   label=f"rank_99 = {r99}")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    ax.set_title("Phase 1 transient window: DMD error vs. fixed rank")
    fig.tight_layout()
    fig.savefig(out_dir / "dmd_error_vs_rank.png", dpi=150)
    plt.close(fig)
    print("  dmd_error_vs_rank.png")


def plot_pod_reconstruction_vs_rank(results_dir: Path, out_dir: Path) -> None:
    rows = _load_csv(results_dir / "pod_reconstruction_vs_rank.csv")
    if not rows:
        return
    ranks = [int(r["rank"]) for r in rows]
    errs = [float(r["relative_reconstruction_error"]) for r in rows]

    fig, ax = plt.subplots()
    ax.plot(ranks, errs, "o-", color="tab:green")
    ax.set_yscale("log")
    ax.set_xlabel("POD rank")
    ax.set_ylabel("relative reconstruction error (log)")
    ax.grid(True, which="both", alpha=0.3)
    ax.set_title("POD reconstruction error vs. rank")
    fig.tight_layout()
    fig.savefig(out_dir / "pod_reconstruction_vs_rank.png", dpi=150)
    plt.close(fig)
    print("  pod_reconstruction_vs_rank.png")


def plot_data_efficiency_dmd(results_dir: Path, out_dir: Path) -> None:
    data = _load_json(results_dir / "data_efficiency_dmd.json")
    if not data:
        return
    counts = [int(k) for k in data.get("errors", {})]
    errs = [data["errors"][str(k)] for k in counts]

    fig, ax = plt.subplots()
    ax.plot(counts, errs, "o-", color="tab:orange")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("training snapshots used (log)")
    ax.set_ylabel("DMD extrapolation relative L2 error (log)")
    ax.grid(True, which="both", alpha=0.3)
    ax.set_title("Data efficiency: DMD extrapolation error vs. data used")
    fig.tight_layout()
    fig.savefig(out_dir / "data_efficiency_dmd.png", dpi=150)
    plt.close(fig)
    print("  data_efficiency_dmd.png")


def _eigen_scatter(ax, eig_rows, dt) -> None:
    reals = []
    imags = []
    stable = []
    unstable = []
    for e in eig_rows:
        mag = e["magnitude"]
        angle = 2 * math.pi * e.get("frequency_hz", 0.0) * dt
        re, im = mag * math.cos(angle), mag * math.sin(angle)
        (stable if mag <= 1.0 else unstable).append((re, im))
    for pts, color, label in ((stable, "tab:blue", "stable (|λ|<=1)"),
                              (unstable, "tab:red", "unstable (|λ|>1)")):
        if pts:
            xs, ys = zip(*pts)
            ax.scatter(xs, ys, s=12, color=color, label=label, zorder=3)
    theta = np.linspace(0, 2 * math.pi, 200)
    ax.plot(np.cos(theta), np.sin(theta), "k-", lw=0.8, alpha=0.5)
    ax.axhline(0, color="gray", lw=0.5, alpha=0.4)
    ax.axvline(0, color="gray", lw=0.5, alpha=0.4)
    ax.set_aspect("equal")


def plot_eigenvalues_unit_circle(results_dir: Path, out_dir: Path) -> None:
    data = _load_json(results_dir / "dmd_eigenvalues.json")
    if not data:
        return
    eig_rows = data.get("eigenvalues", [])
    dt = data.get("key_parameters", {}).get("dt", 0.025)
    rank = data.get("key_parameters", {}).get("rank_used", "?")

    fig, ax = plt.subplots()
    _eigen_scatter(ax, eig_rows, dt)
    ax.legend(loc="upper left", fontsize=8)
    ax.set_title(f"DMD eigenvalues, rank {rank} (transient window)\n"
                 f"unstable: {sum(e['magnitude'] > 1.0 for e in eig_rows)}/{len(eig_rows)}")
    fig.tight_layout()
    fig.savefig(out_dir / "eigenvalues_unit_circle.png", dpi=150)
    plt.close(fig)
    print("  eigenvalues_unit_circle.png")


def plot_eigenvalues_by_rank(results_dir: Path, out_dir: Path) -> None:
    data = _load_json(results_dir / "dmd_rank_sweep.json")
    if not data:
        return
    dt = data.get("key_parameters", {}).get("dt", 0.025)
    ranks = data.get("ranks", [])
    per_rank = data.get("per_rank", {})
    ncols = min(4, len(ranks))
    nrows = math.ceil(len(ranks) / ncols)

    fig, axes = plt.subplots(nrows, ncols, figsize=(4.4 * ncols, 4.0 * nrows),
                             squeeze=False)
    for ax, r in zip(axes.flat, ranks):
        entry = per_rank.get(str(r), {})
        eig_rows = entry.get("eigen_summary", {}).get("eigenvalues", [])
        n_unstable = entry.get("eigen_summary", {}).get("n_unstable", 0)
        _eigen_scatter(ax, eig_rows, dt)
        ax.set_title(f"rank {r}: unstable {n_unstable}/{len(eig_rows)}", fontsize=10)
    for ax in axes.flat[len(ranks):]:
        ax.set_visible(False)
    fig.suptitle("DMD eigenvalues across the fixed-rank sweep (transient window)",
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(out_dir / "eigenvalues_by_rank.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  eigenvalues_by_rank.png")


def plot_comparison_table(results_dir: Path, out_dir: Path) -> None:
    rows = _load_csv(results_dir / "comparison_table.csv")
    if not rows:
        return
    names = [r["baseline"] for r in rows]
    recon = []
    extrap = []
    for r in rows:
        recon.append(float(r["reconstruction_err"]) if r.get("reconstruction_err") else None)
        extrap.append(float(r["extrapolation_err"]) if r.get("extrapolation_err") else None)

    fig, ax = plt.subplots()
    width = 0.35
    x = np.arange(len(names))
    for vals, offset, label, color in (
        (recon, -width / 2, "reconstruction (train)", "tab:blue"),
        (extrap, width / 2, "extrapolation (test)", "tab:red"),
    ):
        idx = [i for i, v in enumerate(vals) if v is not None]
        if idx:
            ax.bar(x[idx] + offset, [vals[i] for i in idx], width, label=label, color=color)
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.set_ylabel("relative L2 error (log)")
    ax.legend()
    ax.set_title("Phase 1 baseline comparison")
    fig.tight_layout()
    fig.savefig(out_dir / "comparison_table.png", dpi=150)
    plt.close(fig)
    print("  comparison_table.png")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Render PNG figures from Phase 1 result artifacts.")
    parser.add_argument("--dir", default="results_dmdpod",
                        help="run directory to read from (under phase1/)")
    parser.add_argument("--out", default="figures",
                        help="output folder (under phase1/)")
    args = parser.parse_args()

    base = Path(__file__).parent
    results_dir = base / args.dir
    out_dir = base / args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Reading {results_dir}/ -> {out_dir}/")
    plot_dmd_error_vs_rank(results_dir, out_dir)
    plot_pod_reconstruction_vs_rank(results_dir, out_dir)
    plot_data_efficiency_dmd(results_dir, out_dir)
    plot_eigenvalues_unit_circle(results_dir, out_dir)
    plot_eigenvalues_by_rank(results_dir, out_dir)
    plot_comparison_table(results_dir, out_dir)
    print("Done.")


if __name__ == "__main__":
    main()