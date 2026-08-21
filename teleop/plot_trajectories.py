"""Top-down (X-Y projection) policy-vs-demonstration trajectory plots.

For each eval episode logged by eval_flipup_policy.py --trajectory-dir, overlays
the policy's tool-tip path against the real demonstration episode its init pose
was bootstrapped from (same start point, so the overlay is meaningful), plus a
sweep-summary plot showing how a chosen parameter (default_perp_stiffness or
replan_every_ticks) affects a quantitative drift metric.

Colors follow the dataviz skill's validated categorical palette: slot 1 (blue,
#2a78d6) = demonstration, slot 2 (orange, #eb6834) = policy. Single-hue,
light->dark not used here since only two discrete series (not a magnitude scale)
are being compared per plot.
"""
from __future__ import annotations

import argparse
import glob
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import zarr

DEMO_COLOR = "#2a78d6"   # categorical slot 1 (blue)
POLICY_COLOR = "#eb6834"  # categorical slot 2 (orange)
BOOK_COLOR = "#7a7a76"    # muted ink, reference annotation not a data series
GRID_COLOR = "#d8d8d3"


def load_demo_xy(dataset_path: str, episode_name: str):
    root = zarr.open(dataset_path, mode="r")
    fb = root["data"][episode_name]["ts_pose_fb_0"][:]
    book_xy = root["data"][episode_name]["book_pose"][0, :2]
    return fb[:, :2], book_xy


def plot_episode(npz_path: str, dataset_path: str, out_path: str):
    d = np.load(npz_path, allow_pickle=True)
    policy_xyz = d["tool_xyz_trace"]
    demo_name = str(d["demo_episode_name"])
    success = bool(d["success"])
    final_angle = float(d["final_book_angle_deg"])

    demo_xy, book_xy = load_demo_xy(dataset_path, demo_name)
    policy_xy = policy_xyz[:, :2]

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_facecolor("#fcfcfb")
    ax.grid(True, color=GRID_COLOR, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)

    ax.plot(demo_xy[:, 0], demo_xy[:, 1], color=DEMO_COLOR, linewidth=2, label="Demonstration", zorder=3)
    ax.plot(policy_xy[:, 0], policy_xy[:, 1], color=POLICY_COLOR, linewidth=2, label="Policy", zorder=3)

    ax.scatter(*demo_xy[0], color=DEMO_COLOR, marker="o", s=70, zorder=4, edgecolors="white", linewidths=1)
    ax.scatter(*demo_xy[-1], color=DEMO_COLOR, marker="s", s=70, zorder=4, edgecolors="white", linewidths=1)
    ax.scatter(*policy_xy[0], color=POLICY_COLOR, marker="o", s=70, zorder=4, edgecolors="white", linewidths=1)
    ax.scatter(*policy_xy[-1], color=POLICY_COLOR, marker="s", s=70, zorder=4, edgecolors="white", linewidths=1)
    ax.scatter(*book_xy, color=BOOK_COLOR, marker="*", s=180, zorder=5, label="Book (start)")

    ax.set_xlabel("World X (m)")
    ax.set_ylabel("World Y (m)")
    ax.set_aspect("equal")
    status = "SUCCESS" if success else "failed"
    ax.set_title(f"{os.path.basename(npz_path).replace('.npz','')} vs {demo_name} "
                 f"— {status} (final angle {final_angle:.1f}°)", fontsize=10)
    ax.legend(loc="best", frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def plot_grid(npz_dir: str, dataset_path: str, out_path: str, title: str):
    files = sorted(glob.glob(f"{npz_dir}/episode_*.npz"), key=lambda p: int(os.path.basename(p).split("_")[1].split(".")[0]))
    n = len(files)
    cols = min(5, n)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(3.2 * cols, 3.2 * rows), squeeze=False)
    for i, f in enumerate(files):
        ax = axes[i // cols][i % cols]
        d = np.load(f, allow_pickle=True)
        policy_xyz = d["tool_xyz_trace"]
        demo_name = str(d["demo_episode_name"])
        success = bool(d["success"])
        demo_xy, book_xy = load_demo_xy(dataset_path, demo_name)
        policy_xy = policy_xyz[:, :2]
        ax.set_facecolor("#fcfcfb")
        ax.grid(True, color=GRID_COLOR, linewidth=0.6, zorder=0)
        ax.set_axisbelow(True)
        ax.plot(demo_xy[:, 0], demo_xy[:, 1], color=DEMO_COLOR, linewidth=1.6, zorder=3)
        ax.plot(policy_xy[:, 0], policy_xy[:, 1], color=POLICY_COLOR, linewidth=1.6, zorder=3)
        ax.scatter(*demo_xy[0], color=DEMO_COLOR, marker="o", s=30, zorder=4)
        ax.scatter(*policy_xy[0], color=POLICY_COLOR, marker="o", s=30, zorder=4)
        ax.scatter(*book_xy, color=BOOK_COLOR, marker="*", s=70, zorder=5)
        ax.set_title(f"ep{i} {'✓' if success else '✗'}", fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect("equal")
    for i in range(n, rows * cols):
        axes[i // cols][i % cols].axis("off")

    handles = [
        plt.Line2D([0], [0], color=DEMO_COLOR, linewidth=2, label="Demonstration"),
        plt.Line2D([0], [0], color=POLICY_COLOR, linewidth=2, label="Policy"),
        plt.Line2D([0], [0], color=BOOK_COLOR, marker="*", linewidth=0, markersize=12, label="Book"),
    ]
    fig.legend(handles=handles, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.02))
    fig.suptitle(title, y=1.06, fontsize=13)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def compute_drift(npz_path: str, dataset_path: str) -> float:
    """Mean per-tick XY distance between policy and demo trajectories, over
    the overlapping duration (both start at the same point by construction)."""
    d = np.load(npz_path, allow_pickle=True)
    policy_xy = d["tool_xyz_trace"][:, :2]
    demo_xy, _ = load_demo_xy(dataset_path, str(d["demo_episode_name"]))
    n = min(len(policy_xy), len(demo_xy))
    return float(np.linalg.norm(policy_xy[:n] - demo_xy[:n], axis=1).mean())


def plot_sweep(sweep_dirs: dict, dataset_path: str, out_path: str, xlabel: str, title: str):
    """sweep_dirs: {param_value: npz_dir} for a single series, or
    {series_label: {param_value: npz_dir}} for multiple series."""
    fig, ax = plt.subplots(figsize=(6.5, 5))
    ax.set_facecolor("#fcfcfb")
    ax.grid(True, color=GRID_COLOR, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)

    is_multi_series = all(isinstance(v, dict) for v in sweep_dirs.values())
    series = sweep_dirs if is_multi_series else {"": sweep_dirs}
    colors = [DEMO_COLOR, POLICY_COLOR, "#1baf7a", "#eda100"]

    for i, (label, param_to_dir) in enumerate(series.items()):
        xs, means, stds = [], [], []
        for param_val in sorted(param_to_dir.keys()):
            files = glob.glob(f"{param_to_dir[param_val]}/episode_*.npz")
            drifts = [compute_drift(f, dataset_path) for f in files]
            if not drifts:
                continue
            xs.append(param_val)
            means.append(np.mean(drifts))
            stds.append(np.std(drifts))
        xs, means, stds = np.array(xs), np.array(means), np.array(stds)
        color = colors[i % len(colors)]
        ax.plot(xs, means, color=color, linewidth=2, marker="o", markersize=7,
                 markeredgecolor="white", markeredgewidth=1, label=label if label else None, zorder=3)
        ax.fill_between(xs, means - stds, means + stds, color=color, alpha=0.15, zorder=2)

    ax.set_xlabel(xlabel)
    ax.set_ylabel("Mean drift from demonstration (m)")
    ax.set_title(title, fontsize=12)
    if is_multi_series:
        ax.legend(loc="best", frameon=False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--trajectory-dir", required=True)
    parser.add_argument("--dataset-path", default="/local/real/jvclark/mujoco_data/try_2_flipup_sim_20hz.zarr")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--title", default="")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    files = sorted(glob.glob(f"{args.trajectory_dir}/episode_*.npz"))
    for f in files:
        name = os.path.basename(f).replace(".npz", "")
        plot_episode(f, args.dataset_path, f"{args.out_dir}/{name}.png")
    plot_grid(args.trajectory_dir, args.dataset_path, f"{args.out_dir}/_grid.png", args.title or args.trajectory_dir)
    print(f"wrote {len(files)} per-episode plots + grid to {args.out_dir}")
