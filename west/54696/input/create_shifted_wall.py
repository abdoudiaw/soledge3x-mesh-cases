#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from scipy.io import loadmat, savemat

plt = None


def get_pyplot(show: bool):
    global plt
    if plt is not None:
        return plt

    try:
        import matplotlib
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "matplotlib is required for plotting. Install it in the active environment first."
        ) from exc

    if not show:
        matplotlib.use("Agg")

    import matplotlib.pyplot as pyplot

    plt = pyplot
    return plt


def find_longest_contiguous_segment(indices: np.ndarray) -> tuple[int, int]:
    if indices.size == 0:
        raise ValueError("No wall points matched the requested outboard-midplane mask.")

    breaks = np.where(np.diff(indices) > 1)[0]
    starts = np.r_[0, breaks + 1]
    ends = np.r_[breaks, indices.size - 1]
    lengths = ends - starts + 1
    best = int(np.argmax(lengths))
    return int(indices[starts[best]]), int(indices[ends[best]])


def apply_midplane_limiter_shift(
    rwall: np.ndarray,
    zwall: np.ndarray,
    shift_m: float,
    r_threshold: float,
    z_threshold: float,
) -> tuple[np.ndarray, np.ndarray]:
    mask = (rwall > r_threshold) & (np.abs(zwall) < z_threshold)
    start, end = find_longest_contiguous_segment(np.where(mask)[0])

    shifted = rwall.copy()
    segment = np.arange(start, end + 1)
    s = np.linspace(0.0, 1.0, segment.size)
    weight = np.sin(np.pi * s) ** 2
    shifted[segment] += shift_m * weight
    return shifted, segment


def apply_support_arm_shift(
    rwall: np.ndarray,
    zwall: np.ndarray,
    shift_m: float,
    r_threshold: float,
    z_start: float,
    z_stop: float,
) -> tuple[np.ndarray, np.ndarray]:
    shifted = rwall.copy()

    # Keep the limiter nose fixed and only move the upper/lower support-arm
    # sections behind it on the outboard side.
    mask = (rwall > r_threshold) & (np.abs(zwall) >= z_start) & (np.abs(zwall) <= z_stop)
    indices = np.where(mask)[0]
    if indices.size == 0:
        raise ValueError("No wall points matched the requested support-arm mask.")

    # Smooth ramp from zero at the limiter attachment to the full shift deeper
    # into the support arm.
    z_abs = np.abs(zwall[indices])
    weight = np.sin(0.5 * np.pi * (z_abs - z_start) / (z_stop - z_start)) ** 2
    shifted[indices] += shift_m * weight
    return shifted, indices


def apply_rigid_limiter_shift(
    rwall: np.ndarray,
    zwall: np.ndarray,
    shift_m: float,
    start_idx: int,
    end_idx: int,
) -> tuple[np.ndarray, np.ndarray]:
    shifted = rwall.copy()
    indices = np.arange(start_idx, end_idx + 1)
    # Include the duplicated closure point when the contour is explicitly
    # closed by repeating the first limiter point at the end of the array.
    if np.isclose(rwall[-1], rwall[start_idx]) and np.isclose(zwall[-1], zwall[start_idx]):
        indices = np.r_[indices, len(rwall) - 1]
    shifted[indices] += shift_m
    return shifted, indices


def make_plot(
    rwall: np.ndarray,
    zwall: np.ndarray,
    shifted: np.ndarray,
    changed: np.ndarray,
    output: Path,
    mode: str,
    show: bool,
) -> Path:
    pyplot = get_pyplot(show)

    plot_path = output.with_suffix(".png")
    fig, ax = pyplot.subplots(figsize=(7, 7))
    ax.plot(rwall, zwall, color="black", linewidth=1.8, label="Original wall")
    ax.plot(shifted, zwall, color="tab:red", linewidth=1.8, linestyle="--", label="Shifted wall")
    if changed.size:
        ax.scatter(
            shifted[changed],
            zwall[changed],
            s=10,
            color="tab:blue",
            alpha=0.7,
            label="Shifted points",
        )
    ax.set_xlabel("R [m]")
    ax.set_ylabel("Z [m]")
    ax.set_title(f"Wall comparison ({mode})")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(plot_path, dpi=200)
    if show:
        pyplot.show()
    pyplot.close(fig)
    return plot_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create a shifted WEST wall MAT file by moving the outboard midplane wall radially outward."
    )
    parser.add_argument("--input", default="wall_geometry.mat", help="Input wall MAT file")
    parser.add_argument(
        "--output",
        default="wall_geometry_shifted_outboard_3cm.mat",
        help="Output wall MAT file",
    )
    parser.add_argument(
        "--mode",
        choices=("midplane-limiter", "support-arm", "rigid-limiter"),
        default="rigid-limiter",
        help="Which outboard structure to shift",
    )
    parser.add_argument(
        "--shift-cm",
        type=float,
        default=3.0,
        help="Maximum outward radial shift applied at the outboard midplane",
    )
    parser.add_argument(
        "--r-threshold",
        type=float,
        default=2.90,
        help="Minimum R used to identify the outboard limiter section",
    )
    parser.add_argument(
        "--z-threshold",
        type=float,
        default=0.45,
        help="Maximum |Z| used to identify the outboard midplane section",
    )
    parser.add_argument(
        "--z-start",
        type=float,
        default=0.42,
        help="Minimum |Z| where the support-arm shift begins",
    )
    parser.add_argument(
        "--z-stop",
        type=float,
        default=0.78,
        help="Maximum |Z| where the support-arm shift reaches full strength",
    )
    parser.add_argument(
        "--limiter-start-idx",
        type=int,
        default=0,
        help="First wall index belonging to the rigidly shifted limiter contour",
    )
    parser.add_argument(
        "--limiter-end-idx",
        type=int,
        default=133,
        help="Last wall index belonging to the rigidly shifted limiter contour",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Save an overlay plot of the original and shifted wall contours",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display the overlay plot interactively when a GUI backend is available",
    )
    args = parser.parse_args()

    data = loadmat(args.input, squeeze_me=True)
    rwall = np.asarray(data["Rwall"], dtype=float).ravel()
    zwall = np.asarray(data["Zwall"], dtype=float).ravel()

    shift_m = 0.01 * args.shift_cm
    if args.mode == "midplane-limiter":
        shifted, changed = apply_midplane_limiter_shift(
            rwall,
            zwall,
            shift_m,
            args.r_threshold,
            args.z_threshold,
        )
    elif args.mode == "rigid-limiter":
        shifted, changed = apply_rigid_limiter_shift(
            rwall,
            zwall,
            shift_m,
            args.limiter_start_idx,
            args.limiter_end_idx,
        )
    else:
        shifted, changed = apply_support_arm_shift(
            rwall,
            zwall,
            shift_m,
            args.r_threshold,
            args.z_start,
            args.z_stop,
        )

    output = Path(args.output)
    savemat(
        output,
        {
            "Rwall": shifted.reshape(data["Rwall"].shape),
            "Zwall": zwall.reshape(data["Zwall"].shape),
        },
    )

    d_r = shifted - rwall
    mid_idx = int(np.argmax(d_r))
    print(f"input: {args.input}")
    print(f"output: {output}")
    print(f"mode: {args.mode}")
    print(f"changed_points: {changed.size}")
    print(f"changed_index_range: {int(changed.min())}:{int(changed.max())}")
    print(f"max_shift_cm: {args.shift_cm:.3f}")
    print(f"midplane_reference_point: index={mid_idx}, R={rwall[mid_idx]:.6f}, Z={zwall[mid_idx]:.6f}")
    print(f"shifted_reference_point: index={mid_idx}, R={shifted[mid_idx]:.6f}, Z={zwall[mid_idx]:.6f}")
    if args.plot or args.show:
        plot_path = make_plot(rwall, zwall, shifted, changed, output, args.mode, args.show)
        print(f"plot: {plot_path}")


if __name__ == "__main__":
    main()
