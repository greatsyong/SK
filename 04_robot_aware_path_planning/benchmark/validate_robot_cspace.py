from pathlib import Path
import math

import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import distance_transform_edt

from benchmark.metric_map_loader import load_ros_metric_map


# ============================================================
# CONFIG
# ============================================================

MAP_YAML = Path(
    "data/metric_maps/stech_lab/stech_lab_completed.yaml"
)

RESULT_DIR = Path(
    "results/stech_lab"
)

ROBOTS = {
    "waffle": {
        "display_name": "TurtleBot3 Waffle Pi",
        "length_m": 0.281,
        "width_m": 0.306,
    },
    "ridgeback": {
        "display_name": "Clearpath Ridgeback",
        "length_m": 0.960,
        "width_m": 0.793,
    },
}


# ============================================================
# ROBOT GEOMETRY
# ============================================================

def circumscribed_radius(length_m, width_m):
    return 0.5 * math.hypot(
        length_m,
        width_m,
    )


# ============================================================
# MAIN
# ============================================================

def main():

    RESULT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Load metric map
    # --------------------------------------------------------

    m = load_ros_metric_map(
        MAP_YAML
    )

    grid = m["grid"]

    resolution = m["resolution"]

    free = (
        grid == 0
    )

    # --------------------------------------------------------
    # Metric obstacle clearance
    #
    # distance_transform_edt() returns distance in cells.
    # Multiply by resolution to obtain metres.
    # --------------------------------------------------------

    clearance_m = (
        distance_transform_edt(
            free
        )
        * resolution
    )

    cspaces = {}

    # --------------------------------------------------------
    # Build robot-specific C-spaces
    # --------------------------------------------------------

    print()
    print("=" * 72)
    print("STECH LAB ROBOT C-SPACE VALIDATION")
    print("=" * 72)

    print(
        f"Map resolution : "
        f"{resolution:.3f} m/cell"
    )

    print(
        f"Map size       : "
        f"{m['width']} x {m['height']} cells"
    )

    print(
        f"Physical size  : "
        f"{m['width'] * resolution:.2f} x "
        f"{m['height'] * resolution:.2f} m"
    )

    print()

    for key, robot in ROBOTS.items():

        radius = circumscribed_radius(
            robot["length_m"],
            robot["width_m"],
        )

        robot[
            "circumscribed_radius_m"
        ] = radius

        cspace_free = (
            free
            &
            (
                clearance_m
                >= radius
            )
        )

        cspaces[
            key
        ] = cspace_free

        print(
            f"{robot['display_name']}"
        )

        print(
            f"  size         : "
            f"{robot['length_m']:.3f} x "
            f"{robot['width_m']:.3f} m"
        )

        print(
            f"  circ radius  : "
            f"{radius:.4f} m"
        )

        print(
            f"  radius cells : "
            f"{radius / resolution:.3f}"
        )

        print(
            f"  feasible     : "
            f"{int(cspace_free.sum())} cells"
        )

        print()

    # --------------------------------------------------------
    # Waffle-only area
    # --------------------------------------------------------

    waffle_only = (
        cspaces["waffle"]
        &
        ~cspaces["ridgeback"]
    )

    common_free = (
        cspaces["waffle"]
        &
        cspaces["ridgeback"]
    )

    print(
        "Waffle-only feasible :",
        int(
            waffle_only.sum()
        ),
        "cells",
    )

    print(
        "Common feasible      :",
        int(
            common_free.sum()
        ),
        "cells",
    )

    # ========================================================
    # Figure 1: three-way comparison
    # ========================================================

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(18, 7),
    )

    # Original
    axes[0].imshow(
        grid,
        cmap="gray_r",
        origin="upper",
        interpolation="nearest",
    )

    axes[0].set_title(
        "Original Occupancy"
    )

    # Waffle
    axes[1].imshow(
        ~cspaces["waffle"],
        cmap="gray_r",
        origin="upper",
        interpolation="nearest",
    )

    axes[1].set_title(
        "Waffle Pi C-Space"
    )

    # Ridgeback
    axes[2].imshow(
        ~cspaces["ridgeback"],
        cmap="gray_r",
        origin="upper",
        interpolation="nearest",
    )

    axes[2].set_title(
        "Ridgeback C-Space"
    )

    for ax in axes:

        ax.set_xticks([])
        ax.set_yticks([])

    fig.suptitle(
        (
            "STech Lab — "
            "Robot-Specific Configuration Spaces"
        ),
        fontsize=16,
    )

    fig.tight_layout()

    comparison_path = (
        RESULT_DIR
        / "robot_cspace_comparison.png"
    )

    fig.savefig(
        comparison_path,
        dpi=180,
        bbox_inches="tight",
    )

    plt.close(fig)

    # ========================================================
    # Figure 2: Waffle-only regions
    # ========================================================

    fig, ax = plt.subplots(
        figsize=(12, 10)
    )

    ax.imshow(
        grid,
        cmap="gray_r",
        origin="upper",
        interpolation="nearest",
    )

    overlay = np.zeros(
        (
            grid.shape[0],
            grid.shape[1],
            4,
        ),
        dtype=float,
    )

    overlay[
        ...,
        0
    ] = waffle_only.astype(
        float
    )

    overlay[
        ...,
        3
    ] = (
        waffle_only.astype(float)
        * 0.55
    )

    ax.imshow(
        overlay,
        origin="upper",
    )

    ax.set_title(
        (
            "STech Lab — "
            "Waffle-Feasible / Ridgeback-Infeasible Regions"
        )
    )

    ax.set_xlabel(
        (
            "x [cell]  |  "
            f"{resolution:.2f} m/cell"
        )
    )

    ax.set_ylabel(
        "y [cell]"
    )

    diff_path = (
        RESULT_DIR
        / "waffle_only_regions.png"
    )

    fig.savefig(
        diff_path,
        dpi=180,
        bbox_inches="tight",
    )

    plt.close(fig)

    # ========================================================
    # Figure 3: clearance field
    # ========================================================

    fig, ax = plt.subplots(
        figsize=(12, 10)
    )

    masked_clearance = (
        np.ma.masked_where(
            ~free,
            clearance_m,
        )
    )

    im = ax.imshow(
        masked_clearance,
        origin="upper",
    )

    ax.set_title(
        "STech Lab — Obstacle Clearance [m]"
    )

    ax.set_xlabel(
        (
            "x [cell]  |  "
            f"{resolution:.2f} m/cell"
        )
    )

    ax.set_ylabel(
        "y [cell]"
    )

    fig.colorbar(
        im,
        ax=ax,
        label=(
            "Distance to nearest obstacle [m]"
        ),
    )

    clearance_path = (
        RESULT_DIR
        / "clearance_map.png"
    )

    fig.savefig(
        clearance_path,
        dpi=180,
        bbox_inches="tight",
    )

    plt.close(fig)

    print()
    print("=" * 72)
    print("C-SPACE VALIDATION COMPLETE")
    print("=" * 72)

    print(
        "Saved:",
        comparison_path,
    )

    print(
        "Saved:",
        diff_path,
    )

    print(
        "Saved:",
        clearance_path,
    )


if __name__ == "__main__":
    main()