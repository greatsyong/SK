from pathlib import Path
import math

import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import distance_transform_edt

from benchmark.metric_map_loader import load_ros_metric_map
from benchmark.find_robot_dependent_scenarios import astar


# ============================================================
# CONFIG
# ============================================================

MAP_YAML = Path(
    "data/metric_maps/stech_lab/stech_lab_completed.yaml"
)

RESULT_DIR = Path(
    "results/stech_lab/scenario01"
)

START = (360, 136)
GOAL = (744, 840)

ROBOTS = {
    "waffle": {
        "name": "TurtleBot3 Waffle Pi",
        "length_m": 0.281,
        "width_m": 0.306,
    },
    "ridgeback": {
        "name": "Clearpath Ridgeback",
        "length_m": 0.960,
        "width_m": 0.793,
    },
}

# Actual footprint interior sampling resolution.
# 0.025 m = half of map resolution.
FOOTPRINT_SAMPLE_M = 0.025

# Orientation-independent Ridgeback test:
# 0, 5, 10, ... 175 deg
ORIENTATION_STEP_DEG = 5.0


# ============================================================
# BASIC GEOMETRY
# ============================================================

def circumscribed_radius(
    length_m,
    width_m,
):
    return 0.5 * math.hypot(
        length_m,
        width_m,
    )


def path_heading(
    path,
    index,
):
    """
    Estimate tangent heading from neighboring path points.
    """

    if index == 0:
        x0, y0 = path[0]
        x1, y1 = path[1]

    elif index == len(path) - 1:
        x0, y0 = path[-2]
        x1, y1 = path[-1]

    else:
        x0, y0 = path[index - 1]
        x1, y1 = path[index + 1]

    return math.atan2(
        y1 - y0,
        x1 - x0,
    )


# ============================================================
# RECTANGULAR FOOTPRINT
# ============================================================

def make_local_footprint_samples(
    length_m,
    width_m,
    sample_spacing_m,
):
    """
    Fill the rectangular footprint with sample points.

    x_local:
        robot forward direction

    y_local:
        robot lateral direction
    """

    xs = np.arange(
        -length_m / 2.0,
        length_m / 2.0
        + sample_spacing_m * 0.5,
        sample_spacing_m,
    )

    ys = np.arange(
        -width_m / 2.0,
        width_m / 2.0
        + sample_spacing_m * 0.5,
        sample_spacing_m,
    )

    xx, yy = np.meshgrid(
        xs,
        ys,
    )

    return np.column_stack(
        (
            xx.ravel(),
            yy.ravel(),
        )
    )


def rotate_samples(
    samples,
    theta,
):
    c = math.cos(theta)
    s = math.sin(theta)

    x = (
        c * samples[:, 0]
        -
        s * samples[:, 1]
    )

    y = (
        s * samples[:, 0]
        +
        c * samples[:, 1]
    )

    return np.column_stack(
        (
            x,
            y,
        )
    )


def footprint_collision(
    grid,
    resolution,
    center,
    rotated_samples,
):
    """
    grid:
        0 = free
        1 = occupied / unknown

    center:
        grid-cell coordinate (x, y)
    """

    cx, cy = center

    dx = np.rint(
        rotated_samples[:, 0]
        / resolution
    ).astype(int)

    dy = np.rint(
        rotated_samples[:, 1]
        / resolution
    ).astype(int)

    xs = cx + dx
    ys = cy + dy

    outside = (
        (xs < 0)
        |
        (xs >= grid.shape[1])
        |
        (ys < 0)
        |
        (ys >= grid.shape[0])
    )

    if np.any(outside):
        return True

    return bool(
        np.any(
            grid[
                ys,
                xs,
            ]
            != 0
        )
    )


# ============================================================
# PATH VALIDATION
# ============================================================

def validate_path_tangent_heading(
    grid,
    resolution,
    path,
    footprint_samples,
):
    """
    Place rectangle at every path point using path tangent heading.
    """

    collision_indices = []

    for i, center in enumerate(path):

        theta = path_heading(
            path,
            i,
        )

        rotated = rotate_samples(
            footprint_samples,
            theta,
        )

        collision = footprint_collision(
            grid,
            resolution,
            center,
            rotated,
        )

        if collision:
            collision_indices.append(
                i
            )

    return collision_indices


def validate_path_any_orientation(
    grid,
    resolution,
    path,
    footprint_samples,
    angle_step_deg,
):
    """
    At each path center, test whether the footprint can fit
    in ANY sampled orientation.

    If no orientation fits, the center location is geometrically
    impossible for the robot regardless of path heading.

    This is especially useful for checking whether Ridgeback can
    physically occupy the Waffle shortcut.
    """

    angles = np.deg2rad(
        np.arange(
            0.0,
            180.0,
            angle_step_deg,
        )
    )

    rotated_sets = [
        rotate_samples(
            footprint_samples,
            theta,
        )
        for theta in angles
    ]

    impossible_indices = []

    feasible_orientation_counts = []

    for i, center in enumerate(path):

        feasible_count = 0

        for rotated in rotated_sets:

            if not footprint_collision(
                grid,
                resolution,
                center,
                rotated,
            ):
                feasible_count += 1

        feasible_orientation_counts.append(
            feasible_count
        )

        if feasible_count == 0:
            impossible_indices.append(
                i
            )

    return (
        impossible_indices,
        feasible_orientation_counts,
    )


# ============================================================
# PATH LENGTH
# ============================================================

def path_length_m(
    path,
    resolution,
):
    total = 0.0

    for a, b in zip(
        path[:-1],
        path[1:],
    ):

        total += (
            math.hypot(
                b[0] - a[0],
                b[1] - a[1],
            )
            * resolution
        )

    return total


# ============================================================
# MAIN
# ============================================================

def main():

    RESULT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Load map
    # --------------------------------------------------------

    m = load_ros_metric_map(
        MAP_YAML
    )

    grid = m[
        "grid"
    ]

    resolution = m[
        "resolution"
    ]

    raw_free = (
        grid == 0
    )

    clearance_m = (
        distance_transform_edt(
            raw_free
        )
        * resolution
    )

    # --------------------------------------------------------
    # Reproduce circular-screening paths
    # --------------------------------------------------------

    cspaces = {}

    for key, robot in ROBOTS.items():

        radius = circumscribed_radius(
            robot["length_m"],
            robot["width_m"],
        )

        robot[
            "circ_radius_m"
        ] = radius

        cspaces[
            key
        ] = (
            raw_free
            &
            (
                clearance_m
                >= radius
            )
        )

    waffle_result = astar(
        cspaces["waffle"],
        START,
        GOAL,
    )

    ridge_result = astar(
        cspaces["ridgeback"],
        START,
        GOAL,
    )

    if waffle_result is None:
        raise RuntimeError(
            "Waffle path not found."
        )

    if ridge_result is None:
        raise RuntimeError(
            "Ridgeback path not found."
        )

    waffle_path = (
        waffle_result[
            "path"
        ]
    )

    ridge_path = (
        ridge_result[
            "path"
        ]
    )

    # --------------------------------------------------------
    # Actual rectangular footprints
    # --------------------------------------------------------

    waffle_samples = (
        make_local_footprint_samples(
            ROBOTS["waffle"]["length_m"],
            ROBOTS["waffle"]["width_m"],
            FOOTPRINT_SAMPLE_M,
        )
    )

    ridge_samples = (
        make_local_footprint_samples(
            ROBOTS["ridgeback"]["length_m"],
            ROBOTS["ridgeback"]["width_m"],
            FOOTPRINT_SAMPLE_M,
        )
    )

    # ========================================================
    # Test A
    # Own path + tangent orientation
    # ========================================================

    waffle_own_collisions = (
        validate_path_tangent_heading(
            grid,
            resolution,
            waffle_path,
            waffle_samples,
        )
    )

    ridge_own_collisions = (
        validate_path_tangent_heading(
            grid,
            resolution,
            ridge_path,
            ridge_samples,
        )
    )

    # ========================================================
    # Test B
    # Ridgeback rectangle on Waffle shortcut
    # ========================================================

    ridge_on_waffle_tangent = (
        validate_path_tangent_heading(
            grid,
            resolution,
            waffle_path,
            ridge_samples,
        )
    )

    (
        ridge_on_waffle_impossible,
        ridge_orientation_counts,
    ) = validate_path_any_orientation(
        grid,
        resolution,
        waffle_path,
        ridge_samples,
        ORIENTATION_STEP_DEG,
    )

    # ========================================================
    # Test C
    # Waffle rectangle on Ridgeback route
    # ========================================================

    waffle_on_ridge = (
        validate_path_tangent_heading(
            grid,
            resolution,
            ridge_path,
            waffle_samples,
        )
    )

    # --------------------------------------------------------
    # Print results
    # --------------------------------------------------------

    waffle_length = path_length_m(
        waffle_path,
        resolution,
    )

    ridge_length = path_length_m(
        ridge_path,
        resolution,
    )

    print()
    print("=" * 80)
    print("SCENARIO 01 — RECTANGULAR FOOTPRINT VALIDATION")
    print("=" * 80)

    print(
        "Start:",
        START,
    )

    print(
        "Goal :",
        GOAL,
    )

    print()

    print(
        f"Waffle path length    : "
        f"{waffle_length:.2f} m"
    )

    print(
        f"Ridgeback path length : "
        f"{ridge_length:.2f} m"
    )

    print(
        f"R/W ratio             : "
        f"{ridge_length / waffle_length:.3f}"
    )

    print()

    print(
        "Waffle own path — "
        "tangent-heading collisions:",
        len(
            waffle_own_collisions
        ),
    )

    print(
        "Ridgeback own path — "
        "tangent-heading collisions:",
        len(
            ridge_own_collisions
        ),
    )

    print()

    print(
        "Ridgeback on Waffle shortcut — "
        "tangent-heading collisions:",
        len(
            ridge_on_waffle_tangent
        ),
    )

    print(
        "Ridgeback on Waffle shortcut — "
        "NO feasible orientation:",
        len(
            ridge_on_waffle_impossible
        ),
    )

    print()

    print(
        "Waffle on Ridgeback route — "
        "tangent-heading collisions:",
        len(
            waffle_on_ridge
        ),
    )

    # ========================================================
    # Visualization
    # ========================================================

    fig, ax = plt.subplots(
        figsize=(12, 10)
    )

    ax.imshow(
        grid,
        cmap="gray_r",
        origin="upper",
    )

    wx = [
        p[0]
        for p in waffle_path
    ]

    wy = [
        p[1]
        for p in waffle_path
    ]

    rx = [
        p[0]
        for p in ridge_path
    ]

    ry = [
        p[1]
        for p in ridge_path
    ]

    ax.plot(
        wx,
        wy,
        linewidth=2.0,
        label=(
            f"Waffle path "
            f"{waffle_length:.1f} m"
        ),
    )

    ax.plot(
        rx,
        ry,
        linewidth=2.0,
        label=(
            f"Ridgeback path "
            f"{ridge_length:.1f} m"
        ),
    )

    # Locations where Ridgeback physically cannot fit
    # at ANY sampled orientation on the Waffle shortcut.
    if ridge_on_waffle_impossible:

        bad_points = [
            waffle_path[i]
            for i
            in ridge_on_waffle_impossible
        ]

        bx = [
            p[0]
            for p in bad_points
        ]

        by = [
            p[1]
            for p in bad_points
        ]

        ax.scatter(
            bx,
            by,
            s=14,
            marker="x",
            label=(
                "Ridgeback impossible "
                "on Waffle route"
            ),
            zorder=10,
        )

    ax.scatter(
        [START[0]],
        [START[1]],
        s=70,
        marker="o",
        label="Start",
        zorder=11,
    )

    ax.scatter(
        [GOAL[0]],
        [GOAL[1]],
        s=80,
        marker="X",
        label="Goal",
        zorder=11,
    )

    ax.set_title(
        (
            "STech Lab Scenario 01 — "
            "Rectangular Footprint Validation"
        )
    )

    ax.set_xlabel(
        "x [cell]"
    )

    ax.set_ylabel(
        "y [cell]"
    )

    ax.legend(
        fontsize=8,
        loc="best",
    )

    save_path = (
        RESULT_DIR
        / "scenario01_footprint_validation.png"
    )

    fig.savefig(
        save_path,
        dpi=180,
        bbox_inches="tight",
    )

    plt.close(fig)

    # --------------------------------------------------------
    # Save text report
    # --------------------------------------------------------

    report_path = (
        RESULT_DIR
        / "scenario01_footprint_validation.txt"
    )

    with report_path.open(
        "w"
    ) as f:

        f.write(
            "STech Lab Scenario 01\n"
        )

        f.write(
            "Rectangular Footprint Validation\n"
        )

        f.write(
            "=" * 60
            + "\n"
        )

        f.write(
            f"Start: {START}\n"
        )

        f.write(
            f"Goal : {GOAL}\n\n"
        )

        f.write(
            f"Waffle path: "
            f"{waffle_length:.3f} m\n"
        )

        f.write(
            f"Ridgeback path: "
            f"{ridge_length:.3f} m\n"
        )

        f.write(
            f"R/W ratio: "
            f"{ridge_length / waffle_length:.4f}\n\n"
        )

        f.write(
            "Own-path tangent-heading collisions\n"
        )

        f.write(
            f"Waffle: "
            f"{len(waffle_own_collisions)}\n"
        )

        f.write(
            f"Ridgeback: "
            f"{len(ridge_own_collisions)}\n\n"
        )

        f.write(
            "Ridgeback applied to Waffle shortcut\n"
        )

        f.write(
            f"Tangent collisions: "
            f"{len(ridge_on_waffle_tangent)}\n"
        )

        f.write(
            f"No feasible orientation "
            f"({ORIENTATION_STEP_DEG:.1f} deg sampling): "
            f"{len(ridge_on_waffle_impossible)}\n\n"
        )

        f.write(
            "Waffle applied to Ridgeback route\n"
        )

        f.write(
            f"Tangent collisions: "
            f"{len(waffle_on_ridge)}\n"
        )

    print()
    print(
        "Saved:",
        save_path,
    )

    print(
        "Saved:",
        report_path,
    )


if __name__ == "__main__":
    main()