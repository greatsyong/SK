from pathlib import Path
import math

import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import distance_transform_edt

from benchmark.metric_map_loader import load_ros_metric_map
from benchmark.find_robot_dependent_scenarios import astar
from benchmark.refine_scenario01_paths import simplify_path_los


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
        "length_m": 0.281,
        "width_m": 0.306,
    },
    "ridgeback": {
        "length_m": 0.960,
        "width_m": 0.793,
    },
}

# Fraction of the shorter adjacent segment initially used
# for corner rounding.
INITIAL_CORNER_FRACTION = 0.30

# Reduce smoothing distance if collision occurs.
SHRINK_FACTOR = 0.70

# Stop attempting to smooth below this distance.
MIN_CORNER_DISTANCE_M = 0.05

# Bézier samples per corner.
CURVE_SAMPLES = 30


# ============================================================
# GEOMETRY
# ============================================================

def circumscribed_radius(length_m, width_m):
    return 0.5 * math.hypot(
        length_m,
        width_m,
    )


def unit_vector(a, b):
    v = np.asarray(
        b,
        dtype=float,
    ) - np.asarray(
        a,
        dtype=float,
    )

    norm = np.linalg.norm(v)

    if norm == 0.0:
        return np.zeros(2)

    return v / norm


def distance_cells(a, b):
    return float(
        np.linalg.norm(
            np.asarray(a, dtype=float)
            -
            np.asarray(b, dtype=float)
        )
    )


def path_length_m(path, resolution):

    points = np.asarray(
        path,
        dtype=float,
    )

    if len(points) < 2:
        return 0.0

    diff = np.diff(
        points,
        axis=0,
    )

    return float(
        np.sum(
            np.linalg.norm(
                diff,
                axis=1,
            )
        )
        * resolution
    )


# ============================================================
# COLLISION CHECKING
# ============================================================

def point_free(
    free_grid,
    point,
):

    x = int(
        round(point[0])
    )

    y = int(
        round(point[1])
    )

    if not (
        0 <= x < free_grid.shape[1]
        and
        0 <= y < free_grid.shape[0]
    ):
        return False

    return bool(
        free_grid[y, x]
    )


def path_free(
    free_grid,
    points,
):

    for p in points:

        if not point_free(
            free_grid,
            p,
        ):
            return False

    return True


# ============================================================
# QUADRATIC BEZIER
# ============================================================

def quadratic_bezier(
    p0,
    p1,
    p2,
    samples,
):

    p0 = np.asarray(
        p0,
        dtype=float,
    )

    p1 = np.asarray(
        p1,
        dtype=float,
    )

    p2 = np.asarray(
        p2,
        dtype=float,
    )

    ts = np.linspace(
        0.0,
        1.0,
        samples,
    )

    curve = []

    for t in ts:

        p = (
            (1.0 - t) ** 2
            * p0
            +
            2.0
            * (1.0 - t)
            * t
            * p1
            +
            t ** 2
            * p2
        )

        curve.append(
            tuple(p)
        )

    return curve


# ============================================================
# CORNER SMOOTHING
# ============================================================

def smooth_polyline(
    path,
    free_grid,
    resolution,
):

    if len(path) <= 2:
        return [
            tuple(map(float, p))
            for p in path
        ]

    output = [
        tuple(
            map(
                float,
                path[0],
            )
        )
    ]

    for i in range(
        1,
        len(path) - 1,
    ):

        previous = np.asarray(
            path[i - 1],
            dtype=float,
        )

        corner = np.asarray(
            path[i],
            dtype=float,
        )

        following = np.asarray(
            path[i + 1],
            dtype=float,
        )

        len_before_cells = (
            distance_cells(
                previous,
                corner,
            )
        )

        len_after_cells = (
            distance_cells(
                corner,
                following,
            )
        )

        shorter_cells = min(
            len_before_cells,
            len_after_cells,
        )

        smoothing_cells = (
            INITIAL_CORNER_FRACTION
            * shorter_cells
        )

        min_cells = (
            MIN_CORNER_DISTANCE_M
            / resolution
        )

        incoming = (
            unit_vector(
                corner,
                previous,
            )
        )

        outgoing = (
            unit_vector(
                corner,
                following,
            )
        )

        accepted_curve = None

        while (
            smoothing_cells
            >= min_cells
        ):

            entry = (
                corner
                +
                incoming
                * smoothing_cells
            )

            exit_point = (
                corner
                +
                outgoing
                * smoothing_cells
            )

            curve = quadratic_bezier(
                entry,
                corner,
                exit_point,
                CURVE_SAMPLES,
            )

            if path_free(
                free_grid,
                curve,
            ):

                accepted_curve = (
                    curve
                )

                break

            smoothing_cells *= (
                SHRINK_FACTOR
            )

        # ----------------------------------------------------
        # Could not smooth safely
        # ----------------------------------------------------

        if accepted_curve is None:

            output.append(
                tuple(corner)
            )

            continue

        # ----------------------------------------------------
        # Add collision-free rounded corner
        # ----------------------------------------------------

        entry = (
            accepted_curve[0]
        )

        if (
            distance_cells(
                output[-1],
                entry,
            )
            > 1e-6
        ):
            output.append(
                entry
            )

        output.extend(
            accepted_curve[1:]
        )

    final_point = tuple(
        map(
            float,
            path[-1],
        )
    )

    if (
        distance_cells(
            output[-1],
            final_point,
        )
        > 1e-6
    ):
        output.append(
            final_point
        )

    return output


# ============================================================
# CURVATURE
# ============================================================

def discrete_curvature(
    points,
    resolution,
):

    points = np.asarray(
        points,
        dtype=float,
    ) * resolution

    curvature = np.zeros(
        len(points),
        dtype=float,
    )

    for i in range(
        1,
        len(points) - 1,
    ):

        p0 = points[
            i - 1
        ]

        p1 = points[
            i
        ]

        p2 = points[
            i + 1
        ]

        a = np.linalg.norm(
            p1 - p0
        )

        b = np.linalg.norm(
            p2 - p1
        )

        c = np.linalg.norm(
            p2 - p0
        )

        if (
            a < 1e-9
            or
            b < 1e-9
            or
            c < 1e-9
        ):
            continue

        cross = abs(
            np.cross(
                p1 - p0,
                p2 - p0,
            )
        )

        curvature[i] = (
            2.0
            * cross
            /
            (
                a
                * b
                * c
            )
        )

    return curvature


# ============================================================
# MAIN
# ============================================================

def main():

    RESULT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

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

    cspaces = {}

    for key, robot in ROBOTS.items():

        radius = (
            circumscribed_radius(
                robot["length_m"],
                robot["width_m"],
            )
        )

        cspaces[key] = (
            raw_free
            &
            (
                clearance_m
                >= radius
            )
        )

    results = {}

    for key in [
        "waffle",
        "ridgeback",
    ]:

        plan = astar(
            cspaces[key],
            START,
            GOAL,
        )

        if plan is None:
            raise RuntimeError(
                f"{key} path missing."
            )

        los = simplify_path_los(
            plan["path"],
            cspaces[key],
        )

        smooth = smooth_polyline(
            los,
            cspaces[key],
            resolution,
        )

        curvature = (
            discrete_curvature(
                smooth,
                resolution,
            )
        )

        results[key] = {
            "raw":
                plan["path"],
            "los":
                los,
            "smooth":
                smooth,
            "length_los":
                path_length_m(
                    los,
                    resolution,
                ),
            "length_smooth":
                path_length_m(
                    smooth,
                    resolution,
                ),
            "max_curvature":
                float(
                    np.max(
                        curvature
                    )
                ),
            "curvature":
                curvature,
        }

    # ========================================================
    # PRINT
    # ========================================================

    print()
    print("=" * 84)
    print(
        "SCENARIO 01 — COLLISION-AWARE CORNER SMOOTHING"
    )
    print("=" * 84)

    for key in [
        "waffle",
        "ridgeback",
    ]:

        r = results[key]

        print()
        print(
            key.upper()
        )
        print(
            "-" * 40
        )

        print(
            f"LOS waypoints       : "
            f"{len(r['los'])}"
        )

        print(
            f"Smooth samples      : "
            f"{len(r['smooth'])}"
        )

        print(
            f"LOS length          : "
            f"{r['length_los']:.3f} m"
        )

        print(
            f"Smoothed length     : "
            f"{r['length_smooth']:.3f} m"
        )

        print(
            f"Max curvature       : "
            f"{r['max_curvature']:.3f} 1/m"
        )

        if (
            r["max_curvature"]
            > 1e-9
        ):

            print(
                f"Min curvature radius: "
                f"{1.0 / r['max_curvature']:.3f} m"
            )

    # ========================================================
    # VISUALIZATION
    # ========================================================

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(16, 8),
    )

    for ax, key in zip(
        axes,
        [
            "waffle",
            "ridgeback",
        ],
    ):

        r = results[key]

        ax.imshow(
            grid,
            cmap="gray_r",
            origin="upper",
        )

        los = np.asarray(
            r["los"]
        )

        smooth = np.asarray(
            r["smooth"]
        )

        ax.plot(
            los[:, 0],
            los[:, 1],
            linewidth=1.0,
            alpha=0.6,
            marker="o",
            markersize=3,
            label="LOS path",
        )

        ax.plot(
            smooth[:, 0],
            smooth[:, 1],
            linewidth=2.0,
            label="Collision-aware smoothed path",
        )

        ax.scatter(
            [START[0]],
            [START[1]],
            s=50,
            marker="o",
            label="Start",
        )

        ax.scatter(
            [GOAL[0]],
            [GOAL[1]],
            s=60,
            marker="X",
            label="Goal",
        )

        ax.set_title(
            key.capitalize()
        )

        ax.set_xticks([])
        ax.set_yticks([])

        ax.legend(
            fontsize=8,
        )

    fig.suptitle(
        "STech Lab Scenario 01 — "
        "Collision-Aware Path Smoothing",
        fontsize=15,
    )

    fig.tight_layout()

    plot_path = (
        RESULT_DIR
        /
        "scenario01_smoothed_paths.png"
    )

    fig.savefig(
        plot_path,
        dpi=180,
        bbox_inches="tight",
    )

    plt.close(fig)

    # ========================================================
    # SAVE NUMERIC TRAJECTORIES
    # ========================================================

    for key in [
        "waffle",
        "ridgeback",
    ]:

        output_path = (
            RESULT_DIR
            /
            f"scenario01_{key}_smoothed.csv"
        )

        data = np.asarray(
            results[key][
                "smooth"
            ]
        )

        np.savetxt(
            output_path,
            data,
            delimiter=",",
            header="x_cell,y_cell",
            comments="",
        )

        print(
            "Saved:",
            output_path,
        )

    print(
        "Saved:",
        plot_path,
    )


if __name__ == "__main__":
    main()