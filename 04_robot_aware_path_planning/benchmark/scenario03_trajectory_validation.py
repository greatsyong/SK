from pathlib import Path
import math

import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import distance_transform_edt

from benchmark.metric_map_loader import load_ros_metric_map
from benchmark.find_robot_dependent_scenarios import astar
from benchmark.refine_scenario01_paths import simplify_path_los
from benchmark.smooth_scenario01_paths import smooth_polyline


# ============================================================
# SCENARIO 03 — POST-PROCESSING TRAJECTORY VALIDATION
# ============================================================
#
# Purpose:
#   Verify that a planner path remains executable after
#   LOS refinement and collision-aware smoothing.
#
# Common physical condition:
#   robot circumscribed radius + 0.025 m safety margin
#
# This script uses A* only as a pipeline check.
# It does NOT compare planners yet.
# ============================================================


MAP_YAML = Path(
    "data/metric_maps/stech_lab_scenario02/"
    "stech_lab_scenario02.yaml"
)

RESULT_DIR = Path(
    "results/stech_lab/scenario03/"
    "trajectory_validation"
)

START = (360, 136)
GOAL = (744, 840)

SAFETY_MARGIN_M = 0.025
VALIDATION_SPACING_M = 0.01

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


def circumscribed_radius(length_m, width_m):
    return 0.5 * math.hypot(
        length_m,
        width_m,
    )


def path_length_m(points, resolution):
    points = np.asarray(
        points,
        dtype=float,
    )

    if len(points) < 2:
        return 0.0

    delta = np.diff(
        points,
        axis=0,
    )

    return float(
        np.sum(
            np.linalg.norm(
                delta,
                axis=1,
            )
        )
        * resolution
    )


def dense_sample_path(
    points,
    resolution,
    spacing_m=VALIDATION_SPACING_M,
):
    """
    Densely sample every path segment at approximately spacing_m.
    Input/output coordinates remain in map-cell coordinates.
    """

    points = np.asarray(
        points,
        dtype=float,
    )

    if len(points) == 0:
        return np.empty(
            (0, 2),
            dtype=float,
        )

    if len(points) == 1:
        return points.copy()

    samples = [
        points[0]
    ]

    spacing_cells = (
        spacing_m
        / resolution
    )

    for p0, p1 in zip(
        points[:-1],
        points[1:],
    ):
        delta = (
            p1 - p0
        )

        length_cells = float(
            np.linalg.norm(
                delta
            )
        )

        if length_cells < 1e-12:
            continue

        n = max(
            1,
            int(
                math.ceil(
                    length_cells
                    / spacing_cells
                )
            ),
        )

        for i in range(
            1,
            n + 1,
        ):
            alpha = (
                i / n
            )

            samples.append(
                p0
                + alpha
                * delta
            )

    return np.asarray(
        samples,
        dtype=float,
    )


def sample_clearance_nearest(
    clearance_m,
    samples,
):
    """
    Sample the discrete metric clearance map at the nearest cell.
    This keeps the validation consistent with the grid-based
    configuration-space definition used by the planners.
    """

    x = np.rint(
        samples[:, 0]
    ).astype(int)

    y = np.rint(
        samples[:, 1]
    ).astype(int)

    inside = (
        (x >= 0)
        & (x < clearance_m.shape[1])
        & (y >= 0)
        & (y < clearance_m.shape[0])
    )

    values = np.full(
        len(samples),
        -math.inf,
        dtype=float,
    )

    values[
        inside
    ] = clearance_m[
        y[inside],
        x[inside],
    ]

    return (
        values,
        inside,
    )


def validate_path(
    points,
    clearance_m,
    required_clearance_m,
    resolution,
):
    dense = dense_sample_path(
        points,
        resolution,
    )

    values, inside = (
        sample_clearance_nearest(
            clearance_m,
            dense,
        )
    )

    feasible_mask = (
        inside
        &
        (
            values
            >= required_clearance_m
        )
    )

    invalid_mask = (
        ~feasible_mask
    )

    if len(values) == 0:
        min_clearance_m = math.nan
        feasible = False
    else:
        finite_values = values[
            np.isfinite(
                values
            )
        ]

        min_clearance_m = (
            float(
                np.min(
                    finite_values
                )
            )
            if len(finite_values)
            else math.nan
        )

        feasible = bool(
            np.all(
                feasible_mask
            )
        )

    return {
        "feasible":
            feasible,

        "dense_samples":
            dense,

        "clearance_values_m":
            values,

        "invalid_mask":
            invalid_mask,

        "invalid_samples":
            int(
                np.count_nonzero(
                    invalid_mask
                )
            ),

        "sample_count":
            int(
                len(
                    dense
                )
            ),

        "min_clearance_m":
            min_clearance_m,
    }


def analyze_robot(
    robot_key,
    grid,
    raw_free,
    clearance_m,
    resolution,
):
    robot = ROBOTS[
        robot_key
    ]

    footprint_radius_m = (
        circumscribed_radius(
            robot[
                "length_m"
            ],
            robot[
                "width_m"
            ],
        )
    )

    required_clearance_m = (
        footprint_radius_m
        + SAFETY_MARGIN_M
    )

    cspace = (
        raw_free
        &
        (
            clearance_m
            >= required_clearance_m
        )
    )

    plan = astar(
        cspace,
        START,
        GOAL,
    )

    if plan is None:
        return {
            "robot_key":
                robot_key,

            "robot_name":
                robot["name"],

            "planner_success":
                False,

            "required_clearance_m":
                required_clearance_m,

            "cspace":
                cspace,
        }

    raw = (
        plan[
            "path"
        ]
    )

    los = simplify_path_los(
        raw,
        cspace,
    )

    smooth = smooth_polyline(
        los,
        cspace,
        resolution,
    )

    raw_validation = (
        validate_path(
            raw,
            clearance_m,
            required_clearance_m,
            resolution,
        )
    )

    los_validation = (
        validate_path(
            los,
            clearance_m,
            required_clearance_m,
            resolution,
        )
    )

    smooth_validation = (
        validate_path(
            smooth,
            clearance_m,
            required_clearance_m,
            resolution,
        )
    )

    return {
        "robot_key":
            robot_key,

        "robot_name":
            robot["name"],

        "planner_success":
            True,

        "footprint_radius_m":
            footprint_radius_m,

        "required_clearance_m":
            required_clearance_m,

        "cspace":
            cspace,

        "raw":
            raw,

        "los":
            los,

        "smooth":
            smooth,

        "raw_length_m":
            path_length_m(
                raw,
                resolution,
            ),

        "los_length_m":
            path_length_m(
                los,
                resolution,
            ),

        "smooth_length_m":
            path_length_m(
                smooth,
                resolution,
            ),

        "raw_validation":
            raw_validation,

        "los_validation":
            los_validation,

        "smooth_validation":
            smooth_validation,
    }


def print_stage(
    name,
    validation,
):
    status = (
        "PASS"
        if validation[
            "feasible"
        ]
        else "FAIL"
    )

    print(
        f"{name:<10} | "
        f"{status:<4} | "
        f"min clearance="
        f"{validation['min_clearance_m']:.3f} m | "
        f"invalid="
        f"{validation['invalid_samples']}/"
        f"{validation['sample_count']}"
    )


def print_summary(results):
    print()
    print(
        "=" * 92
    )

    print(
        "SCENARIO 03 — POST-PROCESSING TRAJECTORY VALIDATION"
    )

    print(
        "=" * 92
    )

    print(
        f"Safety margin              : "
        f"{SAFETY_MARGIN_M:.3f} m"
    )

    print(
        f"Validation spacing         : "
        f"{VALIDATION_SPACING_M:.3f} m"
    )

    for result in results:
        print()
        print(
            result[
                "robot_name"
            ]
        )

        print(
            "-" * 64
        )

        if not result[
            "planner_success"
        ]:
            print(
                "A* planner result          : NO PATH"
            )
            continue

        print(
            f"Required center clearance  : "
            f"{result['required_clearance_m']:.3f} m"
        )

        print(
            f"Raw path length            : "
            f"{result['raw_length_m']:.2f} m"
        )

        print(
            f"LOS path length            : "
            f"{result['los_length_m']:.2f} m"
        )

        print(
            f"Smoothed path length       : "
            f"{result['smooth_length_m']:.2f} m"
        )

        print()

        print_stage(
            "RAW",
            result[
                "raw_validation"
            ],
        )

        print_stage(
            "LOS",
            result[
                "los_validation"
            ],
        )

        print_stage(
            "SMOOTH",
            result[
                "smooth_validation"
            ],
        )


def save_figure(
    grid,
    results,
):
    RESULT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    fig, axes = plt.subplots(
        1,
        len(
            results
        ),
        figsize=(16, 8),
    )

    if len(
        results
    ) == 1:
        axes = [
            axes
        ]

    for ax, result in zip(
        axes,
        results,
    ):
        ax.imshow(
            grid,
            cmap="gray_r",
            origin="upper",
        )

        if result[
            "planner_success"
        ]:
            raw = np.asarray(
                result[
                    "raw"
                ],
                dtype=float,
            )

            los = np.asarray(
                result[
                    "los"
                ],
                dtype=float,
            )

            smooth = np.asarray(
                result[
                    "smooth"
                ],
                dtype=float,
            )

            ax.plot(
                raw[:, 0],
                raw[:, 1],
                linewidth=0.7,
                alpha=0.35,
                label="Raw A*",
            )

            ax.plot(
                los[:, 0],
                los[:, 1],
                linewidth=1.0,
                alpha=0.65,
                label="LOS",
            )

            ax.plot(
                smooth[:, 0],
                smooth[:, 1],
                linewidth=2.0,
                label="Smoothed",
            )

            validation = (
                result[
                    "smooth_validation"
                ]
            )

            invalid = (
                validation[
                    "dense_samples"
                ][
                    validation[
                        "invalid_mask"
                    ]
                ]
            )

            if len(
                invalid
            ) > 0:
                ax.scatter(
                    invalid[:, 0],
                    invalid[:, 1],
                    s=9,
                    marker="x",
                    label="Invalid samples",
                )

            final_status = (
                "EXECUTABLE"
                if validation[
                    "feasible"
                ]
                else "INFEASIBLE"
            )

        else:
            final_status = (
                "NO PLANNER PATH"
            )

        ax.scatter(
            [START[0]],
            [START[1]],
            s=55,
            marker="o",
            label="Start",
        )

        ax.scatter(
            [GOAL[0]],
            [GOAL[1]],
            s=65,
            marker="X",
            label="Goal",
        )

        ax.set_title(
            (
                f"{result['robot_name']}\n"
                f"{final_status}"
            )
        )

        ax.set_xticks([])
        ax.set_yticks([])

        ax.legend(
            fontsize=8,
        )

    fig.suptitle(
        (
            "Scenario 03 — A* Pipeline Check with "
            "0.025 m Safety Margin"
        ),
        fontsize=15,
    )

    fig.tight_layout()

    output = (
        RESULT_DIR
        / "scenario03_astar_trajectory_validation.png"
    )

    fig.savefig(
        output,
        dpi=180,
        bbox_inches="tight",
    )

    plt.close(
        fig
    )

    return output


def main():
    RESULT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    metric_map = (
        load_ros_metric_map(
            MAP_YAML
        )
    )

    grid = (
        metric_map[
            "grid"
        ]
    )

    resolution = float(
        metric_map[
            "resolution"
        ]
    )

    raw_free = (
        grid == 0
    )

    clearance_m = (
        distance_transform_edt(
            raw_free
        )
        * resolution
    )

    results = []

    for robot_key in [
        "waffle",
        "ridgeback",
    ]:
        results.append(
            analyze_robot(
                robot_key,
                grid,
                raw_free,
                clearance_m,
                resolution,
            )
        )

    print_summary(
        results
    )

    figure_path = save_figure(
        grid,
        results,
    )

    print()
    print(
        "Saved:",
        figure_path,
    )


if __name__ == "__main__":
    main()