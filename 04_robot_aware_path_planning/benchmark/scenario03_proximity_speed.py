from pathlib import Path
import math

import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import distance_transform_edt, map_coordinates

from benchmark.metric_map_loader import load_ros_metric_map
from benchmark.find_robot_dependent_scenarios import astar
from benchmark.refine_scenario01_paths import simplify_path_los
from benchmark.smooth_scenario01_paths import smooth_polyline
from benchmark.run_scenario02_analysis import (
    ROBOTS,
    resample_path_uniform,
    compute_arc_length,
    compute_curvature,
    speed_limit_from_curvature,
    parameterize_velocity,
    integrate_time,
)


# ============================================================
# SCENARIO 03 — PROXIMITY-AWARE VELOCITY POLICY
# ============================================================
#
# Purpose:
#   Add one common execution policy on top of the existing
#   trajectory model:
#
#       reduce speed near temporary / caution-required objects.
#
#   The planner itself is NOT modified or tuned.
#
# Temporary-object interpretation:
#   The objects are static in this benchmark snapshot, but may
#   move or otherwise justify cautious traversal in operation.
#
# Operating-speed policy:
#
#   This benchmark treats the robots as task-performing mobile
#   platforms operating in spaces shared with people.
#
#   - In open space, cruise speed is limited to 80% of the
#     rated platform maximum to preserve maneuvering margin
#     while maintaining useful task throughput.
#
#   - Near temporary / caution-required objects, translational
#     speed is capped at 0.30 m/s.  The purpose is broader than
#     collision avoidance alone: it provides more reaction
#     margin, reduces the need for abrupt emergency braking,
#     preserves motion stability, and supports predictable
#     operation around people and uncertain objects.
#
#   - Existing curvature and acceleration/deceleration limits
#     remain active.  The final commanded velocity is therefore
#     the minimum of the operating-zone cap and the robot's
#     dynamic/path limits.
#
#   The planner itself is NOT modified or tuned.
# ============================================================


MAP_YAML = Path(
    "data/metric_maps/stech_lab_scenario02/"
    "stech_lab_scenario02.yaml"
)

TEMP_DISTANCE_NPY = Path(
    "results/stech_lab/scenario03/"
    "temporary_objects/"
    "scenario03_temporary_object_distance_m.npy"
)

RESULT_DIR = Path(
    "results/stech_lab/scenario03/"
    "proximity_speed"
)

START = (360, 136)
GOAL = (744, 840)

SAFETY_MARGIN_M = 0.025

OPEN_CRUISE_RATIO = 0.80
CAUTION_SPEED_CAP_MPS = 0.30

# Distance measured from the outside of the robot's effective
# safety envelope to the nearest temporary/caution object.
# Within this band, the reduced-speed operating mode is active.
CAUTION_CLEARANCE_M = 1.00


def circumscribed_radius(
    length_m,
    width_m,
):
    return 0.5 * math.hypot(
        length_m,
        width_m,
    )


def build_safety_cspace(
    raw_free,
    resolution,
    robot,
):
    all_obstacle_clearance_m = (
        distance_transform_edt(
            raw_free
        )
        * resolution
    )

    footprint_radius_m = (
        circumscribed_radius(
            robot["length_m"],
            robot["width_m"],
        )
    )

    effective_radius_m = (
        footprint_radius_m
        + SAFETY_MARGIN_M
    )

    cspace = (
        raw_free
        &
        (
            all_obstacle_clearance_m
            >= effective_radius_m
        )
    )

    return (
        cspace,
        all_obstacle_clearance_m,
        footprint_radius_m,
        effective_radius_m,
    )


def sample_field_bilinear(
    field,
    path_cells,
):
    """
    Bilinear sample of a map field at continuous path coordinates.

    path_cells columns:
        x_cell, y_cell
    """

    path_cells = np.asarray(
        path_cells,
        dtype=float,
    )

    coords = np.vstack(
        (
            path_cells[:, 1],
            path_cells[:, 0],
        )
    )

    return map_coordinates(
        field,
        coords,
        order=1,
        mode="nearest",
    )


def proximity_speed_limit(
    object_center_distance_m,
    effective_radius_m,
    v_max,
):
    """
    Zone-based operating-speed policy.

    Open space:
        v <= 0.80 * platform maximum speed

    Caution zone:
        v <= min(open-space cruise, 0.30 m/s)

    The caution-zone distance is measured from the outside of
    the robot's effective safety envelope to the nearest
    temporary/caution-required object.
    """

    available_m = np.maximum(
        0.0,
        object_center_distance_m
        - effective_radius_m,
    )

    open_cruise_mps = (
        OPEN_CRUISE_RATIO
        * float(v_max)
    )

    caution_cap_mps = min(
        open_cruise_mps,
        CAUTION_SPEED_CAP_MPS,
    )

    in_caution_zone = (
        available_m
        <= CAUTION_CLEARANCE_M
    )

    zone_limit = np.full(
        len(available_m),
        open_cruise_mps,
        dtype=float,
    )

    zone_limit[
        in_caution_zone
    ] = caution_cap_mps

    return (
        zone_limit,
        available_m,
        in_caution_zone,
        open_cruise_mps,
        caution_cap_mps,
    )


def trajectory_hard_validation(
    all_obstacle_clearance_m,
    path_cells,
    effective_radius_m,
):
    sampled = sample_field_bilinear(
        all_obstacle_clearance_m,
        path_cells,
    )

    minimum = float(
        np.min(
            sampled
        )
    )

    feasible = bool(
        np.all(
            sampled
            >= effective_radius_m
        )
    )

    return (
        feasible,
        minimum,
    )


def analyze_robot(
    robot_key,
    grid,
    raw_free,
    resolution,
    temporary_distance_m,
):
    robot = ROBOTS[
        robot_key
    ]

    (
        cspace,
        all_obstacle_clearance_m,
        footprint_radius_m,
        effective_radius_m,
    ) = build_safety_cspace(
        raw_free,
        resolution,
        robot,
    )

    plan = astar(
        cspace,
        START,
        GOAL,
    )

    if plan is None:
        raise RuntimeError(
            f"{robot_key}: no A* path found "
            f"with {SAFETY_MARGIN_M:.3f} m safety margin"
        )

    raw = plan[
        "path"
    ]

    los = simplify_path_los(
        raw,
        cspace,
    )

    smooth = smooth_polyline(
        los,
        cspace,
        resolution,
    )

    resampled = resample_path_uniform(
        smooth,
        spacing_m=resolution,
    )

    (
        points_m,
        ds,
        s,
    ) = compute_arc_length(
        resampled
    )

    curvature = compute_curvature(
        points_m
    )

    curvature_limit = (
        speed_limit_from_curvature(
            robot_key,
            curvature,
        )
    )

    temp_center_distance_m = (
        sample_field_bilinear(
            temporary_distance_m,
            resampled,
        )
    )

    (
        proximity_limit,
        temp_available_clearance_m,
        in_caution_zone,
        open_cruise_mps,
        caution_cap_mps,
    ) = proximity_speed_limit(
        temp_center_distance_m,
        effective_radius_m,
        robot["v_max"],
    )

    combined_limit = np.minimum(
        curvature_limit,
        proximity_limit,
    )

    open_cruise_limit = np.minimum(
        curvature_limit,
        open_cruise_mps,
    )

    nominal_velocity = (
        parameterize_velocity(
            ds,
            open_cruise_limit,
            robot["a_max"],
            robot["decel_max"],
        )
    )

    (
        nominal_dt,
        nominal_time,
    ) = integrate_time(
        ds,
        nominal_velocity,
    )

    safety_velocity = (
        parameterize_velocity(
            ds,
            combined_limit,
            robot["a_max"],
            robot["decel_max"],
        )
    )

    (
        safety_dt,
        safety_time,
    ) = integrate_time(
        ds,
        safety_velocity,
    )

    (
        hard_feasible,
        minimum_all_obstacle_clearance_m,
    ) = trajectory_hard_validation(
        all_obstacle_clearance_m,
        resampled,
        effective_radius_m,
    )

    proximity_active = in_caution_zone

    affected_segments = (
        proximity_active[:-1]
        |
        proximity_active[1:]
    )

    slowdown_distance_m = float(
        np.sum(
            ds[
                affected_segments
            ]
        )
    )

    slowdown_time_s = float(
        np.sum(
            safety_dt[
                affected_segments
            ]
        )
    )

    return {
        "robot_key":
            robot_key,

        "robot_name":
            robot["name"],

        "footprint_radius_m":
            footprint_radius_m,

        "effective_radius_m":
            effective_radius_m,

        "open_cruise_mps":
            open_cruise_mps,

        "caution_cap_mps":
            caution_cap_mps,

        "hard_feasible":
            hard_feasible,

        "minimum_all_obstacle_clearance_m":
            minimum_all_obstacle_clearance_m,

        "raw":
            raw,

        "los":
            los,

        "smooth":
            smooth,

        "resampled":
            resampled,

        "s":
            s,

        "curvature":
            curvature,

        "temp_center_distance_m":
            temp_center_distance_m,

        "temp_available_clearance_m":
            temp_available_clearance_m,

        "curvature_limit":
            curvature_limit,

        "proximity_limit":
            proximity_limit,

        "combined_limit":
            combined_limit,

        "nominal_velocity":
            nominal_velocity,

        "safety_velocity":
            safety_velocity,

        "nominal_time":
            nominal_time,

        "safety_time":
            safety_time,

        "nominal_mission_time_s":
            float(
                nominal_time[-1]
            ),

        "safety_mission_time_s":
            float(
                safety_time[-1]
            ),

        "safety_time_penalty_s":
            float(
                safety_time[-1]
                - nominal_time[-1]
            ),

        "minimum_temporary_center_distance_m":
            float(
                np.min(
                    temp_center_distance_m
                )
            ),

        "minimum_temporary_available_clearance_m":
            float(
                np.min(
                    temp_available_clearance_m
                )
            ),

        "slowdown_distance_m":
            slowdown_distance_m,

        "slowdown_time_s":
            slowdown_time_s,
    }


def save_csv(
    result,
):
    output = np.column_stack(
        (
            result["s"],
            result["curvature"],
            result[
                "temp_center_distance_m"
            ],
            result[
                "temp_available_clearance_m"
            ],
            result[
                "curvature_limit"
            ],
            result[
                "proximity_limit"
            ],
            result[
                "combined_limit"
            ],
            result[
                "nominal_velocity"
            ],
            result[
                "safety_velocity"
            ],
            result[
                "nominal_time"
            ],
            result[
                "safety_time"
            ],
        )
    )

    path = (
        RESULT_DIR
        /
        (
            "scenario03_"
            f"{result['robot_key']}_"
            "proximity_speed_profile.csv"
        )
    )

    np.savetxt(
        path,
        output,
        delimiter=",",
        header=(
            "s_m,"
            "curvature_1pm,"
            "temporary_center_distance_m,"
            "temporary_available_clearance_m,"
            "curvature_limit_mps,"
            "operating_zone_limit_mps,"
            "combined_limit_mps,"
            "nominal_velocity_mps,"
            "safety_velocity_mps,"
            "nominal_time_s,"
            "safety_time_s"
        ),
        comments="",
    )

    return path


def save_velocity_figure(
    results,
):
    fig, axes = plt.subplots(
        2,
        1,
        figsize=(11, 10),
        sharex=False,
    )

    for ax, result in zip(
        axes,
        results,
    ):
        ax.plot(
            result["s"],
            result[
                "curvature_limit"
            ],
            linestyle="--",
            linewidth=1.1,
            label="Curvature-based limit",
        )

        ax.plot(
            result["s"],
            result[
                "proximity_limit"
            ],
            linestyle=":",
            linewidth=1.4,
            label="Operating-zone speed limit",
        )

        ax.plot(
            result["s"],
            result[
                "safety_velocity"
            ],
            linewidth=2.0,
            label="Final safety-aware velocity",
        )

        ax.plot(
            result["s"],
            result[
                "nominal_velocity"
            ],
            linewidth=1.1,
            alpha=0.65,
            label="Nominal velocity",
        )

        ax.set_ylabel(
            "Velocity [m/s]"
        )

        ax.set_title(
            result[
                "robot_name"
            ]
        )

        ax.grid(
            True,
            alpha=0.25,
        )

        ax.legend(
            fontsize=8,
        )

    axes[-1].set_xlabel(
        "Path distance [m]"
    )

    fig.suptitle(
        (
            "Scenario 03 — "
            "Human-Environment Operating Speed Policy"
        ),
        fontsize=15,
    )

    fig.tight_layout()

    path = (
        RESULT_DIR
        /
        "scenario03_proximity_velocity_profiles.png"
    )

    fig.savefig(
        path,
        dpi=180,
        bbox_inches="tight",
    )

    plt.close(
        fig
    )

    return path


def save_clearance_figure(
    results,
):
    fig, axes = plt.subplots(
        2,
        1,
        figsize=(11, 9),
        sharex=False,
    )

    for ax, result in zip(
        axes,
        results,
    ):
        ax.plot(
            result["s"],
            result[
                "temp_available_clearance_m"
            ],
            linewidth=1.6,
            label=(
                "Available clearance outside "
                "effective robot envelope"
            ),
        )

        ax.set_ylabel(
            "Clearance [m]"
        )

        ax.set_title(
            result[
                "robot_name"
            ]
        )

        ax.grid(
            True,
            alpha=0.25,
        )

        ax.legend(
            fontsize=8,
        )

    axes[-1].set_xlabel(
        "Path distance [m]"
    )

    fig.suptitle(
        (
            "Scenario 03 — "
            "Temporary-Object Clearance Along Path"
        ),
        fontsize=15,
    )

    fig.tight_layout()

    path = (
        RESULT_DIR
        /
        "scenario03_temporary_clearance_profiles.png"
    )

    fig.savefig(
        path,
        dpi=180,
        bbox_inches="tight",
    )

    plt.close(
        fig
    )

    return path


def print_summary(
    results,
):
    print()
    print(
        "=" * 94
    )

    print(
        "SCENARIO 03 — HUMAN-ENVIRONMENT VELOCITY POLICY"
    )

    print(
        "=" * 94
    )

    print(
        f"Safety margin              : "
        f"{SAFETY_MARGIN_M:.3f} m"
    )

    print(
        f"Open-space cruise          : "
        f"{OPEN_CRUISE_RATIO * 100:.0f}% of rated v_max"
    )

    print(
        f"Caution speed cap          : "
        f"{CAUTION_SPEED_CAP_MPS:.2f} m/s"
    )

    print(
        f"Caution clearance band     : "
        f"{CAUTION_CLEARANCE_M:.2f} m "
        f"outside effective envelope"
    )

    for result in results:
        print()
        print(
            result[
                "robot_name"
            ]
        )

        print(
            "-" * 60
        )

        print(
            f"Effective radius           : "
            f"{result['effective_radius_m']:.3f} m"
        )

        print(
            f"Open-space cruise          : "
            f"{result['open_cruise_mps']:.3f} m/s"
        )

        print(
            f"Robot caution cap          : "
            f"{result['caution_cap_mps']:.3f} m/s"
        )

        print(
            f"Hard trajectory feasible   : "
            f"{'YES' if result['hard_feasible'] else 'NO'}"
        )

        print(
            f"Minimum all-obstacle clr.  : "
            f"{result['minimum_all_obstacle_clearance_m']:.3f} m"
        )

        print(
            f"Min temp center distance   : "
            f"{result['minimum_temporary_center_distance_m']:.3f} m"
        )

        print(
            f"Min temp available clr.    : "
            f"{result['minimum_temporary_available_clearance_m']:.3f} m"
        )

        print(
            f"Slowdown-zone distance     : "
            f"{result['slowdown_distance_m']:.2f} m"
        )

        print(
            f"Slowdown-zone time         : "
            f"{result['slowdown_time_s']:.2f} s"
        )

        print(
            f"Nominal mission time       : "
            f"{result['nominal_mission_time_s']:.3f} s"
        )

        print(
            f"Safety-aware mission time  : "
            f"{result['safety_mission_time_s']:.3f} s"
        )

        print(
            f"Added time from policy     : "
            f"{result['safety_time_penalty_s']:.3f} s"
        )

        if not result[
            "hard_feasible"
        ]:
            print(
                "NOTE: Mission time is a velocity-policy "
                "diagnostic only because the current smoothed "
                "candidate violates the hard clearance check."
            )


def main():
    RESULT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    if not TEMP_DISTANCE_NPY.exists():
        raise FileNotFoundError(
            "Temporary-object distance field not found:\n"
            f"{TEMP_DISTANCE_NPY}\n"
            "Run benchmark.scenario03_temporary_objects first."
        )

    metric_map = (
        load_ros_metric_map(
            MAP_YAML
        )
    )

    grid = metric_map[
        "grid"
    ]

    resolution = float(
        metric_map[
            "resolution"
        ]
    )

    raw_free = (
        grid == 0
    )

    temporary_distance_m = np.load(
        TEMP_DISTANCE_NPY
    )

    if (
        temporary_distance_m.shape
        != grid.shape
    ):
        raise RuntimeError(
            "Temporary-object distance field shape does not "
            "match Scenario 03 map."
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
                resolution,
                temporary_distance_m,
            )
        )

    print_summary(
        results
    )

    csv_paths = [
        save_csv(
            result
        )
        for result in results
    ]

    velocity_figure = (
        save_velocity_figure(
            results
        )
    )

    clearance_figure = (
        save_clearance_figure(
            results
        )
    )

    print()
    print(
        "Saved:"
    )

    for path in csv_paths:
        print(
            " ",
            path,
        )

    print(
        " ",
        velocity_figure,
    )

    print(
        " ",
        clearance_figure,
    )


if __name__ == "__main__":
    main()