from pathlib import Path
import math

import matplotlib.pyplot as plt
import numpy as np

from scipy.ndimage import distance_transform_edt

from benchmark.metric_map_loader import load_ros_metric_map
from benchmark.find_robot_dependent_scenarios import astar
from benchmark.refine_scenario01_paths import simplify_path_los
from benchmark.smooth_scenario01_paths import (
    smooth_polyline,
    discrete_curvature,
)


# ============================================================
# CONFIG
# ============================================================

MAP_YAML = Path(
    "data/metric_maps/stech_lab_scenario02/stech_lab_scenario02.yaml"
)

RESULT_DIR = Path(
    "results/stech_lab/scenario02"
)

START = (360, 136)
GOAL = (744, 840)

RESOLUTION = 0.05


ROBOTS = {

    "waffle": {
        "name":
            "TurtleBot3 Waffle Pi",

        "length_m":
            0.281,

        "width_m":
            0.306,

        "v_max":
            0.22,

        "a_max":
            2.5,

        "decel_max":
            2.5,

        "omega_max":
            1.0,
    },

    "ridgeback": {
        "name":
            "Clearpath Ridgeback",

        "length_m":
            0.960,

        "width_m":
            0.793,

        "v_max":
            1.10,

        "a_max":
            1.0,

        "decel_max":
            1.0,

        "lateral_accel_max":
            1.0,
    },
}


# ============================================================
# GEOMETRY
# ============================================================

def circumscribed_radius(
    length_m,
    width_m,
):
    return 0.5 * math.hypot(
        length_m,
        width_m,
    )


def path_length_m(
    points,
    resolution,
):
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


# ============================================================
# UNIFORM RESAMPLING
# ============================================================

def resample_path_uniform(
    points,
    spacing_m=RESOLUTION,
):
    points_m = (
        np.asarray(
            points,
            dtype=float,
        )
        * RESOLUTION
    )

    delta = np.diff(
        points_m,
        axis=0,
    )

    segment_lengths = np.linalg.norm(
        delta,
        axis=1,
    )

    cumulative = np.concatenate(
        (
            [0.0],
            np.cumsum(
                segment_lengths
            ),
        )
    )

    total_length = (
        cumulative[-1]
    )

    samples = np.arange(
        0.0,
        total_length,
        spacing_m,
    )

    if (
        len(samples) == 0
        or
        samples[-1] < total_length
    ):
        samples = np.append(
            samples,
            total_length,
        )

    x = np.interp(
        samples,
        cumulative,
        points_m[:, 0],
    )

    y = np.interp(
        samples,
        cumulative,
        points_m[:, 1],
    )

    return np.column_stack(
        (
            x / RESOLUTION,
            y / RESOLUTION,
        )
    )


# ============================================================
# ARC LENGTH
# ============================================================

def compute_arc_length(
    points,
):
    points_m = (
        np.asarray(
            points,
            dtype=float,
        )
        * RESOLUTION
    )

    delta = np.diff(
        points_m,
        axis=0,
    )

    ds = np.linalg.norm(
        delta,
        axis=1,
    )

    s = np.concatenate(
        (
            [0.0],
            np.cumsum(
                ds
            ),
        )
    )

    return (
        points_m,
        ds,
        s,
    )


# ============================================================
# CURVATURE
# ============================================================

def compute_curvature(
    points_m,
):
    n = len(
        points_m
    )

    curvature = np.zeros(
        n,
        dtype=float,
    )

    for i in range(
        1,
        n - 1,
    ):

        p0 = points_m[
            i - 1
        ]

        p1 = points_m[
            i
        ]

        p2 = points_m[
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
            a < 1e-10
            or
            b < 1e-10
            or
            c < 1e-10
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
# SPEED LIMIT
# ============================================================

def speed_limit_from_curvature(
    robot_key,
    curvature,
):
    robot = ROBOTS[
        robot_key
    ]

    limits = np.full(
        len(curvature),
        robot[
            "v_max"
        ],
        dtype=float,
    )

    mask = (
        curvature
        > 1e-9
    )

    if robot_key == "waffle":

        limits[
            mask
        ] = np.minimum(
            robot[
                "v_max"
            ],
            robot[
                "omega_max"
            ]
            /
            curvature[
                mask
            ],
        )

    elif robot_key == "ridgeback":

        curve_limit = np.sqrt(
            robot[
                "lateral_accel_max"
            ]
            /
            curvature[
                mask
            ]
        )

        limits[
            mask
        ] = np.minimum(
            robot[
                "v_max"
            ],
            curve_limit,
        )

    return limits


# ============================================================
# VELOCITY PARAMETERIZATION
# ============================================================

def parameterize_velocity(
    ds,
    speed_limit,
    accel,
    decel,
):
    velocity = np.array(
        speed_limit,
        dtype=float,
    )

    velocity[0] = 0.0

    # forward pass
    for i in range(
        len(velocity) - 1
    ):

        reachable = math.sqrt(
            max(
                0.0,
                velocity[i] ** 2
                +
                2.0
                * accel
                * ds[i],
            )
        )

        velocity[
            i + 1
        ] = min(
            velocity[
                i + 1
            ],
            reachable,
        )

    velocity[-1] = 0.0

    # backward pass
    for i in range(
        len(velocity) - 2,
        -1,
        -1,
    ):

        reachable = math.sqrt(
            max(
                0.0,
                velocity[
                    i + 1
                ] ** 2
                +
                2.0
                * decel
                * ds[i],
            )
        )

        velocity[i] = min(
            velocity[i],
            reachable,
        )

    return velocity


# ============================================================
# TIME INTEGRATION
# ============================================================

def integrate_time(
    ds,
    velocity,
):
    dt = np.zeros(
        len(ds),
        dtype=float,
    )

    for i in range(
        len(ds)
    ):

        v0 = velocity[i]
        v1 = velocity[
            i + 1
        ]

        average_velocity = (
            0.5
            * (
                v0
                +
                v1
            )
        )

        if average_velocity < 1e-9:

            if ds[i] > 1e-9:
                raise RuntimeError(
                    "Zero velocity on non-zero segment."
                )

            dt[i] = 0.0

        else:

            dt[i] = (
                ds[i]
                /
                average_velocity
            )

    time = np.concatenate(
        (
            [0.0],
            np.cumsum(
                dt
            ),
        )
    )

    return (
        dt,
        time,
    )


# ============================================================
# ANALYSIS
# ============================================================

def analyze_robot(
    robot_key,
    grid,
    raw_free,
):
    robot = ROBOTS[
        robot_key
    ]

    clearance_m = (
        distance_transform_edt(
            raw_free
        )
        * RESOLUTION
    )

    radius = (
        circumscribed_radius(
            robot[
                "length_m"
            ],
            robot[
                "width_m"
            ],
        )
    )

    cspace = (
        raw_free
        &
        (
            clearance_m
            >= radius
        )
    )

    plan = astar(
        cspace,
        START,
        GOAL,
    )

    if plan is None:
        raise RuntimeError(
            f"{robot_key}: no path found"
        )

    los = simplify_path_los(
        plan[
            "path"
        ],
        cspace,
    )

    smooth = smooth_polyline(
        los,
        cspace,
        RESOLUTION,
    )

    resampled = resample_path_uniform(
        smooth
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

    speed_limit = (
        speed_limit_from_curvature(
            robot_key,
            curvature,
        )
    )

    velocity = (
        parameterize_velocity(
            ds,
            speed_limit,
            robot[
                "a_max"
            ],
            robot[
                "decel_max"
            ],
        )
    )

    (
        dt,
        time,
    ) = integrate_time(
        ds,
        velocity,
    )

    max_curvature = float(
        np.max(
            curvature
        )
    )

    average_velocity = (
        s[-1]
        /
        time[-1]
    )

    below_90 = (
        velocity
        <
        0.90
        * robot[
            "v_max"
        ]
    )

    below_75 = (
        velocity
        <
        0.75
        * robot[
            "v_max"
        ]
    )

    below_50 = (
        velocity
        <
        0.50
        * robot[
            "v_max"
        ]
    )

    return {
        "raw":
            plan[
                "path"
            ],

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

        "speed_limit":
            speed_limit,

        "velocity":
            velocity,

        "time":
            time,

        "raw_length_m":
            path_length_m(
                plan[
                    "path"
                ],
                RESOLUTION,
            ),

        "los_length_m":
            path_length_m(
                los,
                RESOLUTION,
            ),

        "smooth_length_m":
            float(
                s[-1]
            ),

        "los_waypoints":
            len(
                los
            ),

        "max_curvature":
            max_curvature,

        "min_radius_m":
            (
                1.0
                /
                max_curvature
            )
            if max_curvature > 1e-9
            else math.inf,

        "peak_velocity":
            float(
                np.max(
                    velocity
                )
            ),

        "min_moving_velocity":
            float(
                np.min(
                    velocity[
                        1:-1
                    ]
                )
            ),

        "average_velocity":
            average_velocity,

        "mission_time_s":
            float(
                time[-1]
            ),

        "fraction_below_90":
            float(
                np.mean(
                    below_90
                )
            ),

        "fraction_below_75":
            float(
                np.mean(
                    below_75
                )
            ),

        "fraction_below_50":
            float(
                np.mean(
                    below_50
                )
            ),
    }


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

    raw_free = (
        grid == 0
    )

    results = {}

    for key in [
        "waffle",
        "ridgeback",
    ]:

        results[
            key
        ] = analyze_robot(
            key,
            grid,
            raw_free,
        )

    print()
    print(
        "=" * 84
    )

    print(
        "SCENARIO 02 — SMOOTHED TRAJECTORY + CONTINUOUS TIME"
    )

    print(
        "=" * 84
    )

    for key in [
        "waffle",
        "ridgeback",
    ]:

        r = results[
            key
        ]

        robot = ROBOTS[
            key
        ]

        print()
        print(
            robot[
                "name"
            ]
        )

        print(
            "-" * 40
        )

        print(
            f"Raw length           : "
            f"{r['raw_length_m']:.2f} m"
        )

        print(
            f"LOS length           : "
            f"{r['los_length_m']:.2f} m"
        )

        print(
            f"Smoothed length      : "
            f"{r['smooth_length_m']:.2f} m"
        )

        print(
            f"LOS waypoints        : "
            f"{r['los_waypoints']}"
        )

        print(
            f"Max curvature        : "
            f"{r['max_curvature']:.3f} 1/m"
        )

        print(
            f"Min curvature radius : "
            f"{r['min_radius_m']:.3f} m"
        )

        print(
            f"Peak velocity        : "
            f"{r['peak_velocity']:.3f} m/s"
        )

        print(
            f"Minimum moving vel   : "
            f"{r['min_moving_velocity']:.3f} m/s"
        )

        print(
            f"Average velocity     : "
            f"{r['average_velocity']:.3f} m/s"
        )

        print(
            f"Mission time         : "
            f"{r['mission_time_s']:.3f} s"
        )

        print(
            f"Samples < 90% vmax   : "
            f"{100.0 * r['fraction_below_90']:.1f}%"
        )

        print(
            f"Samples < 75% vmax   : "
            f"{100.0 * r['fraction_below_75']:.1f}%"
        )

        print(
            f"Samples < 50% vmax   : "
            f"{100.0 * r['fraction_below_50']:.1f}%"
        )

    waffle = results[
        "waffle"
    ]

    ridge = results[
        "ridgeback"
    ]

    print()
    print(
        "=" * 84
    )

    print(
        "COMPARISON"
    )

    print(
        "=" * 84
    )

    print(
        f"Path ratio R/W       : "
        f"{ridge['smooth_length_m'] / waffle['smooth_length_m']:.3f}"
    )

    print(
        f"Time ratio R/W       : "
        f"{ridge['mission_time_s'] / waffle['mission_time_s']:.3f}"
    )

    print(
        f"Time difference      : "
        f"{waffle['mission_time_s'] - ridge['mission_time_s']:.3f} s"
    )

    # ========================================================
    # PATH FIGURE
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

        r = results[
            key
        ]

        ax.imshow(
            grid,
            cmap="gray_r",
            origin="upper",
        )

        los = np.asarray(
            r[
                "los"
            ]
        )

        smooth = np.asarray(
            r[
                "smooth"
            ]
        )

        ax.plot(
            los[
                :,
                0
            ],
            los[
                :,
                1
            ],
            linewidth=1.0,
            alpha=0.55,
            marker="o",
            markersize=2,
            label="LOS path",
        )

        ax.plot(
            smooth[
                :,
                0
            ],
            smooth[
                :,
                1
            ],
            linewidth=2.0,
            label="Smoothed trajectory",
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
            ROBOTS[
                key
            ][
                "name"
            ]
        )

        ax.set_xticks([])
        ax.set_yticks([])

        ax.legend(
            fontsize=8,
        )

    fig.suptitle(
        (
            "STech Lab Scenario 02 — "
            "Collision-Aware Smoothed Trajectories"
        ),
        fontsize=15,
    )

    fig.tight_layout()

    path_plot = (
        RESULT_DIR
        /
        "scenario02_smoothed_paths.png"
    )

    fig.savefig(
        path_plot,
        dpi=180,
        bbox_inches="tight",
    )

    plt.close(
        fig
    )

    # ========================================================
    # VELOCITY FIGURE
    # ========================================================

    fig, ax = plt.subplots(
        figsize=(10, 6)
    )

    for key in [
        "waffle",
        "ridgeback",
    ]:

        r = results[
            key
        ]

        name = ROBOTS[
            key
        ][
            "name"
        ]

        ax.plot(
            r[
                "s"
            ],
            r[
                "velocity"
            ],
            linewidth=2.0,
            label=f"{name} velocity",
        )

        ax.plot(
            r[
                "s"
            ],
            r[
                "speed_limit"
            ],
            linestyle="--",
            linewidth=1.0,
            alpha=0.75,
            label=f"{name} local limit",
        )

    ax.set_xlabel(
        "Path Distance [m]"
    )

    ax.set_ylabel(
        "Velocity [m/s]"
    )

    ax.set_title(
        (
            "Scenario 02 — "
            "Continuous Velocity Profiles"
        )
    )

    ax.grid(
        True,
        alpha=0.25,
    )

    ax.legend()

    fig.tight_layout()

    velocity_plot = (
        RESULT_DIR
        /
        "scenario02_velocity_profiles.png"
    )

    fig.savefig(
        velocity_plot,
        dpi=180,
        bbox_inches="tight",
    )

    plt.close(
        fig
    )

    # ========================================================
    # SAVE CSV
    # ========================================================

    for key in [
        "waffle",
        "ridgeback",
    ]:

        r = results[
            key
        ]

        output = np.column_stack(
            (
                r[
                    "s"
                ],
                r[
                    "curvature"
                ],
                r[
                    "speed_limit"
                ],
                r[
                    "velocity"
                ],
                r[
                    "time"
                ],
            )
        )

        csv_path = (
            RESULT_DIR
            /
            f"scenario02_{key}_time_profile.csv"
        )

        np.savetxt(
            csv_path,
            output,
            delimiter=",",
            header=(
                "s_m,"
                "curvature_1pm,"
                "speed_limit_mps,"
                "velocity_mps,"
                "time_s"
            ),
            comments="",
        )

        print(
            "Saved:",
            csv_path,
        )

    print(
        "Saved:",
        path_plot,
    )

    print(
        "Saved:",
        velocity_plot,
    )


if __name__ == "__main__":
    main()