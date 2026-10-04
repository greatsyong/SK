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

        "name":
            "TurtleBot3 Waffle Pi",

        "length_m":
            0.281,

        "width_m":
            0.306,

        # ROS 2 Humble Nav2 limits
        "v_max":
            0.22,

        "a_max":
            2.5,

        "decel_max":
            2.5,

        "omega_max":
            1.0,

        "alpha_max":
            3.2,

        "angular_decel_max":
            3.2,
    },

    "ridgeback": {

        "name":
            "Clearpath Ridgeback",

        "length_m":
            0.960,

        "width_m":
            0.793,

        # Platform speed ceiling
        "v_max":
            1.10,

        # Clearpath controller defaults
        "a_max":
            1.0,

        "decel_max":
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


def segment_length_m(
    p0,
    p1,
    resolution,
):

    return (
        math.hypot(
            p1[0] - p0[0],
            p1[1] - p0[1],
        )
        * resolution
    )


def heading(
    p0,
    p1,
):

    return math.atan2(
        p1[1] - p0[1],
        p1[0] - p0[0],
    )


def wrap_angle(
    angle,
):

    return (
        angle + math.pi
    ) % (
        2.0 * math.pi
    ) - math.pi


def path_turn_angles(
    path,
):

    headings = [

        heading(
            p0,
            p1,
        )

        for p0, p1 in zip(
            path[:-1],
            path[1:],
        )
    ]

    return [

        abs(
            wrap_angle(
                h1 - h0
            )
        )

        for h0, h1 in zip(
            headings[:-1],
            headings[1:],
        )
    ]


# ============================================================
# 1-D MOTION PROFILE
# ============================================================

def rest_to_rest_time(
    distance,
    v_max,
    accel,
    decel,
):
    """
    Minimum time for a 1-D segment:

        initial velocity = 0
        final velocity   = 0

    with asymmetric acceleration/deceleration limits.

    Returns:
        total time
        peak velocity
        profile type
    """

    if distance <= 0.0:

        return {
            "time_s": 0.0,
            "peak_velocity": 0.0,
            "profile": "zero",
        }

    accel = abs(
        accel
    )

    decel = abs(
        decel
    )

    # Distances required to accelerate to vmax
    # and then decelerate to zero.

    d_acc = (
        v_max ** 2
        /
        (
            2.0
            * accel
        )
    )

    d_dec = (
        v_max ** 2
        /
        (
            2.0
            * decel
        )
    )

    critical_distance = (
        d_acc
        +
        d_dec
    )

    # --------------------------------------------------------
    # Trapezoidal
    # --------------------------------------------------------

    if (
        distance
        >= critical_distance
    ):

        t_acc = (
            v_max
            /
            accel
        )

        t_dec = (
            v_max
            /
            decel
        )

        cruise_distance = (
            distance
            -
            critical_distance
        )

        t_cruise = (
            cruise_distance
            /
            v_max
        )

        return {

            "time_s":
                t_acc
                +
                t_cruise
                +
                t_dec,

            "peak_velocity":
                v_max,

            "profile":
                "trapezoidal",
        }

    # --------------------------------------------------------
    # Triangular
    # --------------------------------------------------------

    peak_velocity = math.sqrt(

        (
            2.0
            * distance
            * accel
            * decel
        )
        /
        (
            accel
            +
            decel
        )
    )

    t_acc = (
        peak_velocity
        /
        accel
    )

    t_dec = (
        peak_velocity
        /
        decel
    )

    return {

        "time_s":
            t_acc
            +
            t_dec,

        "peak_velocity":
            peak_velocity,

        "profile":
            "triangular",
    }


# ============================================================
# ANGULAR MOTION PROFILE
# ============================================================

def angular_rest_to_rest_time(
    angle_rad,
    omega_max,
    alpha,
    angular_decel,
):

    return rest_to_rest_time(
        distance=abs(
            angle_rad
        ),
        v_max=omega_max,
        accel=alpha,
        decel=angular_decel,
    )


# ============================================================
# WAFFLE EXECUTION MODEL
# ============================================================

def evaluate_waffle(
    path,
    resolution,
):

    robot = ROBOTS[
        "waffle"
    ]

    segment_results = []

    for index, (
        p0,
        p1,
    ) in enumerate(
        zip(
            path[:-1],
            path[1:],
        )
    ):

        length = (
            segment_length_m(
                p0,
                p1,
                resolution,
            )
        )

        motion = (
            rest_to_rest_time(
                distance=length,
                v_max=robot[
                    "v_max"
                ],
                accel=robot[
                    "a_max"
                ],
                decel=robot[
                    "decel_max"
                ],
            )
        )

        segment_results.append({

            "index":
                index,

            "length_m":
                length,

            "time_s":
                motion[
                    "time_s"
                ],

            "peak_velocity":
                motion[
                    "peak_velocity"
                ],

            "profile":
                motion[
                    "profile"
                ],
        })

    turns = (
        path_turn_angles(
            path
        )
    )

    turn_results = []

    for index, angle in enumerate(
        turns
    ):

        motion = (
            angular_rest_to_rest_time(

                angle_rad=
                    angle,

                omega_max=
                    robot[
                        "omega_max"
                    ],

                alpha=
                    robot[
                        "alpha_max"
                    ],

                angular_decel=
                    robot[
                        "angular_decel_max"
                    ],
            )
        )

        turn_results.append({

            "index":
                index,

            "angle_rad":
                angle,

            "angle_deg":
                math.degrees(
                    angle
                ),

            "time_s":
                motion[
                    "time_s"
                ],

            "peak_omega":
                motion[
                    "peak_velocity"
                ],

            "profile":
                motion[
                    "profile"
                ],
        })

    translation_time = sum(
        row["time_s"]
        for row in segment_results
    )

    turning_time = sum(
        row["time_s"]
        for row in turn_results
    )

    return {

        "segments":
            segment_results,

        "turns":
            turn_results,

        "translation_time_s":
            translation_time,

        "turning_time_s":
            turning_time,

        "total_time_s":
            translation_time
            +
            turning_time,
    }


# ============================================================
# RIDGEBACK EXECUTION MODEL
# ============================================================

def evaluate_ridgeback(
    path,
    resolution,
):

    robot = ROBOTS[
        "ridgeback"
    ]

    segment_results = []

    for index, (
        p0,
        p1,
    ) in enumerate(
        zip(
            path[:-1],
            path[1:],
        )
    ):

        length = (
            segment_length_m(
                p0,
                p1,
                resolution,
            )
        )

        motion = (
            rest_to_rest_time(
                distance=length,
                v_max=robot[
                    "v_max"
                ],
                accel=robot[
                    "a_max"
                ],
                decel=robot[
                    "decel_max"
                ],
            )
        )

        segment_results.append({

            "index":
                index,

            "length_m":
                length,

            "time_s":
                motion[
                    "time_s"
                ],

            "peak_velocity":
                motion[
                    "peak_velocity"
                ],

            "profile":
                motion[
                    "profile"
                ],
        })

    translation_time = sum(
        row["time_s"]
        for row in segment_results
    )

    return {

        "segments":
            segment_results,

        # Holonomic baseline:
        # no forced yaw maneuver at each
        # translation direction change.
        "total_time_s":
            translation_time,
    }


# ============================================================
# MAIN
# ============================================================

def main():

    RESULT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Map
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
    # Robot-specific screening C-spaces
    # --------------------------------------------------------

    cspaces = {}

    for key, robot in ROBOTS.items():

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

    # --------------------------------------------------------
    # A*
    # --------------------------------------------------------

    waffle_plan = astar(
        cspaces[
            "waffle"
        ],
        START,
        GOAL,
    )

    ridge_plan = astar(
        cspaces[
            "ridgeback"
        ],
        START,
        GOAL,
    )

    if waffle_plan is None:
        raise RuntimeError(
            "Waffle path not found."
        )

    if ridge_plan is None:
        raise RuntimeError(
            "Ridgeback path not found."
        )

    # --------------------------------------------------------
    # LOS refinement
    # --------------------------------------------------------

    waffle_path = (
        simplify_path_los(
            waffle_plan[
                "path"
            ],
            cspaces[
                "waffle"
            ],
        )
    )

    ridge_path = (
        simplify_path_los(
            ridge_plan[
                "path"
            ],
            cspaces[
                "ridgeback"
            ],
        )
    )

    # --------------------------------------------------------
    # Dynamic estimates
    # --------------------------------------------------------

    waffle = (
        evaluate_waffle(
            waffle_path,
            resolution,
        )
    )

    ridge = (
        evaluate_ridgeback(
            ridge_path,
            resolution,
        )
    )

    # ========================================================
    # OUTPUT
    # ========================================================

    print()
    print(
        "=" * 84
    )

    print(
        "SCENARIO 01 — ACCELERATION-LIMITED EXECUTION ESTIMATE"
    )

    print(
        "=" * 84
    )

    print()

    print(
        "TurtleBot3 Waffle Pi"
    )

    print(
        "-" * 40
    )

    print(
        f"LOS segments       : "
        f"{len(waffle['segments'])}"
    )

    print(
        f"Turns              : "
        f"{len(waffle['turns'])}"
    )

    print(
        f"Translation time   : "
        f"{waffle['translation_time_s']:.2f} s"
    )

    print(
        f"Turning time       : "
        f"{waffle['turning_time_s']:.2f} s"
    )

    print(
        f"Total time         : "
        f"{waffle['total_time_s']:.2f} s"
    )

    print()

    print(
        "Clearpath Ridgeback"
    )

    print(
        "-" * 40
    )

    print(
        f"LOS segments       : "
        f"{len(ridge['segments'])}"
    )

    print(
        f"Total time         : "
        f"{ridge['total_time_s']:.2f} s"
    )

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

    ratio = (
        ridge[
            "total_time_s"
        ]
        /
        waffle[
            "total_time_s"
        ]
    )

    difference = (
        waffle[
            "total_time_s"
        ]
        -
        ridge[
            "total_time_s"
        ]
    )

    print(
        f"Mission-time ratio R/W : "
        f"{ratio:.3f}"
    )

    print(
        f"Time difference        : "
        f"{difference:.2f} s"
    )

    # --------------------------------------------------------
    # Detailed segment report
    # --------------------------------------------------------

    print()
    print(
        "WAFFLE SEGMENTS"
    )

    for row in waffle[
        "segments"
    ]:

        print(
            f"  {row['index']:02d} | "
            f"L={row['length_m']:.3f} m | "
            f"T={row['time_s']:.3f} s | "
            f"v_peak={row['peak_velocity']:.3f} m/s | "
            f"{row['profile']}"
        )

    print()
    print(
        "WAFFLE TURNS"
    )

    for row in waffle[
        "turns"
    ]:

        print(
            f"  {row['index']:02d} | "
            f"dtheta={row['angle_deg']:.2f} deg | "
            f"T={row['time_s']:.3f} s | "
            f"omega_peak={row['peak_omega']:.3f} rad/s | "
            f"{row['profile']}"
        )

    print()
    print(
        "RIDGEBACK SEGMENTS"
    )

    for row in ridge[
        "segments"
    ]:

        print(
            f"  {row['index']:02d} | "
            f"L={row['length_m']:.3f} m | "
            f"T={row['time_s']:.3f} s | "
            f"v_peak={row['peak_velocity']:.3f} m/s | "
            f"{row['profile']}"
        )

    # ========================================================
    # BAR PLOT
    # ========================================================

    labels = [
        "Waffle",
        "Ridgeback",
    ]

    times = [
        waffle[
            "total_time_s"
        ],
        ridge[
            "total_time_s"
        ],
    ]

    fig, ax = plt.subplots(
        figsize=(7, 6)
    )

    bars = ax.bar(
        labels,
        times,
    )

    ax.set_ylabel(
        "Estimated Mission Time [s]"
    )

    ax.set_title(
        "STech Lab Scenario 01 — "
        "Acceleration-Limited Time Estimate"
    )

    for bar, value in zip(
        bars,
        times,
    ):

        ax.text(
            bar.get_x()
            + bar.get_width() / 2.0,
            bar.get_height(),
            f"{value:.1f} s",
            ha="center",
            va="bottom",
        )

    fig.tight_layout()

    plot_path = (
        RESULT_DIR
        /
        "scenario01_dynamic_time.png"
    )

    fig.savefig(
        plot_path,
        dpi=180,
        bbox_inches="tight",
    )

    plt.close(
        fig
    )

    # ========================================================
    # REPORT
    # ========================================================

    report_path = (
        RESULT_DIR
        /
        "scenario01_dynamic_time.txt"
    )

    with report_path.open(
        "w"
    ) as f:

        f.write(
            "STech Lab Scenario 01\n"
        )

        f.write(
            "Acceleration-Limited Execution Estimate\n"
        )

        f.write(
            "=" * 60
            + "\n\n"
        )

        f.write(
            "MODEL ASSUMPTIONS\n"
        )

        f.write(
            "Waffle: stop-turn-go at LOS vertices.\n"
        )

        f.write(
            "Ridgeback: rest-to-rest translation per LOS segment, "
            "no mandatory yaw maneuver.\n\n"
        )

        f.write(
            "Waffle Pi\n"
        )

        f.write(
            f"translation_time_s: "
            f"{waffle['translation_time_s']:.6f}\n"
        )

        f.write(
            f"turning_time_s: "
            f"{waffle['turning_time_s']:.6f}\n"
        )

        f.write(
            f"total_time_s: "
            f"{waffle['total_time_s']:.6f}\n\n"
        )

        f.write(
            "Ridgeback\n"
        )

        f.write(
            f"total_time_s: "
            f"{ridge['total_time_s']:.6f}\n\n"
        )

        f.write(
            "Comparison\n"
        )

        f.write(
            f"ratio_R_over_W: "
            f"{ratio:.6f}\n"
        )

        f.write(
            f"time_difference_s: "
            f"{difference:.6f}\n"
        )

    print()
    print(
        "Saved:",
        plot_path,
    )

    print(
        "Saved:",
        report_path,
    )


if __name__ == "__main__":
    main()