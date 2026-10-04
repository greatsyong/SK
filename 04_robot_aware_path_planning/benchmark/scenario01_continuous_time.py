from pathlib import Path
import math

import matplotlib.pyplot as plt
import numpy as np


# ============================================================
# CONFIG
# ============================================================

RESULT_DIR = Path(
    "results/stech_lab/scenario01"
)

WAFFLE_CSV = (
    RESULT_DIR
    / "scenario01_waffle_smoothed.csv"
)

RIDGEBACK_CSV = (
    RESULT_DIR
    / "scenario01_ridgeback_smoothed.csv"
)

RESOLUTION = 0.05


ROBOTS = {

    "waffle": {
        "name":
            "TurtleBot3 Waffle Pi",

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

        "v_max":
            1.10,

        "a_max":
            1.0,

        "decel_max":
            1.0,

        # Scalar approximation for
        # translational normal acceleration.
        "lateral_accel_max":
            1.0,
    },
}


# ============================================================
# LOAD TRAJECTORY
# ============================================================

def load_path(
    path,
):

    return np.loadtxt(
        path,
        delimiter=",",
        skiprows=1,
    )


# ============================================================
# ARC LENGTH
# ============================================================

def compute_arc_length(
    points,
):

    points_m = (
        points
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
            np.cumsum(ds),
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
# LOCAL SPEED LIMIT
# ============================================================

def waffle_speed_limit(
    curvature,
):

    robot = ROBOTS[
        "waffle"
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

    return limits


def ridgeback_speed_limit(
    curvature,
):

    robot = ROBOTS[
        "ridgeback"
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

    curvature_limit = np.sqrt(
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
        curvature_limit,
    )

    return limits


# ============================================================
# FORWARD / BACKWARD VELOCITY PASS
# ============================================================

def parameterize_velocity(
    ds,
    speed_limit,
    accel,
    decel,
):

    n = len(
        speed_limit
    )

    velocity = np.array(
        speed_limit,
        dtype=float,
    )

    # Start at rest
    velocity[0] = 0.0

    # --------------------------------------------------------
    # Forward acceleration constraint
    # --------------------------------------------------------

    for i in range(
        n - 1
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

    # Goal at rest
    velocity[-1] = 0.0

    # --------------------------------------------------------
    # Backward deceleration constraint
    # --------------------------------------------------------

    for i in range(
        n - 2,
        -1,
        -1,
    ):

        reachable = math.sqrt(
            max(
                0.0,
                velocity[i + 1] ** 2
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
        v1 = velocity[i + 1]

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
                    "Zero velocity on "
                    "non-zero trajectory segment."
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
# EVALUATION
# ============================================================

def evaluate(
    robot_key,
    points,
):

    robot = ROBOTS[
        robot_key
    ]

    (
        points_m,
        ds,
        s,
    ) = compute_arc_length(
        points
    )

    curvature = (
        compute_curvature(
            points_m
        )
    )

    if robot_key == "waffle":

        speed_limit = (
            waffle_speed_limit(
                curvature
            )
        )

    elif robot_key == "ridgeback":

        speed_limit = (
            ridgeback_speed_limit(
                curvature
            )
        )

    else:

        raise ValueError(
            robot_key
        )

    velocity = (
        parameterize_velocity(

            ds=ds,

            speed_limit=
                speed_limit,

            accel=
                robot[
                    "a_max"
                ],

            decel=
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

    return {

        "points":
            points,

        "points_m":
            points_m,

        "ds":
            ds,

        "s":
            s,

        "curvature":
            curvature,

        "speed_limit":
            speed_limit,

        "velocity":
            velocity,

        "dt":
            dt,

        "time":
            time,

        "length_m":
            float(
                s[-1]
            ),

        "total_time_s":
            float(
                time[-1]
            ),

        "max_curvature":
            float(
                np.max(
                    curvature
                )
            ),

        "min_velocity":
            float(
                np.min(
                    velocity[
                        1:-1
                    ]
                )
            )
            if len(
                velocity
            ) > 2
            else 0.0,

        "max_velocity":
            float(
                np.max(
                    velocity
                )
            ),
    }
# ============================================================
# UNIFORM ARC-LENGTH RESAMPLING
# ============================================================

def resample_path_uniform(
    points,
    spacing_m=RESOLUTION,
):
    """
    Resample trajectory at approximately uniform arc-length spacing.

    The default spacing equals the map resolution (0.05 m).
    """

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

    sample_s = np.arange(
        0.0,
        total_length,
        spacing_m,
    )

    if (
        len(sample_s) == 0
        or
        sample_s[-1]
        < total_length
    ):
        sample_s = np.append(
            sample_s,
            total_length,
        )

    x = np.interp(
        sample_s,
        cumulative,
        points_m[:, 0],
    )

    y = np.interp(
        sample_s,
        cumulative,
        points_m[:, 1],
    )

    # Return in cell coordinates so the rest of
    # the existing code remains unchanged.
    resampled_cells = np.column_stack(
        (
            x / RESOLUTION,
            y / RESOLUTION,
        )
    )

    return resampled_cells

# ============================================================
# MAIN
# ============================================================

def main():

    waffle_points = (
        load_path(
            WAFFLE_CSV
        )
    )

    ridge_points = (
        load_path(
            RIDGEBACK_CSV
        )
    )

    # --------------------------------------------------------
    # Uniform arc-length resampling
    # MUST happen before evaluate()
    # --------------------------------------------------------

    waffle_points = (
        resample_path_uniform(
            waffle_points
        )
    )

    ridge_points = (
        resample_path_uniform(
            ridge_points
        )
    )

    print(
        f"Waffle resampled points   : "
        f"{len(waffle_points)}"
    )

    print(
        f"Ridgeback resampled points: "
        f"{len(ridge_points)}"
    )

    # --------------------------------------------------------
    # Evaluate resampled trajectories
    # --------------------------------------------------------

    waffle = evaluate(
        "waffle",
        waffle_points,
    )

    ridge = evaluate(
        "ridgeback",
        ridge_points,
    )

    # ========================================================
    # CONSOLE
    # ========================================================

    print()
    print(
        "=" * 84
    )

    print(
        "SCENARIO 01 — CONTINUOUS TRAJECTORY TIME PARAMETERIZATION"
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
        f"Trajectory length  : "
        f"{waffle['length_m']:.3f} m"
    )

    print(
        f"Max curvature      : "
        f"{waffle['max_curvature']:.3f} 1/m"
    )

    print(
        f"Peak velocity      : "
        f"{waffle['max_velocity']:.3f} m/s"
    )

    print(
        f"Minimum moving vel : "
        f"{waffle['min_velocity']:.3f} m/s"
    )

    print(
        f"Mission time       : "
        f"{waffle['total_time_s']:.3f} s"
    )

    print()

    print(
        "Clearpath Ridgeback"
    )

    print(
        "-" * 40
    )

    print(
        f"Trajectory length  : "
        f"{ridge['length_m']:.3f} m"
    )

    print(
        f"Max curvature      : "
        f"{ridge['max_curvature']:.3f} 1/m"
    )

    print(
        f"Peak velocity      : "
        f"{ridge['max_velocity']:.3f} m/s"
    )

    print(
        f"Minimum moving vel : "
        f"{ridge['min_velocity']:.3f} m/s"
    )

    print(
        f"Mission time       : "
        f"{ridge['total_time_s']:.3f} s"
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
        f"Path ratio R/W       : "
        f"{ridge['length_m'] / waffle['length_m']:.3f}"
    )

    print(
        f"Time ratio R/W       : "
        f"{ratio:.3f}"
    )

    print(
        f"Time difference      : "
        f"{difference:.3f} s"
    )

    # ========================================================
    # VELOCITY PROFILE FIGURE
    # ========================================================

    fig, ax = plt.subplots(
        figsize=(10, 6)
    )

    ax.plot(
        waffle["s"],
        waffle["velocity"],
        linewidth=2.0,
        label="Waffle velocity",
    )

    ax.plot(
        waffle["s"],
        waffle["speed_limit"],
        linestyle="--",
        linewidth=1.2,
        label="Waffle local limit",
    )

    ax.plot(
        ridge["s"],
        ridge["velocity"],
        linewidth=2.0,
        label="Ridgeback velocity",
    )

    ax.plot(
        ridge["s"],
        ridge["speed_limit"],
        linestyle="--",
        linewidth=1.2,
        label="Ridgeback local limit",
    )

    ax.set_xlabel(
        "Path Distance [m]"
    )

    ax.set_ylabel(
        "Velocity [m/s]"
    )

    ax.set_title(
        "STech Lab Scenario 01 — "
        "Continuous Velocity Profiles"
    )

    ax.grid(
        True,
        alpha=0.25,
    )

    ax.legend()

    fig.tight_layout()

    velocity_plot = (
        RESULT_DIR
        / "scenario01_velocity_profiles.png"
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
    # CURVATURE FIGURE
    # ========================================================

    fig, ax = plt.subplots(
        figsize=(10, 6)
    )

    ax.plot(
        waffle["s"],
        waffle["curvature"],
        linewidth=1.5,
        label="Waffle",
    )

    ax.plot(
        ridge["s"],
        ridge["curvature"],
        linewidth=1.5,
        label="Ridgeback",
    )

    ax.set_xlabel(
        "Path Distance [m]"
    )

    ax.set_ylabel(
        "Curvature [1/m]"
    )

    ax.set_title(
        "STech Lab Scenario 01 — "
        "Trajectory Curvature"
    )

    ax.grid(
        True,
        alpha=0.25,
    )

    ax.legend()

    fig.tight_layout()

    curvature_plot = (
        RESULT_DIR
        / "scenario01_curvature_profiles.png"
    )

    fig.savefig(
        curvature_plot,
        dpi=180,
        bbox_inches="tight",
    )

    plt.close(
        fig
    )

    # ========================================================
    # SAVE PROFILE CSV
    # ========================================================

    for key, result in [
        (
            "waffle",
            waffle,
        ),
        (
            "ridgeback",
            ridge,
        ),
    ]:

        profile_path = (
            RESULT_DIR
            /
            f"scenario01_{key}_time_profile.csv"
        )

        output = np.column_stack(
            (
                result[
                    "s"
                ],
                result[
                    "curvature"
                ],
                result[
                    "speed_limit"
                ],
                result[
                    "velocity"
                ],
                result[
                    "time"
                ],
            )
        )

        np.savetxt(
            profile_path,
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
            profile_path,
        )

    print(
        "Saved:",
        velocity_plot,
    )

    print(
        "Saved:",
        curvature_plot,
    )


if __name__ == "__main__":
    main()