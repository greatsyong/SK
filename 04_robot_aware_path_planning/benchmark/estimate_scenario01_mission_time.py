from pathlib import Path
import math

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

START = (360, 136)
GOAL = (744, 840)

RESULT_DIR = Path(
    "results/stech_lab/scenario01"
)

ROBOTS = {
    "waffle": {
        "name": "TurtleBot3 Waffle Pi",
        "length_m": 0.281,
        "width_m": 0.306,

        # Official hardware specifications
        "v_max": 0.26,
        "omega_max": 1.82,
    },

    "ridgeback": {
        "name": "Clearpath Ridgeback",
        "length_m": 0.960,
        "width_m": 0.793,

        # Official maximum platform speed
        "v_max": 1.10,
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


def path_length(
    path,
    resolution,
):
    total = 0.0

    for p0, p1 in zip(
        path[:-1],
        path[1:],
    ):

        dx = (
            p1[0]
            - p0[0]
        )

        dy = (
            p1[1]
            - p0[1]
        )

        total += (
            math.hypot(
                dx,
                dy,
            )
            * resolution
        )

    return total


# ============================================================
# PATH COMPRESSION
# ============================================================

def direction_vector(
    p0,
    p1,
):
    dx = (
        p1[0]
        - p0[0]
    )

    dy = (
        p1[1]
        - p0[1]
    )

    if dx != 0:
        dx = int(
            math.copysign(
                1,
                dx,
            )
        )

    if dy != 0:
        dy = int(
            math.copysign(
                1,
                dy,
            )
        )

    return (
        dx,
        dy,
    )


def compress_collinear_path(
    path,
):
    """
    Remove intermediate grid points while direction
    remains unchanged.

    Example:

    (0,0),(1,0),(2,0),(3,0),(4,1),(5,2)

    becomes:

    (0,0),(3,0),(5,2)
    """

    if len(path) <= 2:
        return path[:]

    compressed = [
        path[0]
    ]

    previous_direction = (
        direction_vector(
            path[0],
            path[1],
        )
    )

    for i in range(
        1,
        len(path) - 1,
    ):

        new_direction = (
            direction_vector(
                path[i],
                path[i + 1],
            )
        )

        if (
            new_direction
            != previous_direction
        ):

            compressed.append(
                path[i]
            )

            previous_direction = (
                new_direction
            )

    compressed.append(
        path[-1]
    )

    return compressed


# ============================================================
# TURN METRICS
# ============================================================

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
        angle
        + math.pi
    ) % (
        2.0
        * math.pi
    ) - math.pi


def heading_changes(
    compressed_path,
):
    headings = []

    for p0, p1 in zip(
        compressed_path[:-1],
        compressed_path[1:],
    ):

        headings.append(
            heading(
                p0,
                p1,
            )
        )

    changes = []

    for h0, h1 in zip(
        headings[:-1],
        headings[1:],
    ):

        changes.append(
            abs(
                wrap_angle(
                    h1
                    - h0
                )
            )
        )

    return (
        headings,
        changes,
    )


# ============================================================
# TIME MODELS
# ============================================================

def waffle_time_estimate(
    path,
    resolution,
    v_max,
    omega_max,
):

    compressed = (
        compress_collinear_path(
            path
        )
    )

    length_m = (
        path_length(
            compressed,
            resolution,
        )
    )

    _, turns = (
        heading_changes(
            compressed
        )
    )

    translation_time = (
        length_m
        / v_max
    )

    total_turn_angle = sum(
        turns
    )

    turning_time = (
        total_turn_angle
        / omega_max
    )

    total_time = (
        translation_time
        +
        turning_time
    )

    return {
        "compressed_path": compressed,
        "length_m": length_m,
        "segment_count":
            len(compressed) - 1,
        "turn_count":
            len(turns),
        "total_turn_angle_rad":
            total_turn_angle,
        "total_turn_angle_deg":
            math.degrees(
                total_turn_angle
            ),
        "translation_time_s":
            translation_time,
        "turning_time_s":
            turning_time,
        "total_time_s":
            total_time,
    }


def ridgeback_time_estimate(
    path,
    resolution,
    v_max,
):

    compressed = (
        compress_collinear_path(
            path
        )
    )

    length_m = (
        path_length(
            compressed,
            resolution,
        )
    )

    _, turns = (
        heading_changes(
            compressed
        )
    )

    translation_time = (
        length_m
        / v_max
    )

    return {
        "compressed_path": compressed,
        "length_m": length_m,
        "segment_count":
            len(compressed) - 1,
        "direction_change_count":
            len(turns),
        "total_direction_change_deg":
            math.degrees(
                sum(turns)
            ),
        "translation_time_s":
            translation_time,

        # First-order holonomic estimate:
        # no mandatory stop-and-yaw term.
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

    m = load_ros_metric_map(
        MAP_YAML
    )

    grid = m[
        "grid"
    ]

    resolution = m[
        "resolution"
    ]

    free = (
        grid == 0
    )

    clearance_m = (
        distance_transform_edt(
            free
        )
        * resolution
    )

    # --------------------------------------------------------
    # Circular screening C-spaces
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
            free
            &
            (
                clearance_m
                >= radius
            )
        )

    waffle_plan = astar(
        cspaces["waffle"],
        START,
        GOAL,
    )

    ridge_plan = astar(
        cspaces["ridgeback"],
        START,
        GOAL,
    )

    if waffle_plan is None:
        raise RuntimeError(
            "Waffle path missing."
        )

    if ridge_plan is None:
        raise RuntimeError(
            "Ridgeback path missing."
        )

    # --------------------------------------------------------
    # Kinematic estimates
    # --------------------------------------------------------

    waffle = waffle_time_estimate(
        waffle_plan["path"],
        resolution,
        ROBOTS["waffle"]["v_max"],
        ROBOTS["waffle"]["omega_max"],
    )

    ridge = ridgeback_time_estimate(
        ridge_plan["path"],
        resolution,
        ROBOTS["ridgeback"]["v_max"],
    )

    # --------------------------------------------------------
    # Print
    # --------------------------------------------------------

    print()
    print("=" * 80)
    print("SCENARIO 01 — FIRST-ORDER MISSION TIME ESTIMATE")
    print("=" * 80)

    print(
        f"Start : {START}"
    )

    print(
        f"Goal  : {GOAL}"
    )

    print()

    print(
        "TurtleBot3 Waffle Pi"
    )

    print(
        f"  path length       : "
        f"{waffle['length_m']:.2f} m"
    )

    print(
        f"  segments          : "
        f"{waffle['segment_count']}"
    )

    print(
        f"  turns             : "
        f"{waffle['turn_count']}"
    )

    print(
        f"  total turn angle  : "
        f"{waffle['total_turn_angle_deg']:.1f} deg"
    )

    print(
        f"  translation time  : "
        f"{waffle['translation_time_s']:.2f} s"
    )

    print(
        f"  turning time      : "
        f"{waffle['turning_time_s']:.2f} s"
    )

    print(
        f"  estimated total   : "
        f"{waffle['total_time_s']:.2f} s"
    )

    print()

    print(
        "Clearpath Ridgeback"
    )

    print(
        f"  path length       : "
        f"{ridge['length_m']:.2f} m"
    )

    print(
        f"  segments          : "
        f"{ridge['segment_count']}"
    )

    print(
        f"  direction changes : "
        f"{ridge['direction_change_count']}"
    )

    print(
        f"  total dir change  : "
        f"{ridge['total_direction_change_deg']:.1f} deg"
    )

    print(
        f"  translation time  : "
        f"{ridge['translation_time_s']:.2f} s"
    )

    print(
        f"  estimated total   : "
        f"{ridge['total_time_s']:.2f} s"
    )

    print()

    print("=" * 80)
    print("COMPARISON")
    print("=" * 80)

    print(
        f"Path length ratio R/W : "
        f"{ridge['length_m'] / waffle['length_m']:.3f}"
    )

    print(
        f"Mission time ratio R/W: "
        f"{ridge['total_time_s'] / waffle['total_time_s']:.3f}"
    )

    print(
        f"Estimated time saved by Ridgeback: "
        f"{waffle['total_time_s'] - ridge['total_time_s']:.2f} s"
    )

    # --------------------------------------------------------
    # Save text report
    # --------------------------------------------------------

    report = (
        RESULT_DIR
        / "scenario01_mission_time_estimate.txt"
    )

    with report.open(
        "w"
    ) as f:

        f.write(
            "STech Lab Scenario 01\n"
        )

        f.write(
            "First-order Mission Time Estimate\n"
        )

        f.write(
            "=" * 60
            + "\n\n"
        )

        f.write(
            f"Start: {START}\n"
        )

        f.write(
            f"Goal : {GOAL}\n\n"
        )

        f.write(
            "Waffle Pi\n"
        )

        for key, value in waffle.items():

            if key != "compressed_path":
                f.write(
                    f"{key}: {value}\n"
                )

        f.write(
            "\nRidgeback\n"
        )

        for key, value in ridge.items():

            if key != "compressed_path":
                f.write(
                    f"{key}: {value}\n"
                )

    print()
    print(
        "Saved:",
        report,
    )


if __name__ == "__main__":
    main()