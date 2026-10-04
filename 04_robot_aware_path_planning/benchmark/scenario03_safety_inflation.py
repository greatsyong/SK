from pathlib import Path
import csv
import math

import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import distance_transform_edt

from benchmark.metric_map_loader import load_ros_metric_map
from benchmark.find_robot_dependent_scenarios import astar


MAP_YAML = Path(
    "data/metric_maps/stech_lab_scenario02/"
    "stech_lab_scenario02.yaml"
)

RESULT_DIR = Path(
    "results/stech_lab/scenario02/"
    "safety_benchmark"
)

START = (360, 136)
GOAL = (744, 840)

SAFETY_MARGINS_M = (
    0.00,
    0.025,
    0.05,
)

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


def path_length_m(path, resolution):
    points = np.asarray(
        path,
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


def cell_is_free(cspace, cell):
    x, y = cell

    if (
        y < 0
        or y >= cspace.shape[0]
        or x < 0
        or x >= cspace.shape[1]
    ):
        return False

    return bool(
        cspace[y, x]
    )


def build_safety_cspace(
    raw_free,
    resolution,
    robot,
    safety_margin_m,
):
    """
    Orientation-independent C-space:

        effective radius =
            circumscribed robot radius
            + explicit safety margin

    The safety margin is a common physical constraint,
    not a planner-specific cost/tuning term.
    """

    clearance_m = (
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
        + safety_margin_m
    )

    cspace = (
        raw_free
        &
        (
            clearance_m
            >= effective_radius_m
        )
    )

    return (
        cspace,
        footprint_radius_m,
        effective_radius_m,
    )


def run_case(
    raw_free,
    resolution,
    robot_key,
    safety_margin_m,
):
    robot = ROBOTS[
        robot_key
    ]

    (
        cspace,
        footprint_radius_m,
        effective_radius_m,
    ) = build_safety_cspace(
        raw_free,
        resolution,
        robot,
        safety_margin_m,
    )

    start_free = cell_is_free(
        cspace,
        START,
    )

    goal_free = cell_is_free(
        cspace,
        GOAL,
    )

    plan = None

    if (
        start_free
        and goal_free
    ):
        plan = astar(
            cspace,
            START,
            GOAL,
        )

    connected = (
        plan is not None
    )

    if connected:
        path = plan["path"]

        raw_path_length_m = (
            path_length_m(
                path,
                resolution,
            )
        )

        raw_path_points = len(
            path
        )

    else:
        path = None
        raw_path_length_m = math.nan
        raw_path_points = 0

    return {
        "robot_key":
            robot_key,

        "robot_name":
            robot["name"],

        "safety_margin_m":
            safety_margin_m,

        "footprint_radius_m":
            footprint_radius_m,

        "effective_radius_m":
            effective_radius_m,

        "start_free":
            start_free,

        "goal_free":
            goal_free,

        "connected":
            connected,

        "raw_path_length_m":
            raw_path_length_m,

        "raw_path_points":
            raw_path_points,

        "cspace":
            cspace,

        "path":
            path,
    }


def save_csv(rows):
    csv_path = (
        RESULT_DIR
        / "scenario02_safety_margin_screen.csv"
    )

    fieldnames = [
        "robot_key",
        "robot_name",
        "safety_margin_m",
        "footprint_radius_m",
        "effective_radius_m",
        "start_free",
        "goal_free",
        "connected",
        "raw_path_length_m",
        "raw_path_points",
    ]

    with csv_path.open(
        "w",
        newline="",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        for row in rows:
            writer.writerow(
                {
                    key:
                        row[key]
                    for key in fieldnames
                }
            )

    return csv_path


def save_figure(rows):
    robot_keys = [
        "waffle",
        "ridgeback",
    ]

    margins = list(
        SAFETY_MARGINS_M
    )

    fig, axes = plt.subplots(
        len(robot_keys),
        len(margins),
        figsize=(18, 10),
    )

    for row_index, robot_key in enumerate(
        robot_keys
    ):

        for col_index, margin in enumerate(
            margins
        ):

            ax = axes[
                row_index,
                col_index,
            ]

            match = next(
                row
                for row in rows
                if (
                    row["robot_key"]
                    == robot_key
                    and
                    math.isclose(
                        row["safety_margin_m"],
                        margin,
                    )
                )
            )

            ax.imshow(
                match["cspace"],
                cmap="gray",
                origin="upper",
            )

            if match["path"] is not None:
                path = np.asarray(
                    match["path"]
                )

                ax.plot(
                    path[:, 0],
                    path[:, 1],
                    linewidth=1.6,
                    label="A* raw path",
                )

            ax.scatter(
                [START[0]],
                [START[1]],
                s=45,
                marker="o",
                label="Start",
            )

            ax.scatter(
                [GOAL[0]],
                [GOAL[1]],
                s=55,
                marker="X",
                label="Goal",
            )

            status = (
                "CONNECTED"
                if match["connected"]
                else "NO PATH"
            )

            ax.set_title(
                (
                    f"{ROBOTS[robot_key]['name']}\n"
                    f"Safety margin = "
                    f"{margin:.2f} m | "
                    f"{status}"
                ),
                fontsize=10,
            )

            ax.set_xticks([])
            ax.set_yticks([])

    fig.suptitle(
        (
            "Scenario 02 — "
            "Robot Footprint + Additional Safety Inflation"
        ),
        fontsize=15,
    )

    fig.tight_layout()

    figure_path = (
        RESULT_DIR
        / "scenario02_safety_margin_cspaces.png"
    )

    fig.savefig(
        figure_path,
        dpi=180,
        bbox_inches="tight",
    )

    plt.close(
        fig
    )

    return figure_path


def print_summary(rows):
    print()
    print(
        "=" * 92
    )
    print(
        "SCENARIO 02 — SAFETY INFLATION SCREEN"
    )
    print(
        "=" * 92
    )

    for row in rows:
        if row["connected"]:
            length_text = (
                f"{row['raw_path_length_m']:.2f} m"
            )
        else:
            length_text = "N/A"

        print(
            f"{row['robot_name']:<28} | "
            f"margin={row['safety_margin_m']:.2f} m | "
            f"r_eff={row['effective_radius_m']:.3f} m | "
            f"start={'OK' if row['start_free'] else 'BLOCKED'} | "
            f"goal={'OK' if row['goal_free'] else 'BLOCKED'} | "
            f"path={'YES' if row['connected'] else 'NO':<3} | "
            f"length={length_text}"
        )

    print()
    print(
        "These are screening cases only."
    )
    print(
        "Freeze one engineering safety margin before "
        "Dijkstra/A*/RRT/RRT* comparison."
    )


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

    rows = []

    for robot_key in [
        "waffle",
        "ridgeback",
    ]:

        for safety_margin_m in (
            SAFETY_MARGINS_M
        ):

            rows.append(
                run_case(
                    raw_free,
                    resolution,
                    robot_key,
                    safety_margin_m,
                )
            )

    print_summary(
        rows
    )

    csv_path = save_csv(
        rows
    )

    figure_path = save_figure(
        rows
    )

    print()
    print(
        "Saved:",
        csv_path,
    )

    print(
        "Saved:",
        figure_path,
    )


if __name__ == "__main__":
    main()
