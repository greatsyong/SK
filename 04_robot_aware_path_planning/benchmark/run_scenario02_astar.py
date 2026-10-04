from pathlib import Path
import math

import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import distance_transform_edt

from benchmark.metric_map_loader import load_ros_metric_map
from benchmark.find_robot_dependent_scenarios import astar
from benchmark.refine_scenario01_paths import (
    simplify_path_los,
    path_metrics,
)


MAP_YAML = Path(
    "data/metric_maps/stech_lab_scenario02/stech_lab_scenario02.yaml"
)

RESULT_DIR = Path(
    "results/stech_lab/scenario02"
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


def circumscribed_radius(length_m, width_m):
    return 0.5 * math.hypot(
        length_m,
        width_m,
    )


def main():

    RESULT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    m = load_ros_metric_map(
        MAP_YAML
    )

    grid = m["grid"]
    resolution = m["resolution"]

    raw_free = (
        grid == 0
    )

    clearance_m = (
        distance_transform_edt(
            raw_free
        )
        * resolution
    )

    results = {}

    for key, robot in ROBOTS.items():

        radius = (
            circumscribed_radius(
                robot["length_m"],
                robot["width_m"],
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
                f"{key}: no path found"
            )

        los = simplify_path_los(
            plan["path"],
            cspace,
        )

        raw_metrics = path_metrics(
            plan["path"],
            resolution,
        )

        los_metrics = path_metrics(
            los,
            resolution,
        )

        results[key] = {
            "raw": plan["path"],
            "los": los,
            "raw_metrics": raw_metrics,
            "los_metrics": los_metrics,
        }

    print()
    print("=" * 84)
    print("SCENARIO 02 — A* ROBOT COMPARISON")
    print("=" * 84)

    for key in [
        "waffle",
        "ridgeback",
    ]:

        r = results[key]

        print()
        print(
            ROBOTS[key]["name"]
        )

        print(
            "-" * 40
        )

        print(
            f"Raw path length     : "
            f"{r['raw_metrics']['length_m']:.2f} m"
        )

        print(
            f"LOS path length     : "
            f"{r['los_metrics']['length_m']:.2f} m"
        )

        print(
            f"LOS waypoints       : "
            f"{r['los_metrics']['points']}"
        )

        print(
            f"LOS segments        : "
            f"{r['los_metrics']['segments']}"
        )

        print(
            f"LOS turns           : "
            f"{r['los_metrics']['turns']}"
        )

        print(
            f"Total turn angle    : "
            f"{r['los_metrics']['total_turn_deg']:.1f} deg"
        )

        print(
            f"Max single turn     : "
            f"{r['los_metrics']['max_turn_deg']:.1f} deg"
        )

    # --------------------------------------------------------
    # Visualization
    # --------------------------------------------------------

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

        raw = np.asarray(
            r["raw"]
        )

        los = np.asarray(
            r["los"]
        )

        ax.plot(
            raw[:, 0],
            raw[:, 1],
            linewidth=0.7,
            alpha=0.45,
            label="Raw A*",
        )

        ax.plot(
            los[:, 0],
            los[:, 1],
            linewidth=2.0,
            marker="o",
            markersize=2.5,
            label="LOS-refined",
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
            ROBOTS[key]["name"]
        )

        ax.set_xticks([])
        ax.set_yticks([])

        ax.legend(
            fontsize=8,
        )

    fig.suptitle(
        "STech Lab Scenario 02 — Dense Temporary Obstacles",
        fontsize=15,
    )

    fig.tight_layout()

    plot_path = (
        RESULT_DIR
        /
        "scenario02_astar_robot_comparison.png"
    )

    fig.savefig(
        plot_path,
        dpi=180,
        bbox_inches="tight",
    )

    plt.close(fig)

    print()
    print(
        "Saved:",
        plot_path,
    )


if __name__ == "__main__":
    main()