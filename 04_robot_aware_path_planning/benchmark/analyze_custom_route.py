from pathlib import Path

import matplotlib.pyplot as plt

from benchmark.map_loader import (
    load_movingai_map,
)

from planners.dijkstra import (
    DijkstraPlanner,
)

from metrics.path_structure import (
    analyze_path_structure,
)


MAP_NAME = "maze512-32-0"

START = (
    312,
    95,
)

GOAL = (
    160,
    240,
)

RRT_STEP_SIZE = 10.0


def main():

    project_root = (
        Path(__file__).resolve().parents[1]
    )

    result_dir = (
        project_root
        / "results"
    )

    result_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ========================================================
    # Load map
    # ========================================================

    map_data = load_movingai_map(
        project_root
        / "data"
        / "maps"
        / f"{MAP_NAME}.map"
    )

    grid = map_data["grid"]

    # ========================================================
    # Dijkstra reference route
    # ========================================================

    planner = DijkstraPlanner()

    result = planner.plan(
        grid,
        START,
        GOAL,
    )

    if not result["success"]:
        raise RuntimeError(
            "Dijkstra failed."
        )

    path = result["path"]

    # ========================================================
    # Route structure
    # ========================================================

    analysis = analyze_path_structure(
        grid,
        path,
        step_size=RRT_STEP_SIZE,
    )

    # ========================================================
    # Print summary
    # ========================================================

    straight_dx = (
        GOAL[0] - START[0]
    )

    straight_dy = (
        GOAL[1] - START[1]
    )

    straight_distance = (
        straight_dx ** 2
        + straight_dy ** 2
    ) ** 0.5

    print()
    print("=" * 78)
    print("CUSTOM ROUTE GEOMETRY ANALYSIS")
    print("=" * 78)

    print("Map                       :", MAP_NAME)
    print("Start                     :", START)
    print("Goal                      :", GOAL)

    print()
    print("GEOMETRY")
    print("-" * 78)

    print(
        "Straight-line distance     :",
        straight_distance,
    )

    print(
        "Dijkstra path length       :",
        result["path_length"],
    )

    print(
        "Detour ratio               :",
        result["path_length"]
        / straight_distance,
    )

    print(
        "Raw Dijkstra path points   :",
        analysis[
            "raw_path_points"
        ],
    )

    print(
        "LOS simplified waypoints  :",
        analysis[
            "simplified_waypoint_count"
        ],
    )

    print(
        "Obstacle-induced turns     :",
        analysis[
            "obstacle_induced_turn_count"
        ],
    )

    print(
        "Simplified path length     :",
        analysis[
            "simplified_path_length"
        ],
    )

    print()
    print("SEGMENTS")
    print("-" * 78)

    for i, (
        length,
        extensions,
    ) in enumerate(
        zip(
            analysis[
                "segment_lengths"
            ],
            analysis[
                "extension_count_per_segment"
            ],
        ),
        start=1,
    ):

        print(
            f"Segment {i:02d}: "
            f"length = {length:8.3f} cells | "
            f"min step-10 extensions = "
            f"{extensions}"
        )

    print()
    print(
        "Route-conditioned minimum "
        "extensions :",
        analysis[
            "route_extension_lower_bound"
        ],
    )

    print()
    print("TURNS")
    print("-" * 78)

    for turn in analysis["turns"]:

        print(
            f"Waypoint "
            f"{turn['waypoint_index']:02d}: "
            f"{turn['point']} | "
            f"turn = "
            f"{turn['turn_angle_deg']:.2f} deg"
        )

    print()
    print("CLEARANCE ALONG DIJKSTRA PATH")
    print("-" * 78)

    clearance = (
        analysis[
            "clearance_stats"
        ]
    )

    print(
        "Minimum clearance          :",
        clearance["minimum"],
        "cells",
    )

    print(
        "10th percentile clearance  :",
        clearance["p10"],
        "cells",
    )

    print(
        "Median clearance           :",
        clearance["median"],
        "cells",
    )

    print(
        "Mean clearance             :",
        clearance["mean"],
        "cells",
    )

    # ========================================================
    # Visualization
    # ========================================================

    fig, ax = plt.subplots(
        figsize=(
            9,
            9,
        )
    )

    ax.imshow(
        grid,
        origin="upper",
        interpolation="nearest",
    )

    # Raw Dijkstra path
    raw_x = [
        p[0] + 0.5
        for p in path
    ]

    raw_y = [
        p[1] + 0.5
        for p in path
    ]

    ax.plot(
        raw_x,
        raw_y,
        linewidth=1.0,
        label="Dijkstra grid path",
    )

    # Simplified route
    simplified = (
        analysis[
            "simplified_waypoints"
        ]
    )

    simplified_x = [
        p[0]
        for p in simplified
    ]

    simplified_y = [
        p[1]
        for p in simplified
    ]

    ax.plot(
        simplified_x,
        simplified_y,
        marker="o",
        linewidth=2.0,
        markersize=4,
        label="LOS simplified route",
    )

    # Number each simplified waypoint
    for i, (
        x,
        y,
    ) in enumerate(
        simplified
    ):

        ax.text(
            x + 3,
            y - 3,
            str(i),
            fontsize=8,
        )

    ax.scatter(
        START[0] + 0.5,
        START[1] + 0.5,
        s=70,
        marker="o",
        label="Start",
        zorder=10,
    )

    ax.scatter(
        GOAL[0] + 0.5,
        GOAL[1] + 0.5,
        s=80,
        marker="x",
        linewidths=2,
        label="Goal",
        zorder=10,
    )

    ax.set_title(
        (
            "Dijkstra Route Structure\n"
            f"L = {result['path_length']:.2f} | "
            f"Turns = "
            f"{analysis['obstacle_induced_turn_count']} | "
            f"Step-10 route lower bound = "
            f"{analysis['route_extension_lower_bound']}"
        )
    )

    ax.legend()

    output_path = (
        result_dir
        / "maze512_custom_route_structure.png"
    )

    fig.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight",
    )

    print()
    print(
        "Saved:",
        output_path,
    )

    plt.show()


if __name__ == "__main__":
    main()