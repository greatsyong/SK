from datetime import datetime
from pathlib import Path
import csv

from benchmark.map_loader import load_movingai_map
from planners.dijkstra import DijkstraPlanner
from planners.rrt_star import RRTStarPlanner


MAP_NAME = "maze512-32-0"
SEED = 42

START = (312, 95)
GOAL = (160, 240)

ITERATION_BUDGETS = [
    2000,
    5000,
    10000,
    20000,
    40000,
]


def main():

    project_root = (
        Path(__file__).resolve().parents[1]
    )

    result_dir = (
        project_root / "results"
    )

    result_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Load map
    # --------------------------------------------------------

    map_data = load_movingai_map(
        project_root
        / "data"
        / "maps"
        / f"{MAP_NAME}.map"
    )

    grid = map_data["grid"]

    # --------------------------------------------------------
    # Reference by Dijkstra
    # --------------------------------------------------------

    dijkstra = DijkstraPlanner()

    ref = dijkstra.plan(
        grid,
        START,
        GOAL,
    )

    if not ref["success"]:
        raise RuntimeError(
            "Reference Dijkstra failed. "
            "Choose another goal."
        )

    reference_length = (
        ref["path_length"]
    )

    rows = []

    print()
    print("=" * 78)
    print("RRT* CUSTOM CONVERGENCE STUDY")
    print("=" * 78)

    print("Map       :", MAP_NAME)
    print("Start     :", START)
    print("Goal      :", GOAL)
    print("Reference :", reference_length)
    print("Seed      :", SEED)

    # --------------------------------------------------------
    # Convergence sweep
    # --------------------------------------------------------

    for budget in ITERATION_BUDGETS:

        planner = RRTStarPlanner(
            step_size=10.0,
            goal_radius=10.0,
            goal_sample_rate=0.05,
            max_iterations=budget,
            neighbor_radius=30.0,
            seed=SEED,
        )

        result = planner.plan(
            grid,
            START,
            GOAL,
        )

        if result["success"]:
            normalized_ratio = (
                result["path_length"]
                / reference_length
            )
        else:
            normalized_ratio = float("inf")

        row = {
            "map":
                MAP_NAME,

            "seed":
                SEED,

            "start_x":
                START[0],

            "start_y":
                START[1],

            "goal_x":
                GOAL[0],

            "goal_y":
                GOAL[1],

            "iteration_budget":
                budget,

            "success":
                result["success"],

            "reference_length":
                reference_length,

            "first_solution_iteration":
                result[
                    "first_solution_iteration"
                ],

            "first_solution_cost":
                result[
                    "first_solution_cost"
                ],

            "final_path_length":
                result["path_length"],

            "normalized_reference_ratio":
                normalized_ratio,

            "tree_nodes":
                result["tree_nodes"],

            "rewires":
                result["rewires"],

            "collision_checks":
                result["collision_checks"],

            "planning_time_s":
                result["planning_time"],
        }

        rows.append(row)

        print()
        print(f"N = {budget:6d}")
        print("  success      :", result["success"])
        print(
            "  first iter   :",
            result["first_solution_iteration"]
        )
        print(
            "  first cost   :",
            result["first_solution_cost"]
        )
        print(
            "  final length :",
            result["path_length"]
        )
        print(
            "  ratio        :",
            normalized_ratio
        )
        print(
            "  tree nodes   :",
            result["tree_nodes"]
        )
        print(
            "  rewires      :",
            result["rewires"]
        )
        print(
            "  time         :",
            result["planning_time"],
            "s"
        )

    # --------------------------------------------------------
    # Save CSV
    # --------------------------------------------------------

    timestamp = datetime.now().strftime(
        "%Y%m%d_%H%M%S"
    )

    output_path = (
        result_dir
        / (
            f"{MAP_NAME}_"
            f"rrt_star_custom_"
            f"s{START[0]}_{START[1]}_"
            f"g{GOAL[0]}_{GOAL[1]}_"
            f"seed{SEED}_"
            f"{timestamp}.csv"
        )
    )

    with output_path.open(
        "w",
        newline="",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=list(rows[0].keys()),
        )
        writer.writeheader()
        writer.writerows(rows)

    print()
    print("=" * 78)
    print("CUSTOM CONVERGENCE STUDY COMPLETE")
    print("=" * 78)
    print("Saved:", output_path)


if __name__ == "__main__":
    main()