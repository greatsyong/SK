from pathlib import Path

import numpy as np

from benchmark.metric_map_loader import load_ros_metric_map
from benchmark.scenario03_common_evaluator import (
    build_clearance_field,
    build_robot_context,
    evaluate_path,
)

from planners.dijkstra import DijkstraPlanner
from planners.astar import AStarPlanner
from planners.rrt import RRTPlanner
from planners.rrt_star import RRTStarPlanner


# ============================================================
# SCENARIO 03 — ALL-PLANNER SMOKE TEST
# ============================================================
#
# Purpose:
#   Verify that all four planner implementations can feed the
#   same Scenario 03 evaluator before launching the full
#   statistical benchmark.
#
# This is NOT the final experiment.
# RRT/RRT* use a deliberately modest iteration budget here.
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

START = (360, 136)
GOAL = (744, 840)

SMOKE_SEED = 42
SMOKE_ITERATIONS = 5000


def make_planners():
    return [
        DijkstraPlanner(),
        AStarPlanner(),
        RRTPlanner(
            max_iterations=SMOKE_ITERATIONS,
            seed=SMOKE_SEED,
        ),
        RRTStarPlanner(
            max_iterations=SMOKE_ITERATIONS,
            seed=SMOKE_SEED,
        ),
    ]


def print_planner_result(
    planner_name,
    plan,
    evaluated,
):
    print()
    print(
        f"{planner_name}"
    )
    print(
        "-" * 56
    )

    print(
        f"Success                    : "
        f"{plan['success']}"
    )

    print(
        f"Planning time              : "
        f"{plan['planning_time']:.3f} s"
    )

    if "expanded_nodes" in plan:
        print(
            f"Expanded/tree nodes        : "
            f"{plan['expanded_nodes']}"
        )

    if "iterations" in plan:
        print(
            f"Iterations                  : "
            f"{plan['iterations']}"
        )

    if "collision_checks" in plan:
        print(
            f"Collision checks            : "
            f"{plan['collision_checks']}"
        )

    if "first_solution_iteration" in plan:
        first_it = plan[
            "first_solution_iteration"
        ]
        print(
            f"First solution iteration    : "
            f"{first_it}"
        )

    if evaluated is None:
        return

    print(
        f"Raw path length             : "
        f"{evaluated['raw_length_m']:.2f} m"
    )

    print(
        f"Final path length           : "
        f"{evaluated['final_length_m']:.2f} m"
    )

    print(
        f"Trajectory feasible         : "
        f"{'YES' if evaluated['trajectory_feasible'] else 'NO'}"
    )

    print(
        f"Minimum clearance           : "
        f"{evaluated['min_clearance_m']:.3f} m"
    )

    print(
        f"Caution-zone ratio          : "
        f"{100.0 * evaluated['caution_ratio']:.1f}%"
    )

    print(
        f"Safety-aware mission time   : "
        f"{evaluated['safety_mission_time_s']:.3f} s"
    )


def main():
    metric_map = load_ros_metric_map(
        MAP_YAML
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

    if temporary_distance_m.shape != grid.shape:
        raise RuntimeError(
            "Temporary-object distance field shape mismatch."
        )

    clearance_m = build_clearance_field(
        raw_free,
        resolution,
    )

    print()
    print("=" * 92)
    print("SCENARIO 03 — ALL-PLANNER SMOKE TEST")
    print("=" * 92)
    print(
        f"RRT/RRT* smoke iterations  : "
        f"{SMOKE_ITERATIONS}"
    )
    print(
        f"RRT/RRT* seed              : "
        f"{SMOKE_SEED}"
    )

    for robot_key in [
        "waffle",
        "ridgeback",
    ]:
        context = build_robot_context(
            robot_key,
            raw_free,
            resolution,
            all_obstacle_clearance_m=clearance_m,
        )

        print()
        print()
        print("#" * 92)
        print(
            context[
                "robot"
            ][
                "name"
            ]
        )
        print("#" * 92)

        for planner in make_planners():
            if planner.name == "rrt":
                plan = planner.plan(
                    context[
                        "planner_grid"
                    ],
                    START,
                    GOAL,
                    return_tree=False,
                )

            elif planner.name == "rrt_star":
                plan = planner.plan(
                    context[
                        "planner_grid"
                    ],
                    START,
                    GOAL,
                    return_tree=False,
                )

            else:
                plan = planner.plan(
                    context[
                        "planner_grid"
                    ],
                    START,
                    GOAL,
                )

            evaluated = None

            if plan[
                "success"
            ]:
                evaluated = evaluate_path(
                    raw_path=plan[
                        "path"
                    ],
                    robot_key=robot_key,
                    robot_context=context,
                    temporary_distance_m=temporary_distance_m,
                    resolution=resolution,
                )

            print_planner_result(
                planner.name,
                plan,
                evaluated,
            )


if __name__ == "__main__":
    main()