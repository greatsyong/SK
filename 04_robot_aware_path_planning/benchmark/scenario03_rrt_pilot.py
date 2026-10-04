from pathlib import Path
import csv
import math
import statistics

import numpy as np

from benchmark.metric_map_loader import load_ros_metric_map
from benchmark.scenario03_common_evaluator import (
    build_clearance_field,
    build_robot_context,
    evaluate_path,
)

from planners.rrt import RRTPlanner
from planners.rrt_star import RRTStarPlanner


# ============================================================
# SCENARIO 03 — RRT / RRT* PILOT
# ============================================================
#
# Purpose:
#   Determine a sensible iteration range for the final
#   stochastic benchmark without wasting runs at many budgets.
#
# Strategy:
#   - 3 fixed seeds
#   - one RRT run per seed, max 50k iterations
#     (RRT stops immediately at first feasible path)
#   - one RRT* run per seed, full 50k iterations
#   - RRT* convergence sampled from its existing history at
#     5k / 10k / 20k / 50k without rerunning those budgets
#   - final successful path is passed once through the common
#     Scenario 03 evaluator
#
# This is a pilot, not the final statistical benchmark.
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
    "rrt_pilot"
)

START = (360, 136)
GOAL = (744, 840)

SEEDS = [
    0,
    1,
    2,
]

MAX_ITERATIONS = 50000

CONVERGENCE_CHECKPOINTS = [
    5000,
    10000,
    20000,
    50000,
]


def best_cost_at_iteration(
    convergence_history,
    checkpoint,
):
    """
    Return the latest recorded best goal cost at or before
    checkpoint.  None means no feasible RRT* solution yet.
    """

    best = None

    for iteration, cost in convergence_history:
        if iteration > checkpoint:
            break

        best = float(
            cost
        )

    return best


def save_csv(
    rows,
    output_path,
):
    if not rows:
        return

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fieldnames = []

    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)

    temp_path = output_path.with_suffix(
        output_path.suffix + ".tmp"
    )

    with temp_path.open(
        "w",
        newline="",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(
            rows
        )

    temp_path.replace(
        output_path
    )


def run_rrt(
    robot_key,
    context,
    temporary_distance_m,
    resolution,
    seed,
):
    planner = RRTPlanner(
        max_iterations=MAX_ITERATIONS,
        seed=seed,
    )

    plan = planner.plan(
        context[
            "planner_grid"
        ],
        START,
        GOAL,
        return_tree=False,
    )

    row = {
        "robot":
            robot_key,

        "planner":
            "rrt",

        "seed":
            seed,

        "max_iterations":
            MAX_ITERATIONS,

        "success":
            plan[
                "success"
            ],

        "iterations_used":
            plan[
                "iterations"
            ],

        "planning_time_s":
            plan[
                "planning_time"
            ],

        "tree_nodes":
            plan[
                "tree_nodes"
            ],

        "collision_checks":
            plan[
                "collision_checks"
            ],

        "first_solution_iteration":
            (
                plan["iterations"]
                if plan["success"]
                else None
            ),

        "rewires":
            0,

        "raw_path_length_m":
            math.nan,

        "final_path_length_m":
            math.nan,

        "trajectory_feasible":
            False,

        "caution_ratio":
            math.nan,

        "safety_mission_time_s":
            math.nan,
    }

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

        row.update(
            {
                "raw_path_length_m":
                    evaluated[
                        "raw_length_m"
                    ],

                "final_path_length_m":
                    evaluated[
                        "final_length_m"
                    ],

                "trajectory_feasible":
                    evaluated[
                        "trajectory_feasible"
                    ],

                "caution_ratio":
                    evaluated[
                        "caution_ratio"
                    ],

                "safety_mission_time_s":
                    evaluated[
                        "safety_mission_time_s"
                    ],
            }
        )

    return (
        row,
        plan,
    )


def run_rrt_star(
    robot_key,
    context,
    temporary_distance_m,
    resolution,
    seed,
):
    planner = RRTStarPlanner(
        max_iterations=MAX_ITERATIONS,
        seed=seed,
    )

    plan = planner.plan(
        context[
            "planner_grid"
        ],
        START,
        GOAL,
        return_tree=False,
    )

    row = {
        "robot":
            robot_key,

        "planner":
            "rrt_star",

        "seed":
            seed,

        "max_iterations":
            MAX_ITERATIONS,

        "success":
            plan[
                "success"
            ],

        "iterations_used":
            plan[
                "iterations"
            ],

        "first_solution_iteration":
            plan[
                "first_solution_iteration"
            ],

        "planning_time_s":
            plan[
                "planning_time"
            ],

        "tree_nodes":
            plan[
                "tree_nodes"
            ],

        "collision_checks":
            plan[
                "collision_checks"
            ],

        "rewires":
            plan[
                "rewires"
            ],

        "raw_path_length_m":
            math.nan,

        "final_path_length_m":
            math.nan,

        "trajectory_feasible":
            False,

        "caution_ratio":
            math.nan,

        "safety_mission_time_s":
            math.nan,
    }

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

        row.update(
            {
                "raw_path_length_m":
                    evaluated[
                        "raw_length_m"
                    ],

                "final_path_length_m":
                    evaluated[
                        "final_length_m"
                    ],

                "trajectory_feasible":
                    evaluated[
                        "trajectory_feasible"
                    ],

                "caution_ratio":
                    evaluated[
                        "caution_ratio"
                    ],

                "safety_mission_time_s":
                    evaluated[
                        "safety_mission_time_s"
                    ],
            }
        )

    convergence_rows = []

    history = plan[
        "convergence_history"
    ]

    for checkpoint in CONVERGENCE_CHECKPOINTS:
        cost_cells = best_cost_at_iteration(
            history,
            checkpoint,
        )

        convergence_rows.append(
            {
                "robot":
                    robot_key,

                "planner":
                    "rrt_star",

                "seed":
                    seed,

                "checkpoint_iteration":
                    checkpoint,

                "solution_available":
                    cost_cells
                    is not None,

                "best_cost_cells":
                    (
                        cost_cells
                        if cost_cells
                        is not None
                        else math.nan
                    ),

                "best_cost_m":
                    (
                        cost_cells
                        * resolution
                        if cost_cells
                        is not None
                        else math.nan
                    ),
            }
        )

    return (
        row,
        convergence_rows,
        plan,
    )


def print_trial(
    row,
):
    print(
        f"{row['planner']:8s} "
        f"seed={row['seed']:2d} "
        f"success={str(row['success']):5s} "
        f"iter={row['iterations_used']:6d} "
        f"time={row['planning_time_s']:7.3f} s",
        end="",
    )

    if row[
        "success"
    ]:
        print(
            f" raw={row['raw_path_length_m']:7.2f} m "
            f"final={row['final_path_length_m']:7.2f} m "
            f"mission={row['safety_mission_time_s']:8.2f} s"
        )

    else:
        print()


def print_robot_summary(
    robot_key,
    rows,
):
    print()
    print(
        f"Pilot summary — {robot_key}"
    )
    print(
        "-" * 72
    )

    for planner_name in [
        "rrt",
        "rrt_star",
    ]:
        subset = [
            row
            for row in rows
            if row[
                "robot"
            ] == robot_key
            and row[
                "planner"
            ] == planner_name
        ]

        successful = [
            row
            for row in subset
            if row[
                "success"
            ]
        ]

        print(
            f"{planner_name:8s}: "
            f"{len(successful)}/{len(subset)} successful",
            end="",
        )

        if successful:
            if planner_name == "rrt":
                iterations = [
                    row[
                        "iterations_used"
                    ]
                    for row in successful
                ]
            else:
                iterations = [
                    row[
                        "first_solution_iteration"
                    ]
                    for row in successful
                    if row[
                        "first_solution_iteration"
                    ] is not None
                ]

            if iterations:
                print(
                    f" | first-feasible iteration "
                    f"median={statistics.median(iterations):.0f} "
                    f"range=[{min(iterations)}, {max(iterations)}]"
                )
            else:
                print()

        else:
            print()


def main():
    RESULT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

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

    trial_rows = []
    convergence_rows = []

    trial_csv = (
        RESULT_DIR
        / "scenario03_rrt_pilot_trials.csv"
    )

    convergence_csv = (
        RESULT_DIR
        / "scenario03_rrt_star_pilot_convergence.csv"
    )

    print()
    print("=" * 92)
    print("SCENARIO 03 — RRT / RRT* PILOT")
    print("=" * 92)
    print(
        f"Seeds                      : "
        f"{SEEDS}"
    )
    print(
        f"Maximum iterations         : "
        f"{MAX_ITERATIONS}"
    )
    print(
        f"RRT* checkpoints           : "
        f"{CONVERGENCE_CHECKPOINTS}"
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
        print("#" * 92)
        print(
            context[
                "robot"
            ][
                "name"
            ]
        )
        print("#" * 92)

        for seed in SEEDS:
            rrt_row, _ = run_rrt(
                robot_key,
                context,
                temporary_distance_m,
                resolution,
                seed,
            )

            trial_rows.append(
                rrt_row
            )

            save_csv(
                trial_rows,
                trial_csv,
            )

            print_trial(
                rrt_row
            )

            (
                rrt_star_row,
                rrt_star_convergence,
                _,
            ) = run_rrt_star(
                robot_key,
                context,
                temporary_distance_m,
                resolution,
                seed,
            )

            trial_rows.append(
                rrt_star_row
            )

            convergence_rows.extend(
                rrt_star_convergence
            )

            save_csv(
                trial_rows,
                trial_csv,
            )

            save_csv(
                convergence_rows,
                convergence_csv,
            )

            print_trial(
                rrt_star_row
            )

    for robot_key in [
        "waffle",
        "ridgeback",
    ]:
        print_robot_summary(
            robot_key,
            trial_rows,
        )

    save_csv(
        trial_rows,
        trial_csv,
    )

    save_csv(
        convergence_rows,
        convergence_csv,
    )

    print()
    print(
        "Saved:"
    )
    print(
        " ",
        trial_csv,
    )
    print(
        " ",
        convergence_csv,
    )


if __name__ == "__main__":
    main()