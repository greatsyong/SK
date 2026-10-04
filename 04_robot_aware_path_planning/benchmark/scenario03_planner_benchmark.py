from pathlib import Path

import argparse
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
from planners.astar import AStarPlanner
from planners.dijkstra import DijkstraPlanner
from planners.rrt import RRTPlanner
from planners.rrt_star import RRTStarPlanner


# ============================================================
# SCENARIO 03 — FINAL PLANNER BENCHMARK
# ============================================================
#
# Purpose:
#   Compare Dijkstra, A*, RRT, and RRT* under one common
#   robot-aware Scenario 03 evaluation pipeline.
#
# Design:
#   - same map, start, goal, safety inflation, and postprocessing
#   - deterministic planners: one run each
#   - stochastic planners: fixed seeds
#   - RRT stops at first feasible raw solution
#   - RRT* runs the full iteration budget
#   - planner success, raw feasibility, and final trajectory
#     feasibility are stored separately
#   - every completed trial is checkpointed immediately
#
# This runner does not tune planners to rescue failed cases.
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
    "final_benchmark"
)

START = (360, 136)
GOAL = (744, 840)

ROBOTS = [
    "waffle",
    "ridgeback",
]

STOCHASTIC_SEEDS = list(
    range(10)
)

RRT_MAX_ITERATIONS = 50000
RRT_STAR_MAX_ITERATIONS = 50000

RRT_STAR_CHECKPOINTS = [
    5000,
    10000,
    20000,
    50000,
]


# ============================================================
# CSV utilities
# ============================================================

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


def safe_mean(values):
    values = [
        float(v)
        for v in values
        if v is not None
        and np.isfinite(v)
    ]

    if not values:
        return math.nan

    return float(
        statistics.mean(values)
    )


def safe_median(values):
    values = [
        float(v)
        for v in values
        if v is not None
        and np.isfinite(v)
    ]

    if not values:
        return math.nan

    return float(
        statistics.median(values)
    )


def safe_stdev(values):
    values = [
        float(v)
        for v in values
        if v is not None
        and np.isfinite(v)
    ]

    if len(values) < 2:
        return math.nan

    return float(
        statistics.stdev(values)
    )


def percentile(
    values,
    q,
):
    values = np.asarray(
        [
            float(v)
            for v in values
            if v is not None
            and np.isfinite(v)
        ],
        dtype=float,
    )

    if values.size == 0:
        return math.nan

    return float(
        np.percentile(
            values,
            q,
        )
    )


# ============================================================
# Common result packing
# ============================================================

def empty_eval_fields():
    return {
        "raw_path_feasible":
            False,

        "raw_min_clearance_m":
            math.nan,

        "raw_clearance_margin_m":
            math.nan,

        "raw_max_clearance_violation_mm":
            math.nan,

        "raw_violation_sample_count":
            math.nan,

        "raw_violation_path_length_m":
            math.nan,

        "raw_path_length_m":
            math.nan,

        "final_path_length_m":
            math.nan,

        "final_trajectory_feasible":
            False,

        "min_clearance_m":
            math.nan,

        "clearance_margin_m":
            math.nan,

        "max_clearance_violation_mm":
            math.nan,

        "violation_sample_count":
            math.nan,

        "violation_path_length_m":
            math.nan,

        "effective_radius_m":
            math.nan,

        "caution_distance_m":
            math.nan,

        "caution_ratio":
            math.nan,

        "open_cruise_mps":
            math.nan,

        "caution_speed_cap_mps":
            math.nan,

        "nominal_mission_time_s":
            math.nan,

        "safety_mission_time_s":
            math.nan,

        "policy_added_time_s":
            math.nan,
    }


def pack_evaluation(
    evaluated,
):
    return {
        "raw_path_feasible":
            evaluated[
                "raw_path_feasible"
            ],

        "raw_min_clearance_m":
            evaluated[
                "raw_min_clearance_m"
            ],

        "raw_clearance_margin_m":
            evaluated[
                "raw_clearance_margin_m"
            ],

        "raw_max_clearance_violation_mm":
            evaluated[
                "raw_max_clearance_violation_mm"
            ],

        "raw_violation_sample_count":
            evaluated[
                "raw_violation_sample_count"
            ],

        "raw_violation_path_length_m":
            evaluated[
                "raw_violation_path_length_m"
            ],

        "raw_path_length_m":
            evaluated[
                "raw_length_m"
            ],

        "final_path_length_m":
            evaluated[
                "final_length_m"
            ],

        "final_trajectory_feasible":
            evaluated[
                "trajectory_feasible"
            ],

        "min_clearance_m":
            evaluated[
                "min_clearance_m"
            ],

        "clearance_margin_m":
            evaluated[
                "clearance_margin_m"
            ],

        "max_clearance_violation_mm":
            evaluated[
                "max_clearance_violation_mm"
            ],

        "violation_sample_count":
            evaluated[
                "violation_sample_count"
            ],

        "violation_path_length_m":
            evaluated[
                "violation_path_length_m"
            ],

        "effective_radius_m":
            evaluated[
                "effective_radius_m"
            ],

        "caution_distance_m":
            evaluated[
                "caution_distance_m"
            ],

        "caution_ratio":
            evaluated[
                "caution_ratio"
            ],

        "open_cruise_mps":
            evaluated[
                "open_cruise_mps"
            ],

        "caution_speed_cap_mps":
            evaluated[
                "caution_cap_mps"
            ],

        "nominal_mission_time_s":
            evaluated[
                "nominal_mission_time_s"
            ],

        "safety_mission_time_s":
            evaluated[
                "safety_mission_time_s"
            ],

        "policy_added_time_s":
            evaluated[
                "safety_mission_time_s"
            ]
            - evaluated[
                "nominal_mission_time_s"
            ],
    }


def evaluate_successful_plan(
    plan,
    robot_key,
    context,
    temporary_distance_m,
    resolution,
):
    if not plan[
        "success"
    ]:
        return empty_eval_fields()

    evaluated = evaluate_path(
        raw_path=plan[
            "path"
        ],
        robot_key=robot_key,
        robot_context=context,
        temporary_distance_m=temporary_distance_m,
        resolution=resolution,
    )

    return pack_evaluation(
        evaluated
    )


# ============================================================
# Planner runners
# ============================================================

def run_deterministic(
    planner_name,
    robot_key,
    context,
    temporary_distance_m,
    resolution,
):
    if planner_name == "dijkstra":
        planner = DijkstraPlanner()
    elif planner_name == "astar":
        planner = AStarPlanner()
    else:
        raise ValueError(
            planner_name
        )

    plan = planner.plan(
        context[
            "planner_grid"
        ],
        START,
        GOAL,
    )

    row = {
        "robot":
            robot_key,

        "planner":
            planner_name,

        "seed":
            "",

        "max_iterations":
            "",

        "planner_success":
            plan[
                "success"
            ],

        "planning_time_s":
            plan[
                "planning_time"
            ],

        "expanded_nodes":
            plan[
                "expanded_nodes"
            ],

        "iterations_used":
            "",

        "first_solution_iteration":
            "",

        "tree_nodes":
            "",

        "collision_checks":
            "",

        "rewires":
            "",
    }

    row.update(
        evaluate_successful_plan(
            plan,
            robot_key,
            context,
            temporary_distance_m,
            resolution,
        )
    )

    return row


def run_rrt(
    robot_key,
    context,
    temporary_distance_m,
    resolution,
    seed,
    max_iterations,
):
    planner = RRTPlanner(
        max_iterations=max_iterations,
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
            max_iterations,

        "planner_success":
            plan[
                "success"
            ],

        "planning_time_s":
            plan[
                "planning_time"
            ],

        "expanded_nodes":
            "",

        "iterations_used":
            plan[
                "iterations"
            ],

        "first_solution_iteration":
            (
                plan[
                    "iterations"
                ]
                if plan[
                    "success"
                ]
                else ""
            ),

        "tree_nodes":
            plan[
                "tree_nodes"
            ],

        "collision_checks":
            plan[
                "collision_checks"
            ],

        "rewires":
            0,
    }

    row.update(
        evaluate_successful_plan(
            plan,
            robot_key,
            context,
            temporary_distance_m,
            resolution,
        )
    )

    return row


def best_cost_at_iteration(
    convergence_history,
    checkpoint,
):
    best = None

    for iteration, cost in convergence_history:
        if iteration > checkpoint:
            break
        best = float(
            cost
        )

    return best


def run_rrt_star(
    robot_key,
    context,
    temporary_distance_m,
    resolution,
    seed,
    max_iterations,
    checkpoints,
):
    planner = RRTStarPlanner(
        max_iterations=max_iterations,
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
            max_iterations,

        "planner_success":
            plan[
                "success"
            ],

        "planning_time_s":
            plan[
                "planning_time"
            ],

        "expanded_nodes":
            "",

        "iterations_used":
            plan[
                "iterations"
            ],

        "first_solution_iteration":
            (
                plan[
                    "first_solution_iteration"
                ]
                if plan[
                    "first_solution_iteration"
                ]
                is not None
                else ""
            ),

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
    }

    row.update(
        evaluate_successful_plan(
            plan,
            robot_key,
            context,
            temporary_distance_m,
            resolution,
        )
    )

    convergence_rows = []

    history = plan[
        "convergence_history"
    ]

    for checkpoint in checkpoints:
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

                "best_raw_goal_cost_cells":
                    (
                        cost_cells
                        if cost_cells
                        is not None
                        else math.nan
                    ),

                "best_raw_goal_cost_m":
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
    )


# ============================================================
# Console output
# ============================================================

def yes_no(
    value,
):
    return (
        "YES"
        if bool(value)
        else "NO"
    )


def print_trial(
    row,
):
    print(
        f"{row['planner']:8s} "
        f"success={yes_no(row['planner_success']):3s} "
        f"time={row['planning_time_s']:8.3f} s",
        end="",
    )

    if row[
        "planner"
    ] in {
        "rrt",
        "rrt_star",
    }:
        print(
            f" seed={int(row['seed']):2d}",
            end="",
        )

    if row[
        "planner_success"
    ]:
        print(
            f" raw={row['raw_path_length_m']:7.2f} m"
            f" final={row['final_path_length_m']:7.2f} m"
            f" raw_ok={yes_no(row['raw_path_feasible']):3s}"
            f" final_ok={yes_no(row['final_trajectory_feasible']):3s}"
            f" viol={row['max_clearance_violation_mm']:6.1f} mm"
            f" mission={row['safety_mission_time_s']:8.2f} s"
        )
    else:
        print()


# ============================================================
# Summary
# ============================================================

def build_summary_rows(
    trial_rows,
):
    summary_rows = []

    for robot_key in ROBOTS:
        for planner_name in [
            "dijkstra",
            "astar",
            "rrt",
            "rrt_star",
        ]:
            subset = [
                row
                for row in trial_rows
                if row[
                    "robot"
                ] == robot_key
                and row[
                    "planner"
                ] == planner_name
            ]

            if not subset:
                continue

            successes = [
                row
                for row in subset
                if row[
                    "planner_success"
                ]
            ]

            raw_feasible = [
                row
                for row in successes
                if row[
                    "raw_path_feasible"
                ]
            ]

            final_feasible = [
                row
                for row in successes
                if row[
                    "final_trajectory_feasible"
                ]
            ]

            summary_rows.append(
                {
                    "robot":
                        robot_key,

                    "planner":
                        planner_name,

                    "trials":
                        len(
                            subset
                        ),

                    "planner_success_count":
                        len(
                            successes
                        ),

                    "planner_success_rate":
                        (
                            len(
                                successes
                            )
                            / len(
                                subset
                            )
                        ),

                    "raw_feasible_count":
                        len(
                            raw_feasible
                        ),

                    "raw_feasible_rate_given_success":
                        (
                            len(
                                raw_feasible
                            )
                            / len(
                                successes
                            )
                            if successes
                            else math.nan
                        ),

                    "final_feasible_count":
                        len(
                            final_feasible
                        ),

                    "final_feasible_rate_given_success":
                        (
                            len(
                                final_feasible
                            )
                            / len(
                                successes
                            )
                            if successes
                            else math.nan
                        ),

                    "planning_time_median_s":
                        safe_median(
                            [
                                row[
                                    "planning_time_s"
                                ]
                                for row in subset
                            ]
                        ),

                    "planning_time_mean_s":
                        safe_mean(
                            [
                                row[
                                    "planning_time_s"
                                ]
                                for row in subset
                            ]
                        ),

                    "planning_time_std_s":
                        safe_stdev(
                            [
                                row[
                                    "planning_time_s"
                                ]
                                for row in subset
                            ]
                        ),

                    "planning_time_q1_s":
                        percentile(
                            [
                                row[
                                    "planning_time_s"
                                ]
                                for row in subset
                            ],
                            25,
                        ),

                    "planning_time_q3_s":
                        percentile(
                            [
                                row[
                                    "planning_time_s"
                                ]
                                for row in subset
                            ],
                            75,
                        ),

                    "raw_length_median_m":
                        safe_median(
                            [
                                row[
                                    "raw_path_length_m"
                                ]
                                for row in successes
                            ]
                        ),

                    "final_length_median_m":
                        safe_median(
                            [
                                row[
                                    "final_path_length_m"
                                ]
                                for row in successes
                            ]
                        ),

                    "mission_time_median_s":
                        safe_median(
                            [
                                row[
                                    "safety_mission_time_s"
                                ]
                                for row in successes
                            ]
                        ),

                    "mission_time_q1_s":
                        percentile(
                            [
                                row[
                                    "safety_mission_time_s"
                                ]
                                for row in successes
                            ],
                            25,
                        ),

                    "mission_time_q3_s":
                        percentile(
                            [
                                row[
                                    "safety_mission_time_s"
                                ]
                                for row in successes
                            ],
                            75,
                        ),

                    "max_violation_mm_median":
                        safe_median(
                            [
                                row[
                                    "max_clearance_violation_mm"
                                ]
                                for row in successes
                            ]
                        ),

                    "max_violation_mm_max":
                        (
                            max(
                                [
                                    float(
                                        row[
                                            "max_clearance_violation_mm"
                                        ]
                                    )
                                    for row in successes
                                    if np.isfinite(
                                        row[
                                            "max_clearance_violation_mm"
                                        ]
                                    )
                                ],
                                default=math.nan,
                            )
                        ),

                    "violation_length_median_m":
                        safe_median(
                            [
                                row[
                                    "violation_path_length_m"
                                ]
                                for row in successes
                            ]
                        ),

                    "first_solution_iteration_median":
                        safe_median(
                            [
                                (
                                    row[
                                        "first_solution_iteration"
                                    ]
                                    if row[
                                        "first_solution_iteration"
                                    ]
                                    != ""
                                    else math.nan
                                )
                                for row in subset
                            ]
                        ),
                }
            )

    return summary_rows


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--smoke",
        action="store_true",
        help=(
            "Short integration test: one stochastic seed and "
            "5000-iteration stochastic budgets."
        ),
    )

    args = parser.parse_args()

    if args.smoke:
        stochastic_seeds = [
            0,
        ]
        rrt_budget = 5000
        rrt_star_budget = 5000
        checkpoints = [
            5000,
        ]
        result_dir = (
            RESULT_DIR
            / "smoke"
        )
    else:
        stochastic_seeds = (
            STOCHASTIC_SEEDS
        )
        rrt_budget = (
            RRT_MAX_ITERATIONS
        )
        rrt_star_budget = (
            RRT_STAR_MAX_ITERATIONS
        )
        checkpoints = (
            RRT_STAR_CHECKPOINTS
        )
        result_dir = (
            RESULT_DIR
        )

    result_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    trial_csv = (
        result_dir
        / "scenario03_planner_trials.csv"
    )

    convergence_csv = (
        result_dir
        / "scenario03_rrt_star_convergence.csv"
    )

    summary_csv = (
        result_dir
        / "scenario03_planner_summary.csv"
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

    print()
    print("=" * 96)
    print(
        "SCENARIO 03 — FINAL PLANNER BENCHMARK"
        if not args.smoke
        else "SCENARIO 03 — FINAL BENCHMARK SMOKE TEST"
    )
    print("=" * 96)
    print(
        f"Stochastic seeds       : "
        f"{stochastic_seeds}"
    )
    print(
        f"RRT iteration budget   : "
        f"{rrt_budget}"
    )
    print(
        f"RRT* iteration budget  : "
        f"{rrt_star_budget}"
    )
    print(
        f"RRT* checkpoints       : "
        f"{checkpoints}"
    )

    for robot_key in ROBOTS:
        context = build_robot_context(
            robot_key,
            raw_free,
            resolution,
            all_obstacle_clearance_m=clearance_m,
        )

        print()
        print("#" * 96)
        print(
            context[
                "robot"
            ][
                "name"
            ]
        )
        print("#" * 96)

        for planner_name in [
            "dijkstra",
            "astar",
        ]:
            row = run_deterministic(
                planner_name,
                robot_key,
                context,
                temporary_distance_m,
                resolution,
            )

            trial_rows.append(
                row
            )

            save_csv(
                trial_rows,
                trial_csv,
            )

            print_trial(
                row
            )

        for seed in stochastic_seeds:
            row = run_rrt(
                robot_key,
                context,
                temporary_distance_m,
                resolution,
                seed,
                rrt_budget,
            )

            trial_rows.append(
                row
            )

            save_csv(
                trial_rows,
                trial_csv,
            )

            print_trial(
                row
            )

            (
                row,
                convergence,
            ) = run_rrt_star(
                robot_key,
                context,
                temporary_distance_m,
                resolution,
                seed,
                rrt_star_budget,
                checkpoints,
            )

            trial_rows.append(
                row
            )

            convergence_rows.extend(
                convergence
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
                row
            )

            summary_rows = build_summary_rows(
                trial_rows
            )

            save_csv(
                summary_rows,
                summary_csv,
            )

    summary_rows = build_summary_rows(
        trial_rows
    )

    save_csv(
        summary_rows,
        summary_csv,
    )

    print()
    print("=" * 96)
    print("Saved")
    print("=" * 96)
    print(
        "Trials      :",
        trial_csv,
    )
    print(
        "Convergence :",
        convergence_csv,
    )
    print(
        "Summary     :",
        summary_csv,
    )


if __name__ == "__main__":
    main()