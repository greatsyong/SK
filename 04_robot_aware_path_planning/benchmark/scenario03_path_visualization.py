from pathlib import Path

import argparse
import csv
import math

import matplotlib.pyplot as plt
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
# SCENARIO 03 — PATH VISUALIZATION
# ============================================================
#
# Purpose:
#   Recreate representative planner paths from the final
#   benchmark and visualize:
#
#   1) Waffle planner comparison
#   2) Ridgeback planner comparison
#   3) Waffle RRT seed variability
#   4) Waffle RRT* convergence paths
#   5) Ridgeback clearance-violation locations
#
# Representative stochastic seeds are selected automatically
# from the final benchmark CSV as the successful trial whose
# safety-aware mission time is closest to the median successful
# mission time for that robot/planner.
#
# IMPORTANT:
#   - This script does not change planner settings.
#   - It reproduces selected benchmark trials using the same
#     fixed seeds and 50k iteration budgets.
#   - RRT* convergence visualization reruns one representative
#     Waffle seed at 5k/10k/20k/50k budgets so actual paths,
#     rather than only convergence costs, can be plotted.
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

TRIAL_CSV = Path(
    "results/stech_lab/scenario03/"
    "final_benchmark/"
    "scenario03_planner_trials.csv"
)

RESULT_DIR = Path(
    "results/stech_lab/scenario03/"
    "final_benchmark/"
    "visualization"
)

START = (360, 136)
GOAL = (744, 840)

RRT_MAX_ITERATIONS = 50000
RRT_STAR_MAX_ITERATIONS = 50000

RRT_STAR_CHECKPOINTS = [
    5000,
    10000,
    20000,
    50000,
]

WAFFLE_RRT_VARIABILITY_SEEDS = list(
    range(10)
)


# ============================================================
# CSV helpers
# ============================================================

def parse_bool(value):
    return str(value).strip().lower() in {
        "true",
        "1",
        "yes",
    }


def parse_float(value):
    try:
        result = float(value)
    except (
        TypeError,
        ValueError,
    ):
        return math.nan

    return result


def parse_int(value):
    try:
        return int(
            float(value)
        )
    except (
        TypeError,
        ValueError,
    ):
        return None


def load_trials(
    path,
):
    if not path.exists():
        raise FileNotFoundError(
            f"Final benchmark CSV not found:\n{path}"
        )

    with path.open(
        newline="",
    ) as f:
        rows = list(
            csv.DictReader(
                f
            )
        )

    return rows


def select_representative_seed(
    rows,
    robot_key,
    planner_name,
):
    candidates = []

    for row in rows:
        if (
            row.get(
                "robot"
            ) != robot_key
            or row.get(
                "planner"
            ) != planner_name
        ):
            continue

        if not parse_bool(
            row.get(
                "planner_success",
                False,
            )
        ):
            continue

        mission = parse_float(
            row.get(
                "safety_mission_time_s"
            )
        )

        seed = parse_int(
            row.get(
                "seed"
            )
        )

        if (
            seed is None
            or not np.isfinite(
                mission
            )
        ):
            continue

        candidates.append(
            (
                seed,
                mission,
            )
        )

    if not candidates:
        return None

    missions = np.asarray(
        [
            item[1]
            for item in candidates
        ],
        dtype=float,
    )

    median = float(
        np.median(
            missions
        )
    )

    representative = min(
        candidates,
        key=lambda item: (
            abs(
                item[1]
                - median
            ),
            item[0],
        ),
    )

    return {
        "seed":
            representative[0],

        "mission_time_s":
            representative[1],

        "successful_median_mission_time_s":
            median,
    }


# ============================================================
# Planner execution
# ============================================================

def run_planner(
    planner_name,
    context,
    seed=None,
    max_iterations=None,
):
    if planner_name == "dijkstra":
        planner = DijkstraPlanner()

        return planner.plan(
            context[
                "planner_grid"
            ],
            START,
            GOAL,
        )

    if planner_name == "astar":
        planner = AStarPlanner()

        return planner.plan(
            context[
                "planner_grid"
            ],
            START,
            GOAL,
        )

    if planner_name == "rrt":
        planner = RRTPlanner(
            max_iterations=(
                RRT_MAX_ITERATIONS
                if max_iterations is None
                else max_iterations
            ),
            seed=seed,
        )

        return planner.plan(
            context[
                "planner_grid"
            ],
            START,
            GOAL,
            return_tree=False,
        )

    if planner_name == "rrt_star":
        planner = RRTStarPlanner(
            max_iterations=(
                RRT_STAR_MAX_ITERATIONS
                if max_iterations is None
                else max_iterations
            ),
            seed=seed,
        )

        return planner.plan(
            context[
                "planner_grid"
            ],
            START,
            GOAL,
            return_tree=False,
        )

    raise ValueError(
        planner_name
    )


def recreate_and_evaluate(
    planner_name,
    robot_key,
    context,
    temporary_distance_m,
    resolution,
    seed=None,
    max_iterations=None,
):
    plan = run_planner(
        planner_name,
        context,
        seed=seed,
        max_iterations=max_iterations,
    )

    result = {
        "planner":
            planner_name,

        "seed":
            seed,

        "max_iterations":
            max_iterations,

        "plan":
            plan,

        "evaluation":
            None,
    }

    if plan[
        "success"
    ]:
        result[
            "evaluation"
        ] = evaluate_path(
            raw_path=plan[
                "path"
            ],
            robot_key=robot_key,
            robot_context=context,
            temporary_distance_m=temporary_distance_m,
            resolution=resolution,
        )

    return result


# ============================================================
# Plot helpers
# ============================================================

def draw_map(
    ax,
    raw_free,
):
    occupancy = (
        ~raw_free
    ).astype(
        float
    )

    ax.imshow(
        occupancy,
        origin="upper",
        alpha=0.35,
    )

    ax.set_aspect(
        "equal"
    )

    ax.set_xlabel(
        "Map x [cell]"
    )

    ax.set_ylabel(
        "Map y [cell]"
    )


def draw_start_goal(
    ax,
):
    ax.scatter(
        [
            START[0],
        ],
        [
            START[1],
        ],
        marker="o",
        s=65,
        label="Start",
        zorder=20,
    )

    ax.scatter(
        [
            GOAL[0],
        ],
        [
            GOAL[1],
        ],
        marker="*",
        s=110,
        label="Goal",
        zorder=20,
    )


def draw_result_path(
    ax,
    result,
    label_prefix,
    draw_raw=True,
    draw_final=True,
):
    if not result[
        "plan"
    ][
        "success"
    ]:
        return

    evaluated = result[
        "evaluation"
    ]

    if draw_raw:
        raw = np.asarray(
            evaluated[
                "raw"
            ],
            dtype=float,
        )

        ax.plot(
            raw[
                :,
                0
            ],
            raw[
                :,
                1
            ],
            linestyle="--",
            linewidth=1.0,
            alpha=0.55,
            label=(
                f"{label_prefix} raw"
            ),
        )

    if draw_final:
        final = np.asarray(
            evaluated[
                "resampled"
            ],
            dtype=float,
        )

        ax.plot(
            final[
                :,
                0
            ],
            final[
                :,
                1
            ],
            linewidth=2.0,
            label=(
                f"{label_prefix} final"
            ),
        )


def draw_violation_samples(
    ax,
    result,
    label_prefix,
):
    if (
        not result[
            "plan"
        ][
            "success"
        ]
        or result[
            "evaluation"
        ] is None
    ):
        return

    evaluated = result[
        "evaluation"
    ]

    path = np.asarray(
        evaluated[
            "resampled"
        ],
        dtype=float,
    )

    mask = np.asarray(
        evaluated[
            "violation_mask"
        ],
        dtype=bool,
    )

    if not np.any(
        mask
    ):
        return

    violated = path[
        mask
    ]

    ax.scatter(
        violated[
            :,
            0
        ],
        violated[
            :,
            1
        ],
        marker="x",
        s=32,
        linewidths=1.2,
        label=(
            f"{label_prefix} violation"
        ),
        zorder=15,
    )


def save_planner_overlay(
    robot_name,
    raw_free,
    selected_results,
    output_path,
):
    fig, ax = plt.subplots(
        figsize=(
            11,
            9,
        )
    )

    draw_map(
        ax,
        raw_free,
    )

    for result in selected_results:
        planner_name = result[
            "planner"
        ]

        seed = result[
            "seed"
        ]

        if seed is None:
            label = planner_name
        else:
            label = (
                f"{planner_name} "
                f"(seed {seed})"
            )

        draw_result_path(
            ax,
            result,
            label,
            draw_raw=False,
            draw_final=True,
        )

    draw_start_goal(
        ax
    )

    ax.set_title(
        f"Scenario 03 — {robot_name}: Representative Final Paths"
    )

    ax.legend(
        fontsize=8,
    )

    fig.tight_layout()

    fig.savefig(
        output_path,
        dpi=180,
        bbox_inches="tight",
    )

    plt.close(
        fig
    )


def save_waffle_rrt_variability(
    raw_free,
    results,
    output_path,
):
    fig, ax = plt.subplots(
        figsize=(
            11,
            9,
        )
    )

    draw_map(
        ax,
        raw_free,
    )

    for result in results:
        seed = result[
            "seed"
        ]

        if not result[
            "plan"
        ][
            "success"
        ]:
            continue

        final = np.asarray(
            result[
                "evaluation"
            ][
                "resampled"
            ],
            dtype=float,
        )

        ax.plot(
            final[
                :,
                0
            ],
            final[
                :,
                1
            ],
            linewidth=1.2,
            alpha=0.7,
            label=(
                f"Seed {seed}"
            ),
        )

    draw_start_goal(
        ax
    )

    ax.set_title(
        "Scenario 03 — Waffle RRT Seed-to-Seed Path Variability"
    )

    ax.legend(
        fontsize=7,
        ncol=2,
    )

    fig.tight_layout()

    fig.savefig(
        output_path,
        dpi=180,
        bbox_inches="tight",
    )

    plt.close(
        fig
    )


def save_waffle_rrtstar_convergence(
    raw_free,
    results,
    representative_seed,
    output_path,
):
    fig, ax = plt.subplots(
        figsize=(
            11,
            9,
        )
    )

    draw_map(
        ax,
        raw_free,
    )

    unavailable = []

    for result in results:
        budget = result[
            "max_iterations"
        ]

        if not result[
            "plan"
        ][
            "success"
        ]:
            unavailable.append(
                budget
            )
            continue

        final = np.asarray(
            result[
                "evaluation"
            ][
                "resampled"
            ],
            dtype=float,
        )

        ax.plot(
            final[
                :,
                0
            ],
            final[
                :,
                1
            ],
            linewidth=1.8,
            label=(
                f"{budget // 1000}k iterations"
            ),
        )

    draw_start_goal(
        ax
    )

    title = (
        "Scenario 03 — Waffle RRT* Path Convergence "
        f"(representative seed {representative_seed})"
    )

    if unavailable:
        title += (
            "\nNo solution at: "
            + ", ".join(
                f"{budget // 1000}k"
                for budget in unavailable
            )
        )

    ax.set_title(
        title
    )

    ax.legend(
        fontsize=8,
    )

    fig.tight_layout()

    fig.savefig(
        output_path,
        dpi=180,
        bbox_inches="tight",
    )

    plt.close(
        fig
    )


def save_ridgeback_violation_overlay(
    raw_free,
    selected_results,
    output_path,
):
    fig, ax = plt.subplots(
        figsize=(
            11,
            9,
        )
    )

    draw_map(
        ax,
        raw_free,
    )

    for result in selected_results:
        planner_name = result[
            "planner"
        ]

        seed = result[
            "seed"
        ]

        if seed is None:
            label = planner_name
        else:
            label = (
                f"{planner_name} "
                f"(seed {seed})"
            )

        draw_result_path(
            ax,
            result,
            label,
            draw_raw=False,
            draw_final=True,
        )

        draw_violation_samples(
            ax,
            result,
            label,
        )

    draw_start_goal(
        ax
    )

    ax.set_title(
        "Scenario 03 — Ridgeback Final Paths and Clearance Violations"
    )

    ax.legend(
        fontsize=7,
        ncol=2,
    )

    fig.tight_layout()

    fig.savefig(
        output_path,
        dpi=180,
        bbox_inches="tight",
    )

    plt.close(
        fig
    )


# ============================================================
# Summary text
# ============================================================

def write_summary(
    output_path,
    representative,
    selected_results,
    convergence_results,
):
    lines = []

    lines.append(
        "SCENARIO 03 — PATH VISUALIZATION SUMMARY"
    )

    lines.append(
        "=" * 72
    )

    lines.append(
        ""
    )

    lines.append(
        "Representative stochastic seeds"
    )

    lines.append(
        "-" * 72
    )

    for key, info in representative.items():
        lines.append(
            (
                f"{key:24s}: "
                f"seed={info['seed']}, "
                f"trial mission={info['mission_time_s']:.3f} s, "
                f"successful median={info['successful_median_mission_time_s']:.3f} s"
            )
        )

    lines.append(
        ""
    )

    lines.append(
        "Recreated representative paths"
    )

    lines.append(
        "-" * 72
    )

    for robot_key, results in selected_results.items():
        lines.append(
            robot_key
        )

        for result in results:
            planner = result[
                "planner"
            ]

            seed = result[
                "seed"
            ]

            plan = result[
                "plan"
            ]

            if not plan[
                "success"
            ]:
                lines.append(
                    f"  {planner:10s} seed={seed}: planner failed"
                )
                continue

            evaluated = result[
                "evaluation"
            ]

            lines.append(
                (
                    f"  {planner:10s} seed={str(seed):>4s}: "
                    f"raw={evaluated['raw_length_m']:.3f} m, "
                    f"final={evaluated['final_length_m']:.3f} m, "
                    f"raw_ok={evaluated['raw_path_feasible']}, "
                    f"final_ok={evaluated['trajectory_feasible']}, "
                    f"max_violation={evaluated['max_clearance_violation_mm']:.1f} mm"
                )
            )

    lines.append(
        ""
    )

    lines.append(
        "Waffle RRT* convergence-path reruns"
    )

    lines.append(
        "-" * 72
    )

    for result in convergence_results:
        budget = result[
            "max_iterations"
        ]

        plan = result[
            "plan"
        ]

        if not plan[
            "success"
        ]:
            lines.append(
                f"  {budget:6d}: no solution"
            )
            continue

        evaluated = result[
            "evaluation"
        ]

        lines.append(
            (
                f"  {budget:6d}: "
                f"raw={evaluated['raw_length_m']:.3f} m, "
                f"final={evaluated['final_length_m']:.3f} m, "
                f"mission={evaluated['safety_mission_time_s']:.3f} s"
            )
        )

    output_path.write_text(
        "\n".join(
            lines
        )
        + "\n",
        encoding="utf-8",
    )


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--skip-convergence",
        action="store_true",
        help=(
            "Skip expensive Waffle RRT* 5k/10k/20k/50k "
            "path reruns."
        ),
    )

    args = parser.parse_args()

    RESULT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    trials = load_trials(
        TRIAL_CSV
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

    contexts = {
        robot_key:
            build_robot_context(
                robot_key,
                raw_free,
                resolution,
                all_obstacle_clearance_m=clearance_m,
            )
        for robot_key in [
            "waffle",
            "ridgeback",
        ]
    }

    representative = {}

    for robot_key in [
        "waffle",
        "ridgeback",
    ]:
        for planner_name in [
            "rrt",
            "rrt_star",
        ]:
            info = select_representative_seed(
                trials,
                robot_key,
                planner_name,
            )

            if info is None:
                raise RuntimeError(
                    (
                        "No successful stochastic trial found for "
                        f"{robot_key}/{planner_name}"
                    )
                )

            representative[
                f"{robot_key}_{planner_name}"
            ] = info

    print()
    print(
        "=" * 88
    )
    print(
        "SCENARIO 03 — PATH VISUALIZATION"
    )
    print(
        "=" * 88
    )

    for key, info in representative.items():
        print(
            f"{key:24s} -> "
            f"seed {info['seed']} "
            f"(mission {info['mission_time_s']:.3f} s, "
            f"successful median "
            f"{info['successful_median_mission_time_s']:.3f} s)"
        )

    selected_results = {}

    for robot_key in [
        "waffle",
        "ridgeback",
    ]:
        print()
        print(
            f"Recreating representative paths — {robot_key}"
        )

        context = contexts[
            robot_key
        ]

        robot_results = []

        for planner_name in [
            "dijkstra",
            "astar",
        ]:
            result = recreate_and_evaluate(
                planner_name,
                robot_key,
                context,
                temporary_distance_m,
                resolution,
            )

            robot_results.append(
                result
            )

            print(
                f"  {planner_name:10s} "
                f"success={result['plan']['success']}"
            )

        for planner_name in [
            "rrt",
            "rrt_star",
        ]:
            seed = representative[
                f"{robot_key}_{planner_name}"
            ][
                "seed"
            ]

            result = recreate_and_evaluate(
                planner_name,
                robot_key,
                context,
                temporary_distance_m,
                resolution,
                seed=seed,
                max_iterations=(
                    RRT_MAX_ITERATIONS
                    if planner_name == "rrt"
                    else RRT_STAR_MAX_ITERATIONS
                ),
            )

            robot_results.append(
                result
            )

            print(
                f"  {planner_name:10s} "
                f"seed={seed:2d} "
                f"success={result['plan']['success']}"
            )

        selected_results[
            robot_key
        ] = robot_results

    waffle_overlay = (
        RESULT_DIR
        / "scenario03_waffle_planner_overlay.png"
    )

    ridgeback_overlay = (
        RESULT_DIR
        / "scenario03_ridgeback_planner_overlay.png"
    )

    ridgeback_violation = (
        RESULT_DIR
        / "scenario03_ridgeback_violation_overlay.png"
    )

    save_planner_overlay(
        contexts[
            "waffle"
        ][
            "robot"
        ][
            "name"
        ],
        raw_free,
        selected_results[
            "waffle"
        ],
        waffle_overlay,
    )

    save_planner_overlay(
        contexts[
            "ridgeback"
        ][
            "robot"
        ][
            "name"
        ],
        raw_free,
        selected_results[
            "ridgeback"
        ],
        ridgeback_overlay,
    )

    save_ridgeback_violation_overlay(
        raw_free,
        selected_results[
            "ridgeback"
        ],
        ridgeback_violation,
    )

    print()
    print(
        "Recreating Waffle RRT seed variability..."
    )

    waffle_rrt_results = []

    for seed in WAFFLE_RRT_VARIABILITY_SEEDS:
        result = recreate_and_evaluate(
            "rrt",
            "waffle",
            contexts[
                "waffle"
            ],
            temporary_distance_m,
            resolution,
            seed=seed,
            max_iterations=RRT_MAX_ITERATIONS,
        )

        waffle_rrt_results.append(
            result
        )

        print(
            f"  seed={seed:2d} "
            f"success={result['plan']['success']}"
        )

    waffle_rrt_variability = (
        RESULT_DIR
        / "scenario03_waffle_rrt_variability.png"
    )

    save_waffle_rrt_variability(
        raw_free,
        waffle_rrt_results,
        waffle_rrt_variability,
    )

    convergence_results = []

    waffle_rrtstar_convergence = (
        RESULT_DIR
        / "scenario03_waffle_rrtstar_convergence.png"
    )

    if not args.skip_convergence:
        representative_seed = representative[
            "waffle_rrt_star"
        ][
            "seed"
        ]

        print()
        print(
            "Recreating Waffle RRT* convergence paths "
            f"for seed {representative_seed}..."
        )

        for budget in RRT_STAR_CHECKPOINTS:
            result = recreate_and_evaluate(
                "rrt_star",
                "waffle",
                contexts[
                    "waffle"
                ],
                temporary_distance_m,
                resolution,
                seed=representative_seed,
                max_iterations=budget,
            )

            convergence_results.append(
                result
            )

            if result[
                "plan"
            ][
                "success"
            ]:
                evaluated = result[
                    "evaluation"
                ]

                print(
                    f"  {budget:6d}: "
                    f"success=True "
                    f"final={evaluated['final_length_m']:.2f} m "
                    f"mission={evaluated['safety_mission_time_s']:.2f} s"
                )
            else:
                print(
                    f"  {budget:6d}: "
                    f"success=False"
                )

        save_waffle_rrtstar_convergence(
            raw_free,
            convergence_results,
            representative_seed,
            waffle_rrtstar_convergence,
        )

    summary_txt = (
        RESULT_DIR
        / "scenario03_visualization_summary.txt"
    )

    write_summary(
        summary_txt,
        representative,
        selected_results,
        convergence_results,
    )

    print()
    print(
        "=" * 88
    )
    print(
        "Saved"
    )
    print(
        "=" * 88
    )

    print(
        waffle_overlay
    )
    print(
        ridgeback_overlay
    )
    print(
        ridgeback_violation
    )
    print(
        waffle_rrt_variability
    )

    if not args.skip_convergence:
        print(
            waffle_rrtstar_convergence
        )

    print(
        summary_txt
    )


if __name__ == "__main__":
    main()