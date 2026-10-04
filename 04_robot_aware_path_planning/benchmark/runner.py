from datetime import datetime
from pathlib import Path
import csv
import math
import statistics

from benchmark.map_loader import load_movingai_map
from benchmark.scenario_loader import load_movingai_scenarios

from planners.dijkstra import DijkstraPlanner
from planners.astar import AStarPlanner


# ============================================================
# BENCHMARK CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

MAP_DIR = PROJECT_ROOT / "data" / "maps"
SCENARIO_DIR = PROJECT_ROOT / "data" / "scenarios"
RESULT_DIR = PROJECT_ROOT / "results"


BENCHMARK_MAPS = [
    {
        "name": "maze512-32-0",
        "map_file": "maze512-32-0.map",
        "scenario_file": "maze512-32-0.map.scen",
    },
    {
        "name": "16room_000",
        "map_file": "16room_000.map",
        "scenario_file": "16room_000.map.scen",
    },
    {
        "name": "AR0011SR",
        "map_file": "AR0011SR.map",
        "scenario_file": "AR0011SR.map.scen",
    },
]


PLANNERS = [
    DijkstraPlanner(),
    AStarPlanner(),
]


MIN_REFERENCE_LENGTH = 200.0
SCENARIOS_PER_MAP = 30


# ============================================================
# UTILITY
# ============================================================

def make_timestamp():
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def planner_tag():
    return "_".join(planner.name for planner in PLANNERS)


def ensure_result_directory():
    RESULT_DIR.mkdir(parents=True, exist_ok=True)


def select_scenarios(scenarios):
    """
    Select sufficiently difficult scenarios so trivial
    start-goal pairs do not dominate the benchmark.
    """

    selected = [
        scenario
        for scenario in scenarios
        if scenario.optimal_length >= MIN_REFERENCE_LENGTH
    ]

    return selected[:SCENARIOS_PER_MAP]


# ============================================================
# SINGLE PLANNER RUN
# ============================================================

def run_single_trial(
    planner,
    grid,
    scenario,
    scenario_index,
    map_name,
):

    result = planner.plan(
        grid,
        scenario.start,
        scenario.goal,
    )

    if result["success"]:
        optimality_error = abs(
            result["path_length"]
            - scenario.optimal_length
        )

        relative_error = (
            optimality_error
            / scenario.optimal_length
            if scenario.optimal_length > 0
            else 0.0
        )

    else:
        optimality_error = math.inf
        relative_error = math.inf

    return {
        "map": map_name,
        "scenario_index": scenario_index,
        "bucket": scenario.bucket,

        "start_x": scenario.start[0],
        "start_y": scenario.start[1],

        "goal_x": scenario.goal[0],
        "goal_y": scenario.goal[1],

        "reference_length": scenario.optimal_length,

        "planner": planner.name,

        "success": result["success"],

        "path_length": result["path_length"],

        "absolute_optimality_error": optimality_error,
        "relative_optimality_error": relative_error,

        "expanded_nodes": result["expanded_nodes"],

        "planning_time_s": result["planning_time"],
        "planning_time_ms": result["planning_time"] * 1000.0,
    }


# ============================================================
# MAP BENCHMARK
# ============================================================

def run_map_benchmark(map_config, timestamp):

    map_name = map_config["name"]

    map_path = (
        MAP_DIR
        / map_config["map_file"]
    )

    scenario_path = (
        SCENARIO_DIR
        / map_config["scenario_file"]
    )

    print()
    print("=" * 72)
    print(f"MAP: {map_name}")
    print("=" * 72)

    map_data = load_movingai_map(map_path)

    scenarios = load_movingai_scenarios(
        scenario_path
    )

    selected_scenarios = select_scenarios(
        scenarios
    )

    print(
        f"Selected scenarios: "
        f"{len(selected_scenarios)}"
    )

    if not selected_scenarios:
        raise RuntimeError(
            f"No scenarios satisfy "
            f"minimum reference length "
            f"{MIN_REFERENCE_LENGTH}"
        )

    rows = []

    for i, scenario in enumerate(
        selected_scenarios
    ):

        print(
            f"\nScenario {i + 1:02d}/"
            f"{len(selected_scenarios)} "
            f"| reference = "
            f"{scenario.optimal_length:.3f}"
        )

        for planner in PLANNERS:

            trial = run_single_trial(
                planner=planner,
                grid=map_data["grid"],
                scenario=scenario,
                scenario_index=i,
                map_name=map_name,
            )

            rows.append(trial)

            print(
                f"  {planner.name:10s}"
                f" success="
                f"{trial['success']} "
                f"time="
                f"{trial['planning_time_ms']:.3f} ms "
                f"expanded="
                f"{trial['expanded_nodes']} "
                f"length="
                f"{trial['path_length']:.6f}"
            )

    filename = (
        f"{map_name}_"
        f"{planner_tag()}_"
        f"{timestamp}.csv"
    )

    output_path = RESULT_DIR / filename

    save_results_csv(
        rows,
        output_path
    )

    return rows, output_path


# ============================================================
# CSV WRITER
# ============================================================

def save_results_csv(rows, output_path):

    if not rows:
        return

    fieldnames = list(
        rows[0].keys()
    )

    with output_path.open(
        "w",
        newline="",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        writer.writerows(rows)

    print()
    print(
        f"Saved raw results:"
    )
    print(output_path)


# ============================================================
# SUMMARY STATISTICS
# ============================================================

def percentile(values, percentile_value):

    if not values:
        return math.nan

    values = sorted(values)

    k = (
        (len(values) - 1)
        * percentile_value
        / 100.0
    )

    lower = math.floor(k)
    upper = math.ceil(k)

    if lower == upper:
        return values[int(k)]

    lower_value = values[lower]
    upper_value = values[upper]

    return (
        lower_value * (upper - k)
        + upper_value * (k - lower)
    )


def generate_summary(all_rows):

    summary_rows = []

    map_names = sorted(
        set(
            row["map"]
            for row in all_rows
        )
    )

    planner_names = [
        planner.name
        for planner in PLANNERS
    ]

    for map_name in map_names:

        for planner_name in planner_names:

            subset = [
                row
                for row in all_rows
                if row["map"] == map_name
                and row["planner"]
                == planner_name
            ]

            successful = [
                row
                for row in subset
                if row["success"]
            ]

            total_trials = len(subset)
            success_count = len(successful)

            success_rate = (
                success_count
                / total_trials
                if total_trials
                else 0.0
            )

            if successful:

                times = [
                    row["planning_time_ms"]
                    for row in successful
                ]

                expanded = [
                    row["expanded_nodes"]
                    for row in successful
                ]

                abs_errors = [
                    row[
                        "absolute_optimality_error"
                    ]
                    for row in successful
                ]

                rel_errors = [
                    row[
                        "relative_optimality_error"
                    ]
                    for row in successful
                ]

                mean_time = statistics.mean(times)
                median_time = statistics.median(times)
                p95_time = percentile(times, 95)

                mean_expanded = statistics.mean(
                    expanded
                )

                median_expanded = (
                    statistics.median(
                        expanded
                    )
                )

                mean_abs_error = (
                    statistics.mean(
                        abs_errors
                    )
                )

                max_abs_error = max(
                    abs_errors
                )

                mean_rel_error = (
                    statistics.mean(
                        rel_errors
                    )
                )

            else:

                mean_time = math.nan
                median_time = math.nan
                p95_time = math.nan

                mean_expanded = math.nan
                median_expanded = math.nan

                mean_abs_error = math.nan
                max_abs_error = math.nan
                mean_rel_error = math.nan

            summary_rows.append(
                {
                    "map": map_name,
                    "planner": planner_name,

                    "trials": total_trials,
                    "success_count": success_count,
                    "success_rate": success_rate,

                    "mean_planning_time_ms":
                        mean_time,

                    "median_planning_time_ms":
                        median_time,

                    "p95_planning_time_ms":
                        p95_time,

                    "mean_expanded_nodes":
                        mean_expanded,

                    "median_expanded_nodes":
                        median_expanded,

                    "mean_absolute_optimality_error":
                        mean_abs_error,

                    "max_absolute_optimality_error":
                        max_abs_error,

                    "mean_relative_optimality_error":
                        mean_rel_error,
                }
            )

    return summary_rows


# ============================================================
# MAIN
# ============================================================

def main():

    ensure_result_directory()

    timestamp = make_timestamp()

    print()
    print("=" * 72)
    print("PATH PLANNING BENCHMARK")
    print("=" * 72)

    print(
        "Planners:",
        ", ".join(
            planner.name
            for planner in PLANNERS
        ),
    )

    print(
        "Maps:",
        ", ".join(
            config["name"]
            for config in BENCHMARK_MAPS
        ),
    )

    print(
        "Minimum reference path length:",
        MIN_REFERENCE_LENGTH,
    )

    print(
        "Scenarios per map:",
        SCENARIOS_PER_MAP,
    )

    all_rows = []

    for map_config in BENCHMARK_MAPS:

        rows, _ = run_map_benchmark(
            map_config,
            timestamp,
        )

        all_rows.extend(rows)

    summary = generate_summary(
        all_rows
    )

    summary_filename = (
        f"summary_"
        f"{planner_tag()}_"
        f"{timestamp}.csv"
    )

    summary_path = (
        RESULT_DIR
        / summary_filename
    )

    save_results_csv(
        summary,
        summary_path,
    )

    print()
    print("=" * 72)
    print("BENCHMARK COMPLETE")
    print("=" * 72)

    print(
        f"Total trials: "
        f"{len(all_rows)}"
    )

    print(
        f"Summary:"
    )
    print(summary_path)


if __name__ == "__main__":
    main()