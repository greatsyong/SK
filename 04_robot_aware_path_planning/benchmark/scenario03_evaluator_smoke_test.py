from pathlib import Path

import numpy as np

from benchmark.metric_map_loader import load_ros_metric_map
from benchmark.scenario03_common_evaluator import (
    build_clearance_field,
    build_robot_context,
    evaluate_path,
)
from planners.astar import AStarPlanner


# ============================================================
# SCENARIO 03 — COMMON EVALUATOR SMOKE TEST
# ============================================================
#
# Purpose:
#   Verify that the new planner-independent evaluator reproduces
#   the current A* Scenario 03 behavior before adding the full
#   planner benchmark.
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
    print("SCENARIO 03 — COMMON EVALUATOR A* SMOKE TEST")
    print("=" * 92)

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

        planner = AStarPlanner()

        plan = planner.plan(
            context[
                "planner_grid"
            ],
            START,
            GOAL,
        )

        print()
        print(
            context[
                "robot"
            ][
                "name"
            ]
        )

        print(
            "-" * 64
        )

        print(
            f"Planner success            : "
            f"{plan['success']}"
        )

        print(
            f"Planning time              : "
            f"{plan['planning_time'] * 1000.0:.3f} ms"
        )

        print(
            f"Expanded nodes             : "
            f"{plan['expanded_nodes']}"
        )

        if not plan[
            "success"
        ]:
            continue

        evaluated = evaluate_path(
            raw_path=plan[
                "path"
            ],
            robot_key=robot_key,
            robot_context=context,
            temporary_distance_m=temporary_distance_m,
            resolution=resolution,
        )

        print(
            f"Raw path length            : "
            f"{evaluated['raw_length_m']:.2f} m"
        )

        print(
            f"Final path length          : "
            f"{evaluated['final_length_m']:.2f} m"
        )

        print(
            f"Trajectory feasible        : "
            f"{'YES' if evaluated['trajectory_feasible'] else 'NO'}"
        )

        print(
            f"Minimum clearance          : "
            f"{evaluated['min_clearance_m']:.3f} m"
        )

        print(
            f"Caution-zone distance      : "
            f"{evaluated['caution_distance_m']:.2f} m"
        )

        print(
            f"Caution-zone ratio         : "
            f"{100.0 * evaluated['caution_ratio']:.1f}%"
        )

        print(
            f"Open-space cruise          : "
            f"{evaluated['open_cruise_mps']:.3f} m/s"
        )

        print(
            f"Caution speed cap          : "
            f"{evaluated['caution_cap_mps']:.3f} m/s"
        )

        print(
            f"Nominal mission time       : "
            f"{evaluated['nominal_mission_time_s']:.3f} s"
        )

        print(
            f"Safety-aware mission time  : "
            f"{evaluated['safety_mission_time_s']:.3f} s"
        )

        print(
            f"Added time from policy     : "
            f"{evaluated['safety_time_penalty_s']:.3f} s"
        )
        print(
            f"Raw path feasible          : "
            f"{'YES' if evaluated['raw_path_feasible'] else 'NO'}"
        )

        print(
            f"Raw minimum clearance      : "
            f"{evaluated['raw_min_clearance_m']:.3f} m"
        )

        print(
            f"Raw max violation          : "
            f"{evaluated['raw_max_clearance_violation_mm']:.1f} mm"
        )

        print(
            f"Raw violation length       : "
            f"{evaluated['raw_violation_path_length_m']:.3f} m"
        )

        print(
            f"Final max violation        : "
            f"{evaluated['max_clearance_violation_mm']:.1f} mm"
        )

        print(
            f"Final violation samples    : "
            f"{evaluated['violation_sample_count']}"
        )

        print(
            f"Final violation length     : "
            f"{evaluated['violation_path_length_m']:.3f} m"
        )


if __name__ == "__main__":
    main()