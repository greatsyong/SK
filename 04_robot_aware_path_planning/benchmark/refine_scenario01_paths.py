from pathlib import Path
import math

import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import distance_transform_edt

from benchmark.metric_map_loader import load_ros_metric_map
from benchmark.find_robot_dependent_scenarios import astar


# ============================================================
# CONFIG
# ============================================================

MAP_YAML = Path(
    "data/metric_maps/stech_lab/stech_lab_completed.yaml"
)

RESULT_DIR = Path(
    "results/stech_lab/scenario01"
)

START = (360, 136)
GOAL = (744, 840)

ROBOTS = {
    "waffle": {
        "name": "TurtleBot3 Waffle Pi",
        "length_m": 0.281,
        "width_m": 0.306,
        "v_max": 0.26,
        "omega_max": 1.82,
    },
    "ridgeback": {
        "name": "Clearpath Ridgeback",
        "length_m": 0.960,
        "width_m": 0.793,
        "v_max": 1.10,
    },
}

# LOS collision sampling resolution in grid-cell units.
LOS_STEP_CELLS = 0.25


# ============================================================
# BASIC GEOMETRY
# ============================================================

def circumscribed_radius(
    length_m,
    width_m,
):
    return 0.5 * math.hypot(
        length_m,
        width_m,
    )


def path_length_m(
    path,
    resolution,
):
    total = 0.0

    for p0, p1 in zip(
        path[:-1],
        path[1:],
    ):
        total += (
            math.hypot(
                p1[0] - p0[0],
                p1[1] - p0[1],
            )
            * resolution
        )

    return total


def heading(
    p0,
    p1,
):
    return math.atan2(
        p1[1] - p0[1],
        p1[0] - p0[0],
    )


def wrap_angle(
    angle,
):
    return (
        angle
        + math.pi
    ) % (
        2.0 * math.pi
    ) - math.pi


def heading_changes(
    path,
):
    if len(path) < 3:
        return []

    headings = [
        heading(p0, p1)
        for p0, p1 in zip(
            path[:-1],
            path[1:],
        )
    ]

    return [
        abs(
            wrap_angle(
                h1 - h0
            )
        )
        for h0, h1 in zip(
            headings[:-1],
            headings[1:],
        )
    ]


# ============================================================
# LINE OF SIGHT
# ============================================================

def line_is_free(
    free_grid,
    p0,
    p1,
    step_cells=LOS_STEP_CELLS,
):
    """
    Dense sampling along a segment.

    The grid is already robot-specific C-space,
    therefore this checks whether the robot center
    can traverse the straight segment without
    intersecting the inflated obstacles.
    """

    x0, y0 = p0
    x1, y1 = p1

    distance = math.hypot(
        x1 - x0,
        y1 - y0,
    )

    sample_count = max(
        2,
        int(
            math.ceil(
                distance
                / step_cells
            )
        )
        + 1,
    )

    xs = np.linspace(
        x0,
        x1,
        sample_count,
    )

    ys = np.linspace(
        y0,
        y1,
        sample_count,
    )

    xi = np.rint(
        xs
    ).astype(int)

    yi = np.rint(
        ys
    ).astype(int)

    inside = (
        (xi >= 0)
        &
        (xi < free_grid.shape[1])
        &
        (yi >= 0)
        &
        (yi < free_grid.shape[0])
    )

    if not np.all(
        inside
    ):
        return False

    return bool(
        np.all(
            free_grid[
                yi,
                xi,
            ]
        )
    )


# ============================================================
# GREEDY LOS SIMPLIFICATION
# ============================================================

def simplify_path_los(
    path,
    free_grid,
):
    """
    From each retained waypoint, connect to the
    farthest later waypoint that has direct LOS.
    """

    if len(path) <= 2:
        return path[:]

    simplified = [
        path[0]
    ]

    current_index = 0

    while (
        current_index
        <
        len(path) - 1
    ):

        next_index = (
            len(path) - 1
        )

        while (
            next_index
            >
            current_index + 1
        ):

            if line_is_free(
                free_grid,
                path[current_index],
                path[next_index],
            ):
                break

            next_index -= 1

        simplified.append(
            path[next_index]
        )

        current_index = (
            next_index
        )

    return simplified


# ============================================================
# METRICS
# ============================================================

def path_metrics(
    path,
    resolution,
):
    turns = heading_changes(
        path
    )

    return {
        "points": len(path),
        "segments": max(
            0,
            len(path) - 1,
        ),
        "turns": len(turns),
        "length_m": path_length_m(
            path,
            resolution,
        ),
        "total_turn_rad": sum(
            turns
        ),
        "total_turn_deg": math.degrees(
            sum(turns)
        ),
        "max_turn_deg": (
            math.degrees(
                max(turns)
            )
            if turns
            else 0.0
        ),
    }


# ============================================================
# FIRST-ORDER TIME ESTIMATE
# ============================================================

def waffle_time(
    metrics,
):
    translation = (
        metrics["length_m"]
        /
        ROBOTS["waffle"]["v_max"]
    )

    turning = (
        metrics["total_turn_rad"]
        /
        ROBOTS["waffle"]["omega_max"]
    )

    return {
        "translation_s":
            translation,
        "turning_s":
            turning,
        "total_s":
            translation
            + turning,
    }


def ridgeback_time(
    metrics,
):
    translation = (
        metrics["length_m"]
        /
        ROBOTS["ridgeback"]["v_max"]
    )

    return {
        "translation_s":
            translation,

        # Holonomic lower-bound model.
        "total_s":
            translation,
    }


# ============================================================
# MAIN
# ============================================================

def main():

    RESULT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    m = load_ros_metric_map(
        MAP_YAML
    )

    grid = m["grid"]
    resolution = m[
        "resolution"
    ]

    raw_free = (
        grid == 0
    )

    clearance_m = (
        distance_transform_edt(
            raw_free
        )
        * resolution
    )

    cspaces = {}

    for key, robot in ROBOTS.items():

        radius = (
            circumscribed_radius(
                robot["length_m"],
                robot["width_m"],
            )
        )

        cspaces[key] = (
            raw_free
            &
            (
                clearance_m
                >= radius
            )
        )

    # --------------------------------------------------------
    # Raw planner paths
    # --------------------------------------------------------

    waffle_plan = astar(
        cspaces["waffle"],
        START,
        GOAL,
    )

    ridge_plan = astar(
        cspaces["ridgeback"],
        START,
        GOAL,
    )

    if waffle_plan is None:
        raise RuntimeError(
            "Waffle path missing."
        )

    if ridge_plan is None:
        raise RuntimeError(
            "Ridgeback path missing."
        )

    waffle_raw = (
        waffle_plan["path"]
    )

    ridge_raw = (
        ridge_plan["path"]
    )

    # --------------------------------------------------------
    # LOS simplification
    # --------------------------------------------------------

    waffle_simplified = (
        simplify_path_los(
            waffle_raw,
            cspaces["waffle"],
        )
    )

    ridge_simplified = (
        simplify_path_los(
            ridge_raw,
            cspaces["ridgeback"],
        )
    )

    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

    waffle_raw_metrics = (
        path_metrics(
            waffle_raw,
            resolution,
        )
    )

    waffle_refined_metrics = (
        path_metrics(
            waffle_simplified,
            resolution,
        )
    )

    ridge_raw_metrics = (
        path_metrics(
            ridge_raw,
            resolution,
        )
    )

    ridge_refined_metrics = (
        path_metrics(
            ridge_simplified,
            resolution,
        )
    )

    waffle_time_result = (
        waffle_time(
            waffle_refined_metrics
        )
    )

    ridge_time_result = (
        ridgeback_time(
            ridge_refined_metrics
        )
    )

    # ========================================================
    # OUTPUT
    # ========================================================

    print()
    print("=" * 84)
    print("SCENARIO 01 — LOS PATH REFINEMENT")
    print("=" * 84)

    print()
    print("TurtleBot3 Waffle Pi")
    print("-" * 40)

    print(
        f"Raw path length       : "
        f"{waffle_raw_metrics['length_m']:.2f} m"
    )

    print(
        f"Refined path length   : "
        f"{waffle_refined_metrics['length_m']:.2f} m"
    )

    print(
        f"Raw points            : "
        f"{waffle_raw_metrics['points']}"
    )

    print(
        f"Refined points        : "
        f"{waffle_refined_metrics['points']}"
    )

    print(
        f"Refined segments      : "
        f"{waffle_refined_metrics['segments']}"
    )

    print(
        f"Refined turns         : "
        f"{waffle_refined_metrics['turns']}"
    )

    print(
        f"Total turn angle      : "
        f"{waffle_refined_metrics['total_turn_deg']:.1f} deg"
    )

    print(
        f"Max single turn       : "
        f"{waffle_refined_metrics['max_turn_deg']:.1f} deg"
    )

    print(
        f"Translation time      : "
        f"{waffle_time_result['translation_s']:.2f} s"
    )

    print(
        f"Turning time          : "
        f"{waffle_time_result['turning_s']:.2f} s"
    )

    print(
        f"Estimated total       : "
        f"{waffle_time_result['total_s']:.2f} s"
    )

    print()
    print("Clearpath Ridgeback")
    print("-" * 40)

    print(
        f"Raw path length       : "
        f"{ridge_raw_metrics['length_m']:.2f} m"
    )

    print(
        f"Refined path length   : "
        f"{ridge_refined_metrics['length_m']:.2f} m"
    )

    print(
        f"Raw points            : "
        f"{ridge_raw_metrics['points']}"
    )

    print(
        f"Refined points        : "
        f"{ridge_refined_metrics['points']}"
    )

    print(
        f"Refined segments      : "
        f"{ridge_refined_metrics['segments']}"
    )

    print(
        f"Refined direction changes : "
        f"{ridge_refined_metrics['turns']}"
    )

    print(
        f"Total direction change    : "
        f"{ridge_refined_metrics['total_turn_deg']:.1f} deg"
    )

    print(
        f"Max single change         : "
        f"{ridge_refined_metrics['max_turn_deg']:.1f} deg"
    )

    print(
        f"Translation lower bound   : "
        f"{ridge_time_result['translation_s']:.2f} s"
    )

    print()

    print("=" * 84)
    print("REFINED COMPARISON")
    print("=" * 84)

    print(
        f"Path ratio R/W       : "
        f"{ridge_refined_metrics['length_m'] / waffle_refined_metrics['length_m']:.3f}"
    )

    print(
        f"Time ratio R/W       : "
        f"{ridge_time_result['total_s'] / waffle_time_result['total_s']:.3f}"
    )

    print(
        f"Time difference      : "
        f"{waffle_time_result['total_s'] - ridge_time_result['total_s']:.2f} s"
    )

    # ========================================================
    # VISUALIZATION
    # ========================================================

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(16, 8),
    )

    cases = [
        (
            axes[0],
            waffle_raw,
            waffle_simplified,
            "Waffle Pi",
        ),
        (
            axes[1],
            ridge_raw,
            ridge_simplified,
            "Ridgeback",
        ),
    ]

    for (
        ax,
        raw_path,
        simplified_path,
        title,
    ) in cases:

        ax.imshow(
            grid,
            cmap="gray_r",
            origin="upper",
        )

        raw_x = [
            p[0]
            for p in raw_path
        ]

        raw_y = [
            p[1]
            for p in raw_path
        ]

        simple_x = [
            p[0]
            for p in simplified_path
        ]

        simple_y = [
            p[1]
            for p in simplified_path
        ]

        ax.plot(
            raw_x,
            raw_y,
            linewidth=0.8,
            alpha=0.5,
            label="Raw A* grid path",
        )

        ax.plot(
            simple_x,
            simple_y,
            marker="o",
            markersize=3,
            linewidth=2.0,
            label="LOS-refined path",
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
            title
        )

        ax.set_xticks([])
        ax.set_yticks([])

        ax.legend(
            fontsize=8,
        )

    fig.suptitle(
        "STech Lab Scenario 01 — Raw vs LOS-Refined Paths",
        fontsize=15,
    )

    fig.tight_layout()

    plot_path = (
        RESULT_DIR
        / "scenario01_los_refinement.png"
    )

    fig.savefig(
        plot_path,
        dpi=180,
        bbox_inches="tight",
    )

    plt.close(fig)

    # --------------------------------------------------------
    # Save report
    # --------------------------------------------------------

    report_path = (
        RESULT_DIR
        / "scenario01_los_refinement.txt"
    )

    with report_path.open(
        "w"
    ) as f:

        f.write(
            "STech Lab Scenario 01 — LOS Path Refinement\n"
        )

        f.write(
            "=" * 60
            + "\n\n"
        )

        f.write(
            f"Start: {START}\n"
        )

        f.write(
            f"Goal : {GOAL}\n\n"
        )

        f.write(
            "Waffle Pi\n"
        )

        f.write(
            f"raw_length_m: "
            f"{waffle_raw_metrics['length_m']:.6f}\n"
        )

        f.write(
            f"refined_length_m: "
            f"{waffle_refined_metrics['length_m']:.6f}\n"
        )

        f.write(
            f"refined_segments: "
            f"{waffle_refined_metrics['segments']}\n"
        )

        f.write(
            f"refined_turns: "
            f"{waffle_refined_metrics['turns']}\n"
        )

        f.write(
            f"total_turn_deg: "
            f"{waffle_refined_metrics['total_turn_deg']:.6f}\n"
        )

        f.write(
            f"estimated_time_s: "
            f"{waffle_time_result['total_s']:.6f}\n\n"
        )

        f.write(
            "Ridgeback\n"
        )

        f.write(
            f"raw_length_m: "
            f"{ridge_raw_metrics['length_m']:.6f}\n"
        )

        f.write(
            f"refined_length_m: "
            f"{ridge_refined_metrics['length_m']:.6f}\n"
        )

        f.write(
            f"refined_segments: "
            f"{ridge_refined_metrics['segments']}\n"
        )

        f.write(
            f"direction_changes: "
            f"{ridge_refined_metrics['turns']}\n"
        )

        f.write(
            f"total_direction_change_deg: "
            f"{ridge_refined_metrics['total_turn_deg']:.6f}\n"
        )

        f.write(
            f"translation_lower_bound_s: "
            f"{ridge_time_result['total_s']:.6f}\n"
        )

    print()
    print(
        "Saved:",
        plot_path,
    )

    print(
        "Saved:",
        report_path,
    )


if __name__ == "__main__":
    main()