from pathlib import Path
import math
import heapq

import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import distance_transform_edt

from benchmark.metric_map_loader import load_ros_metric_map


# ============================================================
# CONFIG
# ============================================================

MAP_YAML = Path(
    "data/metric_maps/stech_lab/stech_lab_completed.yaml"
)

RESULT_DIR = Path(
    "results/stech_lab/robot_dependent_scenarios"
)

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

# Candidate start/goal spacing.
# 80 cells = 4.0 m
SAMPLE_SPACING = 80

# Ignore short missions.
MIN_STRAIGHT_DISTANCE_M = 15.0

# To avoid huge runtime in first screening.
MAX_PAIRS = 120

# Number of candidate scenarios to visualize.
TOP_N = 12


# ============================================================
# GEOMETRY
# ============================================================

def circumscribed_radius(length_m, width_m):
    return 0.5 * math.hypot(
        length_m,
        width_m,
    )


def octile_distance(
    x,
    y,
    gx,
    gy,
):
    dx = abs(gx - x)
    dy = abs(gy - y)

    return (
        max(dx, dy)
        +
        (math.sqrt(2.0) - 1.0)
        * min(dx, dy)
    )


# ============================================================
# A*
# ============================================================

NEIGHBORS = [
    (-1, 0, 1.0),
    (1, 0, 1.0),
    (0, -1, 1.0),
    (0, 1, 1.0),
    (-1, -1, math.sqrt(2.0)),
    (1, -1, math.sqrt(2.0)),
    (-1, 1, math.sqrt(2.0)),
    (1, 1, math.sqrt(2.0)),
]


def astar(
    free_grid,
    start,
    goal,
):

    sx, sy = start
    gx, gy = goal

    if not free_grid[sy, sx]:
        return None

    if not free_grid[gy, gx]:
        return None

    open_heap = []

    heapq.heappush(
        open_heap,
        (
            octile_distance(
                sx,
                sy,
                gx,
                gy,
            ),
            0.0,
            start,
        ),
    )

    g_cost = {
        start: 0.0
    }

    parent = {}

    closed = set()

    while open_heap:

        _, current_g, current = (
            heapq.heappop(
                open_heap
            )
        )

        if current in closed:
            continue

        closed.add(
            current
        )

        if current == goal:

            path = [
                current
            ]

            while (
                path[-1]
                in parent
            ):
                path.append(
                    parent[
                        path[-1]
                    ]
                )

            path.reverse()

            return {
                "path": path,
                "length_cells": current_g,
                "expanded": len(closed),
            }

        x, y = current

        for dx, dy, cost in NEIGHBORS:

            nx = x + dx
            ny = y + dy

            if not (
                0 <= nx < free_grid.shape[1]
                and
                0 <= ny < free_grid.shape[0]
            ):
                continue

            if not free_grid[ny, nx]:
                continue

            # No diagonal corner-cutting
            if (
                abs(dx) == 1
                and
                abs(dy) == 1
            ):

                if not (
                    free_grid[
                        y,
                        nx,
                    ]
                    and
                    free_grid[
                        ny,
                        x,
                    ]
                ):
                    continue

            next_node = (
                nx,
                ny,
            )

            new_g = (
                current_g
                + cost
            )

            if (
                new_g
                <
                g_cost.get(
                    next_node,
                    math.inf,
                )
            ):

                g_cost[
                    next_node
                ] = new_g

                parent[
                    next_node
                ] = current

                priority = (
                    new_g
                    +
                    octile_distance(
                        nx,
                        ny,
                        gx,
                        gy,
                    )
                )

                heapq.heappush(
                    open_heap,
                    (
                        priority,
                        new_g,
                        next_node,
                    ),
                )

    return None


# ============================================================
# PATH METRICS
# ============================================================

def path_jaccard(
    path_a,
    path_b,
):
    a = set(
        path_a
    )

    b = set(
        path_b
    )

    union = (
        a | b
    )

    if not union:
        return 0.0

    return (
        len(a & b)
        / len(union)
    )


def path_clearance_stats(
    path,
    clearance_m,
):
    values = np.array(
        [
            clearance_m[
                y,
                x,
            ]
            for x, y in path
        ],
        dtype=float,
    )

    return {
        "min": float(
            np.min(values)
        ),
        "mean": float(
            np.mean(values)
        ),
        "p10": float(
            np.percentile(
                values,
                10,
            )
        ),
    }


# ============================================================
# CANDIDATE POINTS
# ============================================================

def find_nearest_free(
    common_free,
    x,
    y,
    search_radius=25,
):

    best = None
    best_d2 = math.inf

    height, width = (
        common_free.shape
    )

    for yy in range(
        max(
            0,
            y - search_radius,
        ),
        min(
            height,
            y + search_radius + 1,
        ),
    ):

        for xx in range(
            max(
                0,
                x - search_radius,
            ),
            min(
                width,
                x + search_radius + 1,
            ),
        ):

            if not common_free[
                yy,
                xx,
            ]:
                continue

            d2 = (
                (xx - x) ** 2
                +
                (yy - y) ** 2
            )

            if d2 < best_d2:
                best_d2 = d2
                best = (
                    xx,
                    yy,
                )

    return best


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

    grid = m[
        "grid"
    ]

    resolution = m[
        "resolution"
    ]

    raw_free = (
        grid == 0
    )

    # --------------------------------------------------------
    # Metric clearance
    # --------------------------------------------------------

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
                robot[
                    "length_m"
                ],
                robot[
                    "width_m"
                ],
            )
        )

        robot[
            "radius_m"
        ] = radius

        cspaces[
            key
        ] = (
            raw_free
            &
            (
                clearance_m
                >= radius
            )
        )

    common_free = (
        cspaces["waffle"]
        &
        cspaces["ridgeback"]
    )

    # --------------------------------------------------------
    # Generate candidate points
    # --------------------------------------------------------

    candidate_points = []

    for y in range(
        SAMPLE_SPACING // 2,
        grid.shape[0],
        SAMPLE_SPACING,
    ):

        for x in range(
            SAMPLE_SPACING // 2,
            grid.shape[1],
            SAMPLE_SPACING,
        ):

            point = find_nearest_free(
                common_free,
                x,
                y,
            )

            if point is not None:
                candidate_points.append(
                    point
                )

    candidate_points = sorted(
        set(
            candidate_points
        )
    )

    print()
    print("=" * 80)
    print("ROBOT-DEPENDENT SCENARIO SEARCH")
    print("=" * 80)

    print(
        "Candidate points:",
        len(
            candidate_points
        ),
    )

    # --------------------------------------------------------
    # Candidate pair generation
    # --------------------------------------------------------

    pairs = []

    for i in range(
        len(
            candidate_points
        )
    ):

        for j in range(
            i + 1,
            len(
                candidate_points
            ),
        ):

            start = (
                candidate_points[i]
            )

            goal = (
                candidate_points[j]
            )

            straight_m = (
                math.hypot(
                    goal[0]
                    - start[0],
                    goal[1]
                    - start[1],
                )
                * resolution
            )

            if (
                straight_m
                <
                MIN_STRAIGHT_DISTANCE_M
            ):
                continue

            pairs.append(
                (
                    straight_m,
                    start,
                    goal,
                )
            )

    pairs.sort(
        reverse=True,
        key=lambda item:
            item[0],
    )

    pairs = pairs[
        :MAX_PAIRS
    ]

    print(
        "Pairs tested    :",
        len(pairs),
    )

    # --------------------------------------------------------
    # Evaluate
    # --------------------------------------------------------

    results = []

    for index, (
        straight_m,
        start,
        goal,
    ) in enumerate(
        pairs,
        start=1,
    ):

        waffle = astar(
            cspaces[
                "waffle"
            ],
            start,
            goal,
        )

        ridge = astar(
            cspaces[
                "ridgeback"
            ],
            start,
            goal,
        )

        if (
            waffle is None
            and
            ridge is None
        ):
            continue

        row = {
            "start": start,
            "goal": goal,
            "straight_m": straight_m,
            "waffle": waffle,
            "ridge": ridge,
        }

        if waffle is not None:

            row[
                "waffle_length_m"
            ] = (
                waffle[
                    "length_cells"
                ]
                * resolution
            )

            row[
                "waffle_clearance"
            ] = path_clearance_stats(
                waffle[
                    "path"
                ],
                clearance_m,
            )

        else:

            row[
                "waffle_length_m"
            ] = math.inf

        if ridge is not None:

            row[
                "ridge_length_m"
            ] = (
                ridge[
                    "length_cells"
                ]
                * resolution
            )

            row[
                "ridge_clearance"
            ] = path_clearance_stats(
                ridge[
                    "path"
                ],
                clearance_m,
            )

        else:

            row[
                "ridge_length_m"
            ] = math.inf

        if (
            waffle is not None
            and
            ridge is not None
        ):

            row[
                "length_ratio"
            ] = (
                row[
                    "ridge_length_m"
                ]
                /
                row[
                    "waffle_length_m"
                ]
            )

            row[
                "overlap"
            ] = path_jaccard(
                waffle[
                    "path"
                ],
                ridge[
                    "path"
                ],
            )

        else:

            row[
                "length_ratio"
            ] = math.inf

            row[
                "overlap"
            ] = 0.0

        results.append(
            row
        )

        print(
            f"{index:03d} | "
            f"{start} -> {goal} | "
            f"W="
            f"{row['waffle_length_m']:.2f} | "
            f"R="
            f"{row['ridge_length_m']:.2f} | "
            f"ratio="
            f"{row['length_ratio']:.3f} | "
            f"overlap="
            f"{row['overlap']:.3f}"
        )

    # --------------------------------------------------------
    # Ranking
    #
    # Primary interest:
    # - both robots feasible
    # - Ridgeback route longer
    # - routes spatially different
    # --------------------------------------------------------

    both_feasible = [
        r
        for r in results
        if (
            math.isfinite(
                r[
                    "waffle_length_m"
                ]
            )
            and
            math.isfinite(
                r[
                    "ridge_length_m"
                ]
            )
        )
    ]

    both_feasible.sort(
        key=lambda r: (
            r[
                "length_ratio"
            ],
            -r[
                "overlap"
            ],
        ),
        reverse=True,
    )

    # --------------------------------------------------------
    # Print top candidates
    # --------------------------------------------------------

    print()
    print("=" * 80)
    print("TOP ROBOT-DEPENDENT CANDIDATES")
    print("=" * 80)

    for rank, row in enumerate(
        both_feasible[
            :TOP_N
        ],
        start=1,
    ):

        print(
            f"Rank {rank:02d} | "
            f"S={row['start']} -> "
            f"G={row['goal']}"
        )

        print(
            f"    Waffle    : "
            f"{row['waffle_length_m']:.2f} m"
        )

        print(
            f"    Ridgeback : "
            f"{row['ridge_length_m']:.2f} m"
        )

        print(
            f"    R/W ratio : "
            f"{row['length_ratio']:.3f}"
        )

        print(
            f"    overlap   : "
            f"{row['overlap']:.3f}"
        )

        print(
            f"    W min clr : "
            f"{row['waffle_clearance']['min']:.3f} m"
        )

        print(
            f"    R min clr : "
            f"{row['ridge_clearance']['min']:.3f} m"
        )

    # ========================================================
    # Visualization
    # ========================================================

    selected = (
        both_feasible[
            :TOP_N
        ]
    )

    cols = 3
    rows_n = math.ceil(
        len(selected)
        / cols
    )

    fig, axes = plt.subplots(
        rows_n,
        cols,
        figsize=(
            5.5 * cols,
            5.5 * rows_n,
        ),
    )

    axes = np.array(
        axes
    ).reshape(
        rows_n,
        cols,
    )

    for ax in axes.flat:
        ax.axis(
            "off"
        )

    for idx, row in enumerate(
        selected
    ):

        ax = axes[
            idx // cols,
            idx % cols,
        ]

        ax.axis(
            "on"
        )

        ax.imshow(
            grid,
            cmap="gray_r",
            origin="upper",
        )

        waffle_path = (
            row[
                "waffle"
            ][
                "path"
            ]
        )

        ridge_path = (
            row[
                "ridge"
            ][
                "path"
            ]
        )

        wx = [
            p[0]
            for p in waffle_path
        ]

        wy = [
            p[1]
            for p in waffle_path
        ]

        rx = [
            p[0]
            for p in ridge_path
        ]

        ry = [
            p[1]
            for p in ridge_path
        ]

        ax.plot(
            wx,
            wy,
            linewidth=1.8,
            label=(
                "Waffle "
                f"{row['waffle_length_m']:.1f}m"
            ),
        )

        ax.plot(
            rx,
            ry,
            linewidth=1.8,
            label=(
                "Ridgeback "
                f"{row['ridge_length_m']:.1f}m"
            ),
        )

        sx, sy = row[
            "start"
        ]

        gx, gy = row[
            "goal"
        ]

        ax.scatter(
            [sx],
            [sy],
            s=40,
            marker="o",
        )

        ax.scatter(
            [gx],
            [gy],
            s=50,
            marker="x",
        )

        ax.set_title(
            (
                f"Rank {idx + 1}\n"
                f"ratio="
                f"{row['length_ratio']:.3f}, "
                f"overlap="
                f"{row['overlap']:.3f}"
            ),
            fontsize=9,
        )

        ax.set_xticks([])
        ax.set_yticks([])

        ax.legend(
            fontsize=7,
            loc="best",
        )

    fig.suptitle(
        (
            "STech Lab — "
            "Robot-Dependent Route Candidates"
        ),
        fontsize=16,
    )

    fig.tight_layout()

    save_path = (
        RESULT_DIR
        / "robot_dependent_candidates.png"
    )

    fig.savefig(
        save_path,
        dpi=180,
        bbox_inches="tight",
    )

    plt.close(fig)

    print()
    print(
        "Saved:",
        save_path,
    )


if __name__ == "__main__":
    main()