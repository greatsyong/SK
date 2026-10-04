from pathlib import Path
import csv
import heapq
import math

import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import distance_transform_edt


# ============================================================
# CONFIGURATION
# ============================================================

MAP_NAME = "maze512-32-0"

START = (
    312,
    95,
)

GOALS = [
    ("ID043", (140, 180)),
    ("ID052", (180,  20)),
    ("ID070", (220, 380)),
    ("ID084", (300, 300)),
    ("ID095", (340, 420)),
    ("ID096", (340, 460)),
]

# Desired number of genuinely different route candidates
MAX_ROUTES = 5

# RRT step size currently used in the project.
RRT_STEP_SIZE = 10.0

# A route is considered spatially similar if it remains
# within half one RRT step of another route.
ROUTE_PROXIMITY_RADIUS = (
    RRT_STEP_SIZE / 2.0
)

# Penalization strength is NOT fixed.
# The search progressively increases it until a sufficiently
# different route is generated.
PENALTY_STRENGTHS = [
    0.5,
    1.0,
    2.0,
    4.0,
    8.0,
    16.0,
    32.0,
    64.0,
]

# A new route should have less than this symmetric spatial
# overlap with every accepted route.
MAX_ACCEPTABLE_OVERLAP = 0.70


# ============================================================
# GRID MOTION
# ============================================================

MOVES = [
    (-1,  0, 1.0),
    ( 1,  0, 1.0),
    ( 0, -1, 1.0),
    ( 0,  1, 1.0),

    (-1, -1, math.sqrt(2.0)),
    (-1,  1, math.sqrt(2.0)),
    ( 1, -1, math.sqrt(2.0)),
    ( 1,  1, math.sqrt(2.0)),
]


def valid_neighbors(
    grid,
    x,
    y,
):
    height, width = grid.shape

    for dx, dy, base_cost in MOVES:

        nx = x + dx
        ny = y + dy

        if not (
            0 <= nx < width
            and
            0 <= ny < height
        ):
            continue

        if grid[ny, nx] != 0:
            continue

        # Prevent diagonal corner cutting.
        if dx != 0 and dy != 0:

            if grid[y, x + dx] != 0:
                continue

            if grid[y + dy, x] != 0:
                continue

        yield (
            nx,
            ny,
            base_cost,
        )


# ============================================================
# WEIGHTED DIJKSTRA
# ============================================================

def weighted_dijkstra(
    grid,
    start,
    goal,
    penalty_field=None,
    penalty_strength=0.0,
):
    """
    Compute a shortest path under an optional route-reuse penalty.

    Important:
        penalty affects route DISCOVERY only.

        Final physical/grid length is always recomputed using
        the original unpenalized geometry.
    """

    height, width = grid.shape

    sx, sy = start
    gx, gy = goal

    distance = np.full(
        (height, width),
        np.inf,
        dtype=float,
    )

    distance[sy, sx] = 0.0

    parent = {}

    queue = [
        (
            0.0,
            sx,
            sy,
        )
    ]

    while queue:

        current_cost, x, y = heapq.heappop(
            queue
        )

        if current_cost != distance[y, x]:
            continue

        if (
            x == gx
            and
            y == gy
        ):
            break

        for (
            nx,
            ny,
            base_cost,
        ) in valid_neighbors(
            grid,
            x,
            y,
        ):

            if penalty_field is None:

                penalty = 0.0

            else:

                penalty = (
                    penalty_field[ny, nx]
                )

            edge_cost = (
                base_cost
                * (
                    1.0
                    + penalty_strength
                    * penalty
                )
            )

            candidate = (
                current_cost
                + edge_cost
            )

            if candidate >= distance[ny, nx]:
                continue

            distance[ny, nx] = candidate

            parent[
                (nx, ny)
            ] = (
                x,
                y,
            )

            heapq.heappush(
                queue,
                (
                    candidate,
                    nx,
                    ny,
                ),
            )

    if not math.isfinite(
        distance[gy, gx]
    ):
        return {
            "success": False,
            "path": [],
            "weighted_cost": math.inf,
        }

    path = reconstruct_path(
        parent,
        start,
        goal,
    )

    return {
        "success": True,
        "path": path,
        "weighted_cost":
            distance[gy, gx],
    }


def reconstruct_path(
    parent,
    start,
    goal,
):
    if goal == start:
        return [
            start
        ]

    if goal not in parent:
        return []

    path = [
        goal
    ]

    current = goal

    while current != start:

        current = parent[current]

        path.append(
            current
        )

    path.reverse()

    return path


# ============================================================
# TRUE UNPENALIZED PATH LENGTH
# ============================================================

def true_grid_path_length(
    path,
):
    if len(path) < 2:
        return 0.0

    total = 0.0

    for i in range(
        len(path) - 1
    ):

        x0, y0 = path[i]
        x1, y1 = path[i + 1]

        dx = abs(
            x1 - x0
        )

        dy = abs(
            y1 - y0
        )

        if dx == 1 and dy == 1:
            total += math.sqrt(2.0)
        else:
            total += 1.0

    return total


# ============================================================
# PATH MASK / CORRIDOR PENALTY
# ============================================================

def path_mask(
    shape,
    path,
):
    mask = np.zeros(
        shape,
        dtype=bool,
    )

    for x, y in path:
        mask[y, x] = True

    return mask


def build_penalty_field(
    grid,
    routes,
    radius,
):
    """
    Construct a smooth reuse penalty around all previously
    accepted routes.

    Penalty = 1 on an existing path.

    It decreases linearly to zero at ROUTE_PROXIMITY_RADIUS.

    This encourages exploration of another corridor without
    declaring the previous corridor physically blocked.
    """

    field = np.zeros(
        grid.shape,
        dtype=float,
    )

    if not routes:
        return field

    for route in routes:

        mask = path_mask(
            grid.shape,
            route,
        )

        distance_to_route = (
            distance_transform_edt(
                ~mask
            )
        )

        local_penalty = np.clip(
            1.0
            - (
                distance_to_route
                / radius
            ),
            0.0,
            1.0,
        )

        # Multiple previously accepted paths may overlap.
        # Accumulation makes repeatedly reused regions
        # progressively less attractive.
        field += local_penalty

    # Obstacles themselves are irrelevant because they
    # cannot be entered by Dijkstra anyway.
    field[
        grid != 0
    ] = 0.0

    return field


# ============================================================
# ROUTE SIMILARITY
# ============================================================

def directional_proximity_overlap(
    path_a,
    path_b,
    grid_shape,
    radius,
):
    """
    Fraction of path A that lies within 'radius' cells
    of path B.
    """

    mask_b = path_mask(
        grid_shape,
        path_b,
    )

    distance_to_b = (
        distance_transform_edt(
            ~mask_b
        )
    )

    if not path_a:
        return 0.0

    close_count = 0

    for x, y in path_a:

        if (
            distance_to_b[y, x]
            <= radius
        ):
            close_count += 1

    return (
        close_count
        / len(path_a)
    )


def symmetric_route_overlap(
    path_a,
    path_b,
    grid_shape,
    radius,
):
    """
    Symmetric corridor-aware overlap.

    A small lateral shift inside the same corridor should still
    count as the same route.
    """

    overlap_ab = (
        directional_proximity_overlap(
            path_a,
            path_b,
            grid_shape,
            radius,
        )
    )

    overlap_ba = (
        directional_proximity_overlap(
            path_b,
            path_a,
            grid_shape,
            radius,
        )
    )

    return (
        0.5
        * (
            overlap_ab
            + overlap_ba
        )
    )


def exact_jaccard_overlap(
    path_a,
    path_b,
):
    """
    Exact cell-set Jaccard overlap.

    This is reported for reference, but the corridor-aware
    proximity overlap is more useful for route-family analysis.
    """

    set_a = set(
        path_a
    )

    set_b = set(
        path_b
    )

    union = (
        set_a
        | set_b
    )

    if not union:
        return 0.0

    intersection = (
        set_a
        & set_b
    )

    return (
        len(intersection)
        / len(union)
    )


# ============================================================
# ALTERNATIVE ROUTE GENERATION
# ============================================================

def generate_alternative_routes(
    grid,
    start,
    goal,
):
    """
    Route 1:
        ordinary Dijkstra shortest path.

    Routes 2...K:
        progressively penalize regions already used by accepted
        routes until a sufficiently different path is found.
    """

    # --------------------------------------------------------
    # P1: true shortest path
    # --------------------------------------------------------

    baseline = weighted_dijkstra(
        grid,
        start,
        goal,
    )

    if not baseline["success"]:
        return []

    routes = [
        {
            "path":
                baseline["path"],

            "true_length":
                true_grid_path_length(
                    baseline["path"]
                ),

            "penalty_strength":
                0.0,

            "max_overlap_previous":
                0.0,

            "mean_overlap_previous":
                0.0,
        }
    ]

    # --------------------------------------------------------
    # P2 ... PK
    # --------------------------------------------------------

    while len(routes) < MAX_ROUTES:

        accepted_paths = [
            route["path"]
            for route in routes
        ]

        penalty_field = (
            build_penalty_field(
                grid,
                accepted_paths,
                ROUTE_PROXIMITY_RADIUS,
            )
        )

        best_candidate = None

        # Start from weak penalty and increase automatically.
        for strength in PENALTY_STRENGTHS:

            candidate = weighted_dijkstra(
                grid,
                start,
                goal,
                penalty_field=
                    penalty_field,
                penalty_strength=
                    strength,
            )

            if not candidate[
                "success"
            ]:
                continue

            path = candidate[
                "path"
            ]

            overlaps = [
                symmetric_route_overlap(
                    path,
                    previous["path"],
                    grid.shape,
                    ROUTE_PROXIMITY_RADIUS,
                )
                for previous in routes
            ]

            max_overlap = max(
                overlaps
            )

            mean_overlap = (
                sum(overlaps)
                / len(overlaps)
            )

            candidate_record = {
                "path":
                    path,

                "true_length":
                    true_grid_path_length(
                        path
                    ),

                "penalty_strength":
                    strength,

                "max_overlap_previous":
                    max_overlap,

                "mean_overlap_previous":
                    mean_overlap,
            }

            # Keep the least-overlapping candidate encountered
            # in case no candidate passes the threshold.
            if (
                best_candidate is None
                or
                max_overlap
                <
                best_candidate[
                    "max_overlap_previous"
                ]
            ):
                best_candidate = (
                    candidate_record
                )

            # Sufficiently different route found.
            if (
                max_overlap
                <= MAX_ACCEPTABLE_OVERLAP
            ):
                break

        if best_candidate is None:
            break

        # If even the strongest penalty cannot produce
        # a route that is meaningfully different, stop.
        if (
            best_candidate[
                "max_overlap_previous"
            ]
            >
            MAX_ACCEPTABLE_OVERLAP
        ):
            break

        routes.append(
            best_candidate
        )

    return routes


# ============================================================
# PAIRWISE COMPARISON
# ============================================================

def pairwise_route_metrics(
    routes,
    grid_shape,
):
    rows = []

    for i in range(
        len(routes)
    ):

        for j in range(
            i + 1,
            len(routes),
        ):

            proximity_overlap = (
                symmetric_route_overlap(
                    routes[i]["path"],
                    routes[j]["path"],
                    grid_shape,
                    ROUTE_PROXIMITY_RADIUS,
                )
            )

            exact_overlap = (
                exact_jaccard_overlap(
                    routes[i]["path"],
                    routes[j]["path"],
                )
            )

            rows.append(
                {
                    "route_a":
                        i + 1,

                    "route_b":
                        j + 1,

                    "proximity_overlap":
                        proximity_overlap,

                    "exact_jaccard":
                        exact_overlap,
                }
            )

    return rows


# ============================================================
# VISUALIZATION
# ============================================================

def plot_routes(
    grid,
    start,
    goal,
    label,
    routes,
    output_path,
):
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

    for index, route in enumerate(
        routes,
        start=1,
    ):

        xs = [
            x + 0.5
            for x, y
            in route["path"]
        ]

        ys = [
            y + 0.5
            for x, y
            in route["path"]
        ]

        ax.plot(
            xs,
            ys,
            linewidth=1.8,
            label=(
                f"P{index}: "
                f"L={route['true_length']:.1f}"
            ),
        )

    ax.scatter(
        start[0] + 0.5,
        start[1] + 0.5,
        s=70,
        marker="o",
        label="Start",
        zorder=10,
    )

    ax.scatter(
        goal[0] + 0.5,
        goal[1] + 0.5,
        s=80,
        marker="x",
        linewidths=2,
        label="Goal",
        zorder=10,
    )

    ax.set_title(
        (
            f"Alternative Route Analysis — {label}\n"
            f"Goal = {goal} | "
            f"{len(routes)} distinct route candidates"
        )
    )

    ax.legend(
        loc="best"
    )

    fig.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close(
        fig
    )


# ============================================================
# MAIN
# ============================================================

def main():

    project_root = (
        Path(__file__).resolve().parents[1]
    )

    result_dir = (
        project_root
        / "results"
        / "alternative_routes"
    )

    result_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Import here so this file remains standalone with respect
    # to project-root resolution.
    # --------------------------------------------------------

    from benchmark.map_loader import (
        load_movingai_map,
    )

    map_data = load_movingai_map(
        project_root
        / "data"
        / "maps"
        / f"{MAP_NAME}.map"
    )

    grid = map_data["grid"]

    summary_rows = []
    overlap_rows = []

    print()
    print("=" * 88)
    print("ALTERNATIVE ROUTE ANALYSIS")
    print("=" * 88)

    print("Map                 :", MAP_NAME)
    print("Start               :", START)
    print(
        "Route proximity     :",
        ROUTE_PROXIMITY_RADIUS,
        "cells",
    )
    print(
        "Overlap threshold   :",
        MAX_ACCEPTABLE_OVERLAP,
    )

    # ========================================================
    # Evaluate shortlist
    # ========================================================

    for label, goal in GOALS:

        print()
        print("=" * 88)
        print(
            label,
            "Goal =",
            goal,
        )
        print("=" * 88)

        routes = generate_alternative_routes(
            grid,
            START,
            goal,
        )

        if not routes:

            print(
                "No feasible path."
            )

            continue

        shortest = (
            routes[0]["true_length"]
        )

        for route_index, route in enumerate(
            routes,
            start=1,
        ):

            excess = (
                route["true_length"]
                - shortest
            )

            excess_pct = (
                100.0
                * excess
                / shortest
            )

            print(
                f"P{route_index}: "
                f"L = "
                f"{route['true_length']:8.2f} | "
                f"+{excess_pct:6.2f}% | "
                f"penalty = "
                f"{route['penalty_strength']:5.1f} | "
                f"max previous overlap = "
                f"{route['max_overlap_previous']:.3f}"
            )

            summary_rows.append(
                {
                    "candidate":
                        label,

                    "goal_x":
                        goal[0],

                    "goal_y":
                        goal[1],

                    "route":
                        route_index,

                    "true_length":
                        route[
                            "true_length"
                        ],

                    "length_excess":
                        excess,

                    "length_excess_pct":
                        excess_pct,

                    "penalty_strength":
                        route[
                            "penalty_strength"
                        ],

                    "max_overlap_previous":
                        route[
                            "max_overlap_previous"
                        ],

                    "mean_overlap_previous":
                        route[
                            "mean_overlap_previous"
                        ],
                }
            )

        # ----------------------------------------------------
        # Pairwise route overlap
        # ----------------------------------------------------

        pairwise = (
            pairwise_route_metrics(
                routes,
                grid.shape,
            )
        )

        print()
        print("PAIRWISE OVERLAP")

        for pair in pairwise:

            print(
                f"P{pair['route_a']}"
                f"-P{pair['route_b']}: "
                f"proximity = "
                f"{pair['proximity_overlap']:.3f} | "
                f"exact = "
                f"{pair['exact_jaccard']:.3f}"
            )

            overlap_rows.append(
                {
                    "candidate":
                        label,

                    "goal_x":
                        goal[0],

                    "goal_y":
                        goal[1],

                    **pair,
                }
            )

        # ----------------------------------------------------
        # Figure
        # ----------------------------------------------------

        figure_path = (
            result_dir
            / (
                f"{label}_"
                f"goal_{goal[0]}_{goal[1]}_"
                f"routes.png"
            )
        )

        plot_routes(
            grid,
            START,
            goal,
            label,
            routes,
            figure_path,
        )

        print(
            "Saved figure:",
            figure_path,
        )

    # ========================================================
    # Save CSV
    # ========================================================

    summary_path = (
        result_dir
        / "alternative_route_summary.csv"
    )

    with summary_path.open(
        "w",
        newline="",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=list(
                summary_rows[0].keys()
            ),
        )

        writer.writeheader()
        writer.writerows(
            summary_rows
        )

    overlap_path = (
        result_dir
        / "alternative_route_overlap.csv"
    )

    with overlap_path.open(
        "w",
        newline="",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=list(
                overlap_rows[0].keys()
            ),
        )

        writer.writeheader()
        writer.writerows(
            overlap_rows
        )

    print()
    print("=" * 88)
    print("ANALYSIS COMPLETE")
    print("=" * 88)

    print(
        "Summary:",
        summary_path,
    )

    print(
        "Overlap:",
        overlap_path,
    )


if __name__ == "__main__":
    main()