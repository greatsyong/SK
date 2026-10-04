from pathlib import Path
import csv
import heapq
import math

import matplotlib.pyplot as plt
import numpy as np

from benchmark.map_loader import load_movingai_map

from metrics.path_structure import (
    analyze_path_structure,
)


MAP_NAME = "maze512-32-0"

START = (
    312,
    95,
)

# Candidate goals are sampled on a coarse spatial lattice.
CANDIDATE_SPACING = 40

# Ignore goals that are trivially close to the start.
MIN_EUCLIDEAN_DISTANCE = 120.0

RRT_STEP_SIZE = 10.0


# ============================================================
# Grid motion model
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

    for dx, dy, cost in MOVES:

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

        yield nx, ny, cost


# ============================================================
# Single-source Dijkstra
# ============================================================

def single_source_dijkstra(
    grid,
    start,
):
    """
    Run Dijkstra once from START to all reachable cells.

    Returns:
        distance[y, x]
        parent[(x, y)]
    """

    height, width = grid.shape

    distance = np.full(
        (height, width),
        np.inf,
        dtype=float,
    )

    sx, sy = start

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

        for nx, ny, edge_cost in valid_neighbors(
            grid,
            x,
            y,
        ):

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

    return distance, parent


def reconstruct_path(
    parent,
    start,
    goal,
):
    if goal == start:
        return [start]

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
# Candidate generation
# ============================================================

def nearest_free_cell(
    grid,
    target_x,
    target_y,
    search_radius=15,
):
    """
    Find the nearest free cell to a coarse lattice location.
    """

    height, width = grid.shape

    best = None
    best_distance_sq = math.inf

    for dy in range(
        -search_radius,
        search_radius + 1,
    ):

        for dx in range(
            -search_radius,
            search_radius + 1,
        ):

            x = target_x + dx
            y = target_y + dy

            if not (
                0 <= x < width
                and
                0 <= y < height
            ):
                continue

            if grid[y, x] != 0:
                continue

            d2 = (
                dx * dx
                + dy * dy
            )

            if d2 < best_distance_sq:

                best_distance_sq = d2

                best = (
                    x,
                    y,
                )

    return best


def generate_candidates(
    grid,
    start,
):
    height, width = grid.shape

    candidates = set()

    half = (
        CANDIDATE_SPACING // 2
    )

    for y in range(
        half,
        height,
        CANDIDATE_SPACING,
    ):

        for x in range(
            half,
            width,
            CANDIDATE_SPACING,
        ):

            candidate = nearest_free_cell(
                grid,
                x,
                y,
            )

            if candidate is None:
                continue

            if candidate == start:
                continue

            dx = (
                candidate[0]
                - start[0]
            )

            dy = (
                candidate[1]
                - start[1]
            )

            euclidean = math.hypot(
                dx,
                dy,
            )

            if (
                euclidean
                < MIN_EUCLIDEAN_DISTANCE
            ):
                continue

            candidates.add(
                candidate
            )

    return sorted(
        candidates
    )


# ============================================================
# Main
# ============================================================

def main():

    project_root = (
        Path(__file__).resolve().parents[1]
    )

    results_dir = (
        project_root
        / "results"
    )

    results_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    map_data = load_movingai_map(
        project_root
        / "data"
        / "maps"
        / f"{MAP_NAME}.map"
    )

    grid = map_data["grid"]

    # --------------------------------------------------------
    # Compute shortest path from START to the entire map once.
    # --------------------------------------------------------

    print()
    print("=" * 78)
    print("GOAL CANDIDATE ANALYSIS")
    print("=" * 78)

    print("Map   :", MAP_NAME)
    print("Start :", START)

    print()
    print(
        "Running single-source Dijkstra..."
    )

    distance_map, parent = (
        single_source_dijkstra(
            grid,
            START,
        )
    )

    candidates = generate_candidates(
        grid,
        START,
    )

    print(
        "Candidate goals :",
        len(candidates),
    )

    rows = []

    # --------------------------------------------------------
    # Analyze each candidate
    # --------------------------------------------------------

    for index, goal in enumerate(
        candidates
    ):

        gx, gy = goal

        shortest_length = (
            distance_map[gy, gx]
        )

        if not math.isfinite(
            shortest_length
        ):
            continue

        path = reconstruct_path(
            parent,
            START,
            goal,
        )

        if not path:
            continue

        dx = (
            goal[0]
            - START[0]
        )

        dy = (
            goal[1]
            - START[1]
        )

        euclidean = math.hypot(
            dx,
            dy,
        )

        detour_ratio = (
            shortest_length
            / euclidean
        )

        structure = analyze_path_structure(
            grid,
            path,
            step_size=RRT_STEP_SIZE,
        )

        row = {
            "candidate_id":
                len(rows),

            "goal_x":
                gx,

            "goal_y":
                gy,

            "euclidean_distance":
                euclidean,

            "dijkstra_length":
                shortest_length,

            "detour_ratio":
                detour_ratio,

            "los_waypoints":
                structure[
                    "simplified_waypoint_count"
                ],

            "turn_count":
                structure[
                    "obstacle_induced_turn_count"
                ],

            "route_extension_lower_bound":
                structure[
                    "route_extension_lower_bound"
                ],

            "minimum_clearance":
                structure[
                    "clearance_stats"
                ][
                    "minimum"
                ],

            "p10_clearance":
                structure[
                    "clearance_stats"
                ][
                    "p10"
                ],

            "median_clearance":
                structure[
                    "clearance_stats"
                ][
                    "median"
                ],
        }

        rows.append(
            row
        )

        print(
            f"\rAnalyzed "
            f"{len(rows):3d} goals",
            end="",
            flush=True,
        )

    print()

    # --------------------------------------------------------
    # CSV
    # --------------------------------------------------------

    csv_path = (
        results_dir
        / "maze512_goal_candidate_analysis.csv"
    )

    with csv_path.open(
        "w",
        newline="",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=list(
                rows[0].keys()
            ),
        )

        writer.writeheader()

        writer.writerows(
            rows
        )

    # --------------------------------------------------------
    # Sort and print useful extremes
    # --------------------------------------------------------

    print()
    print("=" * 78)
    print("LONGEST SHORTEST-PATH CANDIDATES")
    print("=" * 78)

    longest = sorted(
        rows,
        key=lambda r:
            r["dijkstra_length"],
        reverse=True,
    )

    for row in longest[:15]:

        print(
            f"ID {row['candidate_id']:03d} | "
            f"G=({row['goal_x']:3d},"
            f"{row['goal_y']:3d}) | "
            f"L*={row['dijkstra_length']:8.2f} | "
            f"d={row['euclidean_distance']:7.2f} | "
            f"ratio={row['detour_ratio']:5.2f} | "
            f"turns={row['turn_count']:2d} | "
            f"ext={row['route_extension_lower_bound']:3d} | "
            f"clear={row['minimum_clearance']:.1f}"
        )

    print()
    print("=" * 78)
    print("HIGHEST DETOUR-RATIO CANDIDATES")
    print("=" * 78)

    detours = sorted(
        rows,
        key=lambda r:
            r["detour_ratio"],
        reverse=True,
    )

    for row in detours[:15]:

        print(
            f"ID {row['candidate_id']:03d} | "
            f"G=({row['goal_x']:3d},"
            f"{row['goal_y']:3d}) | "
            f"L*={row['dijkstra_length']:8.2f} | "
            f"d={row['euclidean_distance']:7.2f} | "
            f"ratio={row['detour_ratio']:5.2f} | "
            f"turns={row['turn_count']:2d} | "
            f"ext={row['route_extension_lower_bound']:3d} | "
            f"clear={row['minimum_clearance']:.1f}"
        )

    # --------------------------------------------------------
    # Map visualization
    # --------------------------------------------------------

    fig, ax = plt.subplots(
        figsize=(
            11,
            7,
        )
    )

    ax.imshow(
        grid,
        origin="upper",
        interpolation="nearest",
    )

    ax.scatter(
        START[0],
        START[1],
        s=90,
        marker="o",
        label="Start",
        zorder=10,
    )

    xs = [
        row["goal_x"]
        for row in rows
    ]

    ys = [
        row["goal_y"]
        for row in rows
    ]

    values = [
        row["dijkstra_length"]
        for row in rows
    ]

    scatter = ax.scatter(
        xs,
        ys,
        c=values,
        s=40,
        zorder=5,
    )

    for row in rows:

        ax.text(
            row["goal_x"] + 4,
            row["goal_y"] - 4,
            str(
                row["candidate_id"]
            ),
            fontsize=6,
        )

    fig.colorbar(
        scatter,
        ax=ax,
        label="Dijkstra shortest-path length",
    )

    ax.set_title(
        (
            "Goal Candidate Survey\n"
            "Color = shortest-path length from fixed start"
        )
    )

    ax.legend()

    figure_path = (
        results_dir
        / "maze512_goal_candidate_analysis.png"
    )

    fig.savefig(
        figure_path,
        dpi=200,
        bbox_inches="tight",
    )

    print()
    print("Saved CSV   :", csv_path)
    print("Saved figure:", figure_path)

    plt.show()


if __name__ == "__main__":
    main()