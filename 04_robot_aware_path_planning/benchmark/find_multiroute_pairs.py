from pathlib import Path
import csv
import math

import numpy as np
from scipy.ndimage import binary_dilation

from benchmark.map_loader import (
    load_movingai_map,
)

from planners.astar import (
    AStarPlanner,
)


# ============================================================
# CONFIGURATION
# ============================================================

MAP_NAME = "16room_000"

# Coarse points distributed over the map.
CANDIDATE_SPACING = 64

# Search around each coarse lattice point for a nearby free cell.
FREE_CELL_SEARCH_RADIUS = 25

# Ignore very short start-goal pairs.
MIN_EUCLIDEAN_DISTANCE = 180.0

# We do not need to test every possible pair initially.
# Long-distance pairs are tested first.
MAX_PAIRS_TO_TEST = 80

# Test whether an alternative route survives after excluding
# increasingly wider regions around the primary shortest path.
EXCLUSION_RADII = [
    2,
    5,
    10,
]

# The start and goal must naturally be shared by every route.
# Therefore the first/last portion of P1 is not blocked.
ENDPOINT_SHARED_PATH_CELLS = 20


# ============================================================
# BASIC GEOMETRY
# ============================================================

def euclidean(
    a,
    b,
):
    return math.hypot(
        b[0] - a[0],
        b[1] - a[1],
    )


def make_disk(
    radius,
):
    y, x = np.ogrid[
        -radius:radius + 1,
        -radius:radius + 1
    ]

    return (
        x * x
        + y * y
        <= radius * radius
    )


# ============================================================
# CANDIDATE POINT GENERATION
# ============================================================

def nearest_free_cell(
    grid,
    center_x,
    center_y,
    search_radius,
):
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

            x = center_x + dx
            y = center_y + dy

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


def generate_candidate_points(
    grid,
):
    height, width = grid.shape

    points = set()

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

            point = nearest_free_cell(
                grid,
                x,
                y,
                FREE_CELL_SEARCH_RADIUS,
            )

            if point is not None:
                points.add(
                    point
                )

    return sorted(
        points
    )


# ============================================================
# PRIMARY-PATH EXCLUSION
# ============================================================

def create_path_exclusion_grid(
    grid,
    path,
    exclusion_radius,
):
    """
    Block a band around the interior of the shortest path.

    Start and goal neighborhoods remain untouched so another
    route is allowed to share the task endpoints.
    """

    blocked_grid = grid.copy()

    path_mask = np.zeros(
        grid.shape,
        dtype=bool,
    )

    if (
        len(path)
        <= 2 * ENDPOINT_SHARED_PATH_CELLS
    ):
        return blocked_grid

    interior_path = path[
        ENDPOINT_SHARED_PATH_CELLS:
        -ENDPOINT_SHARED_PATH_CELLS
    ]

    for x, y in interior_path:
        path_mask[y, x] = True

    structure = make_disk(
        exclusion_radius
    )

    exclusion_mask = binary_dilation(
        path_mask,
        structure=structure,
    )

    blocked_grid[
        exclusion_mask
    ] = 1

    # Guarantee that task endpoints remain valid.
    sx, sy = path[0]
    gx, gy = path[-1]

    blocked_grid[
        sy,
        sx,
    ] = 0

    blocked_grid[
        gy,
        gx,
    ] = 0

    return blocked_grid


# ============================================================
# PATH OVERLAP
# ============================================================

def path_overlap(
    path_a,
    path_b,
):
    """
    Exact-cell Jaccard overlap.

    Used only as a descriptive metric.
    """

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
        / "multiroute_search"
    )

    result_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Load map
    # --------------------------------------------------------

    map_data = load_movingai_map(
        project_root
        / "data"
        / "maps"
        / f"{MAP_NAME}.map"
    )

    grid = map_data[
        "grid"
    ]

    planner = AStarPlanner()

    # --------------------------------------------------------
    # Candidate points
    # --------------------------------------------------------

    points = generate_candidate_points(
        grid
    )

    print()
    print("=" * 88)
    print("MULTI-ROUTE START/GOAL SEARCH")
    print("=" * 88)

    print(
        "Map              :",
        MAP_NAME,
    )

    print(
        "Candidate points :",
        len(points),
    )

    # --------------------------------------------------------
    # Generate candidate pairs
    # --------------------------------------------------------

    pairs = []

    for i in range(
        len(points)
    ):

        for j in range(
            i + 1,
            len(points),
        ):

            start = points[i]
            goal = points[j]

            distance = euclidean(
                start,
                goal,
            )

            if (
                distance
                < MIN_EUCLIDEAN_DISTANCE
            ):
                continue

            pairs.append(
                (
                    distance,
                    start,
                    goal,
                )
            )

    # Test geometrically long pairs first.
    pairs.sort(
        key=lambda x:
            x[0],
        reverse=True,
    )

    pairs = pairs[
        :MAX_PAIRS_TO_TEST
    ]

    print(
        "Pairs to test    :",
        len(pairs),
    )

    rows = []

    # ========================================================
    # Evaluate pairs
    # ========================================================

    for pair_index, (
        straight_distance,
        start,
        goal,
    ) in enumerate(
        pairs,
        start=1,
    ):

        # ----------------------------------------------------
        # Primary shortest path
        # ----------------------------------------------------

        primary = planner.plan(
            grid,
            start,
            goal,
        )

        if not primary[
            "success"
        ]:
            continue

        p1 = primary[
            "path"
        ]

        p1_length = primary[
            "path_length"
        ]

        result_row = {
            "pair_id":
                pair_index,

            "start_x":
                start[0],

            "start_y":
                start[1],

            "goal_x":
                goal[0],

            "goal_y":
                goal[1],

            "straight_distance":
                straight_distance,

            "primary_length":
                p1_length,

            "primary_detour_ratio":
                (
                    p1_length
                    / straight_distance
                ),
        }

        strongest_surviving_radius = 0

        # ----------------------------------------------------
        # Search alternatives after blocking P1
        # ----------------------------------------------------

        for radius in EXCLUSION_RADII:

            modified_grid = (
                create_path_exclusion_grid(
                    grid,
                    p1,
                    exclusion_radius=radius,
                )
            )

            alternative = planner.plan(
                modified_grid,
                start,
                goal,
            )

            key = (
                f"r{radius}"
            )

            if alternative[
                "success"
            ]:

                p2 = alternative[
                    "path"
                ]

                p2_length = alternative[
                    "path_length"
                ]

                strongest_surviving_radius = (
                    radius
                )

                result_row[
                    f"{key}_alternative"
                ] = True

                result_row[
                    f"{key}_length"
                ] = p2_length

                result_row[
                    f"{key}_length_ratio"
                ] = (
                    p2_length
                    / p1_length
                )

                result_row[
                    f"{key}_exact_overlap"
                ] = path_overlap(
                    p1,
                    p2,
                )

            else:

                result_row[
                    f"{key}_alternative"
                ] = False

                result_row[
                    f"{key}_length"
                ] = math.inf

                result_row[
                    f"{key}_length_ratio"
                ] = math.inf

                result_row[
                    f"{key}_exact_overlap"
                ] = math.nan

        result_row[
            "strongest_surviving_radius"
        ] = strongest_surviving_radius

        rows.append(
            result_row
        )

        print(
            f"Pair {pair_index:02d} | "
            f"{start} -> {goal} | "
            f"L1={p1_length:7.1f} | "
            f"detour="
            f"{p1_length / straight_distance:4.2f} | "
            f"alt survives to r="
            f"{strongest_surviving_radius}"
        )

    # ========================================================
    # Rank candidates
    # ========================================================

    rows.sort(
        key=lambda r: (
            r[
                "strongest_surviving_radius"
            ],
            r[
                "primary_length"
            ],
        ),
        reverse=True,
    )

    # Reassign ranked IDs
    for rank, row in enumerate(
        rows,
        start=1,
    ):
        row[
            "rank"
        ] = rank

    # ========================================================
    # Print top results
    # ========================================================

    print()
    print("=" * 88)
    print("TOP MULTI-ROUTE CANDIDATES")
    print("=" * 88)

    for row in rows[:20]:

        print(
            f"Rank {row['rank']:02d} | "
            f"S=({row['start_x']:3d},"
            f"{row['start_y']:3d}) -> "
            f"G=({row['goal_x']:3d},"
            f"{row['goal_y']:3d}) | "
            f"L1={row['primary_length']:7.2f} | "
            f"detour="
            f"{row['primary_detour_ratio']:4.2f} | "
            f"survive r="
            f"{row['strongest_surviving_radius']:2d}"
        )

        for radius in EXCLUSION_RADII:

            key = (
                f"r{radius}"
            )

            if row[
                f"{key}_alternative"
            ]:

                print(
                    f"    r={radius:2d}: "
                    f"L2="
                    f"{row[f'{key}_length']:7.2f} | "
                    f"L2/L1="
                    f"{row[f'{key}_length_ratio']:5.3f} | "
                    f"exact overlap="
                    f"{row[f'{key}_exact_overlap']:5.3f}"
                )

            else:

                print(
                    f"    r={radius:2d}: "
                    f"NO ALTERNATIVE"
                )

    # ========================================================
    # Save CSV
    # ========================================================

    output_path = (
        result_dir
        / (
            f"{MAP_NAME}_"
            f"multiroute_candidates.csv"
        )
    )

    if rows:

        # Put rank first for readability.
        fieldnames = [
            "rank"
        ] + [
            key
            for key in rows[0].keys()
            if key != "rank"
        ]

        with output_path.open(
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

    print()
    print("=" * 88)
    print("SEARCH COMPLETE")
    print("=" * 88)

    print(
        "Saved:",
        output_path,
    )


if __name__ == "__main__":
    main()