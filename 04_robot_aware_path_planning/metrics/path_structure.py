import math

import numpy as np
from scipy.ndimage import distance_transform_edt

from planners.geometry import (
    edge_is_free,
    euclidean_distance,
)


def cell_center(cell):
    """
    Convert integer grid cell (x, y)
    to continuous cell-center coordinates.
    """
    x, y = cell

    return (
        float(x) + 0.5,
        float(y) + 0.5,
    )


def polyline_length(points):
    """
    Euclidean length of a continuous polyline.
    """

    if len(points) < 2:
        return 0.0

    return sum(
        euclidean_distance(
            points[i],
            points[i + 1],
        )
        for i in range(len(points) - 1)
    )


def simplify_path_line_of_sight(
    grid,
    path,
):
    """
    Greedy line-of-sight simplification.

    Starting from the current anchor point, connect directly
    to the farthest later path point that remains collision free.

    Input:
        path:
            Grid cells [(x, y), ...]

    Output:
        Continuous cell-center waypoints.
    """

    if not path:
        return []

    points = [
        cell_center(p)
        for p in path
    ]

    if len(points) <= 2:
        return points

    simplified = [
        points[0]
    ]

    anchor_index = 0

    while anchor_index < len(points) - 1:

        # At minimum, the next point must be selected.
        best_index = anchor_index + 1

        # Search backward from the goal-side end
        # for the farthest visible point.
        for candidate_index in range(
            len(points) - 1,
            anchor_index,
            -1,
        ):

            if edge_is_free(
                grid,
                points[anchor_index],
                points[candidate_index],
            ):
                best_index = candidate_index
                break

        simplified.append(
            points[best_index]
        )

        anchor_index = best_index

    return simplified


def segment_lengths(points):
    """
    Length of each segment in a polyline.
    """

    return [
        euclidean_distance(
            points[i],
            points[i + 1],
        )
        for i in range(len(points) - 1)
    ]


def turning_angle_deg(
    p0,
    p1,
    p2,
):
    """
    Absolute heading change at p1 in degrees.

    0 deg:
        straight

    90 deg:
        right-angle turn

    180 deg:
        reversal
    """

    v1 = np.array(
        [
            p1[0] - p0[0],
            p1[1] - p0[1],
        ],
        dtype=float,
    )

    v2 = np.array(
        [
            p2[0] - p1[0],
            p2[1] - p1[1],
        ],
        dtype=float,
    )

    n1 = np.linalg.norm(v1)
    n2 = np.linalg.norm(v2)

    if n1 == 0.0 or n2 == 0.0:
        return 0.0

    cosine = np.dot(
        v1,
        v2,
    ) / (n1 * n2)

    cosine = np.clip(
        cosine,
        -1.0,
        1.0,
    )

    return math.degrees(
        math.acos(cosine)
    )


def turning_angles(points):
    """
    Return turning information for each interior waypoint.
    """

    turns = []

    for i in range(
        1,
        len(points) - 1,
    ):

        angle = turning_angle_deg(
            points[i - 1],
            points[i],
            points[i + 1],
        )

        turns.append(
            {
                "waypoint_index": i,
                "point": points[i],
                "turn_angle_deg": angle,
            }
        )

    return turns


def minimum_route_extensions(
    segment_lengths_cells,
    step_size,
):
    """
    Number of step_size-limited forward extensions required
    to traverse each simplified route segment.

    Important:
    This is a route-conditioned geometric lower bound,
    not the number of random samples RRT actually requires.
    """

    per_segment = [
        int(
            math.ceil(
                length / step_size
            )
        )
        for length
        in segment_lengths_cells
    ]

    return {
        "per_segment":
            per_segment,

        "total":
            sum(per_segment),
    }


def compute_clearance_map(
    grid,
):
    """
    Distance from every free-cell center to the nearest
    occupied-cell center, measured in grid cells.

    grid:
        0 = free
        1 = occupied
    """

    free_mask = (
        grid == 0
    )

    return distance_transform_edt(
        free_mask
    )


def sample_path_clearance(
    path,
    clearance_map,
):
    """
    Clearance along an integer-grid path.
    """

    values = []

    for x, y in path:

        values.append(
            float(
                clearance_map[y, x]
            )
        )

    return values


def analyze_path_structure(
    grid,
    path,
    step_size,
):
    """
    Full geometric route analysis.
    """

    simplified = (
        simplify_path_line_of_sight(
            grid,
            path,
        )
    )

    lengths = segment_lengths(
        simplified
    )

    turns = turning_angles(
        simplified
    )

    extension_info = (
        minimum_route_extensions(
            lengths,
            step_size,
        )
    )

    clearance_map = (
        compute_clearance_map(
            grid
        )
    )

    clearance_values = (
        sample_path_clearance(
            path,
            clearance_map,
        )
    )

    if clearance_values:

        clearance_stats = {
            "minimum":
                float(
                    np.min(
                        clearance_values
                    )
                ),

            "mean":
                float(
                    np.mean(
                        clearance_values
                    )
                ),

            "median":
                float(
                    np.median(
                        clearance_values
                    )
                ),

            "p10":
                float(
                    np.percentile(
                        clearance_values,
                        10,
                    )
                ),
        }

    else:

        clearance_stats = {
            "minimum": math.inf,
            "mean": math.inf,
            "median": math.inf,
            "p10": math.inf,
        }

    return {
        "raw_path_points":
            len(path),

        "simplified_waypoints":
            simplified,

        "simplified_waypoint_count":
            len(simplified),

        "obstacle_induced_turn_count":
            max(
                len(simplified) - 2,
                0,
            ),

        "segment_lengths":
            lengths,

        "simplified_path_length":
            polyline_length(
                simplified
            ),

        "turns":
            turns,

        "extension_count_per_segment":
            extension_info[
                "per_segment"
            ],

        "route_extension_lower_bound":
            extension_info[
                "total"
            ],

        "clearance_stats":
            clearance_stats,

        "clearance_map":
            clearance_map,
    }