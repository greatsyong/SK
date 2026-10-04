from pathlib import Path
import math

import matplotlib.pyplot as plt
import numpy as np

from scipy.ndimage import distance_transform_edt
from scipy.spatial import cKDTree

from benchmark.metric_map_loader import load_ros_metric_map
from benchmark.find_robot_dependent_scenarios import (
    astar,
    find_nearest_free,
)
from benchmark.refine_scenario01_paths import (
    simplify_path_los,
    path_metrics,
)


# ============================================================
# CONFIG
# ============================================================

MAP_YAML = Path(
    "data/metric_maps/stech_lab/stech_lab_completed.yaml"
)

RESULT_DIR = Path(
    "results/stech_lab/scenario02_search"
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

# Same spatial sampling basis as Scenario 01 search.
SAMPLE_SPACING = 80

# Avoid trivial missions.
MIN_STRAIGHT_DISTANCE_M = 12.0

# Search budget.
MAX_PAIRS = 180

TOP_N = 12


# ============================================================
# GEOMETRY
# ============================================================

def circumscribed_radius(
    length_m,
    width_m,
):
    return 0.5 * math.hypot(
        length_m,
        width_m,
    )


def resample_polyline(
    path,
    resolution,
    spacing_m=0.10,
):
    """
    Uniform arc-length resampling.

    Used only for comparing geometric proximity
    between two paths.
    """

    points = (
        np.asarray(
            path,
            dtype=float,
        )
        * resolution
    )

    delta = np.diff(
        points,
        axis=0,
    )

    segment_lengths = (
        np.linalg.norm(
            delta,
            axis=1,
        )
    )

    cumulative = np.concatenate(
        (
            [0.0],
            np.cumsum(
                segment_lengths
            ),
        )
    )

    total = (
        cumulative[-1]
    )

    if total <= 0.0:
        return points

    samples = np.arange(
        0.0,
        total,
        spacing_m,
    )

    if (
        len(samples) == 0
        or samples[-1] < total
    ):
        samples = np.append(
            samples,
            total,
        )

    x = np.interp(
        samples,
        cumulative,
        points[:, 0],
    )

    y = np.interp(
        samples,
        cumulative,
        points[:, 1],
    )

    return np.column_stack(
        (
            x,
            y,
        )
    )


def symmetric_path_distance(
    path_a,
    path_b,
    resolution,
):
    """
    Symmetric nearest-neighbor distance between paths.

    Returns:
        mean distance [m]
        95th percentile distance [m]
        maximum distance [m]
    """

    a = resample_polyline(
        path_a,
        resolution,
    )

    b = resample_polyline(
        path_b,
        resolution,
    )

    tree_a = cKDTree(a)
    tree_b = cKDTree(b)

    d_a_to_b, _ = tree_b.query(
        a,
        k=1,
    )

    d_b_to_a, _ = tree_a.query(
        b,
        k=1,
    )

    distances = np.concatenate(
        (
            d_a_to_b,
            d_b_to_a,
        )
    )

    return {
        "mean_m":
            float(
                np.mean(distances)
            ),

        "p95_m":
            float(
                np.percentile(
                    distances,
                    95,
                )
            ),

        "max_m":
            float(
                np.max(distances)
            ),
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

    # --------------------------------------------------------
    # Robot C-spaces
    # --------------------------------------------------------

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

    common_free = (
        cspaces["waffle"]
        &
        cspaces["ridgeback"]
    )

    # --------------------------------------------------------
    # Candidate points
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

            point = (
                find_nearest_free(
                    common_free,
                    x,
                    y,
                )
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

    # --------------------------------------------------------
    # Candidate pairs
    # --------------------------------------------------------

    pairs = []

    for i in range(
        len(candidate_points)
    ):

        for j in range(
            i + 1,
            len(candidate_points),
        ):

            start = (
                candidate_points[i]
            )

            goal = (
                candidate_points[j]
            )

            straight_m = (
                math.hypot(
                    goal[0] - start[0],
                    goal[1] - start[1],
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

    # Long missions first so turns have room to matter.
    pairs.sort(
        key=lambda item:
            item[0],
        reverse=True,
    )

    pairs = pairs[
        :MAX_PAIRS
    ]

    print()
    print("=" * 84)
    print("SCENARIO 02 CANDIDATE SEARCH")
    print("=" * 84)

    print(
        "Candidate points:",
        len(candidate_points),
    )

    print(
        "Pairs evaluated :",
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

        waffle_plan = astar(
            cspaces["waffle"],
            start,
            goal,
        )

        ridge_plan = astar(
            cspaces["ridgeback"],
            start,
            goal,
        )

        if (
            waffle_plan is None
            or
            ridge_plan is None
        ):
            continue

        waffle_los = (
            simplify_path_los(
                waffle_plan["path"],
                cspaces["waffle"],
            )
        )

        ridge_los = (
            simplify_path_los(
                ridge_plan["path"],
                cspaces["ridgeback"],
            )
        )

        waffle_metrics = (
            path_metrics(
                waffle_los,
                resolution,
            )
        )

        ridge_metrics = (
            path_metrics(
                ridge_los,
                resolution,
            )
        )

        proximity = (
            symmetric_path_distance(
                waffle_los,
                ridge_los,
                resolution,
            )
        )

        mean_turn_deg = (
            0.5
            * (
                waffle_metrics[
                    "total_turn_deg"
                ]
                +
                ridge_metrics[
                    "total_turn_deg"
                ]
            )
        )

        mean_length = (
            0.5
            * (
                waffle_metrics[
                    "length_m"
                ]
                +
                ridge_metrics[
                    "length_m"
                ]
            )
        )

        # Turn complexity normalized by traveled distance.
        turn_density = (
            mean_turn_deg
            /
            mean_length
        )

        results.append(
            {
                "start":
                    start,

                "goal":
                    goal,

                "straight_m":
                    straight_m,

                "waffle_path":
                    waffle_los,

                "ridge_path":
                    ridge_los,

                "waffle_metrics":
                    waffle_metrics,

                "ridge_metrics":
                    ridge_metrics,

                "mean_sep_m":
                    proximity[
                        "mean_m"
                    ],

                "p95_sep_m":
                    proximity[
                        "p95_m"
                    ],

                "max_sep_m":
                    proximity[
                        "max_m"
                    ],

                "turn_density":
                    turn_density,
            }
        )

        print(
            f"{index:03d} | "
            f"{start} -> {goal} | "
            f"mean_sep={proximity['mean_m']:.2f} m | "
            f"p95={proximity['p95_m']:.2f} m | "
            f"turn_density={turn_density:.2f} deg/m"
        )

    # --------------------------------------------------------
    # Ranking
    #
    # First priority:
    #     paths remain spatially close
    #
    # Second priority:
    #     route contains meaningful turning complexity
    #
    # No hard threshold is imposed here.
    # We inspect the best geometric tradeoffs.
    # --------------------------------------------------------

    results.sort(
        key=lambda row: (
            row["mean_sep_m"],
            -row["turn_density"],
        )
    )

    selected = results[
        :TOP_N
    ]

    print()
    print("=" * 84)
    print("TOP SCENARIO 02 CANDIDATES")
    print("=" * 84)

    for rank, row in enumerate(
        selected,
        start=1,
    ):

        wm = row[
            "waffle_metrics"
        ]

        rm = row[
            "ridge_metrics"
        ]

        print()
        print(
            f"Rank {rank:02d} | "
            f"S={row['start']} -> "
            f"G={row['goal']}"
        )

        print(
            f"    mean separation : "
            f"{row['mean_sep_m']:.3f} m"
        )

        print(
            f"    p95 separation  : "
            f"{row['p95_sep_m']:.3f} m"
        )

        print(
            f"    Waffle length   : "
            f"{wm['length_m']:.2f} m"
        )

        print(
            f"    Ridge length    : "
            f"{rm['length_m']:.2f} m"
        )

        print(
            f"    Waffle turns    : "
            f"{wm['turns']} / "
            f"{wm['total_turn_deg']:.1f} deg"
        )

        print(
            f"    Ridge turns     : "
            f"{rm['turns']} / "
            f"{rm['total_turn_deg']:.1f} deg"
        )

        print(
            f"    turn density    : "
            f"{row['turn_density']:.2f} deg/m"
        )

    # ========================================================
    # VISUALIZATION
    # ========================================================

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

    axes = (
        np.asarray(
            axes
        )
        .reshape(
            rows_n,
            cols,
        )
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

        wp = np.asarray(
            row[
                "waffle_path"
            ]
        )

        rp = np.asarray(
            row[
                "ridge_path"
            ]
        )

        ax.plot(
            wp[:, 0],
            wp[:, 1],
            linewidth=2.0,
            marker="o",
            markersize=2,
            label="Waffle",
        )

        ax.plot(
            rp[:, 0],
            rp[:, 1],
            linewidth=2.0,
            marker="o",
            markersize=2,
            label="Ridgeback",
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
            s=35,
            marker="o",
        )

        ax.scatter(
            [gx],
            [gy],
            s=45,
            marker="x",
        )

        ax.set_title(
            (
                f"Rank {idx + 1}\n"
                f"mean sep="
                f"{row['mean_sep_m']:.2f} m\n"
                f"turn density="
                f"{row['turn_density']:.1f}°/m"
            ),
            fontsize=9,
        )

        ax.set_xticks([])
        ax.set_yticks([])

        ax.legend(
            fontsize=7,
        )

    fig.suptitle(
        (
            "STech Lab — Scenario 02 Candidates\n"
            "Similar Route Family + High Turn Complexity"
        ),
        fontsize=15,
    )

    fig.tight_layout()

    output_path = (
        RESULT_DIR
        /
        "scenario02_candidates.png"
    )

    fig.savefig(
        output_path,
        dpi=180,
        bbox_inches="tight",
    )

    plt.close(
        fig
    )

    print()
    print(
        "Saved:",
        output_path,
    )


if __name__ == "__main__":
    main()