from pathlib import Path
import math

import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import binary_dilation

from benchmark.map_loader import load_movingai_map
from planners.astar import AStarPlanner


# ============================================================
# CONFIG
# ============================================================

MAP_NAME = "16room_000"

# Top 20 from your printed ranking
CANDIDATES = [
    ("Rank01", (32, 479), (479, 31)),
    ("Rank02", (31, 31), (479, 479)),
    ("Rank03", (31, 31), (479, 416)),
    ("Rank04", (31, 415), (479, 31)),
    ("Rank05", (95, 31), (479, 479)),
    ("Rank06", (32, 479), (415, 31)),
    ("Rank07", (95, 479), (479, 31)),
    ("Rank08", (31, 31), (415, 479)),
    ("Rank09", (95, 479), (479, 95)),
    ("Rank10", (31, 31), (479, 351)),
    ("Rank11", (31, 415), (415, 31)),
    ("Rank12", (95, 415), (479, 31)),
    ("Rank13", (161, 480), (479, 31)),
    ("Rank14", (32, 479), (351, 31)),
    ("Rank15", (95, 31), (479, 416)),
    ("Rank16", (31, 31), (351, 480)),
    ("Rank17", (32, 479), (415, 95)),
    ("Rank18", (32, 479), (479, 159)),
    ("Rank19", (95, 31), (415, 479)),
    ("Rank20", (95, 479), (415, 31)),
]

EXCLUSION_RADII = [2, 5, 10]
ENDPOINT_SHARED_PATH_CELLS = 20


# ============================================================
# UTILS
# ============================================================

def make_disk(radius):
    y, x = np.ogrid[
        -radius:radius + 1,
        -radius:radius + 1
    ]
    return (x * x + y * y) <= radius * radius


def create_path_exclusion_grid(
    grid,
    path,
    exclusion_radius,
):
    blocked_grid = grid.copy()
    path_mask = np.zeros(
        grid.shape,
        dtype=bool,
    )

    if len(path) <= 2 * ENDPOINT_SHARED_PATH_CELLS:
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

    blocked_grid[exclusion_mask] = 1

    sx, sy = path[0]
    gx, gy = path[-1]

    blocked_grid[sy, sx] = 0
    blocked_grid[gy, gx] = 0

    return blocked_grid


def path_overlap(path_a, path_b):
    a = set(path_a)
    b = set(path_b)
    union = a | b
    if not union:
        return 0.0
    return len(a & b) / len(union)


def plot_path(
    ax,
    path,
    label,
    linewidth=1.8,
):
    xs = [x + 0.5 for x, y in path]
    ys = [y + 0.5 for x, y in path]

    ax.plot(
        xs,
        ys,
        linewidth=linewidth,
        label=label,
    )


def summarize_candidate(
    planner,
    grid,
    start,
    goal,
):
    primary = planner.plan(
        grid,
        start,
        goal,
    )

    if not primary["success"]:
        return None

    result = {
        "P1": {
            "path": primary["path"],
            "length": primary["path_length"],
        }
    }

    p1 = primary["path"]

    for radius in EXCLUSION_RADII:
        modified_grid = create_path_exclusion_grid(
            grid,
            p1,
            exclusion_radius=radius,
        )

        alt = planner.plan(
            modified_grid,
            start,
            goal,
        )

        key = f"r{radius}"

        if alt["success"]:
            result[key] = {
                "success": True,
                "path": alt["path"],
                "length": alt["path_length"],
                "ratio": alt["path_length"] / primary["path_length"],
                "overlap": path_overlap(
                    p1,
                    alt["path"],
                ),
            }
        else:
            result[key] = {
                "success": False,
            }

    return result


# ============================================================
# FIGURE: INDIVIDUAL
# ============================================================

def save_individual_figure(
    grid,
    label,
    start,
    goal,
    result,
    save_path,
):
    fig, ax = plt.subplots(
        figsize=(8.5, 8.5)
    )

    ax.imshow(
        grid,
        origin="upper",
        interpolation="nearest",
    )

    plot_path(
        ax,
        result["P1"]["path"],
        f"P1 shortest: {result['P1']['length']:.1f}",
        linewidth=2.2,
    )

    for radius in EXCLUSION_RADII:
        key = f"r{radius}"
        if result[key]["success"]:
            plot_path(
                ax,
                result[key]["path"],
                (
                    f"r={radius}: "
                    f"{result[key]['length']:.1f} "
                    f"(x{result[key]['ratio']:.3f}, "
                    f"ov={result[key]['overlap']:.3f})"
                ),
                linewidth=1.6,
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
            f"{label}\n"
            f"S={start} -> G={goal} | "
            f"P1={result['P1']['length']:.1f}"
        )
    )

    ax.legend(
        loc="best",
        fontsize=8,
    )

    ax.set_xticks([])
    ax.set_yticks([])

    fig.savefig(
        save_path,
        dpi=180,
        bbox_inches="tight",
    )
    plt.close(fig)


# ============================================================
# FIGURE: CONTACT SHEET
# ============================================================

def save_contact_sheet(
    grid,
    all_results,
    save_path,
):
    n = len(all_results)
    cols = 4
    rows = math.ceil(n / cols)

    fig, axes = plt.subplots(
        rows,
        cols,
        figsize=(4.5 * cols, 4.5 * rows),
    )

    axes = np.array(axes).reshape(rows, cols)

    for ax in axes.flat:
        ax.axis("off")

    for idx, item in enumerate(all_results):
        r = idx // cols
        c = idx % cols
        ax = axes[r, c]
        ax.axis("on")

        label = item["label"]
        start = item["start"]
        goal = item["goal"]
        result = item["result"]

        ax.imshow(
            grid,
            origin="upper",
            interpolation="nearest",
        )

        plot_path(
            ax,
            result["P1"]["path"],
            "P1",
            linewidth=1.6,
        )

        for radius in EXCLUSION_RADII:
            key = f"r{radius}"
            if result[key]["success"]:
                plot_path(
                    ax,
                    result[key]["path"],
                    f"r{radius}",
                    linewidth=1.1,
                )

        ax.scatter(
            start[0] + 0.5,
            start[1] + 0.5,
            s=20,
            marker="o",
            zorder=10,
        )

        ax.scatter(
            goal[0] + 0.5,
            goal[1] + 0.5,
            s=24,
            marker="x",
            linewidths=1.5,
            zorder=10,
        )

        text_lines = [
            label,
            f"S{start} -> G{goal}",
            f"P1={result['P1']['length']:.1f}",
        ]

        for radius in EXCLUSION_RADII:
            key = f"r{radius}"
            if result[key]["success"]:
                text_lines.append(
                    f"r{radius}: x{result[key]['ratio']:.3f}"
                )
            else:
                text_lines.append(
                    f"r{radius}: fail"
                )

        ax.set_title(
            "\n".join(text_lines),
            fontsize=8,
        )

        ax.set_xticks([])
        ax.set_yticks([])

    fig.suptitle(
        "16room_000 Multi-Route Candidate Overview",
        fontsize=16,
    )

    fig.savefig(
        save_path,
        dpi=180,
        bbox_inches="tight",
    )
    plt.close(fig)


# ============================================================
# MAIN
# ============================================================

def main():
    project_root = Path(__file__).resolve().parents[1]

    result_dir = (
        project_root
        / "results"
        / "multiroute_plots"
    )
    result_dir.mkdir(
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
    planner = AStarPlanner()

    all_results = []

    print()
    print("=" * 80)
    print("PLOT MULTI-ROUTE CANDIDATES")
    print("=" * 80)

    for label, start, goal in CANDIDATES:
        print(
            f"Processing {label}: "
            f"{start} -> {goal}"
        )

        result = summarize_candidate(
            planner,
            grid,
            start,
            goal,
        )

        if result is None:
            print("  failed")
            continue

        save_path = (
            result_dir
            / f"{label}_{start[0]}_{start[1]}_{goal[0]}_{goal[1]}.png"
        )

        save_individual_figure(
            grid,
            label,
            start,
            goal,
            result,
            save_path,
        )

        all_results.append(
            {
                "label": label,
                "start": start,
                "goal": goal,
                "result": result,
            }
        )

        print(
            f"  saved: {save_path}"
        )

    sheet_path = (
        result_dir
        / "multiroute_contact_sheet.png"
    )

    save_contact_sheet(
        grid,
        all_results,
        sheet_path,
    )

    print()
    print("=" * 80)
    print("DONE")
    print("=" * 80)
    print("Contact sheet:", sheet_path)


if __name__ == "__main__":
    main()