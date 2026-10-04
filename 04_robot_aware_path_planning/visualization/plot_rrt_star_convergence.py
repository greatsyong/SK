from pathlib import Path

import matplotlib.pyplot as plt

from benchmark.map_loader import load_movingai_map
from benchmark.scenario_loader import load_movingai_scenarios
from planners.rrt_star import RRTStarPlanner


MAP_NAME = "maze512-32-0"
SEED = 42

ITERATION_BUDGETS = [
    1000,
    2000,
    5000,
    10000,
    20000,
]


def draw_tree(
    ax,
    nodes,
    parents,
):
    """
    Draw all RRT* tree edges.
    """

    for i in range(1, len(nodes)):

        parent_index = parents[i]

        if parent_index is None:
            continue

        x0, y0 = nodes[parent_index]
        x1, y1 = nodes[i]

        ax.plot(
            [x0, x1],
            [y0, y1],
            linewidth=0.25,
            alpha=0.25,
        )


def draw_path(
    ax,
    path,
):
    """
    Draw final solution path.
    """

    if not path:
        return

    xs = [
        p[0]
        for p in path
    ]

    ys = [
        p[1]
        for p in path
    ]

    ax.plot(
        xs,
        ys,
        linewidth=2.0,
    )


def main():

    project_root = (
        Path(__file__).resolve().parents[1]
    )

    result_dir = (
        project_root
        / "results"
    )

    result_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ========================================================
    # Load map and scenario
    # ========================================================

    map_data = load_movingai_map(
        project_root
        / "data"
        / "maps"
        / f"{MAP_NAME}.map"
    )

    scenarios = load_movingai_scenarios(
        project_root
        / "data"
        / "scenarios"
        / f"{MAP_NAME}.map.scen"
    )

    long_scenarios = [
        scenario
        for scenario in scenarios
        if scenario.optimal_length >= 200.0
    ]

    scenario = long_scenarios[0]

    grid = map_data["grid"]

    # ========================================================
    # Run all iteration budgets
    # ========================================================

    results = []

    for budget in ITERATION_BUDGETS:

        print(
            f"Running RRT*: "
            f"{budget} iterations..."
        )

        planner = RRTStarPlanner(
            step_size=10.0,
            goal_radius=10.0,
            goal_sample_rate=0.05,
            max_iterations=budget,
            neighbor_radius=30.0,
            seed=SEED,
        )

        result = planner.plan(
            grid,
            scenario.start,
            scenario.goal,
            return_tree=True,
        )

        results.append(
            (
                budget,
                result,
            )
        )

    # ========================================================
    # Plot
    # ========================================================

    fig, axes = plt.subplots(
        1,
        len(ITERATION_BUDGETS),
        figsize=(22, 5),
        constrained_layout=True,
    )

    start_point = (
        scenario.start[0] + 0.5,
        scenario.start[1] + 0.5,
    )

    goal_point = (
        scenario.goal[0] + 0.5,
        scenario.goal[1] + 0.5,
    )

    for ax, (budget, result) in zip(
        axes,
        results,
    ):

        # Map
        ax.imshow(
            grid,
            origin="upper",
            interpolation="nearest",
        )

        # Tree
        draw_tree(
            ax,
            result["nodes"],
            result["parents"],
        )

        # Final path
        if result["success"]:
            draw_path(
                ax,
                result["path"],
            )

        # Start
        ax.scatter(
            start_point[0],
            start_point[1],
            marker="o",
            s=45,
            zorder=10,
        )

        # Goal
        ax.scatter(
            goal_point[0],
            goal_point[1],
            marker="x",
            s=60,
            linewidths=2,
            zorder=10,
        )

        if result["success"]:

            title = (
                f"N = {budget:,}\n"
                f"L = "
                f"{result['path_length']:.2f}\n"
                f"T = "
                f"{result['planning_time']:.2f} s"
            )

        else:

            title = (
                f"N = {budget:,}\n"
                f"No solution\n"
                f"T = "
                f"{result['planning_time']:.2f} s"
            )

        ax.set_title(
            title,
            fontsize=10,
        )

        ax.set_xticks([])
        ax.set_yticks([])

    fig.suptitle(
        (
            "RRT* Convergence — "
            f"{MAP_NAME} | Seed {SEED}\n"
            f"Reference grid path length = "
            f"{scenario.optimal_length:.3f}"
        ),
        fontsize=14,
    )

    output_path = (
        result_dir
        / (
            f"{MAP_NAME}_"
            f"rrt_star_convergence_"
            f"seed{SEED}.png"
        )
    )

    fig.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight",
    )

    print()
    print(
        "Saved:",
        output_path
    )

    plt.show()


if __name__ == "__main__":
    main()