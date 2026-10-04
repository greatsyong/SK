from pathlib import Path

import matplotlib.pyplot as plt


def plot_rrt_result(
    grid,
    start,
    goal,
    result,
    title=None,
    save_path=None,
    show=True,
):
    fig, ax = plt.subplots(
        figsize=(10, 10)
    )

    # --------------------------------------------------------
    # Occupancy map
    # --------------------------------------------------------

    ax.imshow(
        grid,
        origin="upper",
        cmap="gray_r",
        interpolation="nearest",
    )

    # --------------------------------------------------------
    # RRT tree
    # --------------------------------------------------------

    nodes = result.get("nodes", [])
    parents = result.get("parents", [])

    if nodes and parents:

        for i in range(1, len(nodes)):

            parent_index = parents[i]

            if parent_index is None:
                continue

            x1, y1 = nodes[parent_index]
            x2, y2 = nodes[i]

            ax.plot(
                [x1, x2],
                [y1, y2],
                linewidth=0.45,
                alpha=0.35,
            )

    # --------------------------------------------------------
    # Final path
    # --------------------------------------------------------

    path = result.get("path", [])

    if path:

        path_x = [
            point[0]
            for point in path
        ]

        path_y = [
            point[1]
            for point in path
        ]

        ax.plot(
            path_x,
            path_y,
            linewidth=2.5,
            label="RRT path",
        )

    # --------------------------------------------------------
    # Start / Goal
    # --------------------------------------------------------

    start_center = (
        start[0] + 0.5,
        start[1] + 0.5,
    )

    goal_center = (
        goal[0] + 0.5,
        goal[1] + 0.5,
    )

    ax.scatter(
        start_center[0],
        start_center[1],
        s=100,
        marker="o",
        label="Start",
        zorder=10,
    )

    ax.scatter(
        goal_center[0],
        goal_center[1],
        s=130,
        marker="*",
        label="Goal",
        zorder=10,
    )

    # --------------------------------------------------------
    # Formatting
    # --------------------------------------------------------

    ax.set_xlim(
        0,
        grid.shape[1],
    )

    ax.set_ylim(
        grid.shape[0],
        0,
    )

    ax.set_aspect(
        "equal"
    )

    ax.set_xlabel(
        "Grid x"
    )

    ax.set_ylabel(
        "Grid y"
    )

    if title:
        ax.set_title(title)

    ax.legend(
        loc="best"
    )

    fig.tight_layout()

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    if save_path is not None:

        save_path = Path(
            save_path
        )

        save_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        fig.savefig(
            save_path,
            dpi=200,
            bbox_inches="tight",
        )

        print(
            f"Saved visualization: "
            f"{save_path}"
        )

    if show:
        plt.show()

    return fig, ax