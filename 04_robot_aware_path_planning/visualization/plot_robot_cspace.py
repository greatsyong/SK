import matplotlib.pyplot as plt


def plot_robot_cspaces(
    raw_grid,
    robot_results,
    resolution,
    save_path=None,
):
    """
    robot_results:
        [
            (robot_name, inflated_grid, inflation_info),
            ...
        ]
    """

    count = 1 + len(robot_results)

    fig, axes = plt.subplots(
        1,
        count,
        figsize=(7 * count, 7),
    )

    if count == 1:
        axes = [axes]

    # --------------------------------------------------------
    # Raw map
    # --------------------------------------------------------

    ax = axes[0]

    ax.imshow(
        raw_grid,
        cmap="gray_r",
        interpolation="nearest",
    )

    ax.set_title(
        "Original metric occupancy map"
    )

    ax.set_xlabel("Grid x")
    ax.set_ylabel("Grid y")

    # --------------------------------------------------------
    # Robot configuration spaces
    # --------------------------------------------------------

    for ax, result in zip(
        axes[1:],
        robot_results,
    ):

        (
            robot_name,
            inflated_grid,
            info,
        ) = result

        ax.imshow(
            inflated_grid,
            cmap="gray_r",
            interpolation="nearest",
        )

        radius_m = info[
            "inflation_radius_m"
        ]

        radius_cells = info[
            "inflation_radius_cells"
        ]

        ax.set_title(
            f"{robot_name}\n"
            f"r = {radius_m:.3f} m "
            f"({radius_cells:.2f} cells)"
        )

        ax.set_xlabel("Grid x")
        ax.set_ylabel("Grid y")

    fig.suptitle(
        f"Robot-specific configuration spaces "
        f"(resolution = {resolution:.3f} m/cell)"
    )

    fig.tight_layout()

    if save_path is not None:
        fig.savefig(
            save_path,
            dpi=200,
            bbox_inches="tight",
        )

        print(
            "Saved:",
            save_path,
        )

    plt.show()

    return fig, axes