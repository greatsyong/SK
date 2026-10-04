import math

import numpy as np
from scipy.ndimage import binary_dilation


def make_circular_structuring_element(
    radius_cells,
):
    radius_cells = int(
        math.ceil(radius_cells)
    )

    size = (
        2 * radius_cells
        + 1
    )

    y, x = np.ogrid[
        -radius_cells:radius_cells + 1,
        -radius_cells:radius_cells + 1
    ]

    mask = (
        x * x + y * y
        <= radius_cells * radius_cells
    )

    return mask


def inflate_grid_for_robot(
    grid,
    robot,
    resolution_m_per_cell,
):
    """
    Conservative orientation-independent configuration-space
    inflation using the robot's circumscribed radius.

    grid:
        0 = free
        1 = occupied
    """

    if resolution_m_per_cell <= 0:
        raise ValueError(
            "resolution_m_per_cell must be positive"
        )

    radius_m = (
        robot.circumscribed_radius_m
    )

    radius_cells = (
        radius_m
        / resolution_m_per_cell
    )

    structure = (
        make_circular_structuring_element(
            radius_cells
        )
    )

    occupied = grid.astype(bool)

    inflated = binary_dilation(
        occupied,
        structure=structure,
    )

    return inflated.astype(
        np.uint8
    ), {
        "inflation_radius_m":
            radius_m,

        "inflation_radius_cells":
            radius_cells,

        "resolution_m_per_cell":
            resolution_m_per_cell,
    }