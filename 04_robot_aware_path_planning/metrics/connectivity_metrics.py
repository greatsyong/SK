import numpy as np
from scipy.ndimage import label


def analyze_free_space_connectivity(
    grid,
    connectivity=8,
):
    """
    Analyze connected components in free space.

    grid convention:
        0 = free
        1 = occupied
    """

    free_mask = (grid == 0)

    if connectivity == 4:
        structure = np.array(
            [
                [0, 1, 0],
                [1, 1, 1],
                [0, 1, 0],
            ],
            dtype=np.uint8,
        )

    elif connectivity == 8:
        structure = np.ones(
            (3, 3),
            dtype=np.uint8,
        )

    else:
        raise ValueError(
            "connectivity must be 4 or 8"
        )

    labeled, num_components = label(
        free_mask,
        structure=structure,
    )

    component_sizes = np.bincount(
        labeled.ravel()
    )

    # Index 0 corresponds to background / occupied space.
    component_sizes = component_sizes[1:]

    total_free = int(
        free_mask.sum()
    )

    if num_components == 0:
        largest_component = 0
        largest_fraction = 0.0

    else:
        largest_component = int(
            component_sizes.max()
        )

        largest_fraction = (
            largest_component
            / total_free
            if total_free > 0
            else 0.0
        )

    sorted_sizes = sorted(
        component_sizes.tolist(),
        reverse=True,
    )

    return {
        "num_components":
            int(num_components),

        "total_free_cells":
            total_free,

        "largest_component_cells":
            largest_component,

        "largest_component_fraction":
            largest_fraction,

        "component_sizes":
            sorted_sizes,

        "labels":
            labeled,
    }