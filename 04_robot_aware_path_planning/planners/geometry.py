import math


def euclidean_distance(a, b):
    dx = b[0] - a[0]
    dy = b[1] - a[1]

    return math.hypot(dx, dy)


def path_length(path):
    if len(path) < 2:
        return 0.0

    total = 0.0

    for i in range(len(path) - 1):
        total += euclidean_distance(
            path[i],
            path[i + 1],
        )

    return total


def point_is_free(grid, point):
    x, y = point

    height, width = grid.shape

    ix = int(math.floor(x))
    iy = int(math.floor(y))

    if ix < 0 or ix >= width:
        return False

    if iy < 0 or iy >= height:
        return False

    return grid[iy, ix] == 0

def edge_is_free(
    grid,
    p0,
    p1,
    resolution=0.25,
):
    """
    Check the complete line segment between p0 and p1.

    resolution is expressed in grid-cell units.
    A value smaller than one cell prevents skipping
    thin obstacles.
    """

    distance = euclidean_distance(
        p0,
        p1,
    )

    if distance == 0.0:
        return point_is_free(
            grid,
            p0,
        )

    steps = max(
        1,
        int(math.ceil(
            distance / resolution
        )),
    )

    for i in range(steps + 1):

        t = i / steps

        x = (
            p0[0]
            + t * (p1[0] - p0[0])
        )

        y = (
            p0[1]
            + t * (p1[1] - p0[1])
        )

        if not point_is_free(
            grid,
            (x, y),
        ):
            return False

    return True