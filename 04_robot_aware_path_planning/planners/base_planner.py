import math


STRAIGHT_COST = 1.0
DIAGONAL_COST = math.sqrt(2.0)


MOVES = [
    (-1,  0, STRAIGHT_COST),
    ( 1,  0, STRAIGHT_COST),
    ( 0, -1, STRAIGHT_COST),
    ( 0,  1, STRAIGHT_COST),

    (-1, -1, DIAGONAL_COST),
    (-1,  1, DIAGONAL_COST),
    ( 1, -1, DIAGONAL_COST),
    ( 1,  1, DIAGONAL_COST),
]


def is_free(grid, x, y):
    height, width = grid.shape

    if x < 0 or x >= width:
        return False

    if y < 0 or y >= height:
        return False

    return grid[y, x] == 0


def get_neighbors(grid, x, y):
    neighbors = []

    for dx, dy, cost in MOVES:
        nx = x + dx
        ny = y + dy

        if not is_free(grid, nx, ny):
            continue

        # Moving AI rule:
        # diagonal motion cannot cut across obstacle corners.
        if dx != 0 and dy != 0:
            if not is_free(grid, x + dx, y):
                continue

            if not is_free(grid, x, y + dy):
                continue

        neighbors.append((nx, ny, cost))

    return neighbors