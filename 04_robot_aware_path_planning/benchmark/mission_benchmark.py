from dataclasses import dataclass
import random


@dataclass(frozen=True)
class Mission:
    mission_id: int
    start: tuple[int, int]
    goal: tuple[int, int]


def sample_common_missions(
    raw_grid,
    num_missions=100,
    min_distance_cells=100,
    seed=42,
):
    """
    Generate one shared mission set from the ORIGINAL free space.

    These same start/goal pairs are later evaluated against each
    robot-specific configuration space.

    grid convention:
        0 = free
        1 = occupied
    """

    rng = random.Random(seed)

    height, width = raw_grid.shape

    free_cells = [
        (x, y)
        for y in range(height)
        for x in range(width)
        if raw_grid[y, x] == 0
    ]

    missions = []

    attempts = 0
    max_attempts = num_missions * 10000

    while (
        len(missions) < num_missions
        and attempts < max_attempts
    ):
        attempts += 1

        start = rng.choice(free_cells)
        goal = rng.choice(free_cells)

        if start == goal:
            continue

        dx = goal[0] - start[0]
        dy = goal[1] - start[1]

        distance_sq = dx * dx + dy * dy

        if distance_sq < (
            min_distance_cells
            * min_distance_cells
        ):
            continue

        missions.append(
            Mission(
                mission_id=len(missions),
                start=start,
                goal=goal,
            )
        )

    if len(missions) < num_missions:
        raise RuntimeError(
            f"Could only generate "
            f"{len(missions)} missions"
        )

    return missions