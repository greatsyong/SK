from pathlib import Path

import numpy as np


FREE_CHARS = {".", "G", "S"}
BLOCKED_CHARS = {"@", "O", "T", "W"}


def load_movingai_map(path):
    path = Path(path)

    with path.open("r") as f:
        map_type = f.readline().strip()

        height_line = f.readline().strip()
        width_line = f.readline().strip()
        map_line = f.readline().strip()

        if not map_type.startswith("type"):
            raise ValueError(f"Invalid Moving AI map header: {map_type}")

        if not height_line.startswith("height"):
            raise ValueError("Missing map height")

        if not width_line.startswith("width"):
            raise ValueError("Missing map width")

        if map_line != "map":
            raise ValueError("Missing 'map' marker")

        height = int(height_line.split()[1])
        width = int(width_line.split()[1])

        grid = np.zeros((height, width), dtype=np.uint8)

        for y in range(height):
            row = f.readline().rstrip("\n")

            if len(row) != width:
                raise ValueError(
                    f"Row {y}: expected width {width}, got {len(row)}"
                )

            for x, char in enumerate(row):
                if char in FREE_CHARS:
                    grid[y, x] = 0
                else:
                    grid[y, x] = 1

    return {
        "grid": grid,
        "width": width,
        "height": height,
        "type": map_type.split(maxsplit=1)[1],
        "source": str(path),
    }