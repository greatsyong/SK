from pathlib import Path

import numpy as np
import yaml
from PIL import Image


def load_ros_metric_map(yaml_path):
    """
    Load a standard ROS/Nav2 map YAML + image pair.

    Output convention:
        0 = free
        1 = occupied

    Unknown cells are conservatively treated as occupied.
    """

    yaml_path = Path(yaml_path)

    with yaml_path.open("r") as f:
        config = yaml.safe_load(f)

    image_path = yaml_path.parent / config["image"]

    resolution = float(config["resolution"])
    origin = tuple(float(v) for v in config["origin"])

    negate = int(config.get("negate", 0))
    occupied_thresh = float(
        config.get("occupied_thresh", 0.65)
    )
    free_thresh = float(
        config.get("free_thresh", 0.25)
    )

    mode = config.get("mode", "trinary")

    image = Image.open(image_path).convert("L")

    pixels = np.asarray(
        image,
        dtype=np.float64,
    )

    normalized = pixels / 255.0

    if negate == 0:
        occupancy_probability = (
            1.0 - normalized
        )
    else:
        occupancy_probability = normalized

    occupied = (
        occupancy_probability
        > occupied_thresh
    )

    free = (
        occupancy_probability
        < free_thresh
    )

    unknown = ~(occupied | free)

    # Conservative planning representation:
    # unknown space is not considered traversable.
    grid = np.zeros(
        pixels.shape,
        dtype=np.uint8,
    )

    grid[occupied] = 1
    grid[unknown] = 1

    return {
        "grid": grid,
        "resolution": resolution,
        "origin": origin,
        "width": grid.shape[1],
        "height": grid.shape[0],
        "occupied_thresh": occupied_thresh,
        "free_thresh": free_thresh,
        "mode": mode,
        "image_path": str(image_path),
        "yaml_path": str(yaml_path),
        "unknown_mask": unknown,
    }