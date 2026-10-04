from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import distance_transform_edt, label

from benchmark.metric_map_loader import load_ros_metric_map


# ============================================================
# SCENARIO 03 — TEMPORARY / CAUTION-OBJECT EXTRACTION
# ============================================================
#
# Purpose:
#   Separate the additional obstacles introduced in the dense
#   Scenario 02 map from the permanent STech Lab structure.
#
# Interpretation:
#   These objects are static in the benchmark snapshot, but are
#   treated in Scenario 03 as temporary / caution-required
#   objects that may move or otherwise justify reduced speed
#   when the robot passes nearby.
#
# This script does NOT modify the map, planner, or path.
# ============================================================


BASE_MAP_YAML = Path(
    "data/metric_maps/stech_lab/"
    "stech_lab_completed.yaml"
)

DENSE_MAP_YAML = Path(
    "data/metric_maps/stech_lab_scenario02/"
    "stech_lab_scenario02.yaml"
)

RESULT_DIR = Path(
    "results/stech_lab/scenario03/"
    "temporary_objects"
)


def validate_map_compatibility(
    base_map,
    dense_map,
):
    """
    The two maps must share the same metric frame so that a
    pixel-wise occupancy difference has physical meaning.
    """

    keys = (
        "resolution",
        "origin",
        "width",
        "height",
    )

    mismatches = []

    for key in keys:
        if key not in base_map or key not in dense_map:
            continue

        a = base_map[key]
        b = dense_map[key]

        if isinstance(a, np.ndarray):
            same = np.array_equal(
                a,
                b,
            )
        else:
            same = (
                a == b
            )

        if not same:
            mismatches.append(
                (
                    key,
                    a,
                    b,
                )
            )

    if (
        base_map["grid"].shape
        != dense_map["grid"].shape
    ):
        mismatches.append(
            (
                "grid.shape",
                base_map["grid"].shape,
                dense_map["grid"].shape,
            )
        )

    if mismatches:
        lines = [
            "Base and dense maps are not compatible:"
        ]

        for key, a, b in mismatches:
            lines.append(
                f"  {key}: {a} != {b}"
            )

        raise RuntimeError(
            "\n".join(
                lines
            )
        )


def extract_added_obstacles(
    base_grid,
    dense_grid,
):
    """
    grid convention:
        0 = free
        nonzero = occupied / unavailable

    Temporary-object pixels are cells that are occupied in the
    dense map but were free in the original base map.
    """

    base_occupied = (
        base_grid != 0
    )

    dense_occupied = (
        dense_grid != 0
    )

    added_mask = (
        dense_occupied
        &
        ~base_occupied
    )

    removed_mask = (
        base_occupied
        &
        ~dense_occupied
    )

    return (
        added_mask,
        removed_mask,
    )


def save_outputs(
    dense_grid,
    added_mask,
    resolution,
):
    RESULT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    mask_npy = (
        RESULT_DIR
        / "scenario03_temporary_object_mask.npy"
    )

    np.save(
        mask_npy,
        added_mask.astype(
            np.uint8
        ),
    )

    # Distance from every cell centre to the nearest temporary
    # object pixel.  This is saved now so the later slowdown
    # model can reuse the exact same object definition.
    distance_m = (
        distance_transform_edt(
            ~added_mask
        )
        * resolution
    )

    distance_npy = (
        RESULT_DIR
        / "scenario03_temporary_object_distance_m.npy"
    )

    np.save(
        distance_npy,
        distance_m.astype(
            np.float32
        ),
    )

    mask_png = (
        RESULT_DIR
        / "scenario03_temporary_object_mask.png"
    )

    fig, ax = plt.subplots(
        figsize=(12, 10)
    )

    ax.imshow(
        added_mask,
        cmap="gray",
        origin="upper",
    )

    ax.set_title(
        "Scenario 03 — Extracted Temporary / Caution Objects"
    )

    ax.set_xticks([])
    ax.set_yticks([])

    fig.tight_layout()

    fig.savefig(
        mask_png,
        dpi=180,
        bbox_inches="tight",
    )

    plt.close(
        fig
    )

    overlay_png = (
        RESULT_DIR
        / "scenario03_temporary_objects_on_map.png"
    )

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(16, 8),
    )

    axes[0].imshow(
        dense_grid,
        cmap="gray_r",
        origin="upper",
    )

    axes[0].set_title(
        "Dense Benchmark Map"
    )

    axes[1].imshow(
        dense_grid,
        cmap="gray_r",
        origin="upper",
    )

    masked = np.ma.masked_where(
        ~added_mask,
        added_mask,
    )

    axes[1].imshow(
        masked,
        alpha=0.65,
        origin="upper",
    )

    axes[1].set_title(
        "Extracted Temporary / Caution Objects"
    )

    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])

    fig.tight_layout()

    fig.savefig(
        overlay_png,
        dpi=180,
        bbox_inches="tight",
    )

    plt.close(
        fig
    )

    return {
        "mask_npy":
            mask_npy,

        "distance_npy":
            distance_npy,

        "mask_png":
            mask_png,

        "overlay_png":
            overlay_png,
    }


def main():
    base_map = (
        load_ros_metric_map(
            BASE_MAP_YAML
        )
    )

    dense_map = (
        load_ros_metric_map(
            DENSE_MAP_YAML
        )
    )

    validate_map_compatibility(
        base_map,
        dense_map,
    )

    base_grid = (
        base_map["grid"]
    )

    dense_grid = (
        dense_map["grid"]
    )

    resolution = float(
        dense_map["resolution"]
    )

    (
        added_mask,
        removed_mask,
    ) = extract_added_obstacles(
        base_grid,
        dense_grid,
    )

    added_pixels = int(
        np.count_nonzero(
            added_mask
        )
    )

    removed_pixels = int(
        np.count_nonzero(
            removed_mask
        )
    )

    _, component_count = label(
        added_mask
    )

    added_area_m2 = (
        added_pixels
        * resolution
        * resolution
    )

    print()
    print(
        "=" * 88
    )

    print(
        "SCENARIO 03 — TEMPORARY / CAUTION-OBJECT EXTRACTION"
    )

    print(
        "=" * 88
    )

    print(
        f"Resolution                 : "
        f"{resolution:.3f} m/cell"
    )

    print(
        f"Added occupied pixels      : "
        f"{added_pixels}"
    )

    print(
        f"Added occupied area        : "
        f"{added_area_m2:.3f} m^2"
    )

    print(
        f"Connected mask components  : "
        f"{component_count}"
    )

    print(
        f"Removed base-map pixels    : "
        f"{removed_pixels}"
    )

    if removed_pixels != 0:
        print()
        print(
            "WARNING: Dense map also removes occupancy from "
            "the base map. Inspect the map difference."
        )

    outputs = save_outputs(
        dense_grid,
        added_mask,
        resolution,
    )

    print()
    print(
        "Saved:"
    )

    for path in outputs.values():
        print(
            " ",
            path,
        )


if __name__ == "__main__":
    main()