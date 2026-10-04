import math

import numpy as np
from scipy.ndimage import distance_transform_edt, map_coordinates

from benchmark.refine_scenario01_paths import simplify_path_los
from benchmark.smooth_scenario01_paths import smooth_polyline
from benchmark.run_scenario02_analysis import (
    RESOLUTION as ANALYSIS_RESOLUTION,
    ROBOTS,
    resample_path_uniform,
    compute_arc_length,
    compute_curvature,
    speed_limit_from_curvature,
    parameterize_velocity,
    integrate_time,
)


# ============================================================
# SCENARIO 03 — COMMON PATH EVALUATOR
# ============================================================
#
# One planner-independent evaluation pipeline:
#
#   raw planner path
#       -> LOS refinement
#       -> smoothing
#       -> uniform resampling
#       -> hard-clearance validation
#       -> curvature limit
#       -> 80% open-space cruise
#       -> 0.30 m/s caution-zone cap
#       -> acceleration/deceleration pass
#       -> mission-time integration
#
# The planner itself is intentionally outside this module.
# ============================================================


SAFETY_MARGIN_M = 0.025

OPEN_CRUISE_RATIO = 0.80
CAUTION_SPEED_CAP_MPS = 0.30
CAUTION_CLEARANCE_M = 1.00


def circumscribed_radius(
    length_m,
    width_m,
):
    return 0.5 * math.hypot(
        length_m,
        width_m,
    )


def path_length_m(
    points,
    resolution,
):
    points = np.asarray(
        points,
        dtype=float,
    )

    if len(points) < 2:
        return 0.0

    delta = np.diff(
        points,
        axis=0,
    )

    return float(
        np.sum(
            np.linalg.norm(
                delta,
                axis=1,
            )
        )
        * resolution
    )


def build_clearance_field(
    raw_free,
    resolution,
):
    return (
        distance_transform_edt(
            raw_free
        )
        * resolution
    )


def build_robot_context(
    robot_key,
    raw_free,
    resolution,
    all_obstacle_clearance_m=None,
):
    """
    Build the robot-specific Scenario 03 planning context once.

    Returns both:
      cspace_free : bool array, True means feasible/free
      planner_grid: uint8 array, 0 means free and 1 means occupied

    The second representation matches the existing planner classes.
    """

    if robot_key not in ROBOTS:
        raise ValueError(
            f"Unknown robot_key: {robot_key}"
        )

    robot = ROBOTS[
        robot_key
    ]

    if all_obstacle_clearance_m is None:
        all_obstacle_clearance_m = (
            build_clearance_field(
                raw_free,
                resolution,
            )
        )

    footprint_radius_m = (
        circumscribed_radius(
            robot["length_m"],
            robot["width_m"],
        )
    )

    effective_radius_m = (
        footprint_radius_m
        + SAFETY_MARGIN_M
    )

    cspace_free = (
        raw_free
        &
        (
            all_obstacle_clearance_m
            >= effective_radius_m
        )
    )

    planner_grid = np.where(
        cspace_free,
        0,
        1,
    ).astype(
        np.uint8
    )

    return {
        "robot_key":
            robot_key,

        "robot":
            robot,

        "footprint_radius_m":
            footprint_radius_m,

        "effective_radius_m":
            effective_radius_m,

        "all_obstacle_clearance_m":
            all_obstacle_clearance_m,

        "cspace_free":
            cspace_free,

        "planner_grid":
            planner_grid,
    }


def sample_field_bilinear(
    field,
    path_cells,
):
    path_cells = np.asarray(
        path_cells,
        dtype=float,
    )

    if (
        path_cells.ndim != 2
        or path_cells.shape[1] != 2
    ):
        raise ValueError(
            "path_cells must have shape (N, 2)"
        )

    coords = np.vstack(
        (
            path_cells[:, 1],
            path_cells[:, 0],
        )
    )

    return map_coordinates(
        field,
        coords,
        order=1,
        mode="nearest",
    )


def operating_zone_speed_limit(
    object_center_distance_m,
    effective_radius_m,
    v_max,
):
    """
    Human-environment operating policy.

    Open space:
        80% of rated platform maximum speed.

    Caution zone:
        max translational speed = 0.30 m/s.

    For robots whose open cruise is already below 0.30 m/s,
    the caution cap does not reduce speed further.
    """

    available_clearance_m = np.maximum(
        0.0,
        object_center_distance_m
        - effective_radius_m,
    )

    open_cruise_mps = (
        OPEN_CRUISE_RATIO
        * float(v_max)
    )

    caution_cap_mps = min(
        open_cruise_mps,
        CAUTION_SPEED_CAP_MPS,
    )

    in_caution_zone = (
        available_clearance_m
        <= CAUTION_CLEARANCE_M
    )

    zone_limit_mps = np.full(
        len(
            available_clearance_m
        ),
        open_cruise_mps,
        dtype=float,
    )

    zone_limit_mps[
        in_caution_zone
    ] = caution_cap_mps

    return {
        "available_clearance_m":
            available_clearance_m,

        "in_caution_zone":
            in_caution_zone,

        "zone_limit_mps":
            zone_limit_mps,

        "open_cruise_mps":
            open_cruise_mps,

        "caution_cap_mps":
            caution_cap_mps,
    }


def validate_trajectory(
    path_cells,
    all_obstacle_clearance_m,
    effective_radius_m,
):
    sampled_clearance_m = (
        sample_field_bilinear(
            all_obstacle_clearance_m,
            path_cells,
        )
    )

    clearance_margin_m = (
        sampled_clearance_m
        - effective_radius_m
    )

    violation_mask = (
        clearance_margin_m
        < 0.0
    )

    min_clearance_m = float(
        np.min(
            sampled_clearance_m
        )
    )

    min_margin_m = float(
        np.min(
            clearance_margin_m
        )
    )

    max_clearance_violation_m = float(
        max(
            0.0,
            -min_margin_m,
        )
    )

    feasible = bool(
        not np.any(
            violation_mask
        )
    )

    return {
        "feasible":
            feasible,

        "min_clearance_m":
            min_clearance_m,

        "margin_m":
            min_margin_m,

        "max_clearance_violation_m":
            max_clearance_violation_m,

        "max_clearance_violation_mm":
            (
                1000.0
                * max_clearance_violation_m
            ),

        "violation_sample_count":
            int(
                np.count_nonzero(
                    violation_mask
                )
            ),

        "sample_count":
            int(
                len(
                    sampled_clearance_m
                )
            ),

        "sampled_clearance_m":
            sampled_clearance_m,

        "violation_mask":
            violation_mask,
    }


def evaluate_path(
    raw_path,
    robot_key,
    robot_context,
    temporary_distance_m,
    resolution,
):
    """
    Evaluate one successful planner path.

    This function has no knowledge of Dijkstra, A*, RRT, or RRT*.
    Any planner may provide raw_path as long as it is an (N,2)
    sequence in map-cell coordinates.
    """

    if not math.isclose(
        float(resolution),
        float(ANALYSIS_RESOLUTION),
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise ValueError(
            "Existing Scenario 02 trajectory functions assume "
            f"{ANALYSIS_RESOLUTION:.3f} m/cell, but received "
            f"{resolution:.3f} m/cell."
        )

    raw_path = np.asarray(
        raw_path,
        dtype=float,
    )

    if (
        raw_path.ndim != 2
        or raw_path.shape[1] != 2
        or len(raw_path) < 2
    ):
        raise ValueError(
            "raw_path must contain at least two (x, y) points."
        )

    cspace_free = (
        robot_context[
            "cspace_free"
        ]
    )

    effective_radius_m = (
        robot_context[
            "effective_radius_m"
        ]
    )

    all_obstacle_clearance_m = (
        robot_context[
            "all_obstacle_clearance_m"
        ]
    )

    robot = (
        robot_context[
            "robot"
        ]
    )

    # --------------------------------------------------------
    # Raw-path validation
    # --------------------------------------------------------
    #
    # Planner success and continuous clearance feasibility are
    # intentionally kept separate. The raw path is uniformly
    # resampled before validation so long RRT/RRT* edges are not
    # judged only at their endpoints.
    # --------------------------------------------------------

    raw_resampled = resample_path_uniform(
        raw_path,
        spacing_m=resolution,
    )

    (
        _,
        raw_ds,
        _,
    ) = compute_arc_length(
        raw_resampled
    )

    raw_validation = validate_trajectory(
        raw_resampled,
        all_obstacle_clearance_m,
        effective_radius_m,
    )

    raw_violation_segments = (
        raw_validation[
            "violation_mask"
        ][:-1]
        |
        raw_validation[
            "violation_mask"
        ][1:]
    )

    raw_violation_path_length_m = float(
        np.sum(
            raw_ds[
                raw_violation_segments
            ]
        )
    )

    # --------------------------------------------------------
    # Common refinement
    # --------------------------------------------------------

    los = simplify_path_los(
        raw_path,
        cspace_free,
    )

    smooth = smooth_polyline(
        los,
        cspace_free,
        resolution,
    )

    resampled = resample_path_uniform(
        smooth,
        spacing_m=resolution,
    )

    # --------------------------------------------------------
    # Geometry
    # --------------------------------------------------------

    (
        points_m,
        ds,
        s,
    ) = compute_arc_length(
        resampled
    )

    curvature = compute_curvature(
        points_m
    )

    # --------------------------------------------------------
    # Hard-clearance validation
    # --------------------------------------------------------

    validation = validate_trajectory(
        resampled,
        all_obstacle_clearance_m,
        effective_radius_m,
    )

    violation_segments = (
        validation[
            "violation_mask"
        ][:-1]
        |
        validation[
            "violation_mask"
        ][1:]
    )

    violation_path_length_m = float(
        np.sum(
            ds[
                violation_segments
            ]
        )
    )

    # --------------------------------------------------------
    # Existing curvature/dynamic speed constraints
    # --------------------------------------------------------

    curvature_limit_mps = (
        speed_limit_from_curvature(
            robot_key,
            curvature,
        )
    )

    # --------------------------------------------------------
    # Temporary/caution-object operating policy
    # --------------------------------------------------------

    temp_center_distance_m = (
        sample_field_bilinear(
            temporary_distance_m,
            resampled,
        )
    )

    operating = (
        operating_zone_speed_limit(
            temp_center_distance_m,
            effective_radius_m,
            robot["v_max"],
        )
    )

    nominal_limit_mps = np.minimum(
        curvature_limit_mps,
        operating[
            "open_cruise_mps"
        ],
    )

    safety_limit_mps = np.minimum(
        curvature_limit_mps,
        operating[
            "zone_limit_mps"
        ],
    )

    # --------------------------------------------------------
    # Acceleration/deceleration passes and mission time
    # --------------------------------------------------------

    nominal_velocity_mps = (
        parameterize_velocity(
            ds,
            nominal_limit_mps,
            robot["a_max"],
            robot["decel_max"],
        )
    )

    (
        nominal_dt_s,
        nominal_time_s,
    ) = integrate_time(
        ds,
        nominal_velocity_mps,
    )

    safety_velocity_mps = (
        parameterize_velocity(
            ds,
            safety_limit_mps,
            robot["a_max"],
            robot["decel_max"],
        )
    )

    (
        safety_dt_s,
        safety_time_s,
    ) = integrate_time(
        ds,
        safety_velocity_mps,
    )

    # --------------------------------------------------------
    # Caution-zone exposure
    # --------------------------------------------------------

    in_caution_zone = (
        operating[
            "in_caution_zone"
        ]
    )

    caution_segments = (
        in_caution_zone[:-1]
        |
        in_caution_zone[1:]
    )

    caution_distance_m = float(
        np.sum(
            ds[
                caution_segments
            ]
        )
    )

    final_length_m = float(
        s[-1]
    )

    caution_ratio = (
        caution_distance_m
        / final_length_m
        if final_length_m > 1e-12
        else 0.0
    )

    # --------------------------------------------------------
    # Common result
    # --------------------------------------------------------

    return {
        "robot_key":
            robot_key,

        "robot_name":
            robot["name"],

        "raw":
            raw_path,

        "raw_resampled":
            np.asarray(
                raw_resampled,
                dtype=float,
            ),

        "los":
            np.asarray(
                los,
                dtype=float,
            ),

        "smooth":
            np.asarray(
                smooth,
                dtype=float,
            ),

        "resampled":
            np.asarray(
                resampled,
                dtype=float,
            ),

        "raw_length_m":
            path_length_m(
                raw_path,
                resolution,
            ),

        "los_length_m":
            path_length_m(
                los,
                resolution,
            ),

        "final_length_m":
            final_length_m,

        "raw_path_feasible":
            raw_validation[
                "feasible"
            ],

        "raw_min_clearance_m":
            raw_validation[
                "min_clearance_m"
            ],

        "raw_clearance_margin_m":
            raw_validation[
                "margin_m"
            ],

        "raw_max_clearance_violation_mm":
            raw_validation[
                "max_clearance_violation_mm"
            ],

        "raw_violation_sample_count":
            raw_validation[
                "violation_sample_count"
            ],

        "raw_violation_path_length_m":
            raw_violation_path_length_m,

        "trajectory_feasible":
            validation[
                "feasible"
            ],

        "min_clearance_m":
            validation[
                "min_clearance_m"
            ],

        "clearance_margin_m":
            validation[
                "margin_m"
            ],

        "max_clearance_violation_mm":
            validation[
                "max_clearance_violation_mm"
            ],

        "violation_sample_count":
            validation[
                "violation_sample_count"
            ],

        "violation_path_length_m":
            violation_path_length_m,

        "effective_radius_m":
            effective_radius_m,

        "open_cruise_mps":
            operating[
                "open_cruise_mps"
            ],

        "caution_cap_mps":
            operating[
                "caution_cap_mps"
            ],

        "caution_distance_m":
            caution_distance_m,

        "caution_ratio":
            caution_ratio,

        "nominal_mission_time_s":
            float(
                nominal_time_s[-1]
            ),

        "safety_mission_time_s":
            float(
                safety_time_s[-1]
            ),

        "safety_time_penalty_s":
            float(
                safety_time_s[-1]
                - nominal_time_s[-1]
            ),

        # Detailed arrays retained for plots/debugging.
        "raw_sampled_clearance_m":
            raw_validation[
                "sampled_clearance_m"
            ],

        "raw_violation_mask":
            raw_validation[
                "violation_mask"
            ],

        "sampled_clearance_m":
            validation[
                "sampled_clearance_m"
            ],

        "violation_mask":
            validation[
                "violation_mask"
            ],

        "s":
            s,

        "curvature":
            curvature,

        "temp_center_distance_m":
            temp_center_distance_m,

        "temp_available_clearance_m":
            operating[
                "available_clearance_m"
            ],

        "curvature_limit_mps":
            curvature_limit_mps,

        "operating_zone_limit_mps":
            operating[
                "zone_limit_mps"
            ],

        "nominal_velocity_mps":
            nominal_velocity_mps,

        "safety_velocity_mps":
            safety_velocity_mps,

        "nominal_time_s":
            nominal_time_s,

        "safety_time_s":
            safety_time_s,

        "nominal_dt_s":
            nominal_dt_s,

        "safety_dt_s":
            safety_dt_s,
    }