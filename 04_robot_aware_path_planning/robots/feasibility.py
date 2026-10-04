import math

import numpy as np


EPS = 1e-12


def discrete_curvature(
    p0,
    p1,
    p2,
):
    """
    Three-point circumcircle curvature estimate.

    kappa = 1 / R

    Returns curvature in inverse path units.
    """

    a = math.dist(p0, p1)
    b = math.dist(p1, p2)
    c = math.dist(p0, p2)

    if (
        a < EPS
        or b < EPS
        or c < EPS
    ):
        return 0.0

    x1, y1 = p0
    x2, y2 = p1
    x3, y3 = p2

    twice_area = abs(
        (x2 - x1) * (y3 - y1)
        - (y2 - y1) * (x3 - x1)
    )

    if twice_area < EPS:
        return 0.0

    # Area = twice_area / 2
    #
    # Circumradius:
    # R = abc / (4A)
    #
    # Therefore:
    # kappa = 1/R = 4A/(abc)
    #       = 2*twice_area/(abc)

    curvature = (
        2.0
        * twice_area
        / (a * b * c)
    )

    return curvature


def path_curvatures(path):
    if len(path) < 3:
        return []

    values = []

    for i in range(
        1,
        len(path) - 1,
    ):
        values.append(
            discrete_curvature(
                path[i - 1],
                path[i],
                path[i + 1],
            )
        )

    return values


def curvature_metrics(path):
    curvatures = path_curvatures(
        path
    )

    if not curvatures:
        return {
            "max_curvature": 0.0,
            "rms_curvature": 0.0,
            "min_turning_radius": math.inf,
        }

    array = np.asarray(
        curvatures,
        dtype=float,
    )

    max_curvature = float(
        np.max(array)
    )

    rms_curvature = float(
        np.sqrt(
            np.mean(
                array ** 2
            )
        )
    )

    if max_curvature > EPS:
        min_radius = (
            1.0 / max_curvature
        )
    else:
        min_radius = math.inf

    return {
        "max_curvature":
            max_curvature,

        "rms_curvature":
            rms_curvature,

        "min_turning_radius":
            min_radius,
    }

def evaluate_differential_drive(
    path,
    robot,
    target_speed_mps=None,
):
    metrics = curvature_metrics(
        path
    )

    kappa_max = metrics[
        "max_curvature"
    ]

    if kappa_max > EPS:

        curvature_speed_limit = (
            robot.max_angular_velocity_radps
            / kappa_max
        )

    else:
        curvature_speed_limit = (
            math.inf
        )

    max_feasible_speed = min(
        robot.max_linear_velocity_mps,
        curvature_speed_limit,
    )

    if target_speed_mps is None:
        feasible_at_target_speed = None

    else:
        feasible_at_target_speed = (
            target_speed_mps
            <= max_feasible_speed
            + EPS
        )

    return {
        **metrics,

        "robot_drive_type":
            "differential",

        "hardware_max_speed_mps":
            robot.max_linear_velocity_mps,

        "curvature_speed_limit_mps":
            curvature_speed_limit,

        "max_feasible_speed_mps":
            max_feasible_speed,

        "target_speed_mps":
            target_speed_mps,

        "feasible_at_target_speed":
            feasible_at_target_speed,
    }

def evaluate_mecanum(
    path,
    robot,
    target_speed_mps=None,
):
    metrics = curvature_metrics(
        path
    )

    max_feasible_speed = (
        robot.max_linear_velocity_mps
    )

    if target_speed_mps is None:
        feasible_at_target_speed = None

    else:
        feasible_at_target_speed = (
            target_speed_mps
            <= max_feasible_speed
            + EPS
        )

    return {
        **metrics,

        "robot_drive_type":
            "mecanum",

        "hardware_max_speed_mps":
            robot.max_linear_velocity_mps,

        "max_linear_x_mps":
            robot.max_linear_x_mps,

        "max_linear_y_mps":
            robot.max_linear_y_mps,

        "max_angular_velocity_radps":
            robot.max_angular_velocity_radps,

        # Curvature is still useful as a path-quality metric,
        # but it is NOT used here as a translational
        # feasibility constraint.
        "curvature_speed_limit_mps":
            None,

        "max_feasible_speed_mps":
            max_feasible_speed,

        "target_speed_mps":
            target_speed_mps,

        "feasible_at_target_speed":
            feasible_at_target_speed,
    }

def evaluate_robot_path(
    path,
    robot,
    target_speed_mps=None,
):
    if robot.drive_type == "differential":

        return evaluate_differential_drive(
            path,
            robot,
            target_speed_mps,
        )

    if robot.drive_type == "mecanum":

        return evaluate_mecanum(
            path,
            robot,
            target_speed_mps,
        )

    raise ValueError(
        f"Unsupported drive type: "
        f"{robot.drive_type}"
    )