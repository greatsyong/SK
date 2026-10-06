import numpy as np

from dataclasses import dataclass

from models.kinematics import TwoLinkKinematics


@dataclass(frozen=True)
class CartesianTrajectoryParams:
    """
    Cartesian trajectory parameters.

    The desired end-effector trajectory is

        x_d(t) = xc + Ax cos(omega t)
        y_d(t) = yc + Ay sin(omega t)

    All curriculum stages share the same center.
    Difficulty increases through trajectory geometry,
    amplitude, and speed.
    """

    xc: float
    yc: float

    Ax: float
    Ay: float

    omega: float


# ==========================================================
# Curriculum stages
# ==========================================================

TRAJECTORY_STAGES = {
    1: CartesianTrajectoryParams(
        xc=0.45,
        yc=0.15,
        Ax=0.03,
        Ay=0.03,
        omega=2.0 * np.pi / 20.0,
    ),

    2: CartesianTrajectoryParams(
        xc=0.45,
        yc=0.15,
        Ax=0.03,
        Ay=0.05,
        omega=2.0 * np.pi / 15.0,
    ),

    3: CartesianTrajectoryParams(
        xc=0.45,
        yc=0.15,
        Ax=0.10,
        Ay=0.06,
        omega=2.0 * np.pi / 10.0,
    ),
}


def get_stage_parameters(stage):
    """
    Return trajectory parameters for a curriculum stage.
    """

    if stage not in TRAJECTORY_STAGES:
        raise ValueError(
            f"Unknown trajectory stage: {stage}. "
            f"Valid stages are {tuple(TRAJECTORY_STAGES.keys())}."
        )

    return TRAJECTORY_STAGES[stage]


# ==========================================================
# Cartesian reference
# ==========================================================

def cartesian_reference(t, stage):
    """
    Compute desired Cartesian position, velocity,
    and acceleration.

    Parameters
    ----------
    t : float
        Time [s].

    stage : int
        Curriculum stage: 1, 2, or 3.

    Returns
    -------
    p_d : ndarray, shape (2,)
        Desired end-effector position [x, y] [m].

    pd_d : ndarray, shape (2,)
        Desired end-effector velocity [xdot, ydot] [m/s].

    pdd_d : ndarray, shape (2,)
        Desired end-effector acceleration [xddot, yddot] [m/s^2].
    """

    params = get_stage_parameters(stage)

    theta = (
        params.omega
        * t
    )

    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)

    # ------------------------------------------------------
    # Position
    # ------------------------------------------------------

    x_d = (
        params.xc
        + params.Ax * cos_theta
    )

    y_d = (
        params.yc
        + params.Ay * sin_theta
    )

    # ------------------------------------------------------
    # Velocity
    # ------------------------------------------------------

    xd_d = (
        -params.Ax
        * params.omega
        * sin_theta
    )

    yd_d = (
        params.Ay
        * params.omega
        * cos_theta
    )

    # ------------------------------------------------------
    # Acceleration
    # ------------------------------------------------------

    xdd_d = (
        -params.Ax
        * params.omega**2
        * cos_theta
    )

    ydd_d = (
        -params.Ay
        * params.omega**2
        * sin_theta
    )

    p_d = np.array(
        [x_d, y_d],
        dtype=np.float64,
    )

    pd_d = np.array(
        [xd_d, yd_d],
        dtype=np.float64,
    )

    pdd_d = np.array(
        [xdd_d, ydd_d],
        dtype=np.float64,
    )

    return (
        p_d,
        pd_d,
        pdd_d,
    )


# ==========================================================
# Jacobian time derivative
# ==========================================================

def jacobian_dot(
    q,
    qd,
    kinematics,
):
    """
    Compute J_dot(q, q_dot) for the 2-link planar arm.

    Cartesian acceleration satisfies

        p_ddot = J(q) q_ddot + J_dot(q, q_dot) q_dot
    """

    q = np.asarray(
        q,
        dtype=np.float64,
    )

    qd = np.asarray(
        qd,
        dtype=np.float64,
    )

    if q.shape != (2,):
        raise ValueError(
            "q must have shape (2,)"
        )

    if qd.shape != (2,):
        raise ValueError(
            "qd must have shape (2,)"
        )

    q1, q2 = q

    qd1, qd2 = qd

    l1 = kinematics.l1
    l2 = kinematics.l2

    s1 = np.sin(q1)
    c1 = np.cos(q1)

    s12 = np.sin(q1 + q2)
    c12 = np.cos(q1 + q2)

    qd12 = (
        qd1
        + qd2
    )

    J_dot = np.array(
        [
            [
                -l1 * c1 * qd1
                - l2 * c12 * qd12,

                -l2 * c12 * qd12,
            ],
            [
                -l1 * s1 * qd1
                - l2 * s12 * qd12,

                -l2 * s12 * qd12,
            ],
        ],
        dtype=np.float64,
    )

    return J_dot


# ==========================================================
# Cartesian -> joint reference
# ==========================================================

def joint_reference(
    t,
    stage,
    kinematics=None,
    elbow="up",
):
    """
    Convert the Cartesian end-effector reference into
    joint position, velocity, and acceleration references.

    The conversion uses:

        q_d = IK(p_d)

        qd_d = J(q_d)^(-1) p_dot_d

        qdd_d =
            J(q_d)^(-1)
            [p_ddot_d - J_dot(q_d, qd_d) qd_d]

    No explicit matrix inverse is formed. np.linalg.solve()
    is used instead.

    Parameters
    ----------
    t : float
        Time [s].

    stage : int
        Curriculum stage.

    kinematics : TwoLinkKinematics or None
        Kinematics model. A default instance is created
        if none is provided.

    elbow : {"up", "down"}
        Inverse-kinematics branch.

    Returns
    -------
    q_d : ndarray, shape (2,)
        Desired joint position [rad].

    qd_d : ndarray, shape (2,)
        Desired joint velocity [rad/s].

    qdd_d : ndarray, shape (2,)
        Desired joint acceleration [rad/s^2].
    """

    if kinematics is None:
        kinematics = TwoLinkKinematics()

    # ------------------------------------------------------
    # Desired Cartesian motion
    # ------------------------------------------------------

    p_d, pd_d, pdd_d = cartesian_reference(
        t=t,
        stage=stage,
    )

    # ------------------------------------------------------
    # Position reference through inverse kinematics
    # ------------------------------------------------------

    q_d = kinematics.inverse_kinematics(
        x=p_d[0],
        y=p_d[1],
        elbow=elbow,
    )

    # ------------------------------------------------------
    # Velocity reference
    #
    # p_dot = J q_dot
    # ------------------------------------------------------

    J = kinematics.jacobian(
        q_d
    )

    qd_d = np.linalg.solve(
        J,
        pd_d,
    )

    # ------------------------------------------------------
    # Acceleration reference
    #
    # p_ddot = J q_ddot + J_dot q_dot
    #
    # Therefore:
    #
    # q_ddot =
    #     J^-1 (p_ddot - J_dot q_dot)
    # ------------------------------------------------------

    J_dot = jacobian_dot(
        q=q_d,
        qd=qd_d,
        kinematics=kinematics,
    )

    qdd_d = np.linalg.solve(
        J,
        (
            pdd_d
            - J_dot @ qd_d
        ),
    )

    return (
        q_d,
        qd_d,
        qdd_d,
    )