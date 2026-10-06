import numpy as np

from models.two_link_dynamics import TwoLinkDynamics


class ComputedTorqueController:
    """
    Computed Torque Controller (CTC)

    tau =
        M_hat(q) * (
            qdd_d
            + Kd * (qd_d - qd)
            + Kp * (q_d - q)
        )
        + c_hat(q, qd)
        + g_hat(q)
        + friction_hat(qd)
    """

    def __init__(
        self,
        model: TwoLinkDynamics,
        kp: np.ndarray,
        kd: np.ndarray,
        torque_limits: np.ndarray | None = None,
    ):
        self.model = model

        self.kp = np.asarray(kp, dtype=float)
        self.kd = np.asarray(kd, dtype=float)

        self.torque_limits = (
            None
            if torque_limits is None
            else np.asarray(torque_limits, dtype=float)
        )

    def compute(
        self,
        q: np.ndarray,
        qd: np.ndarray,
        q_ref: np.ndarray,
        qd_ref: np.ndarray,
        qdd_ref: np.ndarray,
    ) -> np.ndarray:

        e = q_ref - q
        ed = qd_ref - qd

        v = (
            qdd_ref
            + self.kd * ed
            + self.kp * e
        )

        tau = (
            self.model.mass_matrix(q) @ v
            + self.model.coriolis_vector(q, qd)
            + self.model.gravity_vector(q)
            + self.model.friction_vector(qd)
        )

        if self.torque_limits is not None:
            tau = np.clip(
                tau,
                -self.torque_limits,
                self.torque_limits,
            )

        return tau