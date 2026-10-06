import numpy as np

from models.parameters import TwoLinkParams


class TwoLinkDynamics:
    """
    2-DOF planar RR manipulator in a vertical plane.

    Coordinates
    -----------
    q1 : angle of link 1 from +x horizontal axis
    q2 : relative angle of link 2 with respect to link 1

    Dynamics
    --------
    M(q) qdd + c(q, qd) + g(q) + B qd = tau + tau_ext
    """

    def __init__(self, params: TwoLinkParams):
        self.p = params

    def mass_matrix(self, q: np.ndarray) -> np.ndarray:
        q1, q2 = q
        p = self.p

        c2 = np.cos(q2)

        M11 = (
            p.I1
            + p.I2
            + p.m1 * p.lc1**2
            + p.m2 * (
                p.l1**2
                + p.lc2**2
                + 2.0 * p.l1 * p.lc2 * c2
            )
        )

        M12 = (
            p.I2
            + p.m2 * (
                p.lc2**2
                + p.l1 * p.lc2 * c2
            )
        )

        M22 = p.I2 + p.m2 * p.lc2**2

        return np.array([
            [M11, M12],
            [M12, M22],
        ])

    def coriolis_vector(
        self,
        q: np.ndarray,
        qd: np.ndarray,
    ) -> np.ndarray:
        _, q2 = q
        qd1, qd2 = qd
        p = self.p

        h = p.m2 * p.l1 * p.lc2 * np.sin(q2)

        c1 = -h * (2.0 * qd1 * qd2 + qd2**2)
        c2 =  h * qd1**2

        return np.array([c1, c2])

    def gravity_vector(self, q: np.ndarray) -> np.ndarray:
        q1, q2 = q
        p = self.p

        g1 = (
            (p.m1 * p.lc1 + p.m2 * p.l1)
            * p.g
            * np.cos(q1)
            + p.m2
            * p.lc2
            * p.g
            * np.cos(q1 + q2)
        )

        g2 = (
            p.m2
            * p.lc2
            * p.g
            * np.cos(q1 + q2)
        )

        return np.array([g1, g2])

    def friction_vector(self, qd: np.ndarray) -> np.ndarray:
        p = self.p

        B = np.diag([p.b1, p.b2])

        return B @ qd

    def inverse_dynamics(
        self,
        q: np.ndarray,
        qd: np.ndarray,
        qdd: np.ndarray,
    ) -> np.ndarray:
        return (
            self.mass_matrix(q) @ qdd
            + self.coriolis_vector(q, qd)
            + self.gravity_vector(q)
            + self.friction_vector(qd)
        )

    def forward_dynamics(
        self,
        q: np.ndarray,
        qd: np.ndarray,
        tau: np.ndarray,
        tau_ext: np.ndarray | None = None,
    ) -> np.ndarray:
        if tau_ext is None:
            tau_ext = np.zeros(2)

        rhs = (
            tau
            + tau_ext
            - self.coriolis_vector(q, qd)
            - self.gravity_vector(q)
            - self.friction_vector(qd)
        )

        return np.linalg.solve(
            self.mass_matrix(q),
            rhs
        )

    def state_derivative(
        self,
        state: np.ndarray,
        tau: np.ndarray,
        tau_ext: np.ndarray | None = None,
    ) -> np.ndarray:
        q = state[:2]
        qd = state[2:]

        qdd = self.forward_dynamics(
            q=q,
            qd=qd,
            tau=tau,
            tau_ext=tau_ext,
        )

        return np.concatenate([qd, qdd])

    def kinetic_energy(
        self,
        q: np.ndarray,
        qd: np.ndarray,
    ) -> float:
        """
        Kinetic energy:
            T = 1/2 qd^T M(q) qd
        """
        M = self.mass_matrix(q)

        return 0.5 * float(qd.T @ M @ qd)

    def potential_energy(
        self,
        q: np.ndarray,
    ) -> float:
        """
        Gravitational potential energy.

        Coordinate convention:
        q1 measured from +x horizontal axis.
        q2 relative to link 1.
        """
        q1, q2 = q
        p = self.p

        y1 = p.lc1 * np.sin(q1)

        y2 = (
            p.l1 * np.sin(q1)
            + p.lc2 * np.sin(q1 + q2)
        )

        return (
            p.m1 * p.g * y1
            + p.m2 * p.g * y2
        )

    def total_energy(
        self,
        q: np.ndarray,
        qd: np.ndarray,
    ) -> float:
        return (
            self.kinetic_energy(q, qd)
            + self.potential_energy(q)
        )