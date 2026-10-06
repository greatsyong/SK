import numpy as np

from models.parameters import TwoLinkParams


class TwoLinkKinematics:
    """
    Forward/inverse kinematics for the 2-DOF planar manipulator.

    Joint convention:
        q1: angle of link 1 measured from the +x axis
        q2: relative angle of link 2 with respect to link 1

    End-effector position:
        x = l1 cos(q1) + l2 cos(q1 + q2)
        y = l1 sin(q1) + l2 sin(q1 + q2)
    """

    def __init__(self, params=None):
        if params is None:
            params = TwoLinkParams()

        self.l1 = params.l1
        self.l2 = params.l2

    # ======================================================
    # Forward kinematics
    # ======================================================

    def forward_kinematics(self, q):
        """
        Compute end-effector Cartesian position.

        Parameters
        ----------
        q : array_like, shape (2,)
            Joint angles [q1, q2] in radians.

        Returns
        -------
        position : ndarray, shape (2,)
            End-effector position [x, y] in meters.
        """

        q = np.asarray(q, dtype=np.float64)

        if q.shape != (2,):
            raise ValueError(
                "q must have shape (2,)"
            )

        q1, q2 = q

        x = (
            self.l1 * np.cos(q1)
            + self.l2 * np.cos(q1 + q2)
        )

        y = (
            self.l1 * np.sin(q1)
            + self.l2 * np.sin(q1 + q2)
        )

        return np.array(
            [x, y],
            dtype=np.float64,
        )

    # ======================================================
    # Inverse kinematics
    # ======================================================

    def inverse_kinematics(
        self,
        x,
        y,
        elbow="up",
    ):
        """
        Compute joint angles from end-effector position.

        Parameters
        ----------
        x : float
            End-effector x position [m].

        y : float
            End-effector y position [m].

        elbow : {"up", "down"}
            Select inverse-kinematics branch.

            "up":
                q2 > 0

            "down":
                q2 < 0

        Returns
        -------
        q : ndarray, shape (2,)
            Joint angles [q1, q2] in radians.

        Raises
        ------
        ValueError
            If the point is outside the reachable workspace.
        """

        if elbow not in ("up", "down"):
            raise ValueError(
                "elbow must be 'up' or 'down'"
            )

        r_squared = (
            x**2
            + y**2
        )

        cos_q2 = (
            r_squared
            - self.l1**2
            - self.l2**2
        ) / (
            2.0
            * self.l1
            * self.l2
        )

        # Numerical tolerance:
        #
        # A theoretically reachable point near the workspace
        # boundary can produce values such as
        # cos(q2) = 1.0000000000000002.
        tolerance = 1e-12

        if (
            cos_q2 > 1.0 + tolerance
            or cos_q2 < -1.0 - tolerance
        ):
            raise ValueError(
                f"Target ({x:.6f}, {y:.6f}) "
                "is outside the reachable workspace."
            )

        cos_q2 = np.clip(
            cos_q2,
            -1.0,
            1.0,
        )

        sin_q2_magnitude = np.sqrt(
            max(
                0.0,
                1.0 - cos_q2**2,
            )
        )

        if elbow == "up":
            sin_q2 = sin_q2_magnitude
        else:
            sin_q2 = -sin_q2_magnitude

        q2 = np.arctan2(
            sin_q2,
            cos_q2,
        )

        q1 = (
            np.arctan2(y, x)
            - np.arctan2(
                self.l2 * sin_q2,
                self.l1
                + self.l2 * cos_q2,
            )
        )

        return np.array(
            [q1, q2],
            dtype=np.float64,
        )

    # ======================================================
    # Jacobian
    # ======================================================

    def jacobian(self, q):
        """
        Compute the Cartesian velocity Jacobian.

        p_dot = J(q) q_dot

        Parameters
        ----------
        q : array_like, shape (2,)
            Joint angles [q1, q2] in radians.

        Returns
        -------
        J : ndarray, shape (2, 2)
            End-effector Jacobian.
        """

        q = np.asarray(q, dtype=np.float64)

        if q.shape != (2,):
            raise ValueError(
                "q must have shape (2,)"
            )

        q1, q2 = q

        s1 = np.sin(q1)
        c1 = np.cos(q1)

        s12 = np.sin(q1 + q2)
        c12 = np.cos(q1 + q2)

        J = np.array(
            [
                [
                    -self.l1 * s1
                    - self.l2 * s12,
                    -self.l2 * s12,
                ],
                [
                    self.l1 * c1
                    + self.l2 * c12,
                    self.l2 * c12,
                ],
            ],
            dtype=np.float64,
        )

        return J

    # ======================================================
    # Singularity measure
    # ======================================================

    def jacobian_determinant(self, q):
        """
        Return det(J).

        For the 2-link planar arm:

            det(J) = l1 * l2 * sin(q2)

        Therefore singular configurations occur when:

            q2 = 0, +/-pi, ...
        """

        J = self.jacobian(q)

        return float(
            np.linalg.det(J)
        )