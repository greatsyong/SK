import numpy as np


class EKF:
    """
    State:
        x = [px, py, yaw, v, w]^T
    """

    def __init__(self):
        self.x = np.zeros((5, 1))

        self.P = np.diag([
            0.5,   # px
            0.5,   # py
            0.2,   # yaw
            0.5,   # v
            0.3    # w
        ])

        self.Q = np.diag([
            0.02,
            0.02,
            0.01,
            0.10,
            0.05
        ])

    @staticmethod
    def normalize_angle(angle):
        return np.arctan2(np.sin(angle), np.cos(angle))

    def predict(self, dt):
        if dt <= 0.0:
            return

        px, py, yaw, v, w = self.x.flatten()

        px_new = px + v * np.cos(yaw) * dt
        py_new = py + v * np.sin(yaw) * dt
        yaw_new = self.normalize_angle(yaw + w * dt)

        self.x = np.array([
            [px_new],
            [py_new],
            [yaw_new],
            [v],
            [w]
        ])

        F = np.eye(5)

        F[0, 2] = -v * np.sin(yaw) * dt
        F[0, 3] = np.cos(yaw) * dt

        F[1, 2] = v * np.cos(yaw) * dt
        F[1, 3] = np.sin(yaw) * dt

        F[2, 4] = dt

        self.P = F @ self.P @ F.T + self.Q * dt

    def update_velocity(self, v_measured, variance=0.05):
        z = np.array([[v_measured]])

        H = np.zeros((1, 5))
        H[0, 3] = 1.0

        R = np.array([[variance]])

        self._update(z, H, R)

    def update_yaw_rate(self, w_measured, variance=0.02):
        z = np.array([[w_measured]])

        H = np.zeros((1, 5))
        H[0, 4] = 1.0

        R = np.array([[variance]])

        self._update(z, H, R)

    def _update(self, z, H, R):
        innovation = z - H @ self.x

        S = H @ self.P @ H.T + R

        K = self.P @ H.T @ np.linalg.inv(S)

        self.x = self.x + K @ innovation

        I = np.eye(self.x.shape[0])

        self.P = (
            (I - K @ H)
            @ self.P
            @ (I - K @ H).T
            + K @ R @ K.T
        )

        self.x[2, 0] = self.normalize_angle(self.x[2, 0])