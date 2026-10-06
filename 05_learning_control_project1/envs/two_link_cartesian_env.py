import numpy as np
import gymnasium as gym
from gymnasium import spaces

from models.parameters import TwoLinkParams
from models.two_link_dynamics import TwoLinkDynamics
from models.integrator import rk4_step
from models.kinematics import TwoLinkKinematics
from evaluation.cartesian_trajectory import (
    get_stage_parameters,
    cartesian_reference,
    joint_reference,
)


class TwoLinkCartesianEnv(gym.Env):
    """
    2-DOF planar manipulator environment for direct torque RL.

    Task
    ----
    Track a Cartesian end-effector trajectory using normalized
    joint-torque actions.

    Observation
    -----------
    [q1, q2,
     qd1, qd2,
     x_ref, y_ref,
     xd_ref, yd_ref]

    Action
    ------
    Normalized joint torques in [-1, 1]^2.

    Timing
    ------
    Control rate : 50 Hz
    Physics rate : 500 Hz
    Physics substeps per control step : 10
    """

    metadata = {
        "render_modes": [],
    }

    def __init__(
        self,
        stage=1,
        control_dt=0.02,
        physics_dt=0.002,
        cycles_per_episode=3,
        position_scale=0.05,
        torque_penalty_weight=0.001,
    ):
        super().__init__()

        self.stage = int(stage)

        self.control_dt = float(control_dt)
        self.physics_dt = float(physics_dt)

        self.physics_substeps = int(
            round(
                self.control_dt
                / self.physics_dt
            )
        )

        if not np.isclose(
            self.physics_substeps
            * self.physics_dt,
            self.control_dt,
        ):
            raise ValueError(
                "control_dt must be an integer multiple of physics_dt"
            )

        self.params = TwoLinkParams()

        self.plant = TwoLinkDynamics(
            self.params
        )

        self.kin = TwoLinkKinematics(
            self.params
        )

        self.torque_limits = np.array(
            [
                self.params.tau1_max,
                self.params.tau2_max,
            ],
            dtype=np.float64,
        )

        # --------------------------------------------------
        # Episode duration
        # --------------------------------------------------

        traj_params = get_stage_parameters(
            self.stage
        )

        self.period = (
            2.0
            * np.pi
            / traj_params.omega
        )

        self.cycles_per_episode = int(
            cycles_per_episode
        )

        self.episode_duration = (
            self.cycles_per_episode
            * self.period
        )

        self.max_steps = int(
            round(
                self.episode_duration
                / self.control_dt
            )
        )

        # --------------------------------------------------
        # Reward parameters
        # --------------------------------------------------

        self.position_scale = float(
            position_scale
        )

        self.torque_penalty_weight = float(
            torque_penalty_weight
        )

        # Strong safety limits.
        # These are not normal operating limits;
        # they only terminate clearly unstable rollouts.
        self.max_abs_joint_angle = 2.0 * np.pi
        self.max_abs_joint_velocity = 10.0

        # --------------------------------------------------
        # Gym spaces
        # --------------------------------------------------

        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(2,),
            dtype=np.float32,
        )

        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(8,),
            dtype=np.float32,
        )

        # --------------------------------------------------
        # Runtime state
        # --------------------------------------------------

        self.state = None
        self.time = 0.0
        self.step_count = 0

    def _get_observation(self):
        q = self.state[:2]
        qd = self.state[2:]

        p_ref, pd_ref, _ = cartesian_reference(
            t=self.time,
            stage=self.stage,
        )

        obs = np.concatenate(
            [
                q,
                qd,
                p_ref,
                pd_ref,
            ]
        )

        return obs.astype(
            np.float32
        )

    def _is_unsafe(self):
        if not np.all(
            np.isfinite(self.state)
        ):
            return True

        q = self.state[:2]
        qd = self.state[2:]

        if np.any(
            np.abs(q)
            > self.max_abs_joint_angle
        ):
            return True

        if np.any(
            np.abs(qd)
            > self.max_abs_joint_velocity
        ):
            return True

        return False

    def reset(
        self,
        *,
        seed=None,
        options=None,
    ):
        super().reset(
            seed=seed
        )

        self.time = 0.0
        self.step_count = 0

        q0, qd0, _ = joint_reference(
            t=0.0,
            stage=self.stage,
            kinematics=self.kin,
            elbow="up",
        )

        self.state = np.concatenate(
            [
                q0,
                qd0,
            ]
        ).astype(
            np.float64
        )

        observation = self._get_observation()

        p_actual = self.kin.forward_kinematics(
            self.state[:2]
        )

        p_ref, _, _ = cartesian_reference(
            t=self.time,
            stage=self.stage,
        )

        info = {
            "time": self.time,
            "ee_error": (
                p_ref
                - p_actual
            ).copy(),
        }

        return observation, info

    def step(
        self,
        action,
    ):
        action = np.asarray(
            action,
            dtype=np.float64,
        )

        action = np.clip(
            action,
            -1.0,
            1.0,
        )

        # Normalized SAC action -> physical joint torque.
        tau = (
            action
            * self.torque_limits
        )

        # Zero-order hold:
        # keep torque constant for one 20 ms control interval
        # while integrating physics at 2 ms.
        for _ in range(
            self.physics_substeps
        ):
            self.state = rk4_step(
                self.plant.state_derivative,
                self.state,
                self.physics_dt,
                tau,
            )

        self.time += self.control_dt
        self.step_count += 1

        # --------------------------------------------------
        # Tracking error at the new control instant
        # --------------------------------------------------

        q = self.state[:2]

        p_actual = self.kin.forward_kinematics(
            q
        )

        p_ref, _, _ = cartesian_reference(
            t=self.time,
            stage=self.stage,
        )

        ee_error = (
            p_ref
            - p_actual
        )

        ee_error_norm = float(
            np.linalg.norm(
                ee_error
            )
        )

        # Smooth bounded Cartesian tracking reward.
        tracking_reward = (
            1.0
            /
            (
                1.0
                +
                (
                    ee_error_norm
                    / self.position_scale
                ) ** 2
            )
        )

        # Small normalized torque penalty.
        torque_cost = float(
            np.mean(
                action ** 2
            )
        )

        reward = (
            tracking_reward
            -
            self.torque_penalty_weight
            * torque_cost
        )

        # --------------------------------------------------
        # Termination
        # --------------------------------------------------

        terminated = self._is_unsafe()

        if terminated:
            reward -= 10.0

        truncated = (
            self.step_count
            >= self.max_steps
        )

        observation = self._get_observation()

        info = {
            "time": self.time,
            "ee_error": ee_error.copy(),
            "ee_error_norm": ee_error_norm,
            "tracking_reward": tracking_reward,
            "torque_cost": torque_cost,
            "tau": tau.copy(),
        }

        return (
            observation,
            float(reward),
            terminated,
            truncated,
            info,
        )
