from dataclasses import replace

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from models.parameters import TwoLinkParams
from models.two_link_dynamics import TwoLinkDynamics
from models.integrator import rk4_step
from models.kinematics import TwoLinkKinematics

from controllers.computed_torque import ComputedTorqueController

from evaluation.cartesian_trajectory import (
    get_stage_parameters,
    cartesian_reference,
    joint_reference,
)


class TwoLinkResidualCartesianEnv(gym.Env):
    """
    Residual SAC environment for Cartesian end-effector
    trajectory tracking.

    Control architecture
    --------------------

        tau_total
            =
        tau_CTC_wrong
            +
        delta_tau_SAC

    The physical plant uses the true model.

    The baseline CTC deliberately underestimates link-2 mass,
    producing a structured model mismatch.

    SAC does NOT replace the baseline controller. It receives
    bounded residual torque authority intended to compensate
    the model-induced torque deficit.

    Observation
    -----------

        [
            q1, q2,
            qd1, qd2,
            x_ref, y_ref,
            xd_ref, yd_ref
        ]

    Action
    ------

        a in [-1, 1]^2

    mapped to:

        delta_tau =
            a * residual_limits

    Default residual authority:

        joint 1: +/- 5 Nm
        joint 2: +/- 2 Nm

    These limits were selected after analytical model-deficit
    analysis showed approximately:

        peak deficit joint 1 = 2.78 Nm
        peak deficit joint 2 = 0.55 Nm

    The limits therefore provide reasonable correction margin
    without allowing SAC unrestricted actuator authority.
    """

    metadata = {}

    def __init__(
        self,
        stage=3,
        control_dt=0.02,
        physics_dt=0.002,
        cycles_per_episode=3,
        position_scale=0.03,
        torque_penalty_weight=0.001,
        model_mass_scale=0.70,
        residual_limits=None,
    ):
        super().__init__()

        # ==================================================
        # Experiment configuration
        # ==================================================

        self.stage = int(stage)

        self.control_dt = float(
            control_dt
        )

        self.physics_dt = float(
            physics_dt
        )

        self.cycles_per_episode = int(
            cycles_per_episode
        )

        self.position_scale = float(
            position_scale
        )

        self.torque_penalty_weight = float(
            torque_penalty_weight
        )

        self.model_mass_scale = float(
            model_mass_scale
        )

        # ==================================================
        # Timing
        # ==================================================

        ratio = (
            self.control_dt
            / self.physics_dt
        )

        if not np.isclose(
            ratio,
            round(ratio),
        ):
            raise ValueError(
                "control_dt must be an integer multiple "
                "of physics_dt"
            )

        self.physics_steps_per_control = int(
            round(ratio)
        )

        traj_params = get_stage_parameters(
            self.stage
        )

        self.period = (
            2.0
            * np.pi
            / traj_params.omega
        )

        self.episode_duration = (
            self.cycles_per_episode
            * self.period
        )

        self.max_episode_steps = int(
            round(
                self.episode_duration
                / self.control_dt
            )
        )

        # ==================================================
        # True physical plant
        # ==================================================

        self.true_params = TwoLinkParams()

        self.plant = TwoLinkDynamics(
            self.true_params
        )

        self.kinematics = TwoLinkKinematics(
            self.true_params
        )

        self.torque_limits = np.array(
            [
                self.true_params.tau1_max,
                self.true_params.tau2_max,
            ],
            dtype=np.float64,
        )

        # ==================================================
        # Imperfect controller model
        # ==================================================
        #
        # Only m2 is deliberately wrong.
        #
        # True:
        #
        #   m2 = 1.50 kg
        #
        # CTC model:
        #
        #   m2 = 1.05 kg
        #
        # for the default 0.70 scale.
        # ==================================================

        self.controller_params = replace(
            self.true_params,
            m2=(
                self.true_params.m2
                * self.model_mass_scale
            ),
        )

        self.controller_model = TwoLinkDynamics(
            self.controller_params
        )

        self.ctc = ComputedTorqueController(
            model=self.controller_model,
            kp=np.array(
                [100.0, 100.0],
                dtype=np.float64,
            ),
            kd=np.array(
                [20.0, 20.0],
                dtype=np.float64,
            ),
            torque_limits=None,
        )

        # ==================================================
        # Residual torque authority
        # ==================================================

        if residual_limits is None:
            residual_limits = np.array(
                [5.0, 2.0],
                dtype=np.float64,
            )

        self.residual_limits = np.asarray(
            residual_limits,
            dtype=np.float64,
        )

        if self.residual_limits.shape != (2,):
            raise ValueError(
                "residual_limits must have shape (2,)"
            )

        # ==================================================
        # Action space
        # ==================================================

        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(2,),
            dtype=np.float32,
        )

        # ==================================================
        # Observation space
        # ==================================================
        #
        # Same 8-D observation used by Pure SAC:
        #
        # [
        #   q1, q2,
        #   qd1, qd2,
        #   x_ref, y_ref,
        #   xd_ref, yd_ref
        # ]
        #
        # The baseline CTC internally uses IK-derived
        # q_ref, qd_ref and qdd_ref.
        #
        # SAC itself is still presented with the Cartesian
        # task formulation.
        # ==================================================

        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(8,),
            dtype=np.float32,
        )

        # ==================================================
        # State
        # ==================================================

        self.state = np.zeros(
            4,
            dtype=np.float64,
        )

        self.time = 0.0
        self.current_step = 0

    # ======================================================
    # Observation
    # ======================================================

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

    # ======================================================
    # Reset
    # ======================================================

    def reset(
        self,
        seed=None,
        options=None,
    ):
        super().reset(
            seed=seed
        )

        self.time = 0.0
        self.current_step = 0

        q_ref, qd_ref, _ = joint_reference(
            t=0.0,
            stage=self.stage,
            kinematics=self.kinematics,
            elbow="up",
        )

        # Same initial-condition philosophy as Pure SAC:
        # begin exactly on the desired trajectory.

        self.state = np.concatenate(
            [
                q_ref,
                qd_ref,
            ]
        ).astype(np.float64)

        observation = (
            self._get_observation()
        )

        info = {
            "tau_base": np.zeros(
                2,
                dtype=np.float64,
            ),
            "tau_residual": np.zeros(
                2,
                dtype=np.float64,
            ),
            "tau_total": np.zeros(
                2,
                dtype=np.float64,
            ),
        }

        return observation, info

    # ======================================================
    # Step
    # ======================================================

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

        q = self.state[:2]
        qd = self.state[2:]

        # ==================================================
        # Internal joint reference for CTC
        # ==================================================

        q_ref, qd_ref, qdd_ref = joint_reference(
            t=self.time,
            stage=self.stage,
            kinematics=self.kinematics,
            elbow="up",
        )

        # ==================================================
        # Imperfect model-based baseline
        # ==================================================

        tau_base = self.ctc.compute(
            q=q,
            qd=qd,
            q_ref=q_ref,
            qd_ref=qd_ref,
            qdd_ref=qdd_ref,
        )

        # ==================================================
        # SAC residual correction
        # ==================================================

        tau_residual = (
            action
            * self.residual_limits
        )

        # ==================================================
        # Hybrid control law
        # ==================================================

        tau_total_unclipped = (
            tau_base
            + tau_residual
        )

        tau_total = np.clip(
            tau_total_unclipped,
            -self.torque_limits,
            self.torque_limits,
        )

        # ==================================================
        # Physics integration
        #
        # Zero-order hold:
        # one 50-Hz control action is held across ten
        # 500-Hz RK4 integration steps.
        # ==================================================

        for _ in range(
            self.physics_steps_per_control
        ):
            self.state = rk4_step(
                self.plant.state_derivative,
                self.state,
                self.physics_dt,
                tau_total,
            )

        self.time += self.control_dt
        self.current_step += 1

        # ==================================================
        # Cartesian tracking error after transition
        # ==================================================

        q_next = self.state[:2]

        p_actual = (
            self.kinematics.forward_kinematics(
                q_next
            )
        )

        p_ref_next, _, _ = cartesian_reference(
            t=self.time,
            stage=self.stage,
        )

        ee_error = (
            p_ref_next
            - p_actual
        )

        ee_error_norm = np.linalg.norm(
            ee_error
        )

        # ==================================================
        # Reward
        # ==================================================
        #
        # Tracking reward is identical in form to Pure SAC:
        #
        #       1
        # ----------------
        # 1 + (e / scale)^2
        #
        # For effort cost we use TOTAL physical torque,
        # normalized by actuator limits.
        #
        # In Pure SAC:
        #
        #   action == tau / tau_limit
        #
        # so this is the physically equivalent definition.
        # ==================================================

        tracking_reward = (
            1.0
            /
            (
                1.0
                +
                (
                    ee_error_norm
                    / self.position_scale
                )**2
            )
        )

        normalized_total_torque = (
            tau_total
            / self.torque_limits
        )

        torque_cost = np.mean(
            normalized_total_torque**2
        )

        reward = (
            tracking_reward
            -
            self.torque_penalty_weight
            * torque_cost
        )

        # ==================================================
        # Safety
        # ==================================================

        q_next = self.state[:2]
        qd_next = self.state[2:]

        unsafe = (
            np.any(
                ~np.isfinite(
                    self.state
                )
            )
            or np.any(
                np.abs(q_next)
                > 2.0 * np.pi
            )
            or np.any(
                np.abs(qd_next)
                > 10.0
            )
        )

        terminated = bool(
            unsafe
        )

        if terminated:
            reward -= 10.0

        truncated = (
            self.current_step
            >= self.max_episode_steps
        )

        observation = (
            self._get_observation()
        )

        # ==================================================
        # Diagnostic info
        #
        # These are intentionally exposed so later evaluation
        # can compare:
        #
        # delta_tau_SAC
        #
        # against:
        #
        # delta_tau_model
        # ==================================================

        info = {
            "ee_error": ee_error.copy(),
            "ee_error_norm": float(
                ee_error_norm
            ),
            "tracking_reward": float(
                tracking_reward
            ),
            "torque_cost": float(
                torque_cost
            ),

            "tau_base": np.asarray(
                tau_base,
                dtype=np.float64,
            ).copy(),

            "tau_residual": np.asarray(
                tau_residual,
                dtype=np.float64,
            ).copy(),

            "tau_total_unclipped": np.asarray(
                tau_total_unclipped,
                dtype=np.float64,
            ).copy(),

            "tau_total": np.asarray(
                tau_total,
                dtype=np.float64,
            ).copy(),

            "residual_action": action.copy(),
        }

        return (
            observation,
            float(reward),
            terminated,
            truncated,
            info,
        )
