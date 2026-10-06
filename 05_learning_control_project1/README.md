# Model-Based and Learning-Based Control of a 2-DOF Manipulator

A controlled robotics study comparing classical model-based control, model-free reinforcement learning, and residual reinforcement learning for Cartesian end-effector trajectory tracking.

The central question is:

> When an analytical robot model is imperfect, should learning replace the controller, or should it learn only the dynamics that the model is missing?

---

## Overview

This project compares four control conditions on the same nonlinear 2-DOF manipulator:

1. **Computed Torque Control (CTC) with an exact model**
2. **CTC with structured model mismatch**
3. **Pure Soft Actor-Critic (SAC)**
4. **CTC + Residual SAC**

The final task is defined in Cartesian task space:

$$
\mathbf{x}_{EE}(t) \rightarrow \mathbf{x}_d(t)
$$

where the desired end-effector position is

$$
\mathbf{x}
=
\begin{bmatrix}
x & y
\end{bmatrix}^{T}.
$$

Joint-space references generated through inverse kinematics are used only as internal quantities where required by model-based control.

The project is not intended to show that reinforcement learning is inherently superior to classical control. Instead, it examines where learning provides value when a physically meaningful model error is introduced.

---

## Research Logic

The study is structured as a sequence of engineering questions.

| Step | Research Question | Purpose |
|---|---|---|
| Exact-model CTC | What performance is achievable when the model is correct? | Establish an ideal model-based reference |
| Model mismatch | What fails when the model is wrong? | Quantify sensitivity to structured uncertainty |
| Torque-deficit analysis | Why does tracking degrade? | Identify the actuator-level physical cause |
| Pure SAC | Can the task be learned without relying on an analytical model? | Establish a model-free baseline |
| Residual SAC | Can known physics be retained while learning only the missing dynamics? | Combine analytical control and learning |
| Residual-deficit comparison | What did SAC actually learn? | Test whether the learned correction matches the analytical model deficit |
| Training refinement | Has the policy already converged? | Determine an empirical stopping point |

The overall logic is:

$$
\text{correct model}
\rightarrow
\text{model mismatch}
\rightarrow
\text{torque deficit}
\rightarrow
\text{model-free learning}
\rightarrow
\text{residual learning}
\rightarrow
\text{physical interpretation}.
$$

---

## Why a 2-DOF Manipulator?

The 2-DOF planar manipulator is selected deliberately.

It is the minimum manipulator structure that preserves the main phenomena needed for this study:

- configuration-dependent inertia
- nonlinear coupled dynamics
- Coriolis and centrifugal effects
- gravity loading
- multi-joint coordination
- torque control
- forward and inverse kinematics
- joint-space and task-space coupling

A 1-DOF system removes most dynamic coupling effects.

A higher-DOF manipulator introduces additional geometric and implementation complexity that can obscure the physical cause of tracking error and learned correction.

The 2-DOF system therefore provides:

$$
\boxed{
\text{minimum complexity with meaningful nonlinear manipulator physics}
}
$$

while remaining analytically interpretable.

---

## Manipulator Dynamics

The robot dynamics are modeled as

$$
M(q)\ddot q
+
C(q,\dot q)\dot q
+
g(q)
+
B\dot q
=
\tau
+
\tau_{\mathrm{ext}}.
$$

The simulation uses numerical RK4 integration.

### Simulation Timing

- Control frequency: **50 Hz**
- Control timestep: **0.02 s**
- Physics frequency: **500 Hz**
- Physics timestep: **0.002 s**
- Physics substeps per control action: **10**

The controller action is therefore held constant across ten physics integration steps.

---

## Computed Torque Control

The model-based controller uses nonlinear dynamics compensation with feedback:

$$
\tau
=
M(q)
\left[
\ddot q_d
+
K_d(\dot q_d-\dot q)
+
K_p(q_d-q)
\right]
+
C(q,\dot q)\dot q
+
g(q)
+
B\dot q.
$$

Controller gains:

$$
K_p=[100,\;100]
$$

$$
K_d=[20,\;20].
$$

The exact-model CTC experiment establishes the ideal performance reference for the study.

Under this condition, the controller model and physical plant share the same dynamics.

---

## Structured Model Mismatch

To evaluate sensitivity to modeling error, only the second-link mass is changed.

True plant:

$$
m_2=1.50\ \mathrm{kg}
$$

Controller model:

$$
\hat m_2=1.05\ \mathrm{kg}
$$

corresponding to a **30% underestimate**.

Only one parameter is perturbed intentionally.

If several inertial parameters were randomized simultaneously, degradation could be observed but the physical source of the error would be harder to isolate.

This experiment preserves the causal chain

$$
m_2\ \text{error}
\rightarrow
\text{incorrect dynamics compensation}
\rightarrow
\text{torque deficit}
\rightarrow
\text{Cartesian tracking error}.
$$

---

## Analytical Torque Deficit

The model-induced torque deficit is defined as

$$
\Delta\tau_{\mathrm{model}}
=
\tau_{\mathrm{true\ model}}
-
\tau_{\mathrm{wrong\ model}}
$$

evaluated at the same robot state and reference.

Measured deficit:

| Joint | Mean [Nm] | RMS [Nm] | Peak [Nm] |
|---|---:|---:|---:|
| Joint 1 | 2.302 | 2.318 | 2.782 |
| Joint 2 | 0.263 | 0.338 | 0.546 |

This analysis is used for two purposes:

1. explain the physical origin of the tracking degradation,
2. provide a physically motivated scale for the Residual SAC action authority.

The mismatch study therefore goes beyond simply measuring trajectory error. It identifies the actuator-level quantity that is missing from the incorrect model-based controller.

---

## Cartesian Curriculum

Pure SAC is trained through three Cartesian trajectory stages.

The stages progressively increase task difficulty through trajectory geometry and dynamic demand.

The curriculum is used because the final task simultaneously requires the policy to learn:

- Cartesian geometry
- trajectory phase
- multi-joint coordination
- nonlinear dynamics
- velocity and acceleration demand

The curriculum therefore controls task complexity rather than changing the fundamental control objective.

### Training Logic

The training scripts intentionally distinguish four operations:

- **train**  
  Fresh policy initialization.

- **refine**  
  Same task semantics, but modified optimization settings.

- **transfer**  
  Learned network weights are transferred to a new trajectory stage, while the replay buffer is reset because the task distribution changes.

- **continue**  
  Same task and reward semantics, so both the learned weights and replay buffer are reused.

This distinction is important because replay data are retained only when the meaning of the experience remains unchanged.

---

## Pure SAC

Pure SAC directly generates actuator torque.

### Observation

The policy observation is

$$
[
q_1,\;
q_2,\;
\dot q_1,\;
\dot q_2,\;
x_d,\;
y_d,\;
\dot x_d,\;
\dot y_d
].
$$

The policy therefore receives robot state and Cartesian reference information without direct access to analytical inverse-dynamics terms.

### Action

SAC outputs normalized actions

$$
a\in[-1,1]^2.
$$

These are mapped to the full actuator torque range:

$$
\tau_{\max}
=
[20,\;8]\ \mathrm{Nm}.
$$

Pure SAC therefore learns the complete actuator command.

Its role in the study is to answer:

> Can the controller avoid explicit model dependence by learning the complete control law directly from interaction?

---

## Residual SAC

Residual SAC retains the imperfect-model CTC and learns only a bounded corrective torque.

The control law is

$$
\tau_{\mathrm{total}}
=
\tau_{\mathrm{CTC,wrong}}
+
\Delta\tau_{\mathrm{SAC}}.
$$

The residual action limits are

$$
\Delta\tau_{\max}
=
[5,\;2]\ \mathrm{Nm}.
$$

These limits exceed the measured analytical torque deficit while remaining substantially smaller than the full actuator range.

The learned policy therefore has sufficient authority to compensate for the model error without being allowed to replace the baseline controller entirely.

The architecture represents the design principle

$$
\boxed{
\text{known physics}
+
\text{learned correction}
}
$$

rather than learning the entire control problem from scratch.

---

## Reward

Tracking performance is rewarded using a bounded Cartesian tracking term:

$$
r_{\mathrm{track}}
=
\frac{1}
{
1+
\left(
\frac{\|e_{EE}\|}
{\sigma_p}
\right)^2
}.
$$

A small normalized torque penalty is added:

$$
r
=
r_{\mathrm{track}}
-
\lambda_\tau J_\tau.
$$

The main objective remains Cartesian tracking.

The torque term discourages unnecessarily aggressive actuation without dominating the task objective.

---

## Evaluation Metrics

No single metric is sufficient to describe trajectory-tracking quality.

The project therefore evaluates performance at four levels.

### 1. Tracking Accuracy

- End-effector RMSE
- End-effector maximum error
- Joint-space RMSE

These measure how accurately the controller follows the desired motion over time.

### 2. Geometric Fidelity

For the periodic closed trajectory:

- area bias
- symmetric-difference area
- geometric area error
- Intersection-over-Union (IoU)

These metrics capture distortions in task-space geometry that may not be obvious from time-domain RMSE alone.

### 3. Control Effort

- torque RMS
- peak torque
- saturation fraction

These verify whether improved tracking is achieved with physically reasonable actuator effort.

### 4. Physical Interpretation

For Residual SAC:

- residual-deficit RMSE
- residual-deficit correlation

These compare the learned residual torque directly with the analytically identified model torque deficit.

The combined evaluation therefore addresses:

$$
\boxed{
\text{accuracy}
+
\text{geometry}
+
\text{control effort}
+
\text{physical interpretation}
}
$$

---

## Final Stage-3 Results

| Controller | EE RMSE | Max Error | Geometric Error | IoU |
|---|---:|---:|---:|---:|
| Exact-model CTC | **0.195 mm** | **0.278 mm** | — | — |
| CTC with 30% $m_2$ error | 31.116 mm | 39.161 mm | 62.230% | 53.919% |
| Pure SAC | 4.662 mm | 9.930 mm | 6.230% | 93.994% |
| Residual SAC | **1.290 mm** | **2.226 mm** | **2.353%** | **97.658%** |

No evaluated controller reached actuator saturation.

---

## Residual Learning Interpretation

The final Residual SAC controller improves task-space tracking substantially compared with the mismatched CTC.

More importantly, the learned correction also follows the analytical torque deficit.

Residual-deficit RMSE:

$$
[0.0978,\;0.0140]\ \mathrm{Nm}
$$

Residual-deficit correlation:

$$
[0.880,\;0.996].
$$

This supports a stronger interpretation than trajectory tracking alone.

The learned policy is not merely producing a correction that happens to improve the Cartesian path. Its residual torque is strongly aligned with the missing torque predicted by the analytical dynamics model.

---

## Training Refinement and Model Selection

Residual SAC training was evaluated at multiple stages:

- initial training: **50k steps**
- continued training: **100k total**
- low-learning-rate refinement: **additional 30k**

The learning rate was reduced from

$$
3\times10^{-4}
$$

to

$$
1\times10^{-4}
$$

during the refinement experiment.

The additional refinement did not improve performance.

The 100k continued policy was therefore retained as the final Residual SAC controller.

This provides an empirical stopping criterion rather than assuming that additional training must always improve the policy.

---

## Practical Limitations and Real-World Considerations

The experiments are intentionally conducted in a controlled simulation environment.

This allows the effect of a specific model mismatch to be isolated and interpreted physically.

However, the reported numerical accuracy should not be interpreted as a direct prediction of physical robot performance.

A real manipulator introduces additional uncertainties.

### Multiple Model Errors

Real systems may contain simultaneous errors in:

- link mass
- inertia
- center of mass
- friction
- transmission characteristics
- payload properties
- structural compliance

The current single-parameter mismatch should therefore be interpreted as a controlled identification experiment rather than a complete representation of real-world uncertainty.

### Sensor and State-Estimation Error

The simulation assumes accurate access to

$$
q,\qquad \dot q.
$$

Real systems introduce:

- encoder quantization
- measurement noise
- numerical differentiation error
- estimator error
- timing jitter

These effects can influence both model-based and learned controllers.

### Actuator Dynamics

The current simulation assumes that commanded joint torque is applied directly.

In hardware,

$$
\tau_{\mathrm{command}}
\neq
\tau_{\mathrm{actual}}
$$

in general because of:

- actuator bandwidth
- current-control dynamics
- gearbox effects
- dead zones
- transmission losses
- torque-estimation uncertainty

### Computation and Communication Delay

The current system operates with deterministic timing.

A physical robot introduces sensing, computation, communication, and actuation delay.

These effects become increasingly important as trajectory bandwidth increases.

### External Disturbance and Contact

The current task is free-space tracking.

Real manipulators may experience

$$
\tau_{\mathrm{ext}}
$$

from:

- contact
- payload motion
- cable interaction
- human interaction
- external disturbance

Compensating model error and rejecting external disturbances are related but distinct control problems.

### Trajectory Generalization

The final policies are evaluated on the Stage-3 task distribution.

Strong performance on this trajectory does not by itself prove generalization to arbitrary:

- trajectory shapes
- trajectory centers
- amplitudes
- frequencies
- velocity profiles

Additional transfer experiments are required before claiming general model-error compensation across arbitrary tasks.

### Simulation-to-Real Gap

The near-perfect nominal CTC result is an ideal model-consistency benchmark.

It should not be interpreted as an expected hardware accuracy level.

The main result of the current study is therefore the controlled relationship among:

$$
\text{model mismatch},
\quad
\text{torque deficit},
\quad
\text{tracking degradation},
\quad
\text{learned compensation}.
$$

---

## Repository Structure

```text
05_learning_control_project1/
├── controllers/
│   └── computed_torque.py
│
├── envs/
│   ├── two_link_cartesian_env.py
│   └── two_link_residual_cartesian_env.py
│
├── evaluation/
│   ├── analyze_model_torque_deficit.py
│   ├── cartesian_trajectory.py
│   ├── ctc_cartesian_baseline.py
│   ├── ctc_model_mismatch.py
│   ├── evaluate_pure_sac.py
│   ├── evaluate_residual_sac.py
│   └── plot_sac_trajectory.py
│
├── models/
│   ├── __init__.py
│   ├── integrator.py
│   ├── kinematics.py
│   ├── parameters.py
│   └── two_link_dynamics.py
│
├── training/
│   ├── train_pure_sac_stage1.py
│   ├── refine_pure_sac_stage1.py
│   ├── transfer_pure_sac_stage2.py
│   ├── continue_pure_sac_stage2.py
│   ├── transfer_pure_sac_stage3.py
│   ├── continue_pure_sac_stage3.py
│   ├── train_residual_sac_stage3.py
│   ├── continue_residual_sac_stage3.py
│   └── refine_residual_sac_stage3.py
│
├── tests/
│   ├── test_cartesian_trajectory.py
│   ├── test_dynamics.py
│   ├── test_integration.py
│   └── test_kinematics.py
│
├── results/
├── requirements.txt
├── .gitignore
└── README.md
