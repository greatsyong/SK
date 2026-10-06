from dataclasses import dataclass


@dataclass(frozen=True)
class TwoLinkParams:
    # Geometry [m]
    l1: float = 0.40
    l2: float = 0.30

    # Mass [kg]
    m1: float = 2.00
    m2: float = 1.50

    # Center of mass location from proximal joint [m]
    lc1: float = 0.20
    lc2: float = 0.15

    # Link inertia about COM [kg m^2]
    I1: float = 0.02666666666666667
    I2: float = 0.01125

    # Viscous friction [N m s/rad]
    b1: float = 0.15
    b2: float = 0.10

    # Gravity [m/s^2]
    g: float = 9.81

    # Provisional actuator limits [N m]
    tau1_max: float = 20.0
    tau2_max: float = 8.0