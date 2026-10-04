from dataclasses import dataclass
from pathlib import Path
import math

import yaml


@dataclass(frozen=True)
class RobotModel:
    name: str
    manufacturer: str
    drive_type: str

    length_m: float
    width_m: float
    height_m: float

    max_linear_velocity_mps: float

    max_angular_velocity_radps: float | None

    max_linear_x_mps: float | None
    max_linear_y_mps: float | None

    footprint_type: str

    @property
    def circumscribed_radius_m(self):
        """
        Conservative radius enclosing the rectangular footprint.

        Suitable for orientation-independent 2-D inflation.
        """
        return 0.5 * math.hypot(
            self.length_m,
            self.width_m,
        )

    @property
    def inscribed_radius_m(self):
        """
        Largest circle that fits completely inside the rectangle.
        """
        return 0.5 * min(
            self.length_m,
            self.width_m,
        )


def load_robot_models(
    path="config/robot_models.yaml",
):
    path = Path(path)

    with path.open("r") as f:
        data = yaml.safe_load(f)

    models = {}

    for name, config in data.items():

        dimensions = config["dimensions_m"]
        limits = config["limits"]
        footprint = config["footprint"]

        drive_type = config["drive_type"]

        if drive_type == "differential":

            model = RobotModel(
                name=name,
                manufacturer=config["manufacturer"],
                drive_type=drive_type,

                length_m=float(
                    dimensions["length"]
                ),
                width_m=float(
                    dimensions["width"]
                ),
                height_m=float(
                    dimensions["height"]
                ),

                max_linear_velocity_mps=float(
                    limits[
                        "max_linear_velocity_mps"
                    ]
                ),

                max_angular_velocity_radps=float(
                    limits[
                        "max_angular_velocity_radps"
                    ]
                ),

                max_linear_x_mps=None,
                max_linear_y_mps=None,

                footprint_type=footprint["type"],
            )

        elif drive_type == "mecanum":

            controller = limits["controller"]

            model = RobotModel(
                name=name,
                manufacturer=config["manufacturer"],
                drive_type=drive_type,

                length_m=float(
                    dimensions["length"]
                ),
                width_m=float(
                    dimensions["width"]
                ),
                height_m=float(
                    dimensions["height"]
                ),

                # Physical platform specification,
                # not merely controller configuration.
                max_linear_velocity_mps=float(
                    limits[
                        "platform_max_speed_mps"
                    ]
                ),

                max_angular_velocity_radps=float(
                    controller[
                        "max_angular_z_radps"
                    ]
                ),

                max_linear_x_mps=float(
                    controller[
                        "max_linear_x_mps"
                    ]
                ),

                max_linear_y_mps=float(
                    controller[
                        "max_linear_y_mps"
                    ]
                ),

                footprint_type=footprint["type"],
            )

        else:
            raise ValueError(
                f"Unsupported drive type: "
                f"{drive_type}"
            )

        models[name] = model

    return models