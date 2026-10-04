from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Scenario:
    bucket: int
    map_name: str

    width: int
    height: int

    start: tuple[int, int]
    goal: tuple[int, int]

    optimal_length: float


def load_movingai_scenarios(path):
    path = Path(path)

    scenarios = []

    with path.open("r") as f:
        version = f.readline().strip()

        if not version.startswith("version"):
            raise ValueError("Invalid Moving AI scenario file")

        for line_number, line in enumerate(f, start=2):
            line = line.strip()

            if not line:
                continue

            fields = line.split()

            if len(fields) != 9:
                raise ValueError(
                    f"Line {line_number}: expected 9 fields, "
                    f"got {len(fields)}"
                )

            (
                bucket,
                map_name,
                width,
                height,
                start_x,
                start_y,
                goal_x,
                goal_y,
                optimal_length,
            ) = fields

            scenarios.append(
                Scenario(
                    bucket=int(bucket),
                    map_name=map_name,
                    width=int(width),
                    height=int(height),
                    start=(int(start_x), int(start_y)),
                    goal=(int(goal_x), int(goal_y)),
                    optimal_length=float(optimal_length),
                )
            )

    return scenarios