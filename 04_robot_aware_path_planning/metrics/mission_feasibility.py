def evaluate_mission_feasibility(
    mission,
    robot_grid,
    labels,
):
    """
    Determine whether a common mission is feasible
    in a robot-specific configuration space.

    Categories:
        valid_reachable
        start_invalid
        goal_invalid
        both_invalid
        disconnected
    """

    sx, sy = mission.start
    gx, gy = mission.goal

    start_valid = (
        robot_grid[sy, sx] == 0
    )

    goal_valid = (
        robot_grid[gy, gx] == 0
    )

    if not start_valid and not goal_valid:
        status = "both_invalid"

    elif not start_valid:
        status = "start_invalid"

    elif not goal_valid:
        status = "goal_invalid"

    else:
        start_label = labels[sy, sx]
        goal_label = labels[gy, gx]

        if (
            start_label != 0
            and start_label == goal_label
        ):
            status = "valid_reachable"

        else:
            status = "disconnected"

    return {
        "mission_id": mission.mission_id,

        "start_x": sx,
        "start_y": sy,

        "goal_x": gx,
        "goal_y": gy,

        "start_valid": start_valid,
        "goal_valid": goal_valid,

        "status": status,

        "feasible": (
            status == "valid_reachable"
        ),
    }