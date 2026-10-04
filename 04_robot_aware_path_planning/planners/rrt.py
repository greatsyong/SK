import math
import random
import time

from planners.geometry import (
    euclidean_distance,
    edge_is_free,
    path_length,
)


class RRTPlanner:

    def __init__(
        self,
        step_size=10.0,
        goal_radius=10.0,
        goal_sample_rate=0.05,
        max_iterations=20000,
        seed=None,
    ):

        self.name = "rrt"

        self.step_size = step_size
        self.goal_radius = goal_radius
        self.goal_sample_rate = goal_sample_rate
        self.max_iterations = max_iterations

        self.seed = seed

    # ========================================================
    # Sampling
    # ========================================================

    def _sample(
        self,
        grid,
        goal,
        rng,
    ):

        height, width = grid.shape

        # Goal-biased sampling
        if rng.random() < self.goal_sample_rate:
            return (
                float(goal[0]),
                float(goal[1]),
            )

        while True:

            x = rng.uniform(
                0.0,
                width,
            )

            y = rng.uniform(
                0.0,
                height,
            )

            ix = min(
                int(x),
                width - 1,
            )

            iy = min(
                int(y),
                height - 1,
            )

            if grid[iy, ix] == 0:
                return (x, y)

    # ========================================================
    # Nearest node
    # ========================================================

    def _nearest(
        self,
        nodes,
        sample,
    ):

        nearest_index = min(
            range(len(nodes)),
            key=lambda i:
                euclidean_distance(
                    nodes[i],
                    sample,
                ),
        )

        return nearest_index

    # ========================================================
    # Steering
    # ========================================================

    def _steer(
        self,
        source,
        target,
    ):

        distance = euclidean_distance(
            source,
            target,
        )

        if distance <= self.step_size:
            return target

        dx = target[0] - source[0]
        dy = target[1] - source[1]

        scale = (
            self.step_size
            / distance
        )

        return (
            source[0] + dx * scale,
            source[1] + dy * scale,
        )

    # ========================================================
    # Path reconstruction
    # ========================================================

    def _reconstruct_path(
        self,
        nodes,
        parents,
        goal_index,
    ):

        path = []

        index = goal_index

        while index is not None:

            path.append(
                nodes[index]
            )

            index = parents[index]

        path.reverse()

        return path

    # ========================================================
    # Result builder
    # ========================================================

    def _build_result(
        self,
        success,
        path,
        path_len,
        nodes,
        parents,
        iterations,
        collision_checks,
        planning_time,
        return_tree,
    ):

        result = {
            "success": success,

            "path": path,

            "path_length": path_len,

            "expanded_nodes":
                len(nodes),

            "tree_nodes":
                len(nodes),

            "iterations":
                iterations,

            "collision_checks":
                collision_checks,

            "planning_time":
                planning_time,
        }

        if return_tree:
            result["nodes"] = nodes
            result["parents"] = parents

        return result

    # ========================================================
    # Planner
    # ========================================================

    def plan(
        self,
        grid,
        start,
        goal,
        return_tree=True,
    ):

        start_time = (
            time.perf_counter()
        )

        rng = random.Random(
            self.seed
        )

        start_point = (
            float(start[0]) + 0.5,
            float(start[1]) + 0.5,
        )

        goal_point = (
            float(goal[0]) + 0.5,
            float(goal[1]) + 0.5,
        )

        nodes = [
            start_point
        ]

        parents = [
            None
        ]

        collision_checks = 0

        for iteration in range(
            1,
            self.max_iterations + 1,
        ):

            sample = self._sample(
                grid,
                goal_point,
                rng,
            )

            nearest_index = (
                self._nearest(
                    nodes,
                    sample,
                )
            )

            nearest_node = (
                nodes[nearest_index]
            )

            new_node = self._steer(
                nearest_node,
                sample,
            )

            collision_checks += 1

            if not edge_is_free(
                grid,
                nearest_node,
                new_node,
            ):
                continue

            nodes.append(
                new_node
            )

            parents.append(
                nearest_index
            )

            new_index = (
                len(nodes) - 1
            )

            distance_to_goal = (
                euclidean_distance(
                    new_node,
                    goal_point,
                )
            )

            if (
                distance_to_goal
                <= self.goal_radius
            ):

                collision_checks += 1

                if edge_is_free(
                    grid,
                    new_node,
                    goal_point,
                ):

                    nodes.append(
                        goal_point
                    )

                    parents.append(
                        new_index
                    )

                    goal_index = (
                        len(nodes) - 1
                    )

                    path = (
                        self._reconstruct_path(
                            nodes,
                            parents,
                            goal_index,
                        )
                    )

                    planning_time = (
                        time.perf_counter()
                        - start_time
                    )

                    return self._build_result(
                        success=True,
                        path=path,
                        path_len=path_length(path),
                        nodes=nodes,
                        parents=parents,
                        iterations=iteration,
                        collision_checks=collision_checks,
                        planning_time=planning_time,
                        return_tree=return_tree,
                    )

        planning_time = (
            time.perf_counter()
            - start_time
        )

        return self._build_result(
            success=False,
            path=[],
            path_len=math.inf,
            nodes=nodes,
            parents=parents,
            iterations=self.max_iterations,
            collision_checks=collision_checks,
            planning_time=planning_time,
            return_tree=return_tree,
        )