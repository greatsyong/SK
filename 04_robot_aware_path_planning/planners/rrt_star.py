import math
import random
import time

from planners.geometry import (
    euclidean_distance,
    edge_is_free,
    path_length,
)


class RRTStarPlanner:

    def __init__(
        self,
        step_size=10.0,
        goal_radius=10.0,
        goal_sample_rate=0.05,
        max_iterations=20000,
        neighbor_radius=30.0,
        seed=None,
    ):
        self.name = "rrt_star"

        self.step_size = step_size
        self.goal_radius = goal_radius
        self.goal_sample_rate = goal_sample_rate
        self.max_iterations = max_iterations
        self.neighbor_radius = neighbor_radius
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

        # Goal bias
        if rng.random() < self.goal_sample_rate:
            return goal

        # Sample only from free cells
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
        return min(
            range(len(nodes)),
            key=lambda i:
                euclidean_distance(
                    nodes[i],
                    sample,
                ),
        )

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
    # Neighbor search
    # ========================================================

    def _near_indices(
        self,
        nodes,
        new_node,
    ):
        return [
            i
            for i, node in enumerate(nodes)
            if euclidean_distance(
                node,
                new_node,
            ) <= self.neighbor_radius
        ]

    # ========================================================
    # Descendant cost propagation
    # ========================================================

    def _propagate_costs(
        self,
        node_index,
        nodes,
        children,
        costs,
    ):
        """
        After rewiring a node, update the cost-to-come of all
        descendants so that

            J(child)
            =
            J(parent)
            +
            distance(parent, child)

        remains valid throughout the subtree.
        """

        for child_index in children[node_index]:

            costs[child_index] = (
                costs[node_index]
                + euclidean_distance(
                    nodes[node_index],
                    nodes[child_index],
                )
            )

            self._propagate_costs(
                child_index,
                nodes,
                children,
                costs,
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
    # Main planner
    # ========================================================

    def plan(
        self,
        grid,
        start,
        goal,
        return_tree=False,
    ):
        start_time = time.perf_counter()

        rng = random.Random(
            self.seed
        )

        # Use cell centers for continuous-space planning
        start_point = (
            float(start[0]) + 0.5,
            float(start[1]) + 0.5,
        )

        goal_point = (
            float(goal[0]) + 0.5,
            float(goal[1]) + 0.5,
        )

        # ----------------------------------------------------
        # Tree data
        # ----------------------------------------------------

        nodes = [
            start_point
        ]

        parents = [
            None
        ]

        children = [
            set()
        ]

        costs = [
            0.0
        ]

        # ----------------------------------------------------
        # Statistics
        # ----------------------------------------------------

        collision_checks = 0
        rewires = 0

        # ----------------------------------------------------
        # Goal tracking
        # ----------------------------------------------------

        # Any tree node that can connect directly to the goal
        # is stored here.
        goal_candidates = set()

        best_goal_parent = None
        best_goal_cost = math.inf

        first_solution_iteration = None
        first_solution_cost = math.inf

        # Stores:
        # (iteration, current best goal cost)
        convergence_history = []

        # ====================================================
        # Main RRT* loop
        # ====================================================

        for iteration in range(
            1,
            self.max_iterations + 1,
        ):

            # ------------------------------------------------
            # 1. Sample
            # ------------------------------------------------

            sample = self._sample(
                grid,
                goal_point,
                rng,
            )

            # ------------------------------------------------
            # 2. Find nearest node
            # ------------------------------------------------

            nearest_index = self._nearest(
                nodes,
                sample,
            )

            nearest_node = (
                nodes[nearest_index]
            )

            # ------------------------------------------------
            # 3. Steer toward sample
            # ------------------------------------------------

            new_node = self._steer(
                nearest_node,
                sample,
            )

            # ------------------------------------------------
            # 4. Check nearest -> new edge
            # ------------------------------------------------

            collision_checks += 1

            if not edge_is_free(
                grid,
                nearest_node,
                new_node,
            ):
                continue

            # ------------------------------------------------
            # 5. Find nearby nodes
            # ------------------------------------------------

            near_indices = (
                self._near_indices(
                    nodes,
                    new_node,
                )
            )

            # ------------------------------------------------
            # 6. Choose minimum-cost parent
            # ------------------------------------------------

            best_parent = nearest_index

            best_cost = (
                costs[nearest_index]
                + euclidean_distance(
                    nearest_node,
                    new_node,
                )
            )

            for i in near_indices:

                candidate_cost = (
                    costs[i]
                    + euclidean_distance(
                        nodes[i],
                        new_node,
                    )
                )

                if candidate_cost >= best_cost:
                    continue

                collision_checks += 1

                if edge_is_free(
                    grid,
                    nodes[i],
                    new_node,
                ):
                    best_parent = i
                    best_cost = candidate_cost

            # ------------------------------------------------
            # 7. Insert new node
            # ------------------------------------------------

            nodes.append(
                new_node
            )

            parents.append(
                best_parent
            )

            children.append(
                set()
            )

            costs.append(
                best_cost
            )

            new_index = (
                len(nodes) - 1
            )

            children[
                best_parent
            ].add(
                new_index
            )

            # ------------------------------------------------
            # 8. Rewire nearby nodes
            # ------------------------------------------------

            for i in near_indices:

                if i == best_parent:
                    continue

                # Root must never be rewired
                if parents[i] is None:
                    continue

                candidate_new_cost = (
                    costs[new_index]
                    + euclidean_distance(
                        new_node,
                        nodes[i],
                    )
                )

                if candidate_new_cost >= costs[i]:
                    continue

                collision_checks += 1

                if not edge_is_free(
                    grid,
                    new_node,
                    nodes[i],
                ):
                    continue

                old_parent = (
                    parents[i]
                )

                # Remove previous tree edge
                children[
                    old_parent
                ].discard(
                    i
                )

                # Install new tree edge
                parents[i] = (
                    new_index
                )

                children[
                    new_index
                ].add(
                    i
                )

                # Update this node's cost
                costs[i] = (
                    candidate_new_cost
                )

                # Update all descendants
                self._propagate_costs(
                    i,
                    nodes,
                    children,
                    costs,
                )

                rewires += 1

            # ------------------------------------------------
            # 9. Check whether this node can connect to goal
            # ------------------------------------------------

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
                    goal_candidates.add(
                        new_index
                    )

                    # Record the first feasible solution only once
                    if (
                        first_solution_iteration
                        is None
                    ):
                        first_solution_iteration = (
                            iteration
                        )

                        first_solution_cost = (
                            costs[new_index]
                            + distance_to_goal
                        )

            # ------------------------------------------------
            # 10. Re-evaluate best goal connection
            #
            # Important:
            # rewiring may reduce the cost of an existing
            # goal candidate or one of its ancestors.
            # Therefore best_goal_cost must be recomputed from
            # the current tree costs.
            # ------------------------------------------------

            if goal_candidates:

                best_goal_parent = min(
                    goal_candidates,
                    key=lambda i:
                        costs[i]
                        + euclidean_distance(
                            nodes[i],
                            goal_point,
                        ),
                )

                best_goal_cost = (
                    costs[best_goal_parent]
                    + euclidean_distance(
                        nodes[best_goal_parent],
                        goal_point,
                    )
                )

                convergence_history.append(
                    (
                        iteration,
                        best_goal_cost,
                    )
                )

        # ====================================================
        # Finalize result
        # ====================================================

        planning_time = (
            time.perf_counter()
            - start_time
        )

        # ----------------------------------------------------
        # No solution
        # ----------------------------------------------------

        if best_goal_parent is None:

            result = {
                "success": False,

                "path": [],

                "path_length":
                    math.inf,

                "expanded_nodes":
                    len(nodes),

                "tree_nodes":
                    len(nodes),

                "iterations":
                    self.max_iterations,

                "collision_checks":
                    collision_checks,

                "rewires":
                    rewires,

                "first_solution_iteration":
                    None,

                "first_solution_cost":
                    math.inf,

                "best_goal_cost":
                    math.inf,

                "planning_time":
                    planning_time,

                "convergence_history":
                    convergence_history,
            }

        # ----------------------------------------------------
        # Solution found
        # ----------------------------------------------------

        else:

            # Add goal as a final tree node
            nodes.append(
                goal_point
            )

            parents.append(
                best_goal_parent
            )

            children.append(
                set()
            )

            costs.append(
                best_goal_cost
            )

            goal_index = (
                len(nodes) - 1
            )

            children[
                best_goal_parent
            ].add(
                goal_index
            )

            path = (
                self._reconstruct_path(
                    nodes,
                    parents,
                    goal_index,
                )
            )

            final_path_length = (
                path_length(path)
            )

            result = {
                "success": True,

                "path":
                    path,

                "path_length":
                    final_path_length,

                "expanded_nodes":
                    len(nodes),

                "tree_nodes":
                    len(nodes),

                "iterations":
                    self.max_iterations,

                "collision_checks":
                    collision_checks,

                "rewires":
                    rewires,

                "first_solution_iteration":
                    first_solution_iteration,

                "first_solution_cost":
                    first_solution_cost,

                "best_goal_cost":
                    best_goal_cost,

                "planning_time":
                    planning_time,

                "convergence_history":
                    convergence_history,
            }

        # ----------------------------------------------------
        # Optional tree output
        # ----------------------------------------------------

        if return_tree:

            result["nodes"] = nodes
            result["parents"] = parents
            result["children"] = children
            result["costs"] = costs

        return result