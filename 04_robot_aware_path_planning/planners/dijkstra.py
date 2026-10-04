import heapq
import math
import time

from planners.base_planner import get_neighbors


class DijkstraPlanner:

    def __init__(self):
        self.name = "dijkstra"

    def plan(self, grid, start, goal):

        start_time = time.perf_counter()

        if start == goal:
            return {
                "success": True,
                "path": [start],
                "path_length": 0.0,
                "expanded_nodes": 0,
                "planning_time": time.perf_counter() - start_time,
            }

        frontier = []

        # heap item:
        # (cost_from_start, x, y)
        heapq.heappush(
            frontier,
            (0.0, start[0], start[1])
        )

        cost_to_come = {
            start: 0.0
        }

        parent = {
            start: None
        }

        expanded_nodes = 0

        while frontier:

            current_cost, x, y = heapq.heappop(frontier)
            current = (x, y)

            # Ignore stale priority-queue entries.
            if current_cost > cost_to_come[current]:
                continue

            expanded_nodes += 1

            if current == goal:
                break

            for nx, ny, move_cost in get_neighbors(grid, x, y):

                neighbor = (nx, ny)

                new_cost = current_cost + move_cost

                if (
                    neighbor not in cost_to_come
                    or new_cost < cost_to_come[neighbor]
                ):
                    cost_to_come[neighbor] = new_cost
                    parent[neighbor] = current

                    heapq.heappush(
                        frontier,
                        (new_cost, nx, ny)
                    )

        planning_time = time.perf_counter() - start_time

        if goal not in cost_to_come:
            return {
                "success": False,
                "path": [],
                "path_length": math.inf,
                "expanded_nodes": expanded_nodes,
                "planning_time": planning_time,
            }

        path = []

        current = goal

        while current is not None:
            path.append(current)
            current = parent[current]

        path.reverse()

        return {
            "success": True,
            "path": path,
            "path_length": cost_to_come[goal],
            "expanded_nodes": expanded_nodes,
            "planning_time": planning_time,
        }