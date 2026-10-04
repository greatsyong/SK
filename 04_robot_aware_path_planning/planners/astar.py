import heapq
import math
import time

from planners.base_planner import get_neighbors


def octile_distance(a, b):
    dx = abs(a[0] - b[0])
    dy = abs(a[1] - b[1])

    return (
        max(dx, dy)
        + (math.sqrt(2.0) - 1.0) * min(dx, dy)
    )


class AStarPlanner:

    def __init__(self):
        self.name = "astar"

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

        g_cost = {
            start: 0.0
        }

        parent = {
            start: None
        }

        h0 = octile_distance(start, goal)

        heapq.heappush(
            frontier,
            (h0, 0.0, start[0], start[1])
        )

        expanded_nodes = 0

        while frontier:

            f_current, g_current, x, y = heapq.heappop(frontier)

            current = (x, y)

            if g_current > g_cost[current]:
                continue

            expanded_nodes += 1

            if current == goal:
                break

            for nx, ny, move_cost in get_neighbors(grid, x, y):

                neighbor = (nx, ny)

                tentative_g = g_current + move_cost

                if (
                    neighbor not in g_cost
                    or tentative_g < g_cost[neighbor]
                ):

                    g_cost[neighbor] = tentative_g
                    parent[neighbor] = current

                    h = octile_distance(
                        neighbor,
                        goal
                    )

                    f = tentative_g + h

                    heapq.heappush(
                        frontier,
                        (
                            f,
                            tentative_g,
                            nx,
                            ny
                        )
                    )

        planning_time = time.perf_counter() - start_time

        if goal not in g_cost:
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
            "path_length": g_cost[goal],
            "expanded_nodes": expanded_nodes,
            "planning_time": planning_time,
        }