#!/usr/bin/env python3
import math

import cavarage as base

from cavarage import *  # noqa: F401,F403


INFO_GAIN_DISTANCE_GAIN = 0.35
INFO_GAIN_BEAM_SKIP = 3


def visible_unknown_count(known_map, cell):
    y, x = cell
    max_cells = int(LIDAR_RANGE / RES)
    visible_unknown = set()
    angles = np.linspace(-math.pi, math.pi, LIDAR_BEAMS)[::INFO_GAIN_BEAM_SKIP]

    for a in angles:
        ey = int(round(y + max_cells * math.sin(a)))
        ex = int(round(x + max_cells * math.cos(a)))

        for cy, cx in bresenham(y, x, ey, ex):
            if cy < 0 or cy >= H or cx < 0 or cx >= W:
                break

            cell_state = known_map[cy, cx]

            if cell_state == KNOWN_WALL:
                break

            if cell_state == UNKNOWN:
                visible_unknown.add((cy, cx))

    return len(visible_unknown)


def choose_frontier_goal_with_candidates(known_map, robot_cell):
    fs = ranked_frontier_candidates(known_map, robot_cell)

    if not fs:
        return None, [], []

    dist_map, came = bfs_distances(known_map, robot_cell)
    candidates = []

    for y, x, unknown_count in fs[:MAX_EVALUATED_FRONTIER_CANDIDATES]:
        goal = (y, x)
        dist = dist_map.get(goal)

        if dist is None:
            continue

        gain = visible_unknown_count(known_map, goal)
        utility = gain - INFO_GAIN_DISTANCE_GAIN * dist
        candidates.append((y, x, utility, gain, dist, unknown_count))

    if not candidates:
        return None, [], []

    candidates.sort(key=lambda v: v[2], reverse=True)
    best = candidates[0]
    goal = (best[0], best[1])
    path = rebuild_path(came, robot_cell, goal)

    return goal, path, candidates


def evaluated_frontier_candidates(known_map, robot_cell):
    _, _, candidates = choose_frontier_goal_with_candidates(known_map, robot_cell)

    return candidates


def choose_frontier_goal(known_map, robot_cell):
    goal, path, _ = choose_frontier_goal_with_candidates(known_map, robot_cell)

    return goal, path


def main():
    base.choose_frontier_goal_with_candidates = choose_frontier_goal_with_candidates
    base.evaluated_frontier_candidates = evaluated_frontier_candidates
    base.choose_frontier_goal = choose_frontier_goal
    base.main()


if __name__ == "__main__":
    main()
